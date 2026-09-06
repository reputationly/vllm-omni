#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Start every production GPUStack deployment against a candidate image.

Answers one question before a base-image or upstream-merge change reaches
production: does each currently-deployed model still come up? It reproduces the
launch GPUStack actually performs -- same weights, same backend parameters, same
env -- rather than a hand-written approximation, because the failures worth
catching here (a moved import, a dropped quantization format, a deploy-config
key that changed meaning) only appear on the real argv.

The model list and every launch parameter come from the GPUStack postgres
`models` table, exported to JSON beforehand:

  ssh manager 'docker exec -u postgres gpustack-server psql -d gpustack -At -c "
    select json_agg(row_to_json(t)) from (
      select name, replicas, local_path, backend_parameters, env
      from models where backend=''vLLMOmni'' and replicas > 0) t;"' > prod_models.json

Usage:

  python tools/validation/run_prod_startup_matrix.py prod_models.json \
      --image <acr>/vllm-omni:<tag> --hosts gpu41,gpu42,...
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Bind-mounts every production instance has (from `docker inspect` of a live
# instance). /deploy-configs is NOT here: it is baked into the image, so a
# deploy-config that a merge silently renamed fails here exactly as it would in
# production rather than being masked by a host mount.
MOUNTS = ["/nfs-models:/nfs-models", "/nfs-output:/nfs-output", "/nfs-data:/nfs-data"]

# Set by the GPUStack worker on every instance, not by the model row.
BASE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_XET": "1",
    "DIFFUSION_ATTENTION_BACKEND": "FLASH_ATTN",
}

PORT = 40099


def ssh(host: str, command: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", host, command],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def build_run_command(model: dict, image: str, container: str) -> str:
    # backend_parameters rows are inconsistent: some entries are one flag per
    # element ("--num-gpus", "2"), others pack flag and value into a single
    # string ("--residency-config /path"). shlex.split over the joined string
    # normalises both without disturbing quoted values.
    params = shlex.split(" ".join(model["backend_parameters"] or []))

    env_args: list[str] = []
    for key, value in {**BASE_ENV, **(model["env"] or {})}.items():
        env_args += ["-e", f"{key}={value}"]

    mount_args: list[str] = []
    for mount in MOUNTS:
        mount_args += ["-v", mount]

    argv = [
        "docker",
        "run",
        "-d",
        "--name",
        container,
        "--gpus",
        "all",
        "--shm-size=64g",
        "--network",
        "host",
        *mount_args,
        *env_args,
        image,
        "vllm",
        "serve",
        model["local_path"],
        "--omni",
        *params,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--trust-remote-code",
    ]
    return " ".join(shlex.quote(a) for a in argv)


def launch(host: str, model: dict, image: str) -> tuple[str, str, str]:
    container = f"val-{model['name'].replace('.', '-')}"
    ssh(host, f"docker rm -f {container} >/dev/null 2>&1; true")
    result = ssh(host, build_run_command(model, image, container), timeout=180)
    if result.returncode != 0:
        return model["name"], "LAUNCH_FAILED", result.stderr.strip()[-400:]
    return model["name"], "STARTED", container


def wait_ready(host: str, model: dict, container: str, deadline_s: int) -> tuple[str, str]:
    """Poll until the server answers, the container dies, or we run out of time.

    A dead container is checked before the timeout so a model that crashes in
    30s is reported in 30s, not after the full init budget that H3-class models
    legitimately need.
    """
    started = time.monotonic()
    while time.monotonic() - started < deadline_s:
        health = ssh(host, f"curl -s -m 5 -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{PORT}/health")
        if health.stdout.strip() == "200":
            return "READY", f"{int(time.monotonic() - started)}s"

        alive = ssh(host, f"docker inspect -f '{{{{.State.Running}}}}' {container} 2>/dev/null")
        if alive.stdout.strip() != "true":
            logs = ssh(host, f"docker logs --tail 60 {container} 2>&1")
            return "DIED", logs.stdout.strip()[-3000:]
        time.sleep(20)

    logs = ssh(host, f"docker logs --tail 40 {container} 2>&1")
    return "TIMEOUT", logs.stdout.strip()[-2000:]


def run_one(host: str, model: dict, image: str, deadline_s: int) -> dict:
    name, status, detail = launch(host, model, image)
    if status != "STARTED":
        return {"model": name, "host": host, "status": status, "detail": detail}
    status, detail = wait_ready(host, model, detail, deadline_s)
    return {"model": name, "host": host, "status": status, "detail": detail}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models_json", type=Path)
    parser.add_argument("--image", required=True)
    parser.add_argument("--hosts", required=True, help="comma-separated, one model per host")
    parser.add_argument("--only", help="comma-separated model names to run instead of all")
    parser.add_argument(
        "--deadline",
        type=int,
        default=2700,
        help="seconds to wait per model; H3 deploy-configs set --init-timeout 2400 (default: 2700)",
    )
    parser.add_argument("--out", type=Path, default=Path("startup_matrix_result.json"))
    args = parser.parse_args()

    models = json.loads(args.models_json.read_text())
    if args.only:
        wanted = set(args.only.split(","))
        models = [m for m in models if m["name"] in wanted]
    hosts = args.hosts.split(",")
    if len(models) > len(hosts):
        raise SystemExit(f"{len(models)} models but only {len(hosts)} hosts; run in batches or pass --only")

    pairs = list(zip(hosts, models))
    print(f"启动 {len(pairs)} 个，每机一个，镜像 {args.image}\n")
    for host, model in pairs:
        print(f"  {host:<7} {model['name']}")
    print()

    with ThreadPoolExecutor(max_workers=len(pairs)) as pool:
        results = list(pool.map(lambda p: run_one(p[0], p[1], args.image, args.deadline), pairs))

    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n{'模型':<26}{'机器':<8}{'结果':<10}")
    for r in sorted(results, key=lambda x: x["status"] != "READY"):
        print(f"{r['model']:<26}{r['host']:<8}{r['status']:<10}{r['detail'] if r['status'] == 'READY' else ''}")

    failed = [r for r in results if r["status"] != "READY"]
    print(f"\n就绪 {len(results) - len(failed)}/{len(results)}；明细见 {args.out}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
