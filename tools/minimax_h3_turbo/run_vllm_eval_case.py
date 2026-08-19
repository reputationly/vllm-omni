#!/usr/bin/env python3
"""Run one vLLM-Omni H3 video request and persist reproducible measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any


def http_json(url: str, *, body: dict[str, Any] | None = None, timeout: float = 30) -> dict[str, Any]:
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def gpu_sample() -> list[dict[str, float | int]]:
    query = "index,memory.used,memory.total,utilization.gpu,power.draw"
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        text=True,
    )
    rows = []
    for line in output.splitlines():
        index, used, total, utilization, power = (item.strip() for item in line.split(","))
        rows.append(
            {
                "index": int(index),
                "memory_used_mib": int(used),
                "memory_total_mib": int(total),
                "utilization_percent": int(utilization),
                "power_watts": float(power),
            }
        )
    return rows


def host_memory() -> dict[str, float]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.strip().split()[0])
    return {
        "used_gib": (values["MemTotal"] - values["MemAvailable"]) / 1024 / 1024,
        "available_gib": values["MemAvailable"] / 1024 / 1024,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def container_log_count(container: str, marker: str) -> int:
    output = subprocess.check_output(["docker", "logs", container], stderr=subprocess.STDOUT, text=True)
    return output.count(marker)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:42080")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-output/h3_turbo_eval/results"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--container", help="Engine container used to attribute denoise log lines.")
    parser.add_argument("--poll-seconds", type=float, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=7200)
    args = parser.parse_args()

    case_dir = args.output_root / args.tag
    case_dir.mkdir(parents=True, exist_ok=False)
    output_path = case_dir / "output.mp4"
    request_body = json.loads(args.request.read_text(encoding="utf-8"))
    request_body["save_result_path"] = str(output_path)
    (case_dir / "request.json").write_text(
        json.dumps(request_body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    models = http_json(f"{args.endpoint}/v1/models")
    (case_dir / "models.json").write_text(json.dumps(models, indent=2) + "\n")

    log_marker = "rows video="
    log_count_before = container_log_count(args.container, log_marker) if args.container else None
    samples: list[dict[str, Any]] = []
    stop = threading.Event()
    started = time.monotonic()

    def sample_loop() -> None:
        while not stop.is_set():
            try:
                samples.append(
                    {
                        "elapsed_seconds": time.monotonic() - started,
                        "gpu": gpu_sample(),
                        "host_memory": host_memory(),
                    }
                )
            except Exception as exc:  # Measurement must not abort generation.
                samples.append({"elapsed_seconds": time.monotonic() - started, "sample_error": repr(exc)})
            stop.wait(args.poll_seconds)

    sampler = threading.Thread(target=sample_loop, daemon=True)
    sampler.start()
    progress: list[dict[str, Any]] = []
    try:
        submit = http_json(f"{args.endpoint}/v1/tasks/video/", body=request_body, timeout=60)
        task_id = submit.get("task_id")
        if not task_id:
            raise RuntimeError(f"submission returned no task_id: {submit}")
        deadline = started + args.timeout_seconds
        status: dict[str, Any] = {}
        while time.monotonic() < deadline:
            status = http_json(f"{args.endpoint}/v1/tasks/{task_id}/status", timeout=30)
            progress.append({"elapsed_seconds": time.monotonic() - started, "status": status})
            if status.get("status") in {"completed", "failed", "cancelled"}:
                break
            time.sleep(args.poll_seconds)
        else:
            raise TimeoutError(f"task {task_id} exceeded {args.timeout_seconds}s")
    except Exception as exc:
        status = {"status": "runner_error", "error": repr(exc)}
    finally:
        stop.set()
        sampler.join(timeout=args.poll_seconds + 2)

    elapsed = time.monotonic() - started
    (case_dir / "progress.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n")
    with (case_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample) + "\n")

    gpu_rows = [gpu for sample in samples for gpu in sample.get("gpu", [])]
    summary: dict[str, Any] = {
        "tag": args.tag,
        "task_id": locals().get("task_id"),
        "status": status,
        "wall_seconds": elapsed,
        "sample_count": len(samples),
        "peak_gpu_memory_mib": max((row["memory_used_mib"] for row in gpu_rows), default=None),
        "peak_host_memory_gib": max(
            (sample["host_memory"]["used_gib"] for sample in samples if "host_memory" in sample), default=None
        ),
    }
    if args.container:
        log_count_after = container_log_count(args.container, log_marker)
        summary["engine_log"] = {
            "container": args.container,
            "row_lines_before": log_count_before,
            "row_lines_after": log_count_after,
            "new_row_lines": log_count_after - int(log_count_before or 0),
        }
    if output_path.is_file():
        summary["output"] = {
            "path": str(output_path),
            "size_bytes": output_path.stat().st_size,
            "sha256": sha256(output_path),
        }
    (case_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if status.get("status") != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
