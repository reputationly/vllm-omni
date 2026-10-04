#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Fleet-wide GPU co-tenancy audit for the GPUStack A100 fleet.

Answers one question per GPU: *is the only thing running on this card the thing
GPUStack says is running on it?*

A card that hosts a process GPUStack does not know about is the failure mode
behind the "CUDA out of memory. Tried to allocate 128.00 MiB" class of report,
where the failing process holds only a fraction of the card and the real
occupant is a neighbour. See ``docs/`` for the worked example that motivated
this tool: a hand-started validation container sharing four cards with a
production instance, costing 21.4 GiB per card.

Three findings are reported:

``COLLISION``
    More than one compute process on a single card. Always worth a look.
``UNDECLARED``
    A compute process on a card that no GPUStack instance claims.
``MISSING``
    GPUStack says RUNNING but no process is on the card. Usually a dead or
    never-started instance.

Usage::

    python tools/gpu_cotenant_audit.py                    # full fleet
    python tools/gpu_cotenant_audit.py --hosts 33 34 35  # subset
    python tools/gpu_cotenant_audit.py --manager mgr      # override manager alias
    python tools/gpu_cotenant_audit.py --json             # machine-readable

Requires passwordless SSH to the manager node and to every GPU host alias
(``gpu1``..``gpuNN``) listed in ``~/.ssh/config``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

SSH_OPTS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "ConnectTimeout=12",
]

# Pulled from the manager. GPUStack keeps model state in PostgreSQL; the
# container's postgres OS user authenticates over a peer socket, so the query
# has to run inside the container as that user rather than over TCP.
INSTANCE_QUERY = (
    "select mi.name, mi.state, mi.worker_name, mi.port, "
    "coalesce(mi.gpu_indexes::text, '-') "
    "from model_instances mi where mi.deleted_at is null"
)

# Emitted as one tab-free line per (gpu, pid, mem, container, cmd) so the caller
# never has to guess field boundaries.
PROBE = r"""
echo "HOSTOK|$(hostname)"
for i in 0 1 2 3; do
  nvidia-smi -i $i --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null |
  while IFS=, read pid mem; do
    pid=$(echo $pid | tr -d ' ')
    [ -z "$pid" ] && continue
    cg=$(grep -oE '[0-9a-f]{64}' /proc/$pid/cgroup 2>/dev/null | head -1)
    if [ -n "$cg" ]; then
      cid=$(echo $cg | cut -c1-12)
      cname=$(docker inspect -f "{{.Name}}" "$cid" 2>/dev/null | sed 's|^/||')
    else
      cname="BARE(no-container)"
    fi
    cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | cut -c1-60)
    echo "PROC|$i|$pid|$(echo $mem | tr -d ' ')|${cname:-?}|${cmd}"
  done
done
"""


@dataclass
class Proc:
    gpu: int
    pid: int
    mem_mib: int
    container: str
    cmd: str


@dataclass
class Instance:
    name: str
    state: str
    worker: str
    port: str
    gpus: list[int] = field(default_factory=list)


def ssh(host: str, script: str, timeout: int = 60) -> str:
    """Run *script* on *host*; return stdout, or "" if the host is unreachable."""
    try:
        proc = subprocess.run(
            ["ssh", *SSH_OPTS, host, "bash -s"],
            input=script,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ""
    return proc.stdout


def parse_gpu_indexes(raw: str) -> list[int]:
    """Parse a PostgreSQL ``int[]`` literal such as ``[0, 1, 2, 3]``.

    The elements carry no padding of their own, but ``split(",")`` on the
    bracketed form leaves a leading space on every element after the first
    (``" 1"``). Stripping is not optional here: an unstripped " 1" silently
    stops matching the "GPU1" keys built from the probe, and the result is a
    full-page of phantom UNDECLARED findings.
    """
    cleaned = raw.strip().strip("[]{}")
    if not cleaned:
        return []
    out = []
    for part in cleaned.split(","):
        part = part.strip()  # <- the load-bearing strip
        if part.isdigit():
            out.append(int(part))
    return out


def fetch_instances(manager: str) -> list[Instance]:
    sql = f'docker exec -u postgres gpustack-server psql -d gpustack -F"|" -Atc "{INSTANCE_QUERY}"'
    raw = ssh(manager, sql)
    if not raw.strip():
        sys.exit(f"error: could not read GPUStack instances from {manager!r}")

    instances: list[Instance] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 5:
            continue
        name, state, worker, port, gpus_raw = parts[:5]
        instances.append(
            Instance(
                name=name,
                state=state,
                worker=worker,
                port=port,
                gpus=parse_gpu_indexes(gpus_raw),
            )
        )
    return instances


def fetch_procs(host: str) -> tuple[bool, list[Proc]]:
    """Return ``(reachable, procs)`` for *host*.

    The two must stay distinguishable: a reachable host whose cards are empty
    is the healthy post-cleanup state, while an empty string from ssh means the
    host never answered. Collapsing them reports every drained machine as
    unreachable.
    """
    raw = ssh(host, PROBE)
    reachable = any(line.startswith("HOSTOK|") for line in raw.splitlines())
    procs: list[Proc] = []
    for line in raw.splitlines():
        if not line.startswith("PROC|"):
            continue
        _, gpu, pid, mem, container, cmd = line.split("|", 5)
        try:
            procs.append(
                Proc(
                    int(gpu),
                    int(pid),
                    int(mem.replace("MiB", "").strip()),
                    container,
                    cmd,
                )
            )
        except ValueError:
            continue
    return reachable, procs


def host_alias(worker_name: str) -> str:
    """``dev-gpustack-a100-0033`` -> ``gpu33`` (matches the ssh config aliases).

    ``int()`` matters: the numeric suffix is zero-padded in the worker name, so
    a plain slice would yield ``gpu0033`` and match no ssh alias.
    """
    m = re.search(r"(\d+)$", worker_name)
    return f"gpu{int(m.group(1))}" if m else worker_name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manager", default="mgr", help="ssh alias of the GPUStack manager (default: mgr)")
    ap.add_argument("--hosts", nargs="*", help="only audit these gpu aliases (default: all referenced hosts)")
    ap.add_argument("--jobs", type=int, default=8, help="parallel ssh workers (default: 8)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a text report")
    args = ap.parse_args()

    instances = fetch_instances(args.manager)

    expected: dict[tuple[str, int], list[str]] = defaultdict(list)
    for inst in instances:
        alias = host_alias(inst.worker)
        for gpu in inst.gpus:
            expected[(alias, gpu)].append(f"{inst.name}({inst.state})")

    hosts = sorted(args.hosts) if args.hosts else sorted({h for h, _ in expected})
    if args.hosts:
        # Audit a subset means the untouched slots are out of scope, not broken.
        scope = set(hosts)
        expected = {k: v for k, v in expected.items() if k[0] in scope}

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = dict(zip(hosts, pool.map(fetch_procs, hosts)))

    actual: dict[tuple[str, int], list[Proc]] = defaultdict(list)
    unreachable = [h for h in hosts if not results[h][0]]
    for host, (_reachable, procs) in results.items():
        for p in procs:
            actual[(host, p.gpu)].append(p)

    # Only distinct containers on one card are a collision. A single instance
    # routinely runs several processes per card (EngineCore beside its
    # DiffusionWorker for the TTS/music models), and that is by design — so
    # counting processes would bury the real signal under routine noise.
    collisions = [
        {"host": h, "gpu": g, "procs": [p.__dict__ for p in ps]}
        for (h, g), ps in sorted(actual.items())
        if len({p.container for p in ps}) > 1
    ]
    undeclared = [
        {"host": h, "gpu": g, "procs": [p.__dict__ for p in ps]}
        for (h, g), ps in sorted(actual.items())
        if (h, g) not in expected
    ]
    missing = [
        {"host": h, "gpu": g, "instances": expected[(h, g)]} for (h, g) in sorted(expected) if (h, g) not in actual
    ]

    report = {
        "instances": len(instances),
        "hosts_audited": len(hosts),
        "expected_slots": len(expected),
        "occupied_slots": len(actual),
        "collision": collisions,
        "undeclared": undeclared,
        "missing": missing,
        "unreachable": unreachable,
    }

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1 if (collisions or undeclared or missing) else 0

    print(f"GPUStack 实例: {len(instances)}  审计机器: {len(hosts)}")
    print(f"期望占位 {len(expected)}  实际有进程 {len(actual)}")
    print()

    def section(title: str, rows: list[dict], fmt) -> None:
        print(f"── {title}: {len(rows)} ──")
        for row in rows:
            print("   " + fmt(row))
        if not rows:
            print("   (无)")
        print()

    def slot(row: dict) -> str:
        return f"{row['host']} GPU{row['gpu']}"

    def procs_fmt(row: dict) -> str:
        detail = ", ".join(f"{p['container']}({p['mem_mib']}MiB)" for p in row["procs"])
        return f"{slot(row)} -> {detail}"

    def inst_fmt(row: dict) -> str:
        return f"{slot(row)} -> {', '.join(row['instances'])}"

    section("COLLISION 单卡多进程", collisions, procs_fmt)
    section("UNDECLARED 有进程但 GPUStack 未声明", undeclared, procs_fmt)
    section("MISSING 声明 RUNNING 但卡上无进程", missing, inst_fmt)

    if unreachable:
        print(f"── 不可达 ──\n   {', '.join(unreachable)}\n")

    return 1 if (collisions or undeclared or missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
