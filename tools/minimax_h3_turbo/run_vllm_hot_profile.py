#!/usr/bin/env python3
"""Run one warmup plus one measured MiniMax-H3 request against a TP engine."""

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


def _http_json(url: str, *, body: dict[str, Any] | None = None, timeout: float = 30) -> dict[str, Any]:
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _gpu_sample() -> list[dict[str, float | int]]:
    query = "index,memory.used,memory.total,utilization.gpu,power.draw"
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        text=True,
    )
    result = []
    for line in output.splitlines():
        index, used, total, utilization, power = (item.strip() for item in line.split(","))
        result.append(
            {
                "index": int(index),
                "memory_used_mib": int(used),
                "memory_total_mib": int(total),
                "utilization_percent": int(utilization),
                "power_watts": float(power),
            }
        )
    return result


def _host_memory() -> dict[str, float]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.strip().split()[0])
    return {
        "used_gib": (values["MemTotal"] - values["MemAvailable"]) / 1024**2,
        "available_gib": values["MemAvailable"] / 1024**2,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _container_logs(container: str) -> str:
    return subprocess.check_output(["docker", "logs", container], stderr=subprocess.STDOUT).decode(
        "utf-8", errors="replace"
    )


def _phase_durations(progress: list[dict[str, Any]], total: float) -> dict[str, float]:
    changes: list[tuple[float, str]] = []
    for entry in progress:
        phase = str(entry["status"].get("phase") or "unknown")
        if not changes or changes[-1][1] != phase:
            changes.append((float(entry["elapsed_seconds"]), phase))
    durations: dict[str, float] = {}
    for index, (started, phase) in enumerate(changes):
        ended = changes[index + 1][0] if index + 1 < len(changes) else total
        durations[phase] = durations.get(phase, 0.0) + max(0.0, ended - started)
    return durations


def _resource_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_gpu: dict[int, list[tuple[float, dict[str, Any]]]] = {}
    for sample in samples:
        elapsed = float(sample["elapsed_seconds"])
        for gpu in sample.get("gpus", []):
            by_gpu.setdefault(int(gpu["index"]), []).append((elapsed, gpu))
    per_gpu = {}
    for index, rows in sorted(by_gpu.items()):
        energy_ws = 0.0
        for (left_t, left), (right_t, right) in zip(rows, rows[1:]):
            energy_ws += (right_t - left_t) * (float(left["power_watts"]) + float(right["power_watts"])) / 2
        per_gpu[str(index)] = {
            "peak_memory_mib": max(row["memory_used_mib"] for _, row in rows),
            "mean_utilization_percent": sum(row["utilization_percent"] for _, row in rows) / len(rows),
            "mean_power_watts": sum(row["power_watts"] for _, row in rows) / len(rows),
            "estimated_energy_wh": energy_ws / 3600,
        }
    host_rows = [sample["host_memory"] for sample in samples if "host_memory" in sample]
    return {
        "per_gpu": per_gpu,
        "sum_peak_memory_mib": sum(row["peak_memory_mib"] for row in per_gpu.values()),
        "max_card_peak_memory_mib": max((row["peak_memory_mib"] for row in per_gpu.values()), default=None),
        "sum_estimated_energy_wh": sum(row["estimated_energy_wh"] for row in per_gpu.values()),
        "peak_host_used_gib": max((row["used_gib"] for row in host_rows), default=None),
        "min_host_available_gib": min((row["available_gib"] for row in host_rows), default=None),
    }


def _run_request(
    *,
    endpoint: str,
    body: dict[str, Any],
    poll_seconds: float,
    timeout_seconds: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], float]:
    started = time.monotonic()
    stop = threading.Event()
    samples: list[dict[str, Any]] = []

    def sample_loop() -> None:
        while not stop.is_set():
            try:
                samples.append(
                    {
                        "elapsed_seconds": time.monotonic() - started,
                        "gpus": _gpu_sample(),
                        "host_memory": _host_memory(),
                    }
                )
            except Exception as exc:
                samples.append({"elapsed_seconds": time.monotonic() - started, "sample_error": repr(exc)})
            stop.wait(1.0)

    sampler = threading.Thread(target=sample_loop, daemon=True)
    sampler.start()
    progress = []
    status: dict[str, Any] = {}
    try:
        submitted = _http_json(f"{endpoint}/v1/tasks/video/", body=body, timeout=60)
        task_id = submitted.get("task_id")
        if not task_id:
            raise RuntimeError(f"submission returned no task_id: {submitted}")
        deadline = started + timeout_seconds
        while time.monotonic() < deadline:
            status = _http_json(f"{endpoint}/v1/tasks/{task_id}/status", timeout=30)
            progress.append({"elapsed_seconds": time.monotonic() - started, "status": status})
            if status.get("status") in {"completed", "failed", "cancelled"}:
                break
            time.sleep(poll_seconds)
        else:
            raise TimeoutError(f"task {task_id} exceeded {timeout_seconds}s")
    finally:
        stop.set()
        sampler.join(timeout=3)
    elapsed = time.monotonic() - started
    if status.get("status") != "completed":
        raise RuntimeError(f"video request did not complete: {status}")
    return status, progress, samples, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--expected-nfe", type=int, required=True)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=7200)
    args = parser.parse_args()

    case_dir = args.output_root / args.tag
    case_dir.mkdir(parents=True, exist_ok=False)
    request = json.loads(args.request.read_text(encoding="utf-8"))
    if int(request.get("num_inference_steps", -1)) != args.expected_nfe:
        raise ValueError("request num_inference_steps does not match --expected-nfe")
    (case_dir / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n")
    (case_dir / "models.json").write_text(json.dumps(_http_json(f"{args.endpoint}/v1/models"), indent=2) + "\n")

    phases = {}
    for phase in ("warmup", "hot"):
        body = dict(request)
        output_path = case_dir / f"{phase}.mp4"
        body["save_result_path"] = str(output_path)
        logs_before = _container_logs(args.container)
        status, progress, samples, elapsed = _run_request(
            endpoint=args.endpoint,
            body=body,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
        )
        logs_after = _container_logs(args.container)
        new_logs = logs_after[len(logs_before) :] if logs_after.startswith(logs_before) else logs_after
        (case_dir / f"{phase}.engine.log").write_text(new_logs, encoding="utf-8")
        (case_dir / f"{phase}.progress.json").write_text(
            json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (case_dir / f"{phase}.resource_samples.json").write_text(
            json.dumps(samples, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        row_lines = new_logs.count("rows video=")
        expected_row_lines = args.expected_nfe * args.tp_size
        if row_lines != expected_row_lines:
            raise RuntimeError(
                f"{phase} NFE evidence mismatch: rows lines={row_lines}, expected={expected_row_lines} "
                f"({args.expected_nfe} NFE * TP{args.tp_size})"
            )
        phases[phase] = {
            "status": status,
            "wall_seconds": elapsed,
            "phase_seconds_from_polling": _phase_durations(progress, elapsed),
            "resource": _resource_summary(samples),
            "nfe_evidence": {
                "row_lines": row_lines,
                "tp_size": args.tp_size,
                "actual_nfe": row_lines // args.tp_size,
            },
            "output": {
                "path": str(output_path),
                "size_bytes": output_path.stat().st_size,
                "sha256": _sha256(output_path),
            },
        }

    summary = {"tag": args.tag, "measurement_mode": "same_process_full_shape_warmup_then_hot", "phases": phases}
    (case_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
