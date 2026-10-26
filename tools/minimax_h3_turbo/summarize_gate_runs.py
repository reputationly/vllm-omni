# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Summarize a gate-suite baseline run directory into a markdown table.

Two things this checks that a plain ``ls`` cannot:

* **Provenance** — every run records ``prompt_sha256``; it is compared against the
  current case file, so a result produced by an older prompt revision is flagged
  instead of silently entering the baseline.
* **Queue contamination** — the gateway serialises requests (``max_num_seqs=1``),
  so a wall time can be several times the compute time. Runs are compared against
  the median for their clip length and anything above the factor is marked as
  queue-affected rather than reported as a performance number.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any

QUEUE_FACTOR = 1.25


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def case_id_from_tag(tag: str) -> tuple[str, int | None]:
    match = re.match(r"^prod_(.+?)(?:_s(\d+))?$", tag)
    if not match:
        return tag, None
    return match.group(1), int(match.group(2)) if match.group(2) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--queue-factor", type=float, default=QUEUE_FACTOR)
    parser.add_argument(
        "--scores", type=Path, default=None, help="optional JSON of subjective scores keyed by run directory name"
    )
    args = parser.parse_args()

    cases: dict[str, dict[str, Any]] = {}
    for path in sorted(args.suite_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        cases[path.stem] = doc

    scores: dict[str, Any] = {}
    if args.scores and args.scores.is_file():
        scores = json.loads(args.scores.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    for run_dir in sorted(p for p in args.runs_dir.iterdir() if p.is_dir()):
        summary_path = run_dir / "summary.json"
        request_path = run_dir / "request.json"
        if not summary_path.is_file() or not request_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        request = json.loads(request_path.read_text(encoding="utf-8"))
        cid, seed = case_id_from_tag(run_dir.name)
        case = cases.get(cid)
        recorded = request.get("prompt_sha256")
        expected = sha(case["prompt"]) if case else None
        if recorded is None:
            provenance = "无哈希"
        elif expected is None:
            provenance = "无对应用例"
        elif recorded == expected:
            provenance = "一致"
        else:
            provenance = "陈旧"
        rows.append(
            {
                "tag": run_dir.name,
                "case": cid,
                "task": (case or {}).get("extra_params", {}).get("task", "?"),
                "seconds": int(str((case or request).get("seconds", "0"))),
                "seed": seed,
                "status": summary.get("status"),
                "wall": float(summary.get("wall_seconds") or 0.0),
                "provenance": provenance,
                "bytes": (summary.get("output") or {}).get("bytes"),
                "score": scores.get(run_dir.name) or scores.get(run_dir.name.removeprefix("prod_")) or {},
            }
        )

    by_len: dict[int, list[float]] = {}
    for row in rows:
        if row["status"] == "completed" and row["provenance"] == "一致":
            by_len.setdefault(row["seconds"], []).append(row["wall"])
    # The floor is the cleanest estimate of compute time: the gateway serialises
    # requests, so every sample above it carries some queue wait.
    floors = {k: min(v) for k, v in by_len.items() if v}
    medians = {k: statistics.median(v) for k, v in by_len.items() if v}

    lines = [
        "# 现网基线（prod-v2）",
        "",
        f"来源目录：`{args.runs_dir}`；队列判定阈值：中位数 ×{args.queue_factor:g}",
        "",
        "| 用例 | 任务 | 时长 | seed | 状态 | 墙钟 s | 相对最快 | 来源 | 幻影 | 结构 | 文字 | 音频 | 备注 |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | --- | :-: | :-: | :-: | ---: | --- |",
    ]
    for row in rows:
        med = floors.get(row["seconds"])
        ratio = (row["wall"] / med) if med else None
        note = ""
        if row["status"] != "completed":
            note = "未完成"
        elif row["provenance"] != "一致":
            note = "不可追溯，不纳入基线"
        elif ratio and ratio > args.queue_factor:
            note = f"含排队（最快 {med:.0f} s 的 {ratio:.1f}×）"
        else:
            note = "计算时间"
        sc = row["score"]

        def g(key: str) -> str:
            value = sc.get(key)
            return "-" if value is None else str(value)

        audio = sc.get("audio")
        db = sc.get("audio_db")
        if audio is None and db is None:
            audio_cell = "-"
        elif audio is None:
            audio_cell = f"{db} dB"
        elif isinstance(audio, str):
            audio_cell = f"{audio} ({db} dB)" if db is not None else audio
        else:
            audio_cell = f"{audio} ({db} dB)"
        lines.append(
            f"| `{row['case']}` | {row['task']} | {row['seconds']}s | {row['seed'] or '-'} | "
            f"{row['status']} | {row['wall']:.0f} | "
            f"{'-' if ratio is None else f'{ratio:.2f}×'} | {row['provenance']} | "
            f"{g('ghost')} | {g('structure')} | {g('text')} | {audio_cell} | {note} |"
        )

    completed = [r for r in rows if r["status"] == "completed" and r["provenance"] == "一致"]
    lines += [
        "",
        "## 汇总",
        "",
        f"- 运行总数：{len(rows)}；完成且可追溯：{len(completed)}；"
        f"失败/跳过：{sum(1 for r in rows if r['status'] != 'completed')}；"
        f"陈旧：{sum(1 for r in rows if r['provenance'] == '陈旧')}",
        "- 各时长最快 / 中位墙钟：" + "；".join(f"{k}s → {floors[k]:.0f}s / {medians[k]:.0f}s" for k in sorted(floors)),
        "",
    ]
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:6]))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
