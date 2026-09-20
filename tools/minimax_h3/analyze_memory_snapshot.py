# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Attribute a CUDA allocator snapshot to the call sites that allocated it.

Written after three wrong guesses about which tensor drove a 19 GiB peak: aggregate
counters say how much, never who. ``torch.cuda.memory._dump_snapshot`` records a stack
per allocation, so this groups live bytes by frame and prints the largest owners.

Two views, because they answer different questions:

* **live** -- blocks still allocated at snapshot time. Explains steady-state footprint.
* **peak-window** -- every allocation recorded in the trace, whether freed or not. A
  transient that is allocated and released inside one step never appears in ``live`` but
  is exactly what sets ``max_memory_allocated``.
"""

from __future__ import annotations

import argparse
import collections
import pickle
from pathlib import Path


def _frame_label(frame: dict) -> str:
    name = frame.get("name", "?")
    filename = str(frame.get("filename", "?"))
    # Keep the tail of the path: the package prefix is identical for every frame here.
    short = "/".join(filename.rsplit("/", 3)[-3:])
    return f"{short}:{frame.get('line', '?')} {name}"


def _pick_frame(frames: list[dict], skip: tuple[str, ...]) -> str:
    """The shallowest frame that is not allocator/dispatch plumbing."""
    for frame in frames:
        label = _frame_label(frame)
        if not any(token in label for token in skip):
            return label
    return _frame_label(frames[0]) if frames else "<no stack>"


def live_by_frame(snapshot: dict, skip: tuple[str, ...]) -> collections.Counter:
    totals: collections.Counter = collections.Counter()
    for segment in snapshot.get("segments", []):
        for block in segment.get("blocks", []):
            if block.get("state") != "active_allocated":
                continue
            frames = block.get("frames") or []
            totals[_pick_frame(frames, skip)] += int(block.get("size", 0))
    return totals


def live_at_peak(snapshot: dict, skip: tuple[str, ...]) -> tuple[collections.Counter, int]:
    """Composition of the live set at the moment the trace's high-water mark is reached.

    Cumulative "bytes ever allocated by frame" ranks churn, not footprint: a small buffer
    reallocated every layer outranks the one huge tensor that actually sets the peak.
    Replaying the trace and snapshotting the live set at its maximum names the owners of
    the peak itself.
    """
    live: dict[int, tuple[int, str]] = {}
    current = 0
    peak = 0
    peak_live: dict[int, tuple[int, str]] = {}
    for trace in snapshot.get("device_traces", []):
        for event in trace:
            action = event.get("action")
            size = int(event.get("size", 0))
            addr = int(event.get("addr", 0))
            if action == "alloc":
                live[addr] = (size, _pick_frame(event.get("frames") or [], skip))
                current += size
                if current > peak:
                    peak = current
                    peak_live = dict(live)
            elif action in ("free_completed", "free_requested"):
                if addr in live:
                    current -= live.pop(addr)[0]
    totals: collections.Counter = collections.Counter()
    for size, label in peak_live.values():
        totals[label] += size
    return totals, peak


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument(
        "--skip",
        default="torch/cuda/memory,_dynamo,_inductor,torch/_ops,autograd",
        help="comma-separated tokens; frames matching any are walked past",
    )
    args = parser.parse_args()

    with args.snapshot.open("rb") as handle:
        snapshot = pickle.load(handle)
    skip = tuple(token for token in args.skip.split(",") if token)

    gib = 1024**3
    live = live_by_frame(snapshot, skip)
    print(f"=== live at snapshot ({sum(live.values()) / gib:.2f} GiB) ===")
    for label, size in live.most_common(args.top):
        print(f"  {size / gib:8.3f} GiB  {label}")

    at_peak, peak = live_at_peak(snapshot, skip)
    print(f"\n=== live AT THE PEAK of the recorded window ({peak / gib:.2f} GiB) ===")
    for label, size in at_peak.most_common(args.top):
        print(f"  {size / gib:8.3f} GiB  {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
