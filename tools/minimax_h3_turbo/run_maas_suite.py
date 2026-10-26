# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run a set of H3 gate cases against the MaaS gateway, expanding multi-seed cases.

Resumable: a case whose ``<output-root>/<tag>/summary.json`` already reports a
completed run is skipped, so an interrupted baseline can be continued.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--only", default="", help="comma-separated case ids/prefixes to run")
    parser.add_argument("--skip-tasks", default="", help="comma-separated tasks to skip, e.g. ref2va")
    parser.add_argument("--runner", type=Path, default=Path(__file__).with_name("run_maas_case.py"))
    args = parser.parse_args()

    manifest = json.loads((args.suite_dir / "manifest.json").read_text(encoding="utf-8"))
    only = [x for x in args.only.split(",") if x]
    skip_tasks = {x for x in args.skip_tasks.split(",") if x}

    failures = []
    for entry in manifest["cases"]:
        cid = entry["id"]
        if only and not any(cid.startswith(prefix) for prefix in only):
            continue
        if entry["task"] in skip_tasks:
            print(f"SKIP {cid} (task {entry['task']})", flush=True)
            continue
        for seed in entry["seeds"]:
            tag = f"prod_{cid}_s{seed}"
            out = args.output_root / tag
            summary = out / "summary.json"
            if summary.is_file():
                done = json.loads(summary.read_text(encoding="utf-8")).get("status")
                if done == "completed":
                    print(f"SKIP {tag} (already completed)", flush=True)
                    continue

            case = json.loads((args.suite_dir / f"{cid}.json").read_text(encoding="utf-8"))
            case["seed"] = seed
            tmp = args.output_root / f".case_{tag}.json"
            args.output_root.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(case, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            cmd = [sys.executable, str(args.runner), "--base-url", args.base_url,
                   "--api-key", args.api_key, "--request", str(tmp), "--tag", tag,
                   "--output-root", str(args.output_root)]
            print(f"RUN  {tag}", flush=True)
            rc = subprocess.call(cmd)
            tmp.unlink(missing_ok=True)
            if rc != 0:
                failures.append(tag)
                print(f"FAIL {tag} rc={rc}", flush=True)

    print(json.dumps({"failures": failures}, ensure_ascii=False), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
