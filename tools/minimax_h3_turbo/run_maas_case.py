# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run one canonical H3 gate case against the new-api (MaaS) gateway.

The canonical case files under ``docs/h3-gate-cases`` describe the engine-level
contract (``width``/``height``, ``extra_params``, ``image_path``). The gateway
takes a different body (``size`` as ``WxH``, ``seconds`` as a string, media as
URLs or data URIs under ``metadata``), so the mapping lives here instead of
forking every case file into two shapes.

Only English ``prompt`` is sent: keys starting with ``_`` are human sidecars
(``_prompt_zh``) and never leave this process.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_MODEL = {
    "t2va": "minimax-h3-fl2va",
    "i2v": "minimax-h3-fl2va",
    "l2va": "minimax-h3-fl2va",
    "flf2v": "minimax-h3-fl2va",
    "ref2va": "minimax-h3-ref2va",
}


def http_json(url: str, *, body: dict[str, Any] | None = None, key: str, timeout: float = 60) -> dict[str, Any]:
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(url, data=data, method="POST" if body is not None else "GET")
    req.add_header("Authorization", f"Bearer {key}")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def as_media(value: str) -> str:
    """Absolute path -> data URI; anything already URL-ish passes through."""
    if value.startswith(("http://", "https://", "data:")):
        return value
    path = Path(value)
    if not path.is_file():
        raise SystemExit(f"media not found: {value}")
    return data_uri(path)


def build_body(case: dict[str, Any], model: str, *, send_tuning: bool = False) -> dict[str, Any]:
    task = (case.get("extra_params") or {}).get("task", "t2va")
    body: dict[str, Any] = {
        "model": model,
        "prompt": case["prompt"],
        "seconds": str(case.get("seconds", "5")),
        "size": f"{case['width']}x{case['height']}",
    }
    if case.get("aspect_ratio"):
        # H3 rejects a size whose implied ratio is not one of its six allowed values
        # (1344x768 is 7:4), so the ratio must be stated, not inferred from pixels.
        body["aspect_ratio"] = case["aspect_ratio"]
    for key in ("seed", "negative_prompt", "fps"):
        if key in case:
            body[key] = case[key]
    if send_tuning:
        # Only when re-running the engine contract itself. A production baseline must
        # measure the deployed sampling contract, so these stay out by default.
        for key in ("num_inference_steps", "extra_params"):
            if key in case:
                body[key] = case[key]

    metadata: dict[str, Any] = {}

    # Keyframe modes are selected by metadata.task_type and, for flf2v, by an ordered
    # ``images`` list. A single ``image`` is the i2v first frame or the l2va last frame.
    # Everything else (reference image/video/audio) goes through the ref2va-style keys.
    keyframe = task in ("i2v", "l2va", "flf2v")
    if keyframe:
        metadata["task_type"] = task
        first = [p.strip() for p in (case.get("image_path") or "").split(",") if p.strip()]
        last = [case["last_frame_path"]] if case.get("last_frame_path") else []
        if task == "flf2v":
            # Two accepted spellings: image_path (first) + last_frame_path, or a
            # comma-joined image_path holding both frames in order.
            if not last and len(first) == 2:
                first, last = [first[0]], [first[1]]
            if not (first and last):
                raise SystemExit("flf2v needs both a first and a last frame (image_path + last_frame_path)")
            body["images"] = [as_media(first[0]), as_media(last[0])]
        elif task == "i2v":
            if not first:
                raise SystemExit("i2v needs image_path")
            body["image"] = as_media(first[0])
        else:  # l2va
            if not last:
                raise SystemExit("l2va needs last_frame_path")
            body["image"] = as_media(last[0])
    else:
        if case.get("image_path"):
            metadata["src_ref_images"] = [as_media(p.strip()) for p in case["image_path"].split(",") if p.strip()]
        if case.get("last_frame_path"):
            metadata["src_ref_images"] = [as_media(case["last_frame_path"])]
        if case.get("video_path"):
            metadata["reference_videos"] = [as_media(p.strip()) for p in case["video_path"].split(",") if p.strip()]
        if case.get("audio_path"):
            # Field name mirrors src_ref_images / reference_videos; the engine-side
            # support lives in minimax_h3/reference_audio.py.
            metadata["reference_audios"] = [as_media(p.strip()) for p in case["audio_path"].split(",") if p.strip()]
    if metadata:
        body["metadata"] = metadata

    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default=None, help="override; default derived from the case task")
    parser.add_argument(
        "--send-tuning",
        action="store_true",
        help="also send num_inference_steps / extra_params (engine contract); off by default so the "
        "baseline measures the deployed sampling contract",
    )
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    args = parser.parse_args()

    case = json.loads(args.request.read_text(encoding="utf-8"))
    case = {k: v for k, v in case.items() if not k.startswith("_")}
    model = args.model or DEFAULT_MODEL[(case.get("extra_params") or {}).get("task", "t2va")]
    body = build_body(case, model, send_tuning=args.send_tuning)

    case_dir = args.output_root / args.tag
    case_dir.mkdir(parents=True, exist_ok=False)
    # Persist the body verbatim, plus a prompt digest: a truncated copy makes it
    # impossible to prove later which prompt revision produced an output.
    (case_dir / "request.json").write_text(
        json.dumps(
            {**body, "prompt_sha256": hashlib.sha256(body["prompt"].encode()).hexdigest()}, ensure_ascii=False, indent=2
        )
        + "\n",
        encoding="utf-8",
    )

    started = time.monotonic()
    submit = http_json(f"{args.base_url}/v1/videos", body=body, key=args.api_key)
    task_id = submit.get("task_id") or submit.get("id")
    status: dict[str, Any] = {"status": submit.get("status")}
    while time.monotonic() - started < args.timeout_seconds:
        status = http_json(f"{args.base_url}/v1/videos/{task_id}", key=args.api_key)
        if status.get("status") in ("completed", "failed", "cancelled"):
            break
        time.sleep(args.poll_seconds)

    wall = time.monotonic() - started
    summary: dict[str, Any] = {
        "tag": args.tag,
        "model": model,
        "task_id": task_id,
        "status": status.get("status"),
        "wall_seconds": round(wall, 1),
        "progress": status.get("progress"),
        "metadata": status.get("metadata"),
    }

    if status.get("status") == "completed":
        url = (status.get("metadata") or {}).get("url") or f"{args.base_url}/v1/videos/{task_id}/content"
        req = urllib.request.Request(url, method="GET")
        req.add_header("Authorization", f"Bearer {args.api_key}")
        with urllib.request.urlopen(req, timeout=300) as response:
            payload = response.read()
        out = case_dir / "output.mp4"
        out.write_bytes(payload)
        summary["output"] = {"path": str(out), "bytes": len(payload)}

    (case_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
