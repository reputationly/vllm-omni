"""Batch-run the 8/10/14-reference cases and measure wall time + peak VRAM per request.

Runs ON the GPU node (needs nvidia-smi + python3), e.g.:

  python3 run_batch.py --url http://127.0.0.1:40070 --variant base
  python3 run_batch.py --url ... --variant budget --only 8ref_catalog,10ref_catalog
"""

import argparse
import base64
import json
import mimetypes
import pathlib
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

ROOT = pathlib.Path("/nfs-output/prod-baseline-20261114/h3i-cases")
sys.path.insert(0, str(ROOT))
from cases_8ref import CASES  # noqa: E402


class VramSampler:
    """Sample summed GPU memory every second until stopped."""

    def __init__(self):
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout
                total = sum(int(x) for x in out.split() if x.strip().isdigit())
                self.samples.append((time.time(), total))
            except Exception:
                pass
            self._stop.wait(1.0)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)

    def peak_between(self, t0, t1):
        window = [value for ts, value in self.samples if t0 - 1 <= ts <= t1 + 1]
        return max(window) if window else None


def multipart(fields, images):
    boundary = f"mini-{uuid.uuid4().hex}"
    parts = []
    for key, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    for path in images:
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{pathlib.Path(path).name}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n".encode()
        )
        parts.append(pathlib.Path(path).read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--size", default="2048x2048")
    args = ap.parse_args()

    out_dir = ROOT / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    only = {item for item in args.only.split(",") if item}

    sampler = VramSampler()
    sampler.start()
    rows = []
    try:
        for name, prompt, refs in CASES:
            if only and name not in only:
                continue
            body, content_type = multipart(
                {
                    "prompt": prompt,
                    "size": args.size,
                    "seed": "42",
                    "generator_device": "cpu",
                    "response_format": "b64_json",
                    "output_format": "png",
                },
                refs,
            )
            request = urllib.request.Request(
                f"{args.url}/v1/images/edits", data=body, headers={"Content-Type": content_type}
            )
            start = time.time()
            try:
                with urllib.request.urlopen(request, timeout=1800) as response:
                    raw = response.read()
                    code = response.status
            except urllib.error.HTTPError as exc:
                raw = exc.read()
                code = exc.code
            except Exception as exc:  # timeouts / resets
                raw = str(exc).encode()
                code = 0
            end = time.time()
            wall = round(end - start, 2)
            peak = sampler.peak_between(start, end)
            nbytes = None
            message = None
            if code == 200:
                payload = json.loads(raw)
                data = base64.b64decode(payload["data"][0]["b64_json"])
                nbytes = len(data)
                (out_dir / f"{name}.png").write_bytes(data)
            else:
                message = raw[:300].decode("utf-8", "ignore")
            rows.append(
                {
                    "case": name,
                    "refs": len(refs),
                    "http": code,
                    "wall_s": wall,
                    "peak_vram_mib": peak,
                    "bytes": nbytes,
                    "msg": message,
                }
            )
            status = "ok" if code == 200 else f"FAIL({code})"
            print(
                f"{name:18s} refs={len(refs):2d} {status:10s} wall={wall:7.1f}s "
                f"peak={peak if peak is not None else '-'} MiB",
                flush=True,
            )
    finally:
        sampler.stop()
    (out_dir / "timings_8ref.json").write_text(json.dumps({"variant": args.variant, "rows": rows}, indent=1, ensure_ascii=False))
    print(f"wrote {out_dir}/timings_8ref.json")


if __name__ == "__main__":
    main()
