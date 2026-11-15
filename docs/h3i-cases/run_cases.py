"""Run the 10 H3-Image edit cases against one server and save the PNGs.

    python3 run_cases.py --url http://host:port --variant base
    python3 run_cases.py --url http://host:port --variant kv --only 01_official_podcast_1ref,...

Outputs land next to this script: <root>/<variant>/<case>.png plus <variant>/timings.json.
"""

import argparse
import base64
import json
import mimetypes
import pathlib
import sys
import time
import urllib.error
import urllib.request
import uuid

ROOT = pathlib.Path("/nfs-output/prod-baseline-20261114/h3i-cases")
sys.path.insert(0, str(ROOT))
from cases import CASES  # noqa: E402


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
    ap.add_argument("--seed", default="42")
    args = ap.parse_args()

    out_dir = ROOT / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    only = {item for item in args.only.split(",") if item}

    rows = []
    for name, prompt, refs in CASES:
        if only and name not in only:
            continue
        body, content_type = multipart(
            {
                "prompt": prompt,
                "size": args.size,
                "seed": args.seed,
                "generator_device": "cpu",
                "response_format": "b64_json",
                "output_format": "png",
            },
            refs,
        )
        request = urllib.request.Request(f"{args.url}/v1/images/edits", data=body, headers={"Content-Type": content_type})
        start = time.time()
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                raw = response.read()
                code = response.status
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            code = exc.code
        elapsed = round(time.time() - start, 2)
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
            {"case": name, "refs": len(refs), "http": code, "wall_s": elapsed, "bytes": nbytes, "msg": message}
        )
        status = "ok" if code == 200 else f"FAIL {code}"
        print(f"{name:28s} refs={len(refs)} {status:9s} wall={elapsed:7.2f}s", flush=True)

    (out_dir / "timings.json").write_text(json.dumps({"variant": args.variant, "rows": rows}, indent=1, ensure_ascii=False))
    print(f"wrote {out_dir}/timings.json")


if __name__ == "__main__":
    main()
