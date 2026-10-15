#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Wait for the H3 service, then send N identical same-seed requests.

usage: send3.py <host> <port> <tag> <N>
Writes /nfs-output/qfix-20261014/<tag>_run<i>.mp4
"""

import base64
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

HOST, PORT, TAG, N = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
OUT = "/nfs-output/qfix-20261014"
IMG = "/nfs-output/baseline-20261012/first_frame_small.jpg"
MODEL = "/nfs-models/wuhanjisuan894/models/MiniMax-H3-Singularity-VDN-INT8"
PROMPT = (
    "Cinematic drone shot pushing through misty mountain ridges at sunrise, "
    "golden light catching the clouds, slow and smooth camera move."
)
BASE = "http://%s:%s/v1/videos/sync" % (HOST, PORT)

# ---- wait for readiness (does not touch the model) ----
t0 = time.time()
while True:
    try:
        with urllib.request.urlopen("http://%s:%s/ready" % (HOST, PORT), timeout=5) as r:
            if r.status == 200:
                break
    except Exception:
        pass
    if time.time() - t0 > 3600:
        print("timeout waiting for /ready", flush=True)
        sys.exit(1)
    time.sleep(10)
print("service ready after %.0fs" % (time.time() - t0), flush=True)

data_url = "data:image/jpeg;base64," + base64.b64encode(open(IMG, "rb").read()).decode()
fields = {
    "prompt": PROMPT,
    "model": MODEL,
    "width": "1344",
    "height": "768",
    "seconds": "5",
    "num_frames": "120",
    "fps": "24",
    "seed": "42",
    "image_reference": json.dumps({"image_url": data_url}),
}

for i in range(1, N + 1):
    b = uuid.uuid4().hex
    body = (
        b"".join(
            ('--%s\r\nContent-Disposition: form-data; name="%s"\r\n\r\n%s\r\n' % (b, k, v)).encode()
            for k, v in fields.items()
        )
        + ("--%s--\r\n" % b).encode()
    )
    req = urllib.request.Request(BASE, data=body, headers={"Content-Type": "multipart/form-data; boundary=%s" % b})
    t1 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            raw, code = r.read(), r.status
    except urllib.error.HTTPError as e:
        raw, code = e.read(), e.code
    el = time.time() - t1
    if code == 200 and len(raw) > 10000:
        p = "%s/%s_run%d.mp4" % (OUT, TAG, i)
        with open(p, "wb") as fh:
            fh.write(raw)
            fh.flush()
            import os

            os.fsync(fh.fileno())
        print(
            "run%d http=200 bytes=%d elapsed=%.1fs md5=%s -> %s"
            % (i, len(raw), el, hashlib.md5(raw).hexdigest()[:16], p),
            flush=True,
        )
    else:
        print("run%d http=%s elapsed=%.1fs body=%s" % (i, code, el, raw[:200].decode(errors="replace")), flush=True)
print("done", flush=True)
