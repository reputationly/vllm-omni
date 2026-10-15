# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys

p = "/nfs-output/qfix-20261014/ws_audio/vllm_omni/diffusion/models/minimax_h3/vae.py"
s = open(p, encoding="utf-8").read()
old = """        try:
            with torch.inference_mode():
                self.remote.decode(probe)
        except Exception as exc:  # noqa: BLE001 - a warm-up must never break serving
            logger.warning("MiniMax H3 audio VAE warm-up decode failed: %s", exc)"""
new = """        try:
            _h3_probe("warmup_in", probe_shape=list(probe.shape), device=str(probe.device))
            with torch.inference_mode():
                _w = self.remote.decode(probe)
            _h3_probe("warmup_out", out_shape=list(_w.shape), out_md5=_h3_tensor_md5(_w))
        except Exception as exc:  # noqa: BLE001 - a warm-up must never break serving
            _h3_probe("warmup_failed", err=repr(exc))
            logger.warning("MiniMax H3 audio VAE warm-up decode failed: %s", exc)"""
n = s.count(old)
assert n == 1, "anchor count=%d" % n
s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
import ast, os

ast.parse(open(p, encoding="utf-8").read())
print("instrumented warm-up; parses OK")
