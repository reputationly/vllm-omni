#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Patch MiniMaxH3AudioVAE.decode_latent to log per-call torch.backends state.

Driven by env vars so one tree serves both arms:
  H3_AUDIO_PROBE_LOG=<path>          enable logging
  H3_AUDIO_DECODE_DETERMINISM=1      wrap decode in _AudioVAEDeterminismContext (candidate fix)
"""

import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()

HELPERS = """

# ---- audio decode probe (instrumentation; remove before commit) ----
_H3_AUDIO_DECODE_CALLS: dict[int, int] = {}


def _h3_probe_enabled() -> bool:
    import os as _os

    return bool(_os.environ.get("H3_AUDIO_PROBE_LOG"))


def _h3_probe(tag: str, **kw) -> None:
    import json as _json
    import os as _os
    import time as _time

    p = _os.environ.get("H3_AUDIO_PROBE_LOG")
    if not p:
        return
    rec = {"t": round(_time.time(), 3), "pid": _os.getpid(), "tag": tag}
    rec.update(kw)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(_json.dumps(rec, ensure_ascii=False, default=str) + "\\n")
        fh.flush()
        _os.fsync(fh.fileno())


def _h3_next_call() -> int:
    import os as _os

    pid = _os.getpid()
    _H3_AUDIO_DECODE_CALLS[pid] = _H3_AUDIO_DECODE_CALLS.get(pid, 0) + 1
    return _H3_AUDIO_DECODE_CALLS[pid]


def _h3_backend_state() -> dict:
    b = torch.backends
    return {
        "matmul_tf32": b.cuda.matmul.allow_tf32,
        "cudnn_tf32": b.cudnn.allow_tf32,
        "cudnn_benchmark": b.cudnn.benchmark,
        "cudnn_deterministic": b.cudnn.deterministic,
        "cudnn_enabled": b.cudnn.enabled,
        "flash_sdp": b.cuda.flash_sdp_enabled(),
        "mem_efficient_sdp": b.cuda.mem_efficient_sdp_enabled(),
        "math_sdp": b.cuda.math_sdp_enabled(),
        "f32_precision": torch.get_float32_matmul_precision(),
    }


def _h3_tensor_md5(t) -> str:
    import hashlib as _hashlib

    return _hashlib.md5(
        t.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()
    ).hexdigest()


# ---- end probe ----


"""

OLD_DECODE = "        waveform = self.remote.decode(latent * std + mean)\n"
NEW_DECODE = """        _call = _h3_next_call() if _h3_probe_enabled() else None
        if _h3_probe_enabled():
            _h3_probe(
                "decode_in",
                call=_call,
                latent_shape=list(latent.shape),
                latent_dtype=str(latent.dtype),
                latent_md5=_h3_tensor_md5(latent),
                remote_param_device=str(next(self.remote.parameters()).device),
                remote_param_dtype=str(next(self.remote.parameters()).dtype),
                **_h3_backend_state(),
            )
        import os as _os

        _determinism = _os.environ.get("H3_AUDIO_DECODE_DETERMINISM", "0").strip() in ("1", "true", "True")
        with (_AudioVAEDeterminismContext() if _determinism else nullcontext()):
            waveform = self.remote.decode(latent * std + mean)
        if _h3_probe_enabled():
            _h3_probe(
                "decode_out",
                call=_call,
                waveform_shape=list(waveform.shape),
                waveform_dtype=str(waveform.dtype),
                waveform_md5=_h3_tensor_md5(waveform),
                determinism=_determinism,
                **_h3_backend_state(),
            )
"""

n_old = src.count(OLD_DECODE)
assert n_old == 1, f"decode anchor count={n_old}"
anchor = "class MiniMaxH3AudioVAE(nn.Module):"
assert src.count(anchor) == 1, "class anchor not unique"

src = src.replace(anchor, HELPERS.lstrip("\n") + anchor, 1)
src = src.replace(OLD_DECODE, NEW_DECODE, 1)

with open(path, "w", encoding="utf-8") as fh:
    fh.write(src)
    import os

    fh.flush()
    os.fsync(fh.fileno())

import ast

ast.parse(open(path, encoding="utf-8").read())
print("patched + parses OK:", path)
