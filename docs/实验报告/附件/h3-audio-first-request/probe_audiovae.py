#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Isolate MiniMaxH3AudioVAE.decode_latent: is the FIRST invocation different?

Loads only the checkpoint's audio VAE (no DiT / text encoder) and decodes the
SAME latent several times in one process, logging the output hash each time.
"""

import hashlib
import os
import sys
import torch

from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3AudioVAE

COMPONENT = "/nfs-data/models/MiniMax-H3/FL2VA/audio_vae"


def md5t(t):
    return hashlib.md5(t.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()).hexdigest()[:16]


def main():
    dev = torch.device("cuda", 0)
    vae = MiniMaxH3AudioVAE(COMPONENT, device=dev, trust_remote_code=True)
    vae.eval()
    vae.remote.to(dev)
    print("loaded; remote device:", next(vae.remote.parameters()).device, flush=True)

    ch = int(vae.config_dict["latent_channels"])
    torch.manual_seed(7)
    latent = torch.randn(2, ch, 207, device=dev, dtype=torch.float32)
    print("latent md5:", md5t(latent), "shape:", tuple(latent.shape), flush=True)

    outs = []
    for i in range(1, 5):
        with torch.inference_mode():
            w = vae.decode_latent(latent)
        h = md5t(w)
        outs.append(h)
        print("  decode call %d -> waveform md5 %s  shape=%s dtype=%s" % (i, h, tuple(w.shape), w.dtype), flush=True)
        del w
    print()
    print("first == second ?", outs[0] == outs[1])
    print("second == third ?", outs[1] == outs[2])
    print("third  == fourth?", outs[2] == outs[3])
    print("distinct outputs:", len(set(outs)))

    # Now the same latent through a freshly-constructed wrapper class (same weights).
    return 0


if __name__ == "__main__":
    sys.exit(main())
