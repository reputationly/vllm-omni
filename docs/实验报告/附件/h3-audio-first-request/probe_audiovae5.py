#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Which construction variant does the real service use, and does the fix hold there?"""

import hashlib
import torch

from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3AudioVAE

COMPONENT = "/nfs-data/models/MiniMax-H3/FL2VA/audio_vae"


def md5t(t):
    return hashlib.md5(t.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()).hexdigest()[:16]


def run(decode_only, use_warmup, latent, label):
    dev, cpu = torch.device("cuda", 0), torch.device("cpu")
    vae = MiniMaxH3AudioVAE(COMPONENT, device=dev, load_device=cpu, decode_only=decode_only, trust_remote_code=True)
    vae.eval()
    if not use_warmup:
        vae._first_decode_absorbed = True  # emulate pre-fix behaviour
    out = []
    for i in range(1, 4):
        vae.load_to_device()
        with torch.inference_mode():
            w = vae.decode_latent(latent)
        vae.offload_to_cpu()
        out.append(md5t(w))
    ok = len(set(out)) == 1
    print("  %-42s -> %s  stable=%s" % (label, out, ok), flush=True)
    del vae
    torch.accelerator.empty_cache()
    return out


def main():
    torch.manual_seed(7)
    latent = torch.randn(2, 32, 207, device=torch.device("cuda", 0), dtype=torch.float32)
    print("latent md5:", md5t(latent), flush=True)
    print("\n=== service-path cycles (load -> decode -> offload) x3 ===", flush=True)
    run(False, False, latent, "decode_only=False, NO fix")
    run(False, True, latent, "decode_only=False, WITH fix")
    run(True, False, latent, "decode_only=True , NO fix  <- pipeline default")
    run(True, True, latent, "decode_only=True , WITH fix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
