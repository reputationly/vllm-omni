#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Model the SERVICEPATH: CPU-offload stager, one load/decode/offload cycle per request."""

import hashlib
import torch

from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3AudioVAE

COMPONENT = "/nfs-data/models/MiniMax-H3/FL2VA/audio_vae"


def md5t(t):
    return hashlib.md5(t.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()).hexdigest()[:16]


def cycles(vae, latent, n, label):
    out = []
    for i in range(1, n + 1):
        vae.load_to_device()
        with torch.inference_mode():
            w = vae.decode_latent(latent)
        vae.offload_to_cpu()
        out.append(md5t(w))
        print("  [%s] request %d -> %s" % (label, i, out[-1]), flush=True)
    print("  [%s] all equal: %s   distinct=%d" % (label, len(set(out)) == 1, len(set(out))), flush=True)
    return out


def main():
    dev = torch.device("cuda", 0)
    cpu = torch.device("cpu")
    ch = 32
    torch.manual_seed(7)
    latent = torch.randn(2, ch, 207, device=dev, dtype=torch.float32)
    print("latent md5:", md5t(latent), flush=True)

    print("\n=== A) stager path, NO fix (baseline service behaviour) ===", flush=True)
    a = MiniMaxH3AudioVAE(COMPONENT, device=dev, load_device=cpu, trust_remote_code=True)
    a.eval()
    print("   stager present:", a._stager is not None, flush=True)
    cycles(a, latent, 3, "nofix")
    del a
    torch.accelerator.empty_cache()

    print("\n=== B) stager path, WITH the lazy warm-up ===", flush=True)
    b = MiniMaxH3AudioVAE(COMPONENT, device=dev, load_device=cpu, trust_remote_code=True)
    b.eval()
    res = cycles(b, latent, 3, "fix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
