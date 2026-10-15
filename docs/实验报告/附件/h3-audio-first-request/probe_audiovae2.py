#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Mechanism + fix candidate for the first-call audio divergence."""

import hashlib
import torch

from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3AudioVAE

COMPONENT = "/nfs-data/models/MiniMax-H3/FL2VA/audio_vae"


def md5t(t):
    return hashlib.md5(t.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()).hexdigest()[:16]


def wfp(vae, names=None):
    """Fingerprint the EFFECTIVE (post-weight_norm) weights of selected convs."""
    h = hashlib.md5()
    for name, m in vae.named_modules():
        if names and not any(n in name for n in names):
            continue
        w = getattr(m, "weight", None)
        if isinstance(w, torch.Tensor) and w.dim() >= 2:
            h.update(name.encode())
            h.update(w.detach().to("cpu", torch.float32).numpy().tobytes())
    return h.hexdigest()[:16]


def main():
    dev = torch.device("cuda", 0)
    vae = MiniMaxH3AudioVAE(COMPONENT, device=dev, trust_remote_code=True)
    vae.eval()
    vae.remote.to(dev)

    sample = ["decoder.conv_pre", "decoder.conv_post", "decoder.ups.0"]
    print("effective-weight fingerprint BEFORE any forward:", wfp(vae, sample), flush=True)
    print("  sample of parametrization state:", flush=True)
    for name, m in list(vae.named_modules())[:400]:
        if "decoder.conv_post" in name and hasattr(m, "parametrizations"):
            print("   ", name, type(m).__name__, "->", [p for p in dir(m) if "param" in p.lower()][:5], flush=True)

    ch = int(vae.config_dict["latent_channels"])
    torch.manual_seed(7)
    latent = torch.randn(2, ch, 207, device=dev, dtype=torch.float32)
    print("latent md5:", md5t(latent), flush=True)

    with torch.inference_mode():
        w1 = vae.decode_latent(latent)
    print("call1 out=%s   weights now=%s" % (md5t(w1), wfp(vae, sample)), flush=True)
    with torch.inference_mode():
        w2 = vae.decode_latent(latent)
    print("call2 out=%s   weights now=%s" % (md5t(w2), wfp(vae, sample)), flush=True)
    with torch.inference_mode():
        w3 = vae.decode_latent(latent)
    print("call3 out=%s   weights now=%s" % (md5t(w3), wfp(vae, sample)), flush=True)

    # --- fix candidate: one warm-up decode, then the real ones must be stable ---
    print("\n--- warm-up candidate ---", flush=True)
    vae2 = MiniMaxH3AudioVAE(COMPONENT, device=dev, trust_remote_code=True)
    vae2.eval()
    vae2.remote.to(dev)
    with torch.inference_mode():
        vae2.decode_latent(latent)  # warm-up (discarded)
    got = []
    for i in range(3):
        with torch.inference_mode():
            got.append(md5t(vae2.decode_latent(latent)))
    print("after warm-up, next 3 calls:", got, "all equal:", len(set(got)) == 1, flush=True)
    print("matches un-warmed settled call2 value?", got[0] == md5t(w2), flush=True)

    # --- does a warm-up with a DIFFERENT shape suffice? ---
    print("\n--- warm-up with a different shape ---", flush=True)
    vae3 = MiniMaxH3AudioVAE(COMPONENT, device=dev, trust_remote_code=True)
    vae3.eval()
    vae3.remote.to(dev)
    small = torch.randn(1, ch, 32, device=dev, dtype=torch.float32)
    with torch.inference_mode():
        vae3.decode_latent(small)
    got3 = []
    for i in range(2):
        with torch.inference_mode():
            got3.append(md5t(vae3.decode_latent(latent)))
    print("warm-up(shape=1x32x32) then real:", got3, "stable:", len(set(got3)) == 1, flush=True)
    print("equals settled value?", got3[0] == md5t(w2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
