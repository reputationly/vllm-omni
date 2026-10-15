#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Validate the exact fix primitive: one tiny self.remote.decode() absorbs the transient."""

import hashlib
import time
import torch

from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3AudioVAE

COMPONENT = "/nfs-data/models/MiniMax-H3/FL2VA/audio_vae"


def md5t(t):
    return hashlib.md5(t.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()).hexdigest()[:16]


def build(dev):
    v = MiniMaxH3AudioVAE(COMPONENT, device=dev, trust_remote_code=True)
    v.eval()
    v.remote.to(dev)
    return v


def main():
    dev = torch.device("cuda", 0)
    ch = 32  # latent_channels, from config

    # reference: un-warmed settled value
    ref = build(dev)
    ch_cfg = int(ref.config_dict["latent_channels"])
    ch = ch_cfg
    torch.manual_seed(7)
    latent = torch.randn(2, ch, 207, device=dev, dtype=torch.float32)
    with torch.inference_mode():
        ref.decode_latent(latent)  # transient
        settled = md5t(ref.decode_latent(latent))  # settled
    print("reference settled value:", settled, flush=True)
    del ref
    torch.accelerator.empty_cache()

    # candidate fix primitive: tiny direct remote.decode, minimal length
    v = build(dev)
    probe = torch.zeros(1, ch, 1, device=dev, dtype=torch.float32)
    t0 = time.time()
    with torch.inference_mode():
        v.remote.decode(probe)
    warm_s = time.time() - t0
    got = []
    for _ in range(3):
        with torch.inference_mode():
            got.append(md5t(v.decode_latent(latent)))
    print("warm-up(remote.decode, len=1) took %.3fs" % warm_s, flush=True)
    print("  next 3 real decodes:", got, flush=True)
    print("  all equal:", len(set(got)) == 1, " equals settled:", got[0] == settled, flush=True)

    # also: is the warm-up itself already settled on repeat?
    reps = []
    for _ in range(3):
        with torch.inference_mode():
            reps.append(md5t(v.remote.decode(probe)))
    print("  repeated warm-up probes:", reps, "stable:", len(set(reps)) == 1, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
