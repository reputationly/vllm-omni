#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Make VDN's ``turbo`` adapter fit a curve-pruned H3 partition.

VDN's DMD stage distilled the 8-step ``turbo`` adapter against the frozen Stage-B VDN as
its real score, so the branch and that adapter are a matched pair. Curve pruning refuses
the pair: ``turbo`` edits the 51 AdaLN projections with a ``[out, 2688]`` delta, and a
pruned checkpoint's ``adaln_proj.linear`` is ``[out, 8]``. Substituting some other
few-step adapter is what splits the pair.

This rewrites ``turbo`` instead, projecting each AdaLN update onto the pruned curve with
the 1025-point affine fit in ``vdn_curve_projection`` -- the 208 attention/MLP targets
pass through untouched. The output is a VDN checkpoint directory that
``bake_vdn_adapters.py`` can bake into the pruned partition.

    python3 tools/minimax_h3/project_vdn_turbo_to_pruned_curve.py \\
        --vdn   /nfs-data/models/VDN-MiniMax-H3/stage-dmd-step-250 \\
        --pruned /nfs-data/models/MiniMax-H3-Ref2VA-Pruned-r8-BF16-partition \\
        --full   /nfs-data/models/MiniMax-H3/Ref2VA \\
        --dst    /nfs-data/models/VDN-MiniMax-H3/stage-dmd-step-250-curve-ref2va

``--full`` supplies only ``time_embedder.proj_in/proj_out``: the pruned checkpoint no
longer carries the 2688-wide path, so the curve it was fitted to has to be recomputed
from the model it was pruned FROM. Pass the partition the pruned one was built from, or
the fit describes a different model's timestep response.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.minimax_h3.vdn_curve_projection import (  # noqa: E402
    CURVE_WIDTH,
    GRID_SIZE,
    CurveProjector,
    full_adaln_curve,
)

CURVE_TABLE_KEY = "time_embedder.table"
TIME_KEYS = tuple(f"time_embedder.proj_{part}.{kind}" for part in ("in", "out") for kind in ("weight", "bias"))
# A fit worse than this is not a rounding difference -- it means the pruned checkpoint
# was not built from ``--full``, which is the mistake this tool cannot detect any other
# way and which would otherwise ship as a subtly wrong schedule.
MAX_ACCEPTABLE_ERROR = 1e-3


def _read(root: Path, wanted: set[str]) -> dict[str, torch.Tensor]:
    found: dict[str, torch.Tensor] = {}
    for shard in sorted(glob.glob(os.path.join(root, "*.safetensors"))):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in wanted:
                    found[key] = handle.get_tensor(key)
    missing = sorted(wanted - set(found))
    if missing:
        raise SystemExit(f"{root} is missing {missing}")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vdn", required=True, help="VDN checkpoint directory (with adapters/turbo)")
    parser.add_argument("--pruned", required=True, help="curve-pruned partition to fit against")
    parser.add_argument("--full", required=True, help="the UNPRUNED partition it was pruned from")
    parser.add_argument("--dst", required=True, help="output VDN checkpoint directory")
    parser.add_argument("--adapter", default="turbo", help="which adapter to project (default: turbo)")
    args = parser.parse_args()

    vdn, pruned_t = Path(args.vdn), Path(args.pruned) / "transformer"
    full_t, dst = Path(args.full) / "transformer", Path(args.dst)
    if dst.exists():
        raise SystemExit(f"{dst} exists; refusing to overwrite an artifact")

    table = _read(pruned_t, {CURVE_TABLE_KEY})[CURVE_TABLE_KEY]
    if tuple(table.shape) != (GRID_SIZE, CURVE_WIDTH):
        raise SystemExit(f"{pruned_t} {CURVE_TABLE_KEY} is {tuple(table.shape)}, expected {(GRID_SIZE, CURVE_WIDTH)}")
    time_reference = {k.replace("time_embedder.", ""): v for k, v in _read(full_t, set(TIME_KEYS)).items()}

    projector = CurveProjector.build(table, full_adaln_curve(time_reference))
    print(
        f"curve reconstruction (8-wide table + intercept -> {projector.full_curve.shape[1]}-wide): "
        f"{projector.reconstruction_error:.3e}"
    )
    if projector.reconstruction_error > MAX_ACCEPTABLE_ERROR:
        raise SystemExit(
            f"the pruned table does not reconstruct {full_t}'s timestep curve "
            f"({projector.reconstruction_error:.3e}); --full is probably not the partition "
            f"{pruned_t} was pruned from"
        )

    source = vdn / ADAPTERS / args.adapter / "adapter_model.safetensors"
    if not source.is_file():
        raise SystemExit(f"{source} not found")
    with safe_open(source, framework="pt", device="cpu") as handle:
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        metadata = dict(handle.metadata() or {})

    # Selected by RESOLVED target, not by name. The adapter spells the final AdaLN
    # ``norm_out.linear`` -- no "adaln" in it -- and it is pruned exactly like the other
    # 50. Filtering on the string leaves it full-width, and the bake then fails on the
    # last tensor of the stream, after everything else has already been rewritten.
    from vllm_omni.diffusion.models.minimax_h3.fasth3 import _resolve_native_target
    from vllm_omni.diffusion.models.minimax_h3.vdn import _strip_hybrid_level

    stems = []
    for stem in sorted({k.rsplit(".lora_", 1)[0] for k in tensors if ".lora_" in k}):
        resolved = _resolve_native_target(_strip_hybrid_level(stem))
        if resolved and resolved[0].endswith("adaln_proj.linear"):
            stems.append(stem)
    if not stems:
        raise SystemExit(f"adapter {args.adapter!r} edits no AdaLN projection; nothing to project")

    errors: list[tuple[str, float]] = []
    for stem in stems:
        a_key = next(k for k in tensors if k.startswith(f"{stem}.lora_A"))
        b_key = next(k for k in tensors if k.startswith(f"{stem}.lora_B"))
        a8, diff_b, error = projector.project(tensors[a_key], tensors[b_key])
        # A8 in the adapter's own dtype, the intercept in fp32: it is added to a pruned
        # folded_bias that the forward keeps in fp32 for exactly this reason.
        tensors[a_key] = a8.to(tensors[a_key].dtype).contiguous()
        tensors[f"{stem}{DIFF_B}"] = diff_b.to(torch.float32).contiguous()
        errors.append((stem, error))

    worst_module, worst = max(errors, key=lambda item: item[1])
    values = sorted(error for _, error in errors)
    print(
        f"projected {len(errors)} AdaLN modules; relative error on the applied delta: "
        f"min={values[0]:.3e} median={values[len(values) // 2]:.3e} max={worst:.3e} ({worst_module})"
    )
    if worst > MAX_ACCEPTABLE_ERROR:
        raise SystemExit(f"{worst_module} projected at {worst:.3e}, above {MAX_ACCEPTABLE_ERROR:.0e}")

    # Everything the checkpoint carries, with only the projected adapter rewritten. Copied
    # rather than symlinked: this directory states a different fact about the adapters than
    # its source, and a link would let an edit there change what has been validated here.
    shutil.copytree(vdn, dst, symlinks=False)
    target = dst / ADAPTERS / args.adapter / "adapter_model.safetensors"
    metadata.update(
        {
            "vdn_curve_projection": "adaln_curve_affine_lstsq_pinv1025_with_diff_b",
            "vdn_curve_projection_source": str(pruned_t),
            "vdn_curve_projection_max_error": f"{worst:.6e}",
        }
    )
    save_file(tensors, str(target), metadata=metadata)

    stamp = dst / "curve_projection.json"
    stamp.write_text(
        json.dumps(
            {
                "adapter": args.adapter,
                "algorithm": "adaln_curve_affine_lstsq_pinv1025_with_diff_b",
                "pruned_partition": str(pruned_t),
                "time_reference": str(full_t),
                "curve_reconstruction_error": projector.reconstruction_error,
                "modules": {stem: error for stem, error in errors},
            },
            indent=2,
        )
    )
    print(f"wrote {dst}")
    print(f"  bake with: --adapters default,{args.adapter} --src {args.pruned}")
    return 0


ADAPTERS = "adapters"
DIFF_B = ".diff_b"


if __name__ == "__main__":
    raise SystemExit(main())
