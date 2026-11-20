# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Command-line interface for local MiniMax-H3-Image Context-IR."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from vllm_omni.diffusion.models.minimax_h3.context_ir import enhance, load_pe_prompts, openai_chat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--pe-dir", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--aspect-ratio", default="auto")
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--reasoning-effort")
    args = parser.parse_args()

    if not args.base_url:
        parser.error("--base-url or OPENAI_BASE_URL is required")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        parser.error("OPENAI_API_KEY is required")
    result = enhance(
        openai_chat(args.base_url, api_key),
        model=args.model,
        prompts=load_pe_prompts(args.pe_dir),
        prompt=args.prompt,
        images=args.image or None,
        aspect_ratio=args.aspect_ratio,
        max_repairs=args.max_repairs,
        reasoning_effort=args.reasoning_effort,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
