# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Per-request usage accounting for billable diffusion requests.

This module is deliberately torch-free and deterministic: the same numbers must be
reproducible from the request alone on any machine, because billing reconciles against
them long after the request is gone.

The schema is the one MiniMax's own H3-Image API returns, verified against the live CN
endpoint on 2026-11-16 (14 requests). Every constant below was measured, not guessed:

* ``output_tokens == (width // 32) * (height // 32)``. Confirmed exactly on
  2048x2048 -> 4096, 4096x4096 -> 16384, 2720x1536 -> 4080, 7104x2368 -> 16428,
  5472x3654 -> 19494. The 3654/32 = 114.19 case proves the division is a floor, not a
  round: 171 * 114 = 19494.
* A 1024x1024 request is still billed 4096 output tokens, i.e. the vendor also floors
  the *generated canvas* at 2048x2048. Our pipeline does the same through
  ``min_generation_area``, and the admission gate (>= 4,194,304 px) makes the canvas
  equal the delivered size, so ``basis="delivered"`` is the only basis we ever report.
* ``input_tokens_details.image_tokens`` counts vision tokens of the reference images on
  the text-encoder side. The vendor's per-image counts follow its own dynamic tiling
  (1K ref -> 2050, 4K ref -> 5122, 8K ref -> 5122 saturated, 8192x4096 -> 10242); we
  report our own deterministic count instead (short edge clamped to 1024, 32 px per
  token, plus the two vision marker tokens), and keep the vendor's tiling out of the
  contract.
* ``input_tokens_details.text_tokens`` is multiplied by ``n`` by the vendor (n=2 gave
  text 15 -> 30). We mirror that so a multi-image request is priced the same way.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# One image token covers 32x32 delivered pixels: VAE downsamples by 16 and the DiT
# patchifier folds another 2x2. Verified against the live API (see module docstring).
PIXELS_PER_IMAGE_TOKEN = 32
# ``<vision_start>`` + ``<vision_end>`` per reference image. The vendor's 1K reference
# count (2050 = 2048 + 2) shows the same two markers are counted.
VISION_MARKERS_PER_IMAGE = 2
USAGE_VERSION = "h3-image-v1"


@dataclass(frozen=True)
class UsageSpec:
    """The contract a usage record was built under.

    Bumping ``version`` is how a口径 change stays auditable: old records keep their
    version, and reconciliation can tell "we changed the rule" from "we mis-billed".
    """

    version: str = USAGE_VERSION
    pixels_per_token: int = PIXELS_PER_IMAGE_TOKEN
    vision_markers_per_image: int = VISION_MARKERS_PER_IMAGE

    def image_tokens(self, width: int, height: int) -> int:
        """Image tokens a ``width x height`` image occupies (floor, like the vendor)."""
        if width <= 0 or height <= 0:
            return 0
        return (int(width) // self.pixels_per_token) * (int(height) // self.pixels_per_token)


SPEC = UsageSpec()


@dataclass(frozen=True)
class Usage:
    """One request's billable quantities, in the vendor's schema plus extensions."""

    generated_images: int
    input_text_tokens: int
    input_image_tokens: int
    output_image_tokens: int
    spec: UsageSpec = SPEC
    # Kept for auditing; never billed. ``canvas_tokens`` equals ``output_image_tokens``
    # while the admission gate holds, and differs only if someone lowers
    # ``min_generation_area`` behind the contract.
    canvas_tokens: int = 0
    reference_latent_tokens: int = 0
    per_reference_tokens: tuple[int, ...] = ()
    text_token_factor: int = 1
    source: str = "engine"

    @property
    def total_tokens(self) -> int:
        return self.input_text_tokens + self.input_image_tokens + self.output_image_tokens

    def to_dict(self) -> dict[str, Any]:
        """The wire form: vendor-compatible fields first, extensions after."""
        return {
            "generated_images": int(self.generated_images),
            "total_tokens": int(self.total_tokens),
            "input_tokens": int(self.input_text_tokens + self.input_image_tokens),
            "output_tokens": int(self.output_image_tokens),
            "input_tokens_details": {
                "text_tokens": int(self.input_text_tokens),
                "image_tokens": int(self.input_image_tokens),
            },
            "output_tokens_details": {"image_tokens": int(self.output_image_tokens)},
            # vLLM-Omni extensions (ignored by clients that only know the vendor schema).
            "usage_source": self.source,
            "usage_version": self.spec.version,
            "canvas_tokens": int(self.canvas_tokens or self.output_image_tokens),
            "reference_latent_tokens": int(self.reference_latent_tokens),
            "text_token_factor": int(self.text_token_factor),
        }


def build_usage(
    *,
    width: int,
    height: int,
    generated_images: int,
    text_tokens: int,
    reference_vision_counts: tuple[int, ...] = (),
    canvas: tuple[int, int] | None = None,
    reference_latent_tokens: int = 0,
    spec: UsageSpec = SPEC,
) -> Usage:
    """Assemble a ``Usage`` from per-request facts.

    ``text_tokens`` is what the text encoder actually consumed for one image; it is
    multiplied by ``generated_images`` to match the vendor, which bills the prompt once
    per produced image (measured: n=2 -> text 15 -> 30).
    """
    images = max(int(generated_images), 0)
    per_reference = tuple(
        max(int(count), 0) + spec.vision_markers_per_image for count in reference_vision_counts
    )
    output_tokens = spec.image_tokens(width, height)
    canvas_tokens = spec.image_tokens(*canvas) if canvas is not None else output_tokens
    return Usage(
        generated_images=images,
        input_text_tokens=max(int(text_tokens), 0) * max(images, 1),
        input_image_tokens=sum(per_reference),
        output_image_tokens=output_tokens * max(images, 1),
        spec=spec,
        canvas_tokens=canvas_tokens * max(images, 1),
        reference_latent_tokens=max(int(reference_latent_tokens), 0) * max(images, 1),
        per_reference_tokens=per_reference,
        text_token_factor=max(images, 1),
    )


def validate_usage(usage: Usage, *, text_and_vision_tokens: int | None = None) -> list[str]:
    """Return the invariants this record violates; empty means consistent.

    Violations are reported, never raised: a wrong usage number must degrade the bill
    loudly, not fail a customer's generation after the GPU work is already spent.
    """
    problems: list[str] = []
    if usage.generated_images < 0:
        problems.append("generated_images is negative")
    if usage.output_image_tokens % max(usage.generated_images, 1) != 0 and usage.generated_images:
        problems.append("output_image_tokens is not a multiple of generated_images")
    if usage.canvas_tokens and usage.canvas_tokens != usage.output_image_tokens:
        problems.append(
            f"canvas_tokens={usage.canvas_tokens} differs from output_image_tokens="
            f"{usage.output_image_tokens} (admission gate should keep them equal)"
        )
    if text_and_vision_tokens is not None:
        expected = usage.input_text_tokens + usage.input_image_tokens
        if text_and_vision_tokens < expected:
            problems.append(
                f"encoder saw {text_and_vision_tokens} tokens but usage claims {expected}"
            )
    return problems


# --------------------------------------------------------------------------- audit


_AUDIT_LOCK = threading.Lock()


def usage_log_path() -> Path | None:
    """Where JSONL usage records go, or ``None`` when auditing is disabled."""
    raw = os.environ.get("VLLM_OMNI_USAGE_LOG", "").strip()
    if not raw:
        return None
    return Path(raw)


def record_usage(
    *,
    request_id: str,
    usage: dict[str, Any] | None,
    model: str | None = None,
    status: str = "success",
    problems: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append one usage record to the audit JSONL. Best effort, never raises.

    The bill itself is computed by the gateway from the HTTP response; this file is the
    engine-side ledger used to reconcile ``sum(gateway) == sum(engine)`` and to answer
    "what did we actually charge for" months later.
    """
    path = usage_log_path()
    if path is None:
        return
    record = {
        "ts": round(time.time(), 3),
        "request_id": request_id,
        "model": model,
        "status": status,
        "usage": usage,
        "problems": problems or [],
    }
    if extra:
        record.update(extra)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    try:
        with _AUDIT_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except OSError:
        # Losing an audit line must not lose a customer request.
        pass
