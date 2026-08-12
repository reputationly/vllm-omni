# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Factory for building quantization configs.

build_quant_config() delegates to vLLM's quantization registry.
Out-of-tree integrations can register Omni-specific builders with
register_quantization_override().
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Any

from vllm.logger import init_logger


# ---------------------------------------------------------------------------
# Stub the ``humming`` package so that vLLM's lazy import inside
# ``get_quantization_config()`` (which unconditionally does
# ``from .humming import HummingConfig``) does not crash when the real
# ``humming`` wheel is not installed.  Only populate the bare-minimum
# names that ``humming.py`` accesses at module level.
# ---------------------------------------------------------------------------
def _register_humming_stubs() -> None:
    """Register stub ``humming`` sub-modules so that the optional
    humming quantization backend can be imported without the real wheel."""
    if "humming" in sys.modules:
        return  # already present (real or stub)

    # --- sub-modules ---
    submodules: dict[str, tuple[str, ...]] = {
        "humming": (),
        "humming.config": ("GemmType",),
        "humming.dtypes": ("DataType",),
        "humming.layer": ("HummingLayerMeta", "HummingMethod"),
        "humming.schema": (
            "BaseInputSchema",
            "BaseWeightSchema",
            "HummingInputSchema",
            "HummingWeightSchema",
        ),
        "humming.utils": (),
        "humming.utils.weight": ("quantize_weight",),
    }

    registry: dict[str, ModuleType] = {}
    for name, attrs in submodules.items():
        mod = ModuleType(name)
        for attr in attrs:
            setattr(mod, attr, type(attr, (), {}))
        registry[name] = mod

    # wire parent references
    registry["humming"].config = registry["humming.config"]
    registry["humming"].dtypes = registry["humming.dtypes"]
    registry["humming"].layer = registry["humming.layer"]
    registry["humming"].schema = registry["humming.schema"]
    registry["humming"].utils = registry["humming.utils"]
    registry["humming.utils"].weight = registry["humming.utils.weight"]

    for name, mod in registry.items():
        sys.modules[name] = mod


_register_humming_stubs()

from vllm.model_executor.layers.quantization import (  # noqa: E402
    QUANTIZATION_METHODS,
    get_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import (  # noqa: E402
    QuantizationConfig,
)

from .component_config import ComponentQuantizationConfig  # noqa: E402

logger = init_logger(__name__)


# Checkpoint spellings that mean one of our constructor params. Applied before
# the signature filter, and only when the canonical key is absent, so an
# explicit Omni-side value always wins.
#
# These exist because the override builders construct through ``__init__``
# rather than ``from_config``, and the classes' own ``from_config`` is where
# such translation normally lives. Routing overrides through ``from_config``
# instead is NOT a drop-in swap: ``DiffusionInt8Config.from_config`` derives
# ``is_checkpoint_int8_serialized`` from ``quant_method`` (int8_config.py) where
# ``__init__`` defaults it to False, so the switch would flip long-standing int8
# models between online and offline quantization; and ``_build_single`` never
# receives ``quant_method`` anyway (``_pop_method_name`` strips it upstream), so
# that same ``from_config`` would raise. Aliasing is the narrow fix.
#
# NOT listed, deliberately: ``bnb_4bit_use_double_quant`` -> ``compress_statistics``.
# These aliases only ever reach the ONLINE configs (a pre-quantized checkpoint
# takes the branch in _build_bitsandbytes and never gets here), and on that path
# nested absmax is a live hazard rather than a gap: see
# ``DiffusionBitsAndBytesConfig.__init__``, where True turns MiniMax-H3's
# ``blocks.0.adaln_proj`` output into 1e30 and fails the request with
# "v must be finite". For an already-quantized checkpoint the flag is not a
# runtime knob at all — nestedness travels in the serialized QuantState
# descriptor and the model unpacks it itself (HunyuanImage-3's
# ``dequantize_double_quant``), so nothing is lost by omitting it here either.
_CHECKPOINT_KEY_ALIASES = {
    "modules_to_not_convert": "ignored_layers",  # HF / AutoRound skip list
    "bnb_4bit_quant_type": "quant_type",  # transformers BitsAndBytesConfig
}


def _accepted_params(config_cls: type) -> set[str] | None:
    """Names ``config_cls.__init__`` accepts, or None when it takes anything.

    None means "do not police this class": either the signature is unreadable
    (builtin / C-implemented ``__init__``) or it declares ``**kwargs``, which is
    an explicit opt-in to receiving extras.
    """
    try:
        params = inspect.signature(config_cls.__init__).parameters
    except (TypeError, ValueError):  # builtins / C-implemented __init__
        return None

    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return None
    return {name for name in params if name != "self"}


def _apply_checkpoint_aliases(config_cls: type, kw: dict[str, Any]) -> dict[str, Any]:
    """Rewrite known checkpoint spellings onto the params ``config_cls`` accepts."""
    accepted = _accepted_params(config_cls)
    if accepted is None:  # class takes **kwargs; it parses its own spellings
        return kw

    aliased = dict(kw)
    for alias, canonical in _CHECKPOINT_KEY_ALIASES.items():
        if alias in aliased and canonical in accepted and aliased.get(canonical) is None:
            aliased[canonical] = aliased[alias]
    return aliased


def _construct_override(config_cls: type, kw: dict[str, Any]) -> QuantizationConfig:
    """Instantiate an Omni quant config from checkpoint-derived kwargs.

    Every ``_OVERRIDES`` builder MUST go through here rather than calling its
    config class directly. ``_build_single`` resolves ``_OVERRIDES`` BEFORE the
    vLLM registry and returns immediately, so the ``_drop_unsupported_kwargs``
    call on the registry branch never runs for an overridden method — filtering
    has to happen inside the builder, against the real target class. (Filtering
    against the builder function itself is a no-op: they all take ``**kw``,
    which reads as "accepts everything".)

    This is not hypothetical. The guard was added for
    HunyuanImage-3.0-Instruct-Distil-NF4 while ``bitsandbytes`` still resolved
    through the registry; upstream then added a ``bitsandbytes`` override, and
    the merge of the two silently disarmed it — the DiT stage died on
    ``DiffusionBitsAndBytesConfig.__init__() got an unexpected keyword argument
    '_load_in_4bit'``. Neither side conflicted textually, so nothing flagged it.

    Alias BEFORE filtering: dropping is the last resort, and a checkpoint field
    that has a home must reach it rather than merely be reported as discarded.
    """
    aliased = _apply_checkpoint_aliases(config_cls, kw)
    return config_cls(**_drop_unsupported_kwargs(config_cls, aliased))


def _build_int8(**kw: Any) -> QuantizationConfig:
    """Lazy import for Int8 diffusion config (supports CUDA + NPU)."""
    from .int8_config import DiffusionInt8Config

    return _construct_override(DiffusionInt8Config, kw)


# transformers' ``BitsAndBytesConfig.to_dict()`` vocabulary. Any of these in a
# checkpoint's quantization_config means the weights on disk are ALREADY
# quantized — which is the one case DiffusionBitsAndBytesConfig cannot serve.
_SERIALIZED_BNB_MARKERS = (
    "load_in_4bit",
    "_load_in_4bit",
    "load_in_8bit",
    "_load_in_8bit",
    "bnb_4bit_quant_type",
    "llm_int8_skip_modules",
)


def _build_bitsandbytes(**kw: Any) -> QuantizationConfig:
    """BitsAndBytes: vLLM's config for pre-quantized checkpoints, ours for online.

    Two unrelated callers share this method name, and they need different classes:

    * ONLINE — upstream #5037 registered this override to quantize BF16/FP16
      diffusion transformers at load time. That is what DiffusionBitsAndBytesConfig
      does, and all it does: BnBOnlineLinearMethod calls ``bnb_F.quantize_4bit`` on
      ``layer.weight`` unconditionally (bitsandbytes_config.py).
    * OFFLINE — HunyuanImage-3's DiT ships an already-quantized NF4 checkpoint. Its
      weights are uint8-packed 4-bit with separate ``quant_state.bitsandbytes__nf4``
      descriptors, which the model parses and binds itself
      (hunyuan_image3_transformer.py), and it reads
      ``quant_config.llm_int8_skip_modules`` to keep the router gate in BF16. Both
      are vLLM-BitsAndBytesConfig APIs, written (f27e506c) while this method still
      resolved through the vLLM registry.

    Handing an offline checkpoint to the online config is not a near-miss: it would
    re-quantize 4-bit packed bytes as if they were activations-scale floats. The
    upstream merge that added this override made exactly that swap silently, so
    route by the only signal that separates the two — whether the checkpoint's own
    config says it is already quantized.
    """
    if any(marker in kw for marker in _SERIALIZED_BNB_MARKERS):
        # Reproduce the registry branch of _build_single verbatim: this IS the
        # path these checkpoints took before the override existed.
        config_cls = get_quantization_config("bitsandbytes")
        return config_cls(**_drop_unsupported_kwargs(config_cls, kw))

    from .bitsandbytes_config import DiffusionBitsAndBytesConfig

    return _construct_override(DiffusionBitsAndBytesConfig, kw)


def _build_mxfp8(**kw: Any) -> QuantizationConfig:
    """Lazy import for W8A8 MXFP8 diffusion config (NPU only)."""
    from .mxfp8_config import DiffusionMXFP8Config

    return _construct_override(DiffusionMXFP8Config, kw)


def _build_mxfp4(**kw: Any) -> QuantizationConfig:
    """Lazy import for W4A4 MXFP4 diffusion config (NPU only)."""
    from .mxfp4_config import DiffusionMXFP4Config

    return _construct_override(DiffusionMXFP4Config, kw)


def _build_mxfp4_dualscale(**kw: Any) -> QuantizationConfig:
    """Lazy import for MXFP4 DualScale + BF16 mixed diffusion config (NPU only).

    Offline mode (is_checkpoint_serialized=True):
        ignored_layers from config.json marks interleaved BF16 fallback layers.
        All other linear layers use W4A4 MXFP4 DualScale.

    Online mode (is_checkpoint_serialized=False):
        num_bf16_fallback_layers leading transformer blocks use BF16 original weights
        (default 5 when not specified). Remaining blocks use W4A4 MXFP4 DualScale online.
    """
    from .mxfp4_config import DiffusionMXFP4DualScaleMixedConfig

    return _construct_override(DiffusionMXFP4DualScaleMixedConfig, kw)


def _build_inc(**kw: Any) -> QuantizationConfig:
    """Lazy import for INC/AutoRound config with checkpoint kwarg normalization."""
    from .inc_config import OmniINCConfig

    # Map checkpoint key 'bits' to INCConfig's 'weight_bits'
    if "bits" in kw and "weight_bits" not in kw:
        kw["weight_bits"] = kw.pop("bits")

    # Was a hand-rolled signature filter; _construct_override does the same thing
    # and additionally logs what it discarded, so a checkpoint field this runtime
    # silently ignores becomes visible instead of vanishing.
    return _construct_override(OmniINCConfig, kw)


_OVERRIDES: dict[str, Callable[..., QuantizationConfig]] = {
    "int8": _build_int8,
    "bitsandbytes": _build_bitsandbytes,
    "mxfp8": _build_mxfp8,
    "mxfp4": _build_mxfp4,
    "mxfp4_dualscale": _build_mxfp4_dualscale,
    "inc": _build_inc,
    "auto-round": _build_inc,
    "auto_round": _build_inc,
}


def _compute_supported_quantization_methods() -> list[str]:
    return list(dict.fromkeys(QUANTIZATION_METHODS + list(_OVERRIDES.keys())))


SUPPORTED_QUANTIZATION_METHODS: list[str] = _compute_supported_quantization_methods()


def _refresh_registered_methods() -> None:
    SUPPORTED_QUANTIZATION_METHODS[:] = _compute_supported_quantization_methods()
    global _CACHED_ALIAS_MAP
    _CACHED_ALIAS_MAP = None


def register_quantization_override(method: str, builder: Callable[..., QuantizationConfig]) -> None:
    """Register an Omni-specific quantization config builder."""
    _OVERRIDES[_normalize_method_name(method)] = builder
    _refresh_registered_methods()


def _build_reverse_alias_map() -> dict[str, str]:
    """Build a mapping from normalized method aliases to canonical names.

    All keys in _OVERRIDES that share the same builder function are considered
    aliases of each other. The canonical name is the first key (in definition
    order) that maps to a given builder — i.e. the one returned by
    builder().get_name().
    """
    builder_to_first_key: dict[Callable[..., QuantizationConfig], str] = {}
    for key in _OVERRIDES:
        builder = _OVERRIDES[key]
        if builder not in builder_to_first_key:
            builder_to_first_key[builder] = key

    result: dict[str, str] = {}
    for key, builder in _OVERRIDES.items():
        canonical = builder_to_first_key[builder]
        result[key.lower().replace("-", "_")] = canonical
    return result


_CACHED_ALIAS_MAP: dict[str, str] | None = None


def _normalize_quant_method_alias(method: str | None) -> str | None:
    """Map a method name (or any of its aliases) to its canonical internal name.
    Returns the input unchanged if it is not a known alias.
    """
    if method is None:
        return None
    global _CACHED_ALIAS_MAP
    if _CACHED_ALIAS_MAP is None:
        _CACHED_ALIAS_MAP = _build_reverse_alias_map()
    normalized = method.lower().replace("-", "_")
    return _CACHED_ALIAS_MAP.get(normalized, normalized)


_MODEL_OPT_METHODS = {
    "modelopt",
    "modelopt_fp4",
    "modelopt_mixed",
}
_MODEL_OPT_FP8_ALGOS = {
    "FP8",
    "FP8_PER_CHANNEL_PER_TOKEN",
}
_MODEL_OPT_NVFP4_ALGOS = {
    "NVFP4",
}
_MODEL_OPT_MIXED_ALGOS = {
    "MIXED_PRECISION",
}


def _normalize_method_name(method: Any) -> str:
    return str(method).lower().replace("-", "_")


def _detect_modelopt_method(config: Mapping[str, Any]) -> str | None:
    quantization = config.get("quantization")
    if isinstance(quantization, Mapping):
        quant_algo = str(quantization.get("quant_algo", "")).upper()
    else:
        quant_algo = str(config.get("quant_algo", "")).upper()

    method = config.get("method", config.get("quant_method"))
    normalized_method = _normalize_method_name(method) if method is not None else None

    producer = config.get("producer")
    is_modelopt_config = normalized_method in _MODEL_OPT_METHODS or (
        isinstance(producer, Mapping) and str(producer.get("name", "")).lower() == "modelopt"
    )

    if not is_modelopt_config:
        return None

    if quant_algo:
        if quant_algo in _MODEL_OPT_FP8_ALGOS:
            return "modelopt"
        if quant_algo in _MODEL_OPT_NVFP4_ALGOS:
            return "modelopt_fp4"
        if quant_algo in _MODEL_OPT_MIXED_ALGOS:
            return "modelopt_mixed"
        return None

    if method is not None:
        if normalized_method in _MODEL_OPT_METHODS:
            return normalized_method

    return None


def _build_modelopt_from_config(method: str, config: Mapping[str, Any]) -> QuantizationConfig:
    config_cls = get_quantization_config(method)
    normalized_config = dict(config)
    normalized_config.setdefault("quant_method", method)
    return config_cls.from_config(normalized_config)


def _pop_method_name(spec: dict[str, Any]) -> str | None:
    method = spec.pop("method", None)
    if method is None:
        method = spec.pop("quant_method", None)
    return method


def _build_from_method_and_config(method: str, config: Mapping[str, Any]) -> QuantizationConfig:
    normalized_config = {"quant_method": method, **config}
    modelopt_method = _detect_modelopt_method(normalized_config)
    if modelopt_method is not None:
        return _build_modelopt_from_config(modelopt_method, normalized_config)
    return _build_single(method, **config)


def _build_single(method: str, **kwargs: Any) -> QuantizationConfig:
    """Build a single QuantizationConfig by method name.

    Resolution: _OVERRIDES first, then vLLM registry via from_config().

    Note the asymmetry: the registry branch below filters checkpoint kwargs
    here, but the _OVERRIDES branch returns before ever reaching it. Any builder
    registered in _OVERRIDES is therefore responsible for its own filtering and
    must construct through _construct_override — see its docstring for the
    upstream-merge accident that this rule exists to prevent.
    """
    method = _normalize_method_name(method)

    if method in _OVERRIDES:
        return _OVERRIDES[method](**kwargs)

    # _OVERRIDES is keyed on the normalized (underscored) name, but vLLM's
    # registry keeps each method's original spelling — and several are
    # hyphenated ("compressed-tensors"). Looking the normalized name up in the
    # registry therefore misses them: a compressed-tensors checkpoint died with
    # "Unknown quantization method: 'compressed_tensors'" while
    # SUPPORTED_QUANTIZATION_METHODS listed 'compressed-tensors' two lines
    # later. Map back to the registry's spelling instead. Computed per call
    # because plugins can extend the registry after import.
    registry_method = {_normalize_method_name(name): name for name in QUANTIZATION_METHODS}.get(method)

    if registry_method is None:
        raise ValueError(f"Unknown quantization method: {method!r}. Supported: {SUPPORTED_QUANTIZATION_METHODS}")

    config_cls = get_quantization_config(registry_method)
    checkpoint_kwargs = dict(kwargs)
    kwargs = _drop_unsupported_kwargs(config_cls, kwargs)

    try:
        return config_cls(**kwargs)
    except TypeError:
        # Not every config's __init__ is checkpoint-shaped. CompressedTensorsConfig
        # takes a parsed `target_scheme_map`, not the raw `config_groups` the
        # checkpoint carries, so direct construction can only ever fail for it —
        # from_config() is the classmethod that does that parsing, and is what
        # this function's contract has always promised. Fall back to it rather
        # than construct-or-die, keeping the existing direct path (and its kwarg
        # filtering) intact for the methods that do accept checkpoint fields.
        try:
            return config_cls.from_config(checkpoint_kwargs)
        except Exception as from_config_error:
            sig = inspect.signature(config_cls.__init__)
            raise TypeError(
                f"Cannot instantiate {config_cls.__name__} with kwargs {kwargs} "
                f"(signature: {sig}); {config_cls.__name__}.from_config() also failed "
                f"with {type(from_config_error).__name__}: {from_config_error}"
            ) from from_config_error


def _drop_unsupported_kwargs(config_cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only the kwargs ``config_cls.__init__`` actually accepts.

    A checkpoint's ``quantization_config`` is whatever the producing library
    serialized, not a vLLM constructor signature. Transformers, for example,
    writes BOTH the public ``load_in_4bit`` and its private backing field
    ``_load_in_4bit``; passing the dict through verbatim then dies with
    "Cannot instantiate BitsAndBytesConfig with kwargs {'_load_in_4bit': ...}"
    even though every meaningful value was present under its public name.
    (Seen on HunyuanImage-3.0-Instruct-Distil-NF4; see §4.4 of
    docs/实验报告/vLLM-Omni-HunyuanImage3-A100-NF4-实验与优化复盘.md, whose
    conclusion was exactly "don't hand checkpoint JSON to a runtime quant config
    verbatim — parse only the public fields".)

    Classes taking ``**kwargs`` are left alone: they opted into accepting extras.

    Underscore-prefixed keys are dropped quietly as serialization artifacts.
    Anything else that is dropped is logged at WARNING, because that is a value
    the checkpoint meant to express and this runtime does not honor.
    """
    accepted = _accepted_params(config_cls)
    if accepted is None:
        return kwargs

    kept = {key: value for key, value in kwargs.items() if key in accepted}
    dropped = sorted(set(kwargs) - set(kept))
    if dropped:
        private = [key for key in dropped if key.startswith("_")]
        unknown = [key for key in dropped if not key.startswith("_")]
        if private:
            logger.debug(
                "%s: ignoring serialization-private checkpoint quant fields %s",
                config_cls.__name__,
                private,
            )
        if unknown:
            logger.warning(
                "%s: ignoring unsupported checkpoint quantization fields %s; this runtime does not "
                "honor them, so verify the resulting numerics if any of these were load-bearing.",
                config_cls.__name__,
                unknown,
            )
    return kept


def _is_per_component_dict(spec: dict[str, Any]) -> bool:
    """Check if a dict describes per-component quantization.

    A per-component dict has no "method" / "quant_method" key and all values are
    str, dict, or None. To avoid misdetecting a flat config with
    all-string values (e.g. {"activation_scheme": "static"}), we
    require at least one value to be None or a dict with "method" /
    "quant_method".
    """
    if "method" in spec or "quant_method" in spec:
        return False
    if not all(isinstance(v, (dict, str, type(None))) for v in spec.values()):
        return False
    return any(v is None or (isinstance(v, dict) and ("method" in v or "quant_method" in v)) for v in spec.values())


def _build_component_config(spec: dict[str, Any]) -> ComponentQuantizationConfig:
    """Build ComponentQuantizationConfig from a per-component dict."""
    component_configs: dict[str, QuantizationConfig | None] = {}
    default_config: QuantizationConfig | None = None

    for prefix, value in spec.items():
        if value is None:
            config = None
        elif isinstance(value, str):
            config = _build_single(value)
        elif isinstance(value, dict):
            value = dict(value)  # avoid mutating caller's dict
            method = _pop_method_name(value)
            if method is None:
                raise ValueError(f"Component '{prefix}' config dict must have a 'method' or 'quant_method' key")
            config = _build_from_method_and_config(method, value)
        else:
            raise TypeError(f"Component '{prefix}' config must be str, dict, or None, got {type(value).__name__}")

        if prefix == "default":
            default_config = config
        else:
            component_configs[prefix] = config

    logger.info(
        "Per-component quantization: %s",
        {k: (v.get_name() if v else None) for k, v in component_configs.items()},
    )
    return ComponentQuantizationConfig(component_configs, default_config)


def build_quant_config(
    spec: str | dict[str, Any] | QuantizationConfig | None,
    **kwargs: Any,
) -> QuantizationConfig | None:
    """Build a quantization config from a flexible specification.

    Args:
        spec: None/"none", method name str, dict with "method" key,
              per-component dict, or QuantizationConfig passthrough.
        **kwargs: Extra params merged with dict spec.
    """
    if spec is None:
        return None

    if isinstance(spec, QuantizationConfig):
        return spec

    if isinstance(spec, str):
        if spec.lower() == "none":
            return None
        logger.info("Building quantization config: %s", spec)
        return _build_single(spec, **kwargs)

    if isinstance(spec, Mapping):
        spec = dict(spec)

        if _is_per_component_dict(spec):
            return _build_component_config(spec)

        modelopt_method = _detect_modelopt_method(spec)
        if modelopt_method is not None:
            logger.info("Building quantization config: %s", modelopt_method)
            return _build_modelopt_from_config(modelopt_method, spec)

        method = _pop_method_name(spec)
        if method is None:
            raise ValueError(
                "Dict quantization config must have a 'method' or 'quant_method' key or "
                "be a per-component config with component prefixes as keys."
            )
        merged = {**spec, **kwargs}
        logger.info("Building quantization config: %s", method)
        return _build_from_method_and_config(method, merged)

    raise TypeError(f"quantization config must be str, dict, QuantizationConfig, or None, got {type(spec).__name__}")


def _disk_marks_serialized(qc_kwargs: dict[str, Any], quant_config: object) -> bool:
    """Return True when config.json says serialized but the active quant_config does not.

    Matches any flag following the is_checkpoint_*_serialized naming convention,
    so new quant methods don't require updating an explicit allowlist.
    """
    for key, val in qc_kwargs.items():
        if key.startswith("is_checkpoint_") and key.endswith("_serialized"):
            if val and hasattr(quant_config, key) and not getattr(quant_config, key):
                return True
    return False


def resolve_quant_config_from_disk(
    quant_config: QuantizationConfig | None,
    disk_qc: dict[str, Any] | str | None,
) -> QuantizationConfig | None:
    """Reconcile an active quant_config against quantization_config from a transformer's config.json.

    Used when loading individual transformer blocks that each have their own config.json
    (e.g. cascade models with separate transformer and transformer_2 directories).

    Rules:
      - disk_qc is None: return quant_config unchanged.
      - quant_config is None: auto-detect from disk_qc (full build).
      - Methods mismatch: raise ValueError — prevents silent weight corruption.
      - Disk marks serialized but quant_config is online: rebuild from disk.
      - ignored_layers differ: rebuild from disk (per-transformer BF16 routing).
    """
    if disk_qc is None:
        return quant_config

    if isinstance(disk_qc, str):
        if quant_config is None:
            logger.info("Auto-detected quantization from config.json: method=%s", disk_qc)
            return build_quant_config(disk_qc)
        return quant_config

    if not isinstance(disk_qc, Mapping) or "quant_method" not in disk_qc:
        return quant_config

    qc_method: str = disk_qc["quant_method"]
    qc_kwargs: dict[str, Any] = {k: v for k, v in disk_qc.items() if k != "quant_method"}

    if quant_config is None:
        logger.info(
            "Auto-detected quantization from config.json: method=%s kwargs=%s",
            qc_method,
            qc_kwargs,
        )
        return build_quant_config(qc_method, **qc_kwargs)

    active_method = _normalize_quant_method_alias(quant_config.get_name())
    disk_method = _normalize_quant_method_alias(qc_method)
    if active_method != disk_method:
        raise ValueError(
            f"Checkpoint config.json declares quant_method={qc_method!r} but the "
            f"active quantization config is {quant_config.get_name()!r}. "
            "Pass a matching --quantization flag or omit it for auto-detection."
        )

    if _disk_marks_serialized(qc_kwargs, quant_config):
        logger.info(
            "config.json marks checkpoint as serialized; switching to offline %s mode.",
            qc_method,
        )
        return build_quant_config(qc_method, **qc_kwargs)

    # AutoRound MXFP8 checkpoints use data_type="mx_fp" instead of
    # is_checkpoint_*_serialized; rebuild so the offline path is selected.
    if qc_kwargs.get("data_type") == "mx_fp":
        logger.info("config.json declares data_type='mx_fp'; rebuilding as offline AutoRound MXFP8.")
        return build_quant_config(qc_method, **qc_kwargs)

    if (
        "ignored_layers" in qc_kwargs
        and hasattr(quant_config, "ignored_layers")
        and set(qc_kwargs.get("ignored_layers") or []) != set(quant_config.ignored_layers or [])
    ):
        logger.info("config.json ignored_layers differs from active config; rebuilding quant_config.")
        return build_quant_config(qc_method, **qc_kwargs)

    return quant_config
