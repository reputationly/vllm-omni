# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Every VAE call must sit inside ``_component_on_device``.

Once the VAEs are staged on the host between uses, an unguarded call reaches a module
whose parameters are on the CPU while its input is on the device. That is a hard failure,
but only on the request shape that happens to reach that particular encoder: the standalone
audio path was unguarded for exactly this reason -- the guard was unreachable in production
(it was gated behind layerwise offload), so nothing exercised it, and a reference-image
request never calls it at all.

This is a source check rather than a behavioural one because the invariant is "no call
site is missed", which no fixture can demonstrate: passing it would only prove that the
paths the test happens to drive are guarded.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

STAGED = ("video_vae", "audio_vae")
GUARD = "_component_on_device"
# The suffix marking a body that assumes its component is ALREADY on the device. Such a
# body is legitimate only if a same-named wrapper opens the guard around it, which
# ``test_every_resident_body_has_a_guarded_wrapper`` is what pins.
RESIDENT = "_resident"


def _pipeline_class() -> ast.ClassDef:
    tree = ast.parse(inspect.getsource(pipeline_minimax_h3))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3Pipeline":
            return node
    raise AssertionError("MiniMaxH3Pipeline not found; this test is pinned to that class")


def _methods() -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node for node in _pipeline_class().body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }


# Staged components are staged at *half* granularity in the two places that
# matter: image encoding runs inside `_component_on_device(self.video_vae
# .encoder_component)` and decoding inside `...decoder_component`, because only one
# half is needed at a time and the comment at each site says so. A call on the
# parent is therefore covered by a guard on either half. That mapping is coarse --
# it would not catch a decoder-half call sitting inside an encoder-half guard --
# but the alternative is modelling which method touches which half, and the two
# call sites this rule exists for are both on the correct half today.
_STAGED_SUBCOMPONENTS = ("encoder_component", "decoder_component")

# Methods that answer from configuration or from the argument's own shape, never
# from device tensors, so they are safe to call while the component is parked on the
# host. Listed explicitly with what each one reads, because "does it touch a
# parameter?" is not decidable from the syntax:
#   * `is_distributed_enabled` -> `self.parallel_size` and `dist.is_initialized()`.
#   * `predict_video_latent_shape` -> `self.model.processor` (a CPU-side shape
#     calculator) and the numpy array's dimensions.
# Anything absent from this tuple is assumed to touch parameters, which is the safe
# default: widening it is a deliberate edit here, not a silent one.
DEVICE_FREE_QUERIES = ("is_distributed_enabled", "predict_video_latent_shape")


def _guarded_component(item: ast.withitem, hoisted: dict[str, str]) -> str | None:
    """``with self._component_on_device(self.audio_vae)`` -> ``"audio_vae"``.

    Also resolves the hoisted form, where the guard is built into a variable first
    (``ctx = self._component_on_device(...) if needs else nullcontext()``) and the
    `with` names that variable.
    """
    context = item.context_expr
    if isinstance(context, ast.Name):
        return hoisted.get(context.id)
    if not isinstance(context, ast.Call) or not isinstance(context.func, ast.Attribute):
        return None
    if context.func.attr != GUARD or not context.args:
        return None
    arg = context.args[0]
    if not isinstance(arg, ast.Attribute):
        return None
    if arg.attr in STAGED:
        return arg.attr
    inner = arg.value
    if isinstance(inner, ast.Attribute) and inner.attr in STAGED and arg.attr in _STAGED_SUBCOMPONENTS:
        return inner.attr
    return None


def _hoisted_guards(function: ast.AST) -> dict[str, str]:
    """``ctx = self._component_on_device(self.video_vae.encoder_component)`` -> {"ctx": "video_vae"}.

    The right-hand side may be a conditional expression or any other expression, so
    this looks for the guard call anywhere inside it.
    """
    hoisted: dict[str, str] = {}
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        for inner in ast.walk(node.value):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == GUARD
                and inner.args
                and isinstance(inner.args[0], ast.Attribute)
            ):
                arg = inner.args[0]
                if arg.attr in STAGED:
                    hoisted[target.id] = arg.attr
                elif (
                    isinstance(arg.value, ast.Attribute)
                    and arg.value.attr in STAGED
                    and arg.attr in _STAGED_SUBCOMPONENTS
                ):
                    hoisted[target.id] = arg.value.attr
    return hoisted


def _unguarded_calls(function: ast.AST) -> list[tuple[str, int]]:
    """Calls on a staged component that no enclosing ``with`` in this function covers."""
    hoisted = _hoisted_guards(function)

    def walk(node: ast.AST, active: frozenset[str]) -> list[tuple[str, int]]:
        found: list[tuple[str, int]] = []
        if isinstance(node, ast.With | ast.AsyncWith):
            entering = {name for item in node.items if (name := _guarded_component(item, hoisted))}
            inner = active | entering
            for statement in node.body:
                found += walk(statement, frozenset(inner))
            for item in node.items:
                found += walk(item.context_expr, active)
            return found
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(value := node.func.value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and value.value.id == "self"
            and value.attr in STAGED
            and value.attr not in active
            and node.func.attr not in DEVICE_FREE_QUERIES
        ):
            found.append((value.attr, node.lineno))
        for child in ast.iter_child_nodes(node):
            found += walk(child, active)
        return found

    return walk(function, frozenset())


def test_no_vae_call_escapes_its_staging_guard() -> None:
    offenders = []
    for name, function in _methods().items():
        if name.endswith(RESIDENT) or name == GUARD:
            continue
        for component, lineno in _unguarded_calls(function):
            offenders.append(f"{name} calls self.{component} at line {lineno} outside {GUARD}")
    assert not offenders, "unguarded component use:\n  " + "\n  ".join(offenders)


def test_every_resident_body_has_a_guarded_wrapper() -> None:
    """The ``_resident`` exemption is only sound while the wrapper actually exists.

    Deleting or renaming a wrapper would otherwise silently widen the exemption above into
    "any function whose name ends in _resident may touch a parked VAE".
    """
    methods = _methods()
    for name, function in methods.items():
        if not name.endswith(RESIDENT):
            continue
        used = {component for component, _ in _unguarded_calls(function)}
        if not used:
            continue
        wrapper = methods.get(name[: -len(RESIDENT)])
        assert wrapper is not None, f"{name} assumes a resident component but has no wrapper"
        wrapper_hoisted = _hoisted_guards(wrapper)
        opened = {
            component
            for node in ast.walk(wrapper)
            if isinstance(node, ast.With | ast.AsyncWith)
            for item in node.items
            if (component := _guarded_component(item, wrapper_hoisted))
        }
        calls_it = any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == name
            for node in ast.walk(wrapper)
        )
        assert calls_it, f"{wrapper.name} does not call {name}, so the exemption is unbacked"
        assert used <= opened, f"{wrapper.name} guards {sorted(opened)} but {name} uses {sorted(used)}"
