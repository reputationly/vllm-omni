"""``diffusion_compile_granularity='none'`` serves the transformer eagerly.

Compilation used to have no off switch: the field admitted only 'regional' and
'full', and the model runner compiled unconditionally. A quantization method
whose ``apply`` re-enters the compiled graph — W8A16's Triton bridge does, and
recurses until the stack blows — could then only be diagnosed by rebuilding the
image, and nothing distinguished a compile-induced failure from a real one.
"""

from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_every_place_that_lists_the_granularities_agrees():
    """The allowed set is written out by hand in four places.

    Widening one and missing another produces exactly the failure this test
    exists to prevent: the runner honours 'none', the config accepts it, and the
    CLI rejects it at parse time — so the flag looks broken rather than missing.
    """
    from pathlib import Path

    import vllm_omni

    root = Path(vllm_omni.__file__).parent
    sources = {
        "config/omni_config.py": '"regional", "full", "none"',
        "diffusion/data.py": '{"regional", "full", "none"}',
        "entrypoints/cli/serve.py": '["regional", "full", "none"]',
    }
    for relative, expected in sources.items():
        contents = (root / relative).read_text(encoding="utf-8")
        assert expected in contents, f"{relative} does not list 'none' as {expected}"


def test_validator_admits_none_and_still_rejects_nonsense():
    """Exercise the validator itself rather than the field annotation.

    ``__post_init__`` is what a serve command actually hits, and it kept its own
    literal set separate from the type hint — so widening only one of them would
    pass a type check and fail at startup.
    """
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    validate = OmniDiffusionConfig.__post_init__
    stub = SimpleNamespace(diffusion_compile_dynamic=True)

    for value in ("regional", "full", "none"):
        stub.diffusion_compile_granularity = value
        with pytest.raises(Exception) as excinfo:  # later fields need a real config
            validate(stub)
        assert "diffusion_compile_granularity" not in str(excinfo.value), value

    stub.diffusion_compile_granularity = "sometimes"
    with pytest.raises(ValueError, match="diffusion_compile_granularity"):
        validate(stub)


def test_runner_skips_compilation_for_none(monkeypatch):
    """The point of the flag: no torch.compile call at all, not a cheaper one."""
    from vllm_omni.diffusion.worker import diffusion_model_runner as runner_module

    called: list[str] = []
    monkeypatch.setattr(
        runner_module, "regionally_compile", lambda *a, **k: called.append("regional") or a[0], raising=True
    )

    class _Model:
        def compile(self, dynamic: bool) -> None:  # pragma: no cover - must not run
            called.append("full")

    runner = runner_module.DiffusionModelRunner.__new__(runner_module.DiffusionModelRunner)
    model = _Model()
    runner.pipeline = SimpleNamespace(transformer=model)
    runner.od_config = SimpleNamespace(diffusion_compile_granularity="none", diffusion_compile_dynamic=True)

    runner._compile_transformer("transformer")

    assert called == [], "granularity 'none' must not compile"
    assert runner.pipeline.transformer is model, "the pipeline must keep the eager model"


@pytest.mark.parametrize("granularity", ["regional", "full"])
def test_runner_still_compiles_for_the_other_values(monkeypatch, granularity):
    """The escape hatch must not quietly disable compilation for everyone else."""
    from vllm_omni.diffusion.worker import diffusion_model_runner as runner_module

    called: list[str] = []
    monkeypatch.setattr(
        runner_module, "regionally_compile", lambda model, dynamic: called.append("regional") or model, raising=True
    )

    class _Model:
        def compile(self, dynamic: bool) -> None:
            called.append("full")

    runner = runner_module.DiffusionModelRunner.__new__(runner_module.DiffusionModelRunner)
    runner.pipeline = SimpleNamespace(transformer=_Model())
    runner.od_config = SimpleNamespace(diffusion_compile_granularity=granularity, diffusion_compile_dynamic=True)

    runner._compile_transformer("transformer")

    assert called == [granularity]
