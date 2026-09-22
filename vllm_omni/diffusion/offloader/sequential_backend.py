# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os
from collections.abc import Collection, Iterator
from contextlib import contextmanager

import torch
from torch import nn
from torch.distributed._tensor import DTensor  # type: ignore[attr-defined]
from vllm.logger import init_logger

from vllm_omni.diffusion.hooks import HookRegistry, ModelHook
from vllm_omni.platforms import current_omni_platform

from .base import OffloadBackend, OffloadConfig, SupportsModelCpuOffload
from .module_residency import PinnedModuleStager
from .plan_resolver import resolve_offload_plan

logger = init_logger(__name__)


def _proc_status_field(field: str) -> str:
    """Read one ``/proc/self/status`` field, for offload memory diagnostics."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith(field + ":"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "n/a"

def _capture_tensor_devices(modules: Collection[nn.Module]) -> list[tuple[torch.Tensor, torch.device]]:
    seen: set[int] = set()
    devices: list[tuple[torch.Tensor, torch.device]] = []
    for module in modules:
        for tensor in (*module.parameters(), *module.buffers()):
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            devices.append((tensor, tensor.device))
    return devices


def _restore_tensor_devices(devices: list[tuple[torch.Tensor, torch.device]]) -> None:
    first_error: BaseException | None = None
    for tensor, device in devices:
        try:
            if tensor.device != device:
                tensor.data = tensor.data.to(device)
        except BaseException as exc:
            logger.exception("Failed to restore tensor placement to %s", device)
            first_error = first_error or exc
    if first_error is not None:
        raise RuntimeError("Failed to restore one or more tensor placements") from first_error


class SequentialOffloadHook(ModelHook):
    """Hook for sequential offloading with mutual exclusion on encoder and DiT modules.

    To be used as a model-level (or "component-level") of CPU offloading method;
    When a module's forward is called, this hook offloads target modules to CPU
    and loads the current module to GPU.
    """

    _HOOK_NAME = "sequential_offload"

    def __init__(
        self,
        offload_targets: list[nn.Module],
        device: torch.device,
        pin_memory: bool = True,
        use_hsdp: bool = False,
        offload_after_context: bool = True,
        persistent_staging: bool = False,
    ):
        # Modules to offload to CPU before this module runs
        self.offload_targets = offload_targets
        self.device = device
        self.pin_memory = pin_memory
        self.use_hsdp = use_hsdp
        self.offload_after_context = offload_after_context
        self.persistent_staging = persistent_staging
        self._stager: PinnedModuleStager | None = None

    def initialize_hook(self, module: nn.Module) -> nn.Module:
        if self.persistent_staging and not self.use_hsdp:
            # Fixed staging storage keeps CUDA-graph-captured weight pointers
            # valid across offload/load swaps. HSDP keeps the plain move
            # semantics (decode graphs refuse HSDP anyway).
            self._stager = PinnedModuleStager(
                module,
                self.device,
                pin_memory=self.pin_memory,
                retain_device_storage=True,
            )
        return module

    # Attribute holding the CPU tensor a parameter or buffer was swapped in
    # *from* — the copy the weights were originally loaded into. It hangs off the
    # ``nn.Parameter``/buffer object, whose identity survives a swap, rather than
    # off ``p.data``, which is replaced wholesale on every swap-in. Attaching
    # attributes to parameters is already how vLLM carries ``weight_loader``, so
    # this is a supported use.
    _CPU_HOME_ATTR = "_omni_cpu_home"

    # Fallback landing buffer, used only for tensors that have no CPU home
    # because they were materialized directly on the device and so were never
    # swapped in from anywhere.
    _CPU_SHADOW_ATTR = "_omni_cpu_shadow"

    # Set VLLM_OMNI_OFFLOAD_COPY_BACK=1 to force the old copy-on-swap-out
    # behaviour. Only needed if something mutates weights on the device between
    # swaps, which inference does not do — see ``_to_cpu_tensor``.
    _COPY_BACK_ENV = "VLLM_OMNI_OFFLOAD_COPY_BACK"

    # Diagnostic accounting, enabled with VLLM_OMNI_OFFLOAD_DEBUG=1. In a healthy
    # run ``_home_hit_bytes`` grows and the two ``_shadow_*`` counters stay at 0:
    # every swap-out lands on a CPU copy that already existed, so no host memory
    # is allocated per request.
    _home_hit_bytes = 0
    _shadow_fresh_bytes = 0
    _shadow_reuse_bytes = 0
    _shadow_dtensor_bytes = 0

    @classmethod
    def _stash_cpu_home(cls, owner: torch.Tensor) -> None:
        """Keep a reference to the CPU tensor a swap-in is about to abandon.

        ``owner.data = owner.data.to(cuda)`` drops the last reference to the CPU
        copy, so it is freed. The next swap-out then has nowhere to land and has
        to allocate — and that allocation is what grew the host footprint. Since
        the CPU copy is exactly the bytes we would have copied back, holding onto
        it costs nothing that was not already resident at load time.
        """
        data = owner.data
        if data.device.type == "cpu" and not isinstance(data, DTensor):
            setattr(owner, cls._CPU_HOME_ATTR, data)

    @classmethod
    def _to_cpu_tensor(
        cls,
        owner: torch.Tensor,
        *,
        non_blocking: bool,
        pin_memory: bool,
    ) -> torch.Tensor:
        """Return the CPU tensor ``owner.data`` should become on swap-out.

        Model weights are read-only during inference: nothing between a swap-in
        and the matching swap-out writes to them. So the CPU copy the parameter
        was swapped in from is still byte-for-byte correct, and swapping out is a
        pointer assignment — no device-to-host copy, no host allocation.

        The copy-back this replaces was expensive twice over. It moved the whole
        model across PCIe on every swap (~28 GB per rank per request on
        MiniMax-H3 at TP=4), and it had to land in a *pinned* buffer that
        PyTorch's caching host allocator never returns to the OS. Combined with
        the per-rank CPU copies that USP replication already requires, the two
        together exceeded the host's 251 GB and the OOM killer ended the server
        during the second request.

        Set ``VLLM_OMNI_OFFLOAD_COPY_BACK=1`` to restore the copy if some future
        caller does mutate weights on the device between swaps.
        """
        data = owner.data
        if isinstance(data, DTensor):
            # DTensor has its own placement machinery; leave it on the original
            # path rather than caching a tensor whose layout we don't own.
            cls._shadow_dtensor_bytes += data.nelement() * data.element_size()
            return data.to(torch.device("cpu"), non_blocking=non_blocking)

        nbytes = data.nelement() * data.element_size()
        home = getattr(owner, cls._CPU_HOME_ATTR, None)
        if (
            home is not None
            and not os.getenv(cls._COPY_BACK_ENV)
            and home.shape == data.shape
            and home.dtype == data.dtype
        ):
            cls._home_hit_bytes += nbytes
            return home

        # No CPU home: this tensor was created on the device and has never been
        # swapped in, so there is nothing to point back at and we must copy. Keep
        # one reusable buffer per tensor rather than allocating each time, so
        # even this path settles at a constant instead of ratcheting upward.
        shadow = getattr(owner, cls._CPU_SHADOW_ATTR, None)
        if (
            shadow is None
            or shadow.shape != data.shape
            or shadow.dtype != data.dtype
            or (pin_memory and not shadow.is_pinned())
        ):
            cls._shadow_fresh_bytes += nbytes
            # ``torch.empty_like`` has no pin_memory argument, so build the
            # buffer explicitly. It is contiguous; ``copy_`` handles any layout
            # difference against the source.
            shadow = torch.empty(
                data.shape,
                dtype=data.dtype,
                device="cpu",
                pin_memory=pin_memory,
            )
            setattr(owner, cls._CPU_SHADOW_ATTR, shadow)
        else:
            cls._shadow_reuse_bytes += nbytes

        shadow.copy_(data, non_blocking=non_blocking)
        return shadow

    @classmethod
    def _move_params(
        cls,
        module: nn.Module,
        target_device: torch.device,
        *,
        non_blocking: bool = False,
        pin_memory: bool = False,
    ) -> bool:
        """Move module parameters and buffers to device.

        This cls method specifically prevents recursion device movement,
        E.g., Cache-DiT CachedBlocks has attr `transformer` as a ref to original
        transformer blocks, thus `module.to(device)` will fail for recursion calling,
        refer to
        https://github.com/vipshop/cache-dit/blob/v1.2.3/src/cache_dit/caching/cache_blocks/__init__.py#L83
        """
        to_cpu = target_device.type == "cpu"
        moved = False
        for owner in (*module.parameters(), *module.buffers()):
            if owner.data.device == target_device:
                continue
            if to_cpu:
                owner.data = cls._to_cpu_tensor(
                    owner,
                    non_blocking=non_blocking,
                    pin_memory=pin_memory,
                )
            else:
                cls._stash_cpu_home(owner)
                owner.data = owner.data.to(target_device, non_blocking=non_blocking)
            moved = True

        # BitsAndBytes keeps its dequantization metadata (absmax, code, and the
        # nested state for double quantization) in a plain ``quant_state``
        # attribute rather than a registered buffer, so the two loops above do
        # not see it. Left behind, the packed weight moves while its scales stay
        # put, and ``matmul_4bit`` reads the scales from the wrong device — an
        # illegal memory access inside the kernel, with no Python-level error to
        # point at the cause. ``LazyWeightMixin._maybe_offload_after_online_quant``
        # already special-cases this for the load-time offload; a module swapped
        # by this hook needs the same treatment on every swap.
        for submodule in module.modules():
            quant_state = getattr(submodule, "quant_state", None)
            quant_state_to = getattr(quant_state, "to", None)
            if not callable(quant_state_to):
                continue
            maybe_state = quant_state_to(target_device)
            if maybe_state is not None:
                submodule.quant_state = maybe_state
        return moved

    @staticmethod
    def _module_stager(module: nn.Module) -> PinnedModuleStager | None:
        """The persistent stager owned by ``module``'s own sequential hook, if any.

        Cross-module swaps call ``_to_cpu(target)`` through the *running*
        module's hook, so staging state must be resolved on the target's hook
        to keep its ``loaded`` bookkeeping consistent.
        """
        registry: HookRegistry | None = getattr(module, "_hook_registry", None)
        hook = registry.get_hook(SequentialOffloadHook._HOOK_NAME) if registry is not None else None
        return hook._stager if isinstance(hook, SequentialOffloadHook) else None

    def _to_cpu(self, module: nn.Module) -> None:
        stager = self._module_stager(module)
        if stager is not None:
            stager.offload()
            return
        # XPU's allocator doesn't respect stream dependencies in empty_cache,
        # so non-blocking copies can race with cache eviction. Use blocking
        # copies on XPU to avoid NULL pointer errors during DMA.
        non_blocking = not self.use_hsdp and not current_omni_platform.is_xpu()
        moved = self._move_params(
            module,
            torch.device("cpu"),
            non_blocking=non_blocking,
            pin_memory=self.pin_memory,
        )
        if moved:
            current_omni_platform.empty_cache()

        if os.getenv("VLLM_OMNI_OFFLOAD_DEBUG"):
            cls = type(self)
            logger.info(
                "[offload-debug] %s -> CPU | home-hit=%.2fGB shadow fresh=%.2fGB "
                "reuse=%.2fGB dtensor=%.2fGB | proc RssAnon=%s RssShmem=%s",
                module.__class__.__name__,
                cls._home_hit_bytes / 1024**3,
                cls._shadow_fresh_bytes / 1024**3,
                cls._shadow_reuse_bytes / 1024**3,
                cls._shadow_dtensor_bytes / 1024**3,
                _proc_status_field("RssAnon"),
                _proc_status_field("RssShmem"),
            )

    def _to_gpu(self, module: nn.Module) -> None:
        stager = self._module_stager(module)
        if stager is not None:
            stager.load()
            return
        self._move_params(module, self.device, non_blocking=False)

    def pre_forward(self, module: nn.Module, *args, **kwargs) -> tuple[tuple, dict]:
        # Offload target modules to CPU
        for target in self.offload_targets:
            self._to_cpu(target)

        # Load current module to GPU
        self._to_gpu(module)
        current_omni_platform.synchronize()

        logger.debug(
            "Swapped: %s -> CPU, %s -> %s, free memory: %.4f GB",
            [t.__class__.__name__ for t in self.offload_targets],
            module.__class__.__name__,
            f"{self.device.type}:{self.device.index}",
            current_omni_platform.get_free_memory() / 1024 / 1024 / 1024,
        )

        return args, kwargs


def apply_sequential_offload(
    dit_modules: list[nn.Module],
    encoder_modules: list[nn.Module],
    device: torch.device,
    pin_memory: bool = True,
    use_hsdp: bool = False,
    offload_initial_dits: bool = False,
    offload_dit_modules: Collection[nn.Module] | None = None,
    offload_encoder_modules: Collection[nn.Module] | None = None,
    persistent_dit_staging: bool = False,
) -> None:
    """Apply sequential offloading hooks to DiT and encoder modules.

    Registers hooks on modules to implement mutual-exclusion GPU allocation.
        - Before DiT runs, encoders are offloaded to CPU.
        - Before encoders run, DiT is offloaded to CPU.

    Args:
        dit_modules: DiT/transformer modules to register hooks on
        encoder_modules: Encoder modules to register hooks on
        device: Target GPU device for loading
        pin_memory: Whether to pin CPU memory for faster transfers
        use_hsdp: Whether HSDP is enabled (affects non_blocking behavior)
        offload_initial_dits: Whether to begin with all DiT modules on CPU.
        offload_dit_modules: DiT modules allowed to move to CPU. None selects
            every DiT for backward compatibility.
        offload_encoder_modules: Encoder/stage modules allowed to move to CPU.
            None selects every supplied module for backward compatibility.
        persistent_dit_staging: Keep DiT weights on fixed device staging
            storage across swaps, so CUDA-graph-captured weight pointers stay
            valid. Encoders keep plain move semantics so their memory is
            actually freed while the DiT runs.

    Example:
        >>> apply_sequential_offload(
        ...     dit_modules=[pipeline.transformer],
        ...     encoder_modules=[pipeline.text_encoder, pipeline.vae],
        ...     device=torch.device("cuda:0"),
        ... )
        >>> # Modules of pipeline now automatically swap between CPU and GPU
    """
    selected_dit_ids = {id(module) for module in (dit_modules if offload_dit_modules is None else offload_dit_modules)}
    selected_encoder_ids = {
        id(module) for module in (encoder_modules if offload_encoder_modules is None else offload_encoder_modules)
    }

    all_modules = [*dit_modules, *encoder_modules]
    initial_devices = _capture_tensor_devices(dit_modules) if offload_initial_dits else []
    try:
        selected_encoders = [encoder for encoder in encoder_modules if id(encoder) in selected_encoder_ids]
        # Register hooks on DiT modules (offload selected encoders and other selected DiTs).
        for i, dit_mod in enumerate(dit_modules):
            other_dits = [d for j, d in enumerate(dit_modules) if j != i and id(d) in selected_dit_ids]
            registry = HookRegistry.get_or_create(dit_mod)
            hook = SequentialOffloadHook(
                offload_targets=selected_encoders + other_dits,
                device=device,
                pin_memory=pin_memory,
                use_hsdp=use_hsdp,
                offload_after_context=id(dit_mod) in selected_dit_ids,
                persistent_staging=persistent_dit_staging,
            )
            registry.register_hook(SequentialOffloadHook._HOOK_NAME, hook)
            logger.debug("Registered offload hook for %s", dit_mod.__class__.__name__)

        # Register hooks on all execution stages so unselected resident modules can
        # still evict selected DiTs before they run.
        selected_dits = [dit for dit in dit_modules if id(dit) in selected_dit_ids]
        for enc in encoder_modules:
            registry = HookRegistry.get_or_create(enc)
            hook = SequentialOffloadHook(
                offload_targets=selected_dits,
                device=device,
                pin_memory=pin_memory,
                use_hsdp=use_hsdp,
                offload_after_context=id(enc) in selected_encoder_ids,
            )
            registry.register_hook(SequentialOffloadHook._HOOK_NAME, hook)
            logger.debug("Registered offload hook for %s", enc.__class__.__name__)

        if offload_initial_dits:
            for dit_mod in dit_modules:
                if id(dit_mod) not in selected_dit_ids:
                    continue
                _get_sequential_offload_hook(dit_mod)._to_cpu(dit_mod)
    except BaseException:
        try:
            remove_sequential_offload(all_modules)
        except BaseException:
            logger.exception("Failed to remove every sequential hook during rollback")
        try:
            _restore_tensor_devices(initial_devices)
        except BaseException:
            logger.exception("Failed to restore initial DiT placement during rollback")
        raise


def remove_sequential_offload(modules: list[nn.Module]) -> None:
    """Remove sequential offloading hooks from modules.

    Args:
        modules: Modules to remove hooks from

    Example:
        >>> all_modules = [*dit_modules, *encoder_modules]
        >>> remove_sequential_offload(all_modules)
    """
    first_error: BaseException | None = None
    for module in modules:
        try:
            registry: HookRegistry | None = getattr(module, "_hook_registry", None)
            if registry is not None:
                registry.remove_hook(SequentialOffloadHook._HOOK_NAME)
                logger.debug("Removed offload hook from %s", module.__class__.__name__)
        except BaseException as exc:
            logger.exception("Failed to remove offload hook from %s", module.__class__.__name__)
            first_error = first_error or exc
    if first_error is not None:
        raise RuntimeError("Failed to remove one or more sequential offload hooks") from first_error


def _get_sequential_offload_hook(module: nn.Module) -> SequentialOffloadHook:
    registry: HookRegistry | None = getattr(module, "_hook_registry", None)
    hook = registry.get_hook(SequentialOffloadHook._HOOK_NAME) if registry is not None else None
    if not isinstance(hook, SequentialOffloadHook):
        raise RuntimeError(f"{module.__class__.__name__} has no sequential offload hook")
    return hook


def sequential_offload_staging_active(module: nn.Module) -> bool:
    """True when the module's sequential offload hook serves fixed device storage.

    Fixed staging storage keeps CUDA-graph-captured weight pointers valid
    across offload/load swaps.
    """
    return SequentialOffloadHook._module_stager(module) is not None


@contextmanager
def sequential_offload_component(module: nn.Module) -> Iterator[None]:
    """Activate and release a hooked component called outside ``forward``."""
    hook = _get_sequential_offload_hook(module)
    try:
        hook.pre_forward(module)
        yield
    except BaseException:
        try:
            if hook.offload_after_context:
                hook._to_cpu(module)
        except Exception:
            logger.exception("Failed to release %s after component failure", module.__class__.__name__)
        raise
    else:
        if hook.offload_after_context:
            hook._to_cpu(module)


class ModelLevelOffloadBackend(OffloadBackend):
    """Model-level (sequential) offloading backend.

    Uses SequentialOffloadHook registered via HookRegistry for automatic module swapping.
    """

    def __init__(self, config: OffloadConfig, device: torch.device):
        super().__init__(config, device)
        self._offload_modules: list[nn.Module] = []  # Track modules with hooks
        self._custom_pipeline: SupportsModelCpuOffload | None = None

    # Headroom, in GiB, that must remain free on the device after the DiT joins
    # the encoder there for the swap to be skipped. Unset disables the check and
    # keeps the historical always-swap behaviour.
    _KEEP_RESIDENT_ENV = "VLLM_OMNI_OFFLOAD_KEEP_RESIDENT_GB"

    def _keep_resident_if_fits(
        self,
        dits: list[nn.Module],
        *,
        dit_names: list[str],
        encoder_names: list[str],
    ) -> bool:
        """Keep the DiT on the device alongside the encoder when there is room.

        Offloading exists to make a model fit, not as an end in itself. Once the
        DiT and the encoder both fit at once, swapping them buys nothing and
        costs a great deal: a full round trip of both models across PCIe on
        every request, plus a host landing buffer per parameter that pinned
        memory never returns to the OS. Measured on MiniMax-H3 (NF4 DiT
        15.5 GiB/rank + Qwen3-VL-32B encoder 12.8 GiB/rank at TP=4, 40 GiB
        cards): 143 GiB of pinned host memory across the four ranks and ~28 GiB
        of PCIe traffic per request, to avoid a 28.3 GiB residency that the card
        had room for all along. The host memory is what eventually draws the OOM
        killer.

        Whether the leftover is actually enough depends on peak activations,
        which vary with resolution and frame count and are not knowable here. So
        the caller states the headroom it needs via the environment and this
        stays off by default; too small a number trades a host OOM for a device
        OOM mid-denoise.
        """
        headroom_gb = os.getenv(self._KEEP_RESIDENT_ENV)
        if not headroom_gb:
            return False

        try:
            required = float(headroom_gb) * 1024**3
        except ValueError:
            logger.warning("%s=%r is not a number; ignoring.", self._KEEP_RESIDENT_ENV, headroom_gb)
            return False

        # Online quantization runs the whole bf16 model through the device and
        # then moves it back to CPU, leaving the caching allocator holding tens
        # of GiB of freed-but-retained blocks. ``cudaMemGetInfo`` counts those as
        # used, so without this the check sees ~0 GiB free and always declines.
        current_omni_platform.empty_cache()

        # Encoders and residents are already on the device at this point, so
        # free memory reflects them; only the DiT is still outstanding.
        dit_bytes = sum(
            owner.data.nelement() * owner.data.element_size()
            for dit in dits
            for owner in (*dit.parameters(), *dit.buffers())
            if owner.data.device.type == "cpu"
        )
        free_after = current_omni_platform.get_free_memory() - dit_bytes
        if free_after < required:
            logger.info(
                "Keeping DiT resident would leave %.2f GiB free, below the %.2f GiB "
                "required by %s; falling back to encoder<->DiT swapping.",
                free_after / 1024**3,
                required / 1024**3,
                self._KEEP_RESIDENT_ENV,
            )
            return False

        for dit in dits:
            SequentialOffloadHook._move_params(dit, self.device)
        current_omni_platform.synchronize()
        self.enabled = True
        logger.info(
            "Offload hooks skipped: %s (%.2f GiB) and %s both fit on %s, "
            "%.2f GiB left free. No per-request swapping, no pinned host buffers.",
            ", ".join(dit_names),
            dit_bytes / 1024**3,
            ", ".join(encoder_names),
            self.device,
            current_omni_platform.get_free_memory() / 1024**3,
        )
        return True

    def enable(self, pipeline: nn.Module) -> None:
        if self.enabled:
            logger.warning("ModelLevelOffloadBackend already enabled")
            return

        # Pipelines with non-forward component entry points own their complete
        # mutual-exclusion lifecycle. Delegate through the explicit protocol.
        if isinstance(pipeline, SupportsModelCpuOffload):
            try:
                if self.config.components is not None:
                    pipeline.enable_omni_model_cpu_offload(
                        device=self.device,
                        pin_memory=self.config.pin_cpu_memory,
                        use_hsdp=self.config.use_hsdp,
                        offload_components=self.config.components,
                    )
                else:
                    pipeline.enable_omni_model_cpu_offload(
                        device=self.device,
                        pin_memory=self.config.pin_cpu_memory,
                        use_hsdp=self.config.use_hsdp,
                    )
            except BaseException:
                try:
                    pipeline.disable_omni_model_cpu_offload()
                except BaseException:
                    logger.exception("Model-level cleanup failed while handling an enable failure")
                raise
            self._custom_pipeline = pipeline
            self.enabled = True
            logger.info(
                "Model-level offloading enabled through %s.enable_omni_model_cpu_offload",
                pipeline.__class__.__name__,
            )
            return

        resolved = resolve_offload_plan(pipeline, self.config)
        dits = [component.module for component in resolved.dits]
        encoders = [component.module for component in resolved.encoders]
        vaes = [component.module for component in resolved.vaes]
        residents = [component.module for component in resolved.residents]
        selected_dits = [component.module for component in resolved.dits if component.selected]
        selected_encoders = [component.module for component in resolved.encoders if component.selected]
        dit_names = [component.path for component in resolved.dits]
        encoder_names = [component.path for component in resolved.encoders]

        all_modules = [*dits, *encoders, *vaes, *residents]
        initial_devices = _capture_tensor_devices(all_modules)
        try:
            # ``_move_params`` rather than ``.to()``: ``nn.Module.to`` rebinds
            # ``param.data`` and drops the last reference to the CPU tensor the
            # loader built, so the encoder reaches its first swap-out with no CPU
            # home and ``_to_cpu_tensor`` has to allocate a pinned landing buffer
            # for it. ``_move_params`` stashes the CPU tensor first, so that
            # swap-out is a pointer assignment onto memory that already exists
            # and is pageable.
            #
            # Measured on MiniMax-H3, 4x A100-40G, VLLM_OMNI_OFFLOAD_DEBUG=1: the
            # encoder's first swap-out logged ``home-hit=0.00GB shadow
            # fresh=12.82GB`` while the DiT's logged ``home-hit=15.49GB`` with no
            # fresh shadow, and host Shmem went 0.3 -> 60.0 GB on the first
            # request and stayed there. That 60 GB is unmovable by compaction,
            # which is what starves the high-order free lists over a long-running
            # server.
            for encoder in encoders:
                SequentialOffloadHook._move_params(encoder, self.device)
            for vae in vaes:
                vae.to(self.device, non_blocking=True)
            for resident in residents:
                resident.to(self.device)

            if not dits:
                logger.warning("No DiT/transformer modules found, skipping model-level offloading")
                return
            if not encoders:
                # Nothing to swap against -- move DiTs to GPU and skip hooks.
                # ``_move_params`` again: bitsandbytes keeps its dequantization
                # scales in a plain ``quant_state`` attribute that
                # ``nn.Module.to`` does not walk, and a packed weight whose
                # scales stayed on CPU faults inside ``matmul_4bit`` with no
                # Python error.
                for dit in dits:
                    SequentialOffloadHook._move_params(dit, self.device)
                logger.warning("No encoder modules found, skipping model-level offloading")
                return

            if self._keep_resident_if_fits(dits, dit_names=dit_names, encoder_names=encoder_names):
                return

            apply_sequential_offload(
                dit_modules=dits,
                encoder_modules=encoders,
                device=self.device,
                pin_memory=self.config.pin_cpu_memory,
                use_hsdp=self.config.use_hsdp,
                offload_dit_modules=selected_dits,
                offload_encoder_modules=selected_encoders,
                # Fixed DiT staging keeps decode-graph weight pointers valid;
                # other platforms keep plain move semantics.
                persistent_dit_staging=self.device.type == "cuda",
            )
        except BaseException:
            try:
                remove_sequential_offload([*dits, *encoders])
            except BaseException:
                logger.exception("Failed to remove every model-level hook during rollback")
            try:
                _restore_tensor_devices(initial_devices)
            except BaseException:
                logger.exception("Failed to restore model placement during rollback")
            raise

        # Track modules for cleanup
        self._offload_modules = [*dits, *encoders]

        self.enabled = True

        logger.info(
            "Model-level offloading enabled: %s <-> %s (mutual exclusion)%s",
            ", ".join(dit_names),
            ", ".join(encoder_names),
            (
                f"; resident on GPU: {', '.join(component.path for component in resolved.residents)}"
                if resolved.residents
                else ""
            ),
        )

    def disable(self) -> None:
        if not self.enabled:
            return

        if self._custom_pipeline is not None:
            self._custom_pipeline.disable_omni_model_cpu_offload()
            self._custom_pipeline = None
            self.enabled = False
            logger.info("Model-level offloading disabled")
            return

        remove_sequential_offload(self._offload_modules)

        self._offload_modules.clear()
        self.enabled = False
        logger.info("Model-level offloading disabled")
