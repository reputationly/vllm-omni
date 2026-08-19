# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""参考素材的几何归一化。

rows 正比于归一后的像素数（VAE 空间压缩 16 倍，再 patch 合并 2x2，即每 32x32 像素 1 row），
而 DiT 内每 row 都要展开到 hidden 5376 穿过 50 层——这条链决定了显存峰值，
所以这里的几何算错会直接表现为 OOM。2026-08-15 在 40G A100 x4 实测标定：
1344x768 参考图在短边 2048 下是 7168 rows、768 下是 1008 rows，引擎日志逐 row 吻合。
"""

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

# 每 32x32 像素 1 row（VAE 16 倍 x patch 2）。
_PIXELS_PER_ROW_EDGE = 32


def _rows(width: int, height: int) -> int:
    return (width // _PIXELS_PER_ROW_EDGE) * (height // _PIXELS_PER_ROW_EDGE)


class _SizeOnlyImage:
    """只暴露 .size 的替身。

    _reference_image_shape 只读 image.size，其余全是算术。下面的全量扫描要调它一万六千多次，
    用 Image.new 会为每次调用分配真实 RGB 缓冲（5760x2304 单次近 40MB，累计上百 GB），
    白白拖慢每一次 PR 的 CPU 测试。真 PIL 对象的接口兼容性由
    test_accepts_real_pil_image 单独把关，替身与真实实现一旦脱节会在那里报错。
    """

    __slots__ = ("size",)

    def __init__(self, width: int, height: int):
        self.size = (width, height)


def _shape(width: int, height: int):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _reference_image_shape

    return _reference_image_shape(_SizeOnlyImage(width, height))


@pytest.mark.parametrize(("width", "height"), [(1344, 768), (1024, 1024), (2560, 1024)])
def test_accepts_real_pil_image(width, height):
    """替身与真 PIL 对象必须给出相同结果——否则上面所有扫描都是在测一个幻觉。"""
    from PIL import Image

    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _reference_image_shape

    assert _reference_image_shape(Image.new("RGB", (width, height))) == _shape(width, height)


def _short_edge() -> int:
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _reference_image_short_edge

    return _reference_image_short_edge()


# ---------------------------------------------------------------- 默认行为


@pytest.mark.parametrize(
    ("width", "height", "expected_rows"),
    [
        # 实测标定点：引擎日志 rows video=115024 = 107856（目标序列）+ 7168（本图）。
        (1344, 768, 7168),
        # 小图被**放大**：scale 没有 min(1.0, …) 上限，1024 -> 2048 是纯插值。
        (1024, 1024, 4096),
        (512, 512, 4096),
        (256, 256, 4096),
    ],
)
def test_default_normalises_short_edge_to_2048_and_upscales_small_images(width, height, expected_rows):
    """默认（未设 env）短边一律归一到 2048，比它小的图会被放大。"""
    out_width, out_height = _shape(width, height)
    assert min(out_width, out_height) == 2048
    assert _rows(out_width, out_height) == expected_rows


def test_reference_image_has_no_area_cap_unlike_video_and_output():
    """参考图是唯一没有面积封顶的一条路——最宽的比例会撑出 10 倍于视频封顶的像素数。

    参考视频与输出画布都按 MINIMAX_H3_MAX_PIXELS = 768*1344 封顶，参考图没有。
    """
    from vllm_omni.diffusion.models.minimax_h3.reference_video import MINIMAX_H3_MAX_PIXELS

    # 比例上限 2.5（超过会被 _reference_image_shape 拒绝）。
    out_width, out_height = _shape(2560, 1024)
    assert (out_width, out_height) == (5120, 2048)
    assert out_width * out_height > 10 * MINIMAX_H3_MAX_PIXELS
    # 单图 10240 rows，是 1.75 比例那档（7168）的 1.43 倍。
    assert _rows(out_width, out_height) == 10240


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (255, 255),  # 短边低于 256
        (5761, 5760),  # 长边超过 5760
        (2600, 1000),  # 比例 2.6 > 2.5
        (1000, 2600),  # 比例 0.385 < 0.4
    ],
)
def test_rejects_out_of_contract_source_images(width, height):
    """上传原图的边长与比例校验（与官方声明的 [256,5760] / [0.4,2.5] 一致）。"""
    from vllm_omni.errors import OmniClientError

    with pytest.raises(OmniClientError):
        _shape(width, height)


# ---------------------------------------------- VLLM_OMNI_H3_REF_IMAGE_SHORT_EDGE


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, 2048),  # 未设 -> 默认
        ("", 2048),  # 空串 -> 默认
        ("   ", 2048),  # 纯空白 -> 默认
        ("768", 768),  # 正常值
        ("32", 32),  # 下界
        ("5760", 5760),  # 上界
        ("16", 32),  # 越下界 -> 夹取（更省，保住意图）
        ("5761", 2048),  # 越上界 -> **回落默认**，绝不夹到 5760（那是最贵的一档）
        ("999999", 2048),
        ("abc", 2048),  # 非整数没有可尊重的意图 -> 回落默认
        ("76.8", 2048),
    ],
)
def test_short_edge_env_parsing(monkeypatch, raw, expected):
    """越界的两侧处理**不对称**，因为"安全方向"本身不对称。

    越下界夹取（16 -> 32）：更小必定更省显存，保住了"调小"的意图。
    越上界回落默认（5761 -> 2048）而非夹到 5760：夹到上界等于把手滑变成最贵的一档。
    """
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE_ENV,
    )

    if raw is None:
        monkeypatch.delenv(MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE_ENV, raising=False)
    else:
        monkeypatch.setenv(MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE_ENV, raw)
    assert _short_edge() == expected


@pytest.mark.parametrize("raw", ["1", "16", "31", "5761", "99999", "abc", "-5", "76.8"])
def test_invalid_env_never_resolves_more_expensive_than_default(monkeypatch, raw):
    """核心不变式：任何非法输入都不得解析出比默认 2048 更贵的目标。

    夹到上界 5760 曾经违反这条——2560x1024 在 5760 下是 81000 rows/图，是 2048 档（10240）
    的 7.9 倍，一个手滑就制造出这个开关本要防的 OOM。
    """
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE_ENV,
    )

    monkeypatch.setenv(MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE_ENV, raw)
    assert _short_edge() <= 2048
    # 换算到最坏比例的单图 rows，同样不得超过默认档。
    assert _rows(*_shape(2560, 1024)) <= 10240


def test_short_edge_768_matches_measured_row_counts(monkeypatch):
    """短边 768 的实测标定：9 张 1344x768 从 64512 rows 压到 9072，这是 9 图解 OOM 的全部原因。"""
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE_ENV,
    )

    monkeypatch.setenv(MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE_ENV, "768")
    out_width, out_height = _shape(1344, 768)
    assert (out_width, out_height) == (1344, 768)
    assert _rows(out_width, out_height) == 1008
    assert 9 * 1008 == 9072


# ---------------------------------------------- VLLM_OMNI_H3_REF_IMAGE_NO_UPSCALE


@pytest.mark.parametrize(
    ("width", "height", "short_edge", "expected"),
    [
        # 短边已小于目标 -> 保持原生（不再插值放大）。
        (1024, 1024, None, (1024, 1024)),
        (512, 512, "768", (512, 512)),
        (1344, 768, "2048", (1344, 768)),
        # 短边大于目标 -> 照常缩小。
        (4096, 4096, "768", (768, 768)),
        (2560, 1024, "768", (1920, 768)),
        # 短边恰好等于目标 -> 恒等。
        (1344, 768, "768", (1344, 768)),
    ],
)
def test_no_upscale_keeps_native_resolution(monkeypatch, width, height, short_edge, expected):
    """置位后 scale 加 min(1.0, …) 上限：只缩不放。"""
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        MINIMAX_H3_REFERENCE_IMAGE_NO_UPSCALE_ENV,
        MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE_ENV,
    )

    monkeypatch.setenv(MINIMAX_H3_REFERENCE_IMAGE_NO_UPSCALE_ENV, "1")
    if short_edge is None:
        monkeypatch.delenv(MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE_ENV, raising=False)
    else:
        monkeypatch.setenv(MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE_ENV, short_edge)
    assert _shape(width, height) == expected


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (1008, 1008),  # 31.5 个 patch：最近取整会抬到 1024
        (1080, 1440),  # 33.75 / 45：只有宽被抬到 1088，顺带改掉比例
        (1000, 1000),
        (257, 257),  # 贴近源边长下限
        (2550, 1023),
        (1919, 1079),
        (5750, 2303),
    ],
)
def test_no_upscale_never_enlarges_non_aligned_sources(monkeypatch, width, height):
    """源尺寸不是 32 的倍数时也绝不放大。

    只加 min(1.0, scale) 不够：_align_multiple 是最近取整，1008 会被抬成 1024、
    1080x1440 会被抬成 1088x1440——既插值了像素，又改了宽高比。
    """
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        MINIMAX_H3_REFERENCE_IMAGE_NO_UPSCALE_ENV,
    )

    monkeypatch.setenv(MINIMAX_H3_REFERENCE_IMAGE_NO_UPSCALE_ENV, "1")
    out_width, out_height = _shape(width, height)
    assert out_width <= width, (width, out_width)
    assert out_height <= height, (height, out_height)
    assert out_width % 32 == 0 and out_height % 32 == 0


def test_no_upscale_leaves_aligned_sources_untouched(monkeypatch):
    """源尺寸本就对齐时，向下封顶不得把它压小。"""
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        MINIMAX_H3_REFERENCE_IMAGE_NO_UPSCALE_ENV,
    )

    monkeypatch.setenv(MINIMAX_H3_REFERENCE_IMAGE_NO_UPSCALE_ENV, "1")
    for width, height in [(1344, 768), (1024, 1024), (512, 512), (256, 256)]:
        assert _shape(width, height) == (width, height)


def test_no_upscale_still_downscales(monkeypatch):
    """向下封顶不能反过来妨碍正常的缩小。"""
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        MINIMAX_H3_REFERENCE_IMAGE_NO_UPSCALE_ENV,
        MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE_ENV,
    )

    monkeypatch.setenv(MINIMAX_H3_REFERENCE_IMAGE_NO_UPSCALE_ENV, "1")
    monkeypatch.setenv(MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE_ENV, "768")
    assert _shape(4096, 4096) == (768, 768)
    assert _shape(2550, 1023) == (1920, 768)


def test_no_upscale_is_never_more_expensive_than_default(monkeypatch):
    """只缩不放对任何合法输入都不会增加 rows——这是它可以安全默认开启的前提。"""
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        MINIMAX_H3_REFERENCE_IMAGE_NO_UPSCALE_ENV,
    )

    sizes = [
        (256, 256),
        (512, 512),
        (1024, 1024),
        (1344, 768),
        (2560, 1024),
        (4096, 2048),
        (5760, 2304),
        # 非 32 倍数：最近取整曾在这里越过契约。
        (1008, 1008),
        (1080, 1440),
        (2550, 1023),
    ]

    monkeypatch.delenv(MINIMAX_H3_REFERENCE_IMAGE_NO_UPSCALE_ENV, raising=False)
    baseline = {size: _rows(*_shape(*size)) for size in sizes}

    monkeypatch.setenv(MINIMAX_H3_REFERENCE_IMAGE_NO_UPSCALE_ENV, "1")
    for size in sizes:
        assert _rows(*_shape(*size)) <= baseline[size], size


def test_no_upscale_defaults_off(monkeypatch):
    """默认必须关闭：线上行为零变化，画质 A/B 之前不翻默认值。"""
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        MINIMAX_H3_REFERENCE_IMAGE_NO_UPSCALE_ENV,
        _reference_image_no_upscale,
    )

    monkeypatch.delenv(MINIMAX_H3_REFERENCE_IMAGE_NO_UPSCALE_ENV, raising=False)
    assert _reference_image_no_upscale() is False
    for raw in ("0", "", "no", "false"):
        monkeypatch.setenv(MINIMAX_H3_REFERENCE_IMAGE_NO_UPSCALE_ENV, raw)
        assert _reference_image_no_upscale() is False
    for raw in ("1", "true", "True"):
        monkeypatch.setenv(MINIMAX_H3_REFERENCE_IMAGE_NO_UPSCALE_ENV, raw)
        assert _reference_image_no_upscale() is True


# ---------------------------------------------- VLLM_OMNI_H3_REF_IMAGE_MAX_PIXELS

# = 768*1344，与参考视频 MINIMAX_H3_MAX_PIXELS、输出画布 MINIMAX_H3_OUTPUT_MAX_PIXELS 对齐。
_ALIGNED_CAP = 768 * 1344


def _set_cap(monkeypatch, value):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        MINIMAX_H3_REFERENCE_IMAGE_MAX_PIXELS_ENV,
    )

    if value is None:
        monkeypatch.delenv(MINIMAX_H3_REFERENCE_IMAGE_MAX_PIXELS_ENV, raising=False)
    else:
        monkeypatch.setenv(MINIMAX_H3_REFERENCE_IMAGE_MAX_PIXELS_ENV, str(value))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, 0),  # 未设 -> 不封顶（既有行为）
        ("", 0),
        ("0", 0),  # 显式关闭
        (str(_ALIGNED_CAP), _ALIGNED_CAP),
        ("-1", 0),  # 负数 -> 关闭，不是夹取（意图不明）
        ("abc", 0),
        ("100", 32 * 32),  # 小于一个 patch 的面积 -> 夹到 1024
    ],
)
def test_max_pixels_env_parsing(monkeypatch, raw, expected):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _reference_image_max_pixels

    _set_cap(monkeypatch, raw)
    assert _reference_image_max_pixels() == expected


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (2560, 1024),  # 比例 2.5，上限
        (1024, 2560),  # 比例 0.4，下限
        (1344, 768),  # 比例 1.75，实测素材
        (1024, 1024),  # 1:1
        (5760, 2304),
    ],
)
def test_area_cap_decouples_row_cost_from_aspect_ratio(monkeypatch, width, height):
    """封顶后任何比例的单图恒 <=1008 rows —— 这正是包络实测那 1.8% 缺口的收口方式。"""
    _set_cap(monkeypatch, _ALIGNED_CAP)
    out_width, out_height = _shape(width, height)
    assert out_width * out_height <= _ALIGNED_CAP, (width, height)
    assert _rows(out_width, out_height) <= 1008, (width, height)


def test_area_cap_matches_the_reference_video_ceiling(monkeypatch):
    """参考图封顶值与参考视频用的是同一个常量口径，不是另发明一个数。"""
    from vllm_omni.diffusion.models.minimax_h3.reference_video import MINIMAX_H3_MAX_PIXELS

    assert _ALIGNED_CAP == MINIMAX_H3_MAX_PIXELS

    _set_cap(monkeypatch, MINIMAX_H3_MAX_PIXELS)
    for width, height in [(2560, 1024), (1024, 2560), (5760, 2304)]:
        out_width, out_height = _shape(width, height)
        assert out_width * out_height <= MINIMAX_H3_MAX_PIXELS


def test_area_cap_is_never_more_expensive_than_uncapped(monkeypatch):
    """封顶只会让 rows 变少或不变，绝不会变多。"""
    sizes = [(256, 256), (512, 512), (1024, 1024), (1344, 768), (2560, 1024), (1024, 2560), (5760, 2304)]

    _set_cap(monkeypatch, None)
    baseline = {size: _rows(*_shape(*size)) for size in sizes}

    _set_cap(monkeypatch, _ALIGNED_CAP)
    for size in sizes:
        assert _rows(*_shape(*size)) <= baseline[size], size


def test_area_cap_defaults_off(monkeypatch):
    """默认不封顶：对极端比例的图封顶是真实降分辨率，翻默认值前需要画质 A/B。"""
    _set_cap(monkeypatch, None)
    assert _shape(2560, 1024) == (5120, 2048)


# --------------------------------------------------------- Turbo `match` policy


@pytest.mark.parametrize(
    ("target_width", "target_height", "expected"),
    [
        (832, 480, (992, 384)),
        (864, 480, (1024, 416)),
        (1344, 768, (1632, 640)),
    ],
)
def test_match_follows_each_request_canvas(target_width, target_height, expected):
    """Pin ModelTC's formula and nearest-32 rounding at common product tiers."""
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _reference_image_shape

    actual = _reference_image_shape(
        _SizeOnlyImage(1664, 656),
        aspect_ratio_range=(0.25, 4.0),
        target_canvas=(target_width, target_height),
    )
    assert actual == expected


def test_match_never_intentionally_upscales_a_small_reference():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _reference_image_shape

    assert _reference_image_shape(
        _SizeOnlyImage(512, 512),
        aspect_ratio_range=(0.25, 4.0),
        target_canvas=(1344, 768),
    ) == (512, 512)


def test_match_480p_is_not_the_old_fixed_768p_cap():
    """The regression this change fixes: these policies coincide only at 768p."""
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _reference_image_shape

    image = _SizeOnlyImage(1664, 656)
    matched = _reference_image_shape(
        image,
        aspect_ratio_range=(0.25, 4.0),
        target_canvas=(832, 480),
    )
    fixed = _reference_image_shape(
        image,
        aspect_ratio_range=(0.25, 4.0),
        short_edge=2048,
        no_upscale=True,
        max_pixels=768 * 1344,
    )
    assert matched == (992, 384)
    assert fixed == (1600, 608)
    assert _rows(*fixed) > 2.5 * _rows(*matched)


@pytest.mark.parametrize(
    ("width", "height"),
    [(256, 256), (512, 512), (1024, 1024), (1664, 656), (2560, 1024), (1024, 2560)],
)
def test_fixed_area_is_behaviorally_identical_to_the_previous_base_profile(width, height):
    """Remove the redundant 2048 term without changing Base image geometry."""
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _reference_image_shape

    image = _SizeOnlyImage(width, height)
    previous = _reference_image_shape(
        image,
        aspect_ratio_range=(0.25, 4.0),
        short_edge=2048,
        no_upscale=True,
        max_pixels=768 * 1344,
    )
    fixed_area = _reference_image_shape(
        image,
        aspect_ratio_range=(0.25, 4.0),
        fixed_area_pixels=768 * 1344,
    )
    assert fixed_area == previous


def test_worst_case_nine_images_within_validated_envelope(monkeypatch):
    """9 张最坏比例的图，封顶后总 rows 回到已验证包络之内。

    未封顶时 9 张 2.5 比例的图是 92160 rows，最坏用例约 223600 —— 超出实测通过的 219744。
    封顶后 9 张恒 <=9072，与 1.75 比例素材完全一致，尾巴被彻底收掉。
    """
    _set_cap(monkeypatch, None)
    uncapped = 9 * _rows(*_shape(2560, 1024))
    assert uncapped == 92160

    _set_cap(monkeypatch, _ALIGNED_CAP)
    capped = 9 * _rows(*_shape(2560, 1024))
    assert capped <= 9072
    # 目标序列 107856 + 15s 参考视频 102816 + 参考图，须落在实测通过的 219744 之内。
    assert 107856 + 102816 + capped <= 219744


# ---------------------------------------------------------------- 参考视频


@pytest.mark.parametrize(
    ("width", "height"),
    [(1344, 768), (3840, 2160), (854, 480), (480, 270)],
)
def test_reference_video_target_depends_only_on_aspect_ratio(width, height):
    """参考视频的目标尺寸只由宽高比推出，原分辨率被丢弃——所以小视频同样会被放大。"""
    from vllm_omni.diffusion.models.minimax_h3.reference_video import _reference_video_shape

    # 16:9 一族（比例相同）无论输入多大多小，目标完全一致。
    assert _reference_video_shape(width, height) == _reference_video_shape(1344, 768)


def test_reference_video_is_area_capped_unlike_reference_image():
    """参考视频有面积封顶，参考图没有——同一份代码里的不对称。"""
    from vllm_omni.diffusion.models.minimax_h3.reference_video import (
        MINIMAX_H3_MAX_PIXELS,
        _reference_video_shape,
    )

    for width, height in [(2560, 1024), (1024, 2560), (5760, 2304)]:
        out_width, out_height = _reference_video_shape(width, height)
        assert out_width * out_height <= MINIMAX_H3_MAX_PIXELS, (width, height)


# ------------------------------------------------------------ 不变式全量扫描
#
# 前面每一节都是"针对某个已知缺陷"的定点测试——这种写法只能追认已经发现的问题，
# 无法收敛：修好夹取又漏掉对齐，修好 no_upscale 的对齐又漏掉面积封顶的对齐（三次都是
# 同一类错误的不同实例）。下面把这个函数的契约整体写成不变式，对
# 「所有 env 组合 x 尺寸网格」全量验证，新的组合性缺陷会被自动覆盖。

_SHORT_EDGES = [None, "32", "768", "1024", "2048", "5760", "5761", "abc"]
_NO_UPSCALES = [None, "1"]
# 含病态小值：上一版只扫了 768*1344 这一个值，因而漏掉了"对齐下限撑破封顶"的缺陷
# （2560x1024 配 cap=1800 会停在 64x32=2048）。枚举组合还不够，取值本身也要扫。
_MAX_PIXELS = [None, "1024", "1800", "2047", "50000", str(768 * 1344)]


def _valid_sizes():
    """覆盖源边长 [256,5760] 与比例 [0.4,2.5] 的合法网格，含大量非 32 倍数。"""
    edges = [256, 257, 320, 511, 512, 768, 1000, 1008, 1024, 1080, 1344, 1440, 1919, 2048, 3000, 4096, 5750, 5760]
    sizes = []
    for width in edges:
        for height in edges:
            if 0.4 <= width / height <= 2.5:
                sizes.append((width, height))
    return sizes


def _apply(monkeypatch, short_edge, no_upscale, max_pixels):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        MINIMAX_H3_REFERENCE_IMAGE_MAX_PIXELS_ENV,
        MINIMAX_H3_REFERENCE_IMAGE_NO_UPSCALE_ENV,
        MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE_ENV,
    )

    for env, value in (
        (MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE_ENV, short_edge),
        (MINIMAX_H3_REFERENCE_IMAGE_NO_UPSCALE_ENV, no_upscale),
        (MINIMAX_H3_REFERENCE_IMAGE_MAX_PIXELS_ENV, max_pixels),
    ):
        if value is None:
            monkeypatch.delenv(env, raising=False)
        else:
            monkeypatch.setenv(env, value)


@pytest.mark.parametrize("short_edge", _SHORT_EDGES)
@pytest.mark.parametrize("no_upscale", _NO_UPSCALES)
@pytest.mark.parametrize("max_pixels", _MAX_PIXELS)
def test_invariants_hold_for_every_env_combination(monkeypatch, short_edge, no_upscale, max_pixels):
    """无论 env 怎么组合，这四条必须恒成立。"""
    _apply(monkeypatch, short_edge, no_upscale, max_pixels)
    cap = int(max_pixels) if max_pixels else 0

    for width, height in _valid_sizes():
        out_width, out_height = _shape(width, height)
        ctx = (width, height, short_edge, no_upscale, max_pixels, out_width, out_height)

        # I1 输出必须可被 patch 整除且非零，否则 rows 计算与下游打包都会错。
        assert out_width % 32 == 0 and out_height % 32 == 0, ctx
        assert out_width >= 32 and out_height >= 32, ctx

        # I2 开了只缩不放就绝不能放大——含源尺寸非 32 倍数的情形。
        if no_upscale:
            assert out_width <= width and out_height <= height, ctx

        # I3 设了面积封顶就必须真的封住——含对齐取整之后。
        if cap:
            assert out_width * out_height <= cap, ctx

        # I4 比例误差必须落在**推导出来**的界内，而不是随手拍的常数。这里有两种机制，
        # 不能套同一个界：
        #   取整区（两边都 > 32）：out = 源边长 x scale ± 16（半个 patch），单边相对误差
        #       <= 16/out = 0.5/patch，故比例的绝对误差 <= r * (0.5/pw + 0.5/ph)。
        #   触底区（任一边被 _align_multiple 的 max(32, …) 钳住）：这不是取整而是**钳位**，
        #       比例不再由取整支配（320x768 配 cap=1024 会被压成 32x32，比例 0.42 -> 1.00）。
        #       此时不对比例作断言，只保证输出合法——这是 32 像素网格的数学下限，也正是
        #       短边不该配到 256 以下、封顶不该配到几个 patch 的原因。
        source_ratio = width / height
        if min(out_width, out_height) > 32:
            # 系数取决于对齐方向：最近取整误差 <= 半个 patch；而只缩不放的钳位与封顶的
            # 向下对齐都是 floor，误差可达**一整个** patch（511 -> 480 就掉了 31 像素）。
            slack = 1.0 if (no_upscale or cap) else 0.5
            patches_width = out_width / 32
            patches_height = out_height / 32
            bound = source_ratio * (slack / patches_width + slack / patches_height)
            assert abs(out_width / out_height - source_ratio) <= bound + 1e-9, ctx


@pytest.mark.parametrize("short_edge", _SHORT_EDGES)
def test_enabling_a_safety_switch_never_increases_rows(monkeypatch, short_edge):
    """两个安全开关都必须是单调的：打开只会更省，绝不会更贵。

    这是它们能被安全推荐给运维的前提——否则"为了省显存打开开关"可能适得其反。
    """
    sizes = _valid_sizes()

    _apply(monkeypatch, short_edge, None, None)
    baseline = {size: _rows(*_shape(*size)) for size in sizes}

    for no_upscale, max_pixels in [("1", None), (None, str(768 * 1344)), ("1", str(768 * 1344))]:
        _apply(monkeypatch, short_edge, no_upscale, max_pixels)
        for size in sizes:
            assert _rows(*_shape(*size)) <= baseline[size], (size, short_edge, no_upscale, max_pixels)


@pytest.mark.parametrize("no_upscale", _NO_UPSCALES)
@pytest.mark.parametrize("max_pixels", _MAX_PIXELS)
def test_smaller_short_edge_never_costs_more_rows(monkeypatch, no_upscale, max_pixels):
    """短边调小必定不增加 rows——运维调这个旋钮的唯一理由就是省显存。"""
    sizes = _valid_sizes()
    ladder = ["5760", "2048", "1024", "768", "32"]

    previous = None
    for short_edge in ladder:
        _apply(monkeypatch, short_edge, no_upscale, max_pixels)
        current = {size: _rows(*_shape(*size)) for size in sizes}
        if previous is not None:
            for size in sizes:
                assert current[size] <= previous[size], (size, short_edge, no_upscale, max_pixels)
        previous = current


@pytest.mark.parametrize("raw", ["5761", "99999", "abc", "-5", "76.8", "0", "1", "31"])
def test_malformed_env_never_costs_more_than_default(monkeypatch, raw):
    """非法 env 在**任何**尺寸上都不得比默认更贵——不只是抽查的那几个尺寸。"""
    sizes = _valid_sizes()

    _apply(monkeypatch, None, None, None)
    default = {size: _rows(*_shape(*size)) for size in sizes}

    _apply(monkeypatch, raw, None, None)
    for size in sizes:
        assert _rows(*_shape(*size)) <= default[size], (size, raw)


@pytest.mark.parametrize(
    ("short_edge", "min_patches"),
    [("32", 1), ("256", 8), ("768", 24), ("2048", 64)],
)
def test_aspect_fidelity_degrades_with_tiny_short_edge(monkeypatch, short_edge, min_patches):
    """短边配得越小，比例失真越大——这是运维不该把它配到 256 以下的量化理由。

    32 像素网格下，输出短边只有 N 个 patch 时比例相对误差上界就是 ~1/N：
    短边 32 只有 1 个 patch，320x256（比例 1.25）会被压成 32x32（比例 1.00）。
    """
    _apply(monkeypatch, short_edge, None, None)
    worst = 0.0
    for width, height in _valid_sizes():
        out_width, out_height = _shape(width, height)
        assert min(out_width, out_height) / 32 >= min_patches or min(width, height) < int(short_edge)
        worst = max(worst, abs(out_width / out_height - width / height) / (width / height))
    # 相对误差与粒度同阶。
    assert worst <= 1.0 / min_patches + 1e-9, (short_edge, worst)


@pytest.mark.parametrize("cap", [1024, 1500, 1800, 2047, 2048, 3000, 50000, 768 * 1344])
def test_cap_is_honoured_even_at_pathological_values(monkeypatch, cap):
    """封顶必须真的封住，包括对齐下限本身就逼近封顶的病态取值。

    _align_multiple_down 有 max(32, …) 下限：短边压到 32 后长边仍是 64，乘积 2048 卡住不动，
    所以 2560x1024 配 cap=1800/2047 曾经返回 64x32=2048 越界。env 保证 cap >= 32*32=1024，
    逐 patch 削长边必定能落进封顶。
    """
    _apply(monkeypatch, None, None, str(cap))
    for width, height in _valid_sizes():
        out_width, out_height = _shape(width, height)
        assert out_width * out_height <= cap, (width, height, cap, out_width, out_height)
        assert out_width >= 32 and out_height >= 32, (width, height, cap)
