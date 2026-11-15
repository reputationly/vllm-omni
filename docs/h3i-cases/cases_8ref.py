"""8/10/14-reference cases for the H3-Image reference-optimisation work.

All references are the 1024² studio-style materials in `refs/` (generated with
H3-Image itself) so the *task* is multi-reference fusion, not a collage fallback.
"""

R = "/nfs-output/prod-baseline-20261114/h3i-cases/refs"
# The release's own showcase images are large (multi-MB, up to 4K), i.e. the
# worst case for reference rows: this is the set that used to OOM at 12 refs.
S = "/nfs-models/wuhanjisuan894/models/MiniMax-H3-Image/assets/showcase"
_SHOWCASE = [
    f"{S}/art.jpg", f"{S}/char.jpg", f"{S}/edit_map.jpg", f"{S}/edit_multi.jpg",
    f"{S}/edit_render.jpg", f"{S}/photo.jpg", f"{S}/scene.jpg", f"{S}/surreal.jpg",
]

_OBJECTS = [
    f"{R}/mug.png",
    f"{R}/chair.png",
    f"{R}/sneaker.png",
    f"{R}/robot.png",
    f"{R}/poster.png",
    f"{R}/interior.png",
    f"{R}/fisherman.png",
    f"{R}/mountain.png",
]
_EXTRA = [
    f"{R}/mug.png",
    f"{R}/chair.png",
    f"{R}/sneaker.png",
    f"{R}/robot.png",
    f"{R}/poster.png",
    f"{R}/interior.png",
]

CASES = [
    (
        "8ref_catalog",
        "Pictures 1 to 8 are eight product and place photographs. Compose ONE promotional catalog page: a clean "
        "2x4 grid with thin white gutters, one short caption under each tile, consistent studio lighting, and the "
        "same colour grading across all tiles.",
        _OBJECTS[:8],
    ),
    (
        "8ref_scene",
        "Pictures 1 to 8 are references for one interior. Produce ONE wide photograph of a single coherent room "
        "that contains the mug, the armchair, the sneaker and the robot as props, uses the poster and the sofa "
        "corner as wall decoration, and looks out onto that mountain lake. Consistent perspective and lighting.",
        _OBJECTS[:8],
    ),
    (
        "8ref_moodboard",
        "Pictures 1 to 8 are a moodboard. Produce one magazine double-page spread that unifies their palette, "
        "textures and subjects into a single art-directed composition with a short headline.",
        _OBJECTS[:8],
    ),
    (
        "10ref_catalog",
        "Pictures 1 to 10 are ten photographs. Compose ONE promotional catalog page: a clean 2x5 grid with thin "
        "white gutters, one short caption under each tile, consistent studio lighting.",
        _OBJECTS + _EXTRA[:2],
    ),
    (
        "14ref_catalog",
        "Pictures 1 to 14 are fourteen photographs. Compose ONE promotional catalog page: a clean grid with thin "
        "white gutters, one short caption under each tile, consistent studio lighting.",
        _OBJECTS + _EXTRA,
    ),
    (
        "12ref_4k_catalog",
        "Pictures 1 to 12 are twelve photographs at full resolution. Compose ONE promotional catalog page: a clean "
        "grid with thin white gutters and one short caption under each tile.",
        (_SHOWCASE * 2)[:12],
    ),
    (
        "16ref_4k_catalog",
        "Pictures 1 to 16 are sixteen photographs at full resolution. Compose ONE promotional catalog page: a clean "
        "4x4 grid with thin white gutters and one short caption under each tile.",
        (_SHOWCASE * 2)[:16],
    ),
]
