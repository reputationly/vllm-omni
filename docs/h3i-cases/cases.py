"""10 H3-Image edit cases used for the reference-optimisation comparison.

Each entry: (case name, prompt, [reference image paths]).
The official case reuses the release's own prompt + reference image.
"""

M = "/nfs-models/wuhanjisuan894/models/MiniMax-H3-Image"
R = "/nfs-output/prod-baseline-20261114/h3i-cases/refs"
OFFICIAL = open(f"{M}/examples/prompt.edit.txt").read()

CASES = [
    ("01_official_podcast_1ref", OFFICIAL, [f"{M}/examples/reference1.png"]),
    (
        "02_mug_restyle_1ref",
        "Picture 1 is the mug. Place that exact mug on a rustic oak table beside a sunlit window, shallow depth of "
        "field, keep the mug's shape, colour and handle unchanged.",
        [f"{R}/mug.png"],
    ),
    (
        "03_poster_text_1ref",
        'Picture 1 is a poster. Keep its layout and graphic language identical, but change the headline text to '
        '"WINTER SALE" and recolour the palette to deep navy and silver.',
        [f"{R}/poster.png"],
    ),
    (
        "04_interior_restyle_1ref",
        "Picture 1 is a living room. Restyle it as a traditional Japanese room: tatami floor, shoji screens, low "
        "wooden table. Keep the same camera angle, room proportions and window position.",
        [f"{R}/interior.png"],
    ),
    (
        "05_portrait_oil_1ref",
        "Picture 1 is a portrait. Render the same person as a 19th-century oil painting with visible brush strokes, "
        "keep the face, pose and lighting direction.",
        [f"{R}/fisherman.png"],
    ),
    (
        "06_mountain_season_1ref",
        "Picture 1 is a mountain lake. Keep the identical composition and camera position, but change the season to "
        "late autumn at dusk with warm golden light.",
        [f"{R}/mountain.png"],
    ),
    (
        "07_style_transfer_2ref",
        "Picture 1 is the subject. Picture 2 is the visual style. Render the subject of Picture 1 using the flat "
        "graphic language, palette and typography feel of Picture 2.",
        [f"{R}/fisherman.png", f"{R}/poster.png"],
    ),
    (
        "08_product_pair_2ref",
        "Pictures 1 and 2 are two separate objects. Put both objects together in one studio photograph on a wooden "
        "desk, same lighting, consistent scale, realistic contact shadows.",
        [f"{R}/mug.png", f"{R}/robot.png"],
    ),
    (
        "09_person_room_view_3ref",
        "Picture 1 is the person, Picture 2 is the room, Picture 3 is the view outside. Produce one photorealistic "
        "image of that person standing in that room with that view through the window.",
        [f"{R}/fisherman.png", f"{R}/interior.png", f"{R}/mountain.png"],
    ),
    (
        "10_contact_sheet_4ref",
        "Pictures 1 to 4 are four images. Arrange them as a clean 2x2 editorial contact sheet with thin white "
        "gutters and a small caption strip under each tile.",
        [f"{R}/robot.png", f"{R}/mug.png", f"{R}/poster.png", f"{R}/interior.png"],
    ),
]
