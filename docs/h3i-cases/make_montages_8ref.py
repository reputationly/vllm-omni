"""Comparison strips + metrics for the 8/10/14/16-reference batch.

Run inside the image (needs PIL + numpy) with /nfs-output mounted.
"""

import json
import pathlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path("/nfs-output/prod-baseline-20261114/h3i-cases")
VARIANTS = [
    ("base", "base (no budget)"),
    ("budget_8.4Mpx_v2", "budget 8.4 Mpx"),
    ("budget_2.1Mpx_v2", "budget 2.1 Mpx"),
]
# capacity-only cases whose base run OOMs
CAPACITY = {"12ref_4k_catalog", "16ref_4k_catalog"}
TILE = 620
LABEL_H = 44


def font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if pathlib.Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def grey(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def ssim(a, b, win=8):
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    h, w = a.shape
    hs, ws = h // win * win, w // win * win
    a = a[:hs, :ws].reshape(hs // win, win, ws // win, win).transpose(0, 2, 1, 3).reshape(-1, win * win)
    b = b[:hs, :ws].reshape(hs // win, win, ws // win, win).transpose(0, 2, 1, 3).reshape(-1, win * win)
    ma, mb = a.mean(1), b.mean(1)
    va, vb, cov = a.var(1), b.var(1), ((a - ma[:, None]) * (b - mb[:, None])).mean(1)
    s = ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma**2 + mb**2 + c1) * (va + vb + c2))
    return float(s.mean())


def main():
    timings = {}
    for variant, _ in VARIANTS:
        for name in ("timings_8ref.json",):
            path = ROOT / variant / name
            if path.is_file():
                timings.setdefault(variant, {}).update(
                    {row["case"]: row for row in json.loads(path.read_text())["rows"]}
                )
    # capacity variants live in their own directories
    for variant, src in (("budget_8.4Mpx_v2", "budget_8.4Mpx_v2"), ("budget_2.1Mpx_v2", "budget_2.1Mpx_4krefs_v2")):
        for name in ("timings_8ref.json",):
            path = ROOT / src / name
            if path.is_file():
                for row in json.loads(path.read_text())["rows"]:
                    timings.setdefault(variant, {}).setdefault(row["case"], row)
    for extra in ("budget_2.1Mpx_v2", "budget_8.4Mpx_v2"):
        for row in (timings.get(extra) or {}).values():
            pass

    cases = sorted({case for rows in timings.values() for case in rows})
    out_dir = ROOT / "compare_8ref"
    out_dir.mkdir(exist_ok=True)
    lines = [
        "case,refs,wall_base_s,wall_8.4M_s,wall_2.1M_s,d_8.4M_pct,d_2.1M_pct,"
        "peak_base_MiB,peak_8.4M_MiB,peak_2.1M_MiB,psnr_8.4M,ssim_8.4M,psnr_2.1M,ssim_2.1M"
    ]
    strips = []
    for case in cases:
        base_png = ROOT / "base" / f"{case}.png"
        base_grey = grey(base_png) if base_png.is_file() else None
        tiles = []
        for variant, label in VARIANTS:
            png = ROOT / variant / f"{case}.png"
            row = timings.get(variant, {}).get(case)
            if not png.is_file():
                canvas = Image.new("RGB", (TILE, TILE + LABEL_H), "#202020")
                text = f"{label}: {'OOM / no image' if row and row.get('http') != 200 else 'not run'}"
                ImageDraw.Draw(canvas).text((8, 12), text, fill="white", font=font(20))
                tiles.append(canvas)
                continue
            wall = row.get("wall_s") if row else None
            delta = ""
            base_row = timings.get("base", {}).get(case)
            if wall and base_row and base_row.get("wall_s"):
                delta = f" ({(wall - base_row['wall_s']) / base_row['wall_s'] * 100:+.0f}%)"
            canvas = Image.new("RGB", (TILE, TILE + LABEL_H), "white")
            canvas.paste(Image.open(png).convert("RGB").resize((TILE, TILE), Image.LANCZOS), (0, LABEL_H))
            ImageDraw.Draw(canvas).text((8, 12), f"{label}: {wall}s{delta}", fill="black", font=font(20))
            tiles.append(canvas)
        strip = Image.new("RGB", (TILE * len(tiles), TILE + LABEL_H), "white")
        for i, tile in enumerate(tiles):
            strip.paste(tile, (TILE * i, 0))
        strip.save(out_dir / f"{case}__compare.png")
        strips.append(strip)

        # metrics
        base_row = timings.get("base", {}).get(case, {})
        row = [case, str(base_row.get("refs", ""))]
        for variant, _ in VARIANTS:
            r = timings.get(variant, {}).get(case, {})
            row.append(str(r.get("wall_s", "")))
        for variant, _ in VARIANTS[1:]:
            r = timings.get(variant, {}).get(case, {})
            if r.get("wall_s") and base_row.get("wall_s"):
                row.append(f"{(r['wall_s'] - base_row['wall_s']) / base_row['wall_s'] * 100:+.1f}")
            else:
                row.append("")
        for variant, _ in VARIANTS:
            row.append(str(timings.get(variant, {}).get(case, {}).get("peak_vram_mib", "")))
        for variant, _ in VARIANTS[1:]:
            png = ROOT / variant / f"{case}.png"
            if base_grey is not None and png.is_file():
                g = grey(png)
                diff = np.abs(g - base_grey)
                mse = float((diff**2).mean())
                row += [f"{10 * np.log10(255**2 / mse):.1f}" if mse else "inf", f"{ssim(g, base_grey):.3f}"]
            else:
                row += ["", ""]
        lines.append(",".join(row))

    (out_dir / "metrics_8ref.csv").write_text("\n".join(lines) + "\n")

    per_row = 2
    rows_img = []
    for start in range(0, len(strips), per_row):
        chunk = strips[start : start + per_row]
        sheet = Image.new("RGB", (strips[0].width * per_row, (TILE + LABEL_H) * len(chunk)), "white")
        for i, strip in enumerate(chunk):
            sheet.paste(strip, (0, (TILE + LABEL_H) * i))
        rows_img.append(sheet)
    if rows_img:
        overview = Image.new("RGB", (rows_img[0].width, sum(r.height for r in rows_img)), "white")
        y = 0
        for r in rows_img:
            overview.paste(r, (0, y))
            y += r.height
        overview.save(out_dir / "overview_8ref_variants.png")
    print("wrote", out_dir / "metrics_8ref.csv")
    print("wrote", out_dir / "overview_8ref_variants.png")
    print("strips:", len(strips))


if __name__ == "__main__":
    main()
