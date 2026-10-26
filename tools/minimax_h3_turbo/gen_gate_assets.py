# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Generate the frozen gate-suite stills through qwen-image-pro.

Prompts follow Qwen-Image's own convention (concrete subject, angle, lighting,
background, lens, style; no abstract adjectives) rather than the H3 video format:
the two models want different prompt shapes, and mixing them degrades both.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

NO_TEXT = (
    "画面中不出现任何文字：不要汉字、字母、数字、符号、标牌、招牌、标语、说明文字、"
    "标注线、箭头、编号、水印或 logo；人物服装与所有物体表面均为无字素面。"
)

ASSETS: dict[str, str] = {
    "ref_person": (
        "写实风格半身人像摄影：30 岁左右东亚女性，正面偏三分之四侧角度看向镜头，自然微笑，"
        "五官完整清晰；深棕色齐肩微卷发，灰色针织开衫内搭白色圆领；柔和均匀的左侧主光与右侧补光，"
        "肤色自然、无强烈阴影；纯净浅灰色无缝影棚背景，无文字无杂物；85mm 人像镜头，浅景深，"
        "面部对焦锐利，皮肤质感真实，高分辨率。"
    ),
    "ref_person_male": (
        "写实风格半身人像摄影：45 岁左右东亚男性，正面偏三分之四侧角度看向镜头，神情沉稳，"
        "五官完整清晰；短寸头、鬓角略有灰白，穿深灰色圆领针织衫；柔和均匀的左侧主光与右侧补光，"
        "肤色自然、无强烈阴影；纯净浅灰色无缝影棚背景，无文字无杂物；85mm 人像镜头，浅景深，"
        "面部对焦锐利，皮肤质感真实，高分辨率。"
    ),
    "first_frame": (
        "写实风格室内中景照片：一位穿米白亚麻衬衫的年轻女性坐在窗边木桌前，双手捧着一只白色陶瓷杯，"
        "杯口有淡淡热气；午后侧逆光从右侧窗户透入，头发边缘有柔和高光；背景是虚化的木质书架与绿植；"
        "35mm 镜头，浅景深，暖色调，画面干净无文字。"
    ),
    "last_frame": (
        "写实风格夜景广角照片：雨后黄昏的城市人行横道，湿漉路面反射着街灯与店铺灯箱的暖黄光；"
        "一位撑透明塑料伞的年轻女性站在斑马线起点，身体朝画面左侧，伞沿倾斜；"
        "画面上方有电车架空线与店面雨棚，无任何招牌文字；35mm 镜头，冷蓝与暖黄对比，无文字。"
    ),
    "first_frame_cycle": (
        "写实风格户外运动照片：一名穿深蓝色骑行服与黑色头盔的男性骑手沿海滨公路骑行，"
        "从背面略侧角度拍摄，身体直立握把；画面左侧有一座红白相间的灯塔，右侧是灰蓝色海面与白色护栏；"
        "阴天漫射光，路面湿润；50mm 镜头，中等景深，构图横向，画面无文字。"
    ),
    "last_frame_cycle": (
        "写实风格户外运动照片：同一名穿深蓝色骑行服与黑色头盔的男性骑手沿海滨公路压弯骑行，"
        "身体前倾向右倾斜，从侧面略后方拍摄；画面背景是灰蓝色海面与连续白色护栏，远处灯塔变小位于左上；"
        "阴天漫射光；50mm 镜头，中等景深，构图横向，画面无文字。"
    ),
    "g9_first": (
        "写实风格清晨火车站台照片：一名背帆布旅行包的男性旅客站在黄色安全线后，侧身望向轨道，"
        "身穿深色外套与牛仔裤；站台尽头有薄雾，顶棚投下冷白荧光；背景是灰色站台柱与空白的墙面，无任何指示牌文字；"
        "35mm 镜头，中景，冷色调，画面中无任何文字。"
    ),
    "g9_last": (
        "写实风格列车车厢内照片：同一名深色外套男性旅客把帆布包放到头顶行李架后，"
        "站在靠窗座位旁侧身回头；车厢内是暖白灯光与深蓝色座椅，窗外是明亮晨光；"
        "35mm 镜头，中景，暖色调，画面无文字。"
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="qwen-image-pro")
    parser.add_argument("--only", default="", help="comma-separated asset names")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    wanted = [x for x in args.only.split(",") if x] or list(ASSETS)
    for name in wanted:
        prompt = ASSETS[name] + " " + NO_TEXT
        body = json.dumps({"model": args.model, "prompt": prompt, "n": 1}).encode()
        req = urllib.request.Request(f"{args.base_url}/v1/images/generations", data=body, method="POST")
        req.add_header("Authorization", f"Bearer {args.api_key}")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=180) as response:
            payload = json.loads(response.read())
        url = payload["data"][0]["url"]
        target = args.out_dir / f"{name}.png"
        with urllib.request.urlopen(url, timeout=180) as response:
            target.write_bytes(response.read())
        print(
            json.dumps({"asset": name, "path": str(target), "bytes": target.stat().st_size}, ensure_ascii=False),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
