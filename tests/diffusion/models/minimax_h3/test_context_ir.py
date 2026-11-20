# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from vllm_omni.diffusion.models.minimax_h3.context_ir import (
    EDIT_APPENDIX,
    MODEL_PROFILES,
    T2I_APPENDIX,
    T2I_RATIOS,
    ContextIRError,
    PEPrompts,
    build_edit_messages,
    build_t2i_messages,
    check_edit_caption,
    check_t2i,
    describe_issues,
    enhance,
    load_pe_prompts,
    parse_json_object,
    reference_media,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]
FIXTURES = Path(__file__).parent / "fixtures"


def _write_fake_prompts(root: Path) -> PEPrompts:
    edit = root / "prompt" / "edit"
    t2i = root / "prompt" / "t2i"
    edit.mkdir(parents=True)
    t2i.mkdir(parents=True)
    (edit / "task_prompt.txt").write_text("fake edit task", encoding="utf-8")
    (t2i / "system_prompt.txt").write_text("fake system\n", encoding="utf-8")
    (t2i / "task_prompt.txt").write_text("fake task: {}", encoding="utf-8")
    (t2i / "request_params.json").write_text(
        json.dumps(
            {
                "temperature": 1.0,
                "top_p": 0.95,
                "response_format": {"type": "json_object"},
            }
        ),
        encoding="utf-8",
    )
    return load_pe_prompts(root)


def _valid_edit_caption() -> str:
    return (
        "A red cup in <Picture 1> is <Subject 1>. "
        "The target image shows the cup on the table. "
        "<Subject 1> is fully preserved."
    )


def _openai_response(content: str) -> dict[str, object]:
    return {"choices": [{"message": {"content": content}}]}


class FakeChat:
    def __init__(self, *contents: str):
        self.contents = list(contents)
        self.calls: list[tuple[list[dict[str, object]], dict[str, object]]] = []

    def __call__(self, messages: list[dict[str, object]], **params: object) -> dict[str, object]:
        self.calls.append((messages, params))
        return _openai_response(self.contents.pop(0))


def test_load_pe_prompts_reads_local_files_and_requires_existing_directory(tmp_path):
    prompts = _write_fake_prompts(tmp_path / "model")

    assert prompts.edit_task == "fake edit task"
    assert prompts.t2i_system == "fake system"
    assert prompts.t2i_task == "fake task: {}"
    assert prompts.t2i_params["top_p"] == 0.95
    with pytest.raises(FileNotFoundError, match="huggingface-cli download MiniMaxAI/MiniMax-H3-Image-PE"):
        load_pe_prompts(tmp_path / "missing")


def test_reference_media_normalizes_exif_ratio_and_resizes_preview(tmp_path):
    first_path = tmp_path / "landscape.png"
    Image.new("RGB", (1600, 1189), "red").save(first_path)
    rotated_path = tmp_path / "rotated.png"
    rotated = Image.new("RGB", (1600, 1189), "blue")
    exif = rotated.getexif()
    exif[274] = 6
    rotated.save(rotated_path, exif=exif)

    uri, ratio = reference_media(first_path)
    rotated_uri, rotated_ratio = reference_media(rotated_path)
    assert ratio == "1600:1189"
    assert rotated_ratio == "1189:1600"
    assert reference_media(first_path.read_bytes())[1] == ratio
    assert reference_media(Image.new("RGB", (1600, 1189)))[1] == ratio
    assert reference_media("https://example.test/image.png") == (
        "https://example.test/image.png",
        None,
    )
    assert reference_media("data:image/png;base64,abc") == (
        "data:image/png;base64,abc",
        None,
    )

    for preview in (uri, rotated_uri):
        header, encoded = preview.split(",", 1)
        assert header == "data:image/jpeg;base64"
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            assert max(image.size) == 1568
            assert image.format == "JPEG"


def test_build_edit_messages_orders_segments_and_adds_appendix_before_apply(tmp_path):
    prompts = _write_fake_prompts(tmp_path / "model")
    first = tmp_path / "first.png"
    Image.new("RGB", (1600, 1189)).save(first)
    second = tmp_path / "second.png"
    rotated = Image.new("RGB", (1600, 1189))
    exif = rotated.getexif()
    exif[274] = 6
    rotated.save(second, exif=exif)

    messages = build_edit_messages(prompts, "make a change", [first, second], "4:3")
    content = messages[0]["content"]
    assert messages[0]["role"] == "user"
    assert content[0] == {"type": "text", "text": "fake edit task"}
    assert content[1]["text"] == ("Reference Image Count:\n2 input image(s), labeled <Picture 1> through <Picture 2>.")
    assert content[2]["text"] == "Reference Media:\n- <Picture 1> [aspect_ratio 1600:1189]"
    assert content[4]["text"] == "Reference Media:\n- <Picture 2> [aspect_ratio 1189:1600]"
    final_text = content[-1]["text"]
    assert 'User Editing Instruction (JSON string):\n"make a change"' in final_text
    assert "User Aspect Ratio Selection:\n4:3" in final_text
    assert final_text.index(EDIT_APPENDIX) < final_text.index("\n\nApply the instructions above")
    assert final_text.endswith(
        "Apply the instructions above to the supplied inputs and reference images. "
        "Output only the required two-field JSON object."
    )


def test_build_edit_messages_preserves_http_image_without_aspect_ratio(tmp_path):
    prompts = _write_fake_prompts(tmp_path / "model")
    image_url = "https://example.com/a.png"

    messages = build_edit_messages(prompts, "make a change", [image_url], appendix=False)
    content = messages[0]["content"]

    assert content[2]["text"] == "Reference Media:\n- <Picture 1> [aspect_ratio unavailable]"
    assert content[3] == {"type": "image_url", "image_url": {"url": image_url}}
    assert "aspect_ratio aspect_ratio" not in content[2]["text"]


def test_build_t2i_messages_adds_t2i_appendix(tmp_path):
    prompts = _write_fake_prompts(tmp_path / "model")

    messages = build_t2i_messages(prompts, "draw a scene")

    assert messages == [
        {"role": "system", "content": "fake system"},
        {"role": "user", "content": f"fake task: draw a scene\n\n{T2I_APPENDIX}"},
    ]
    assert build_t2i_messages(prompts, "draw a scene", appendix=False)[1]["content"] == "fake task: draw a scene"


def test_official_edit_revised_prompt_examples_pass_the_spec():
    examples = json.loads((FIXTURES / "context_ir_official_captions.json").read_text(encoding="utf-8"))

    assert len(examples) == 3
    for example in examples:
        assert (
            check_edit_caption(
                example["revised_prompt"],
                example["aspect_ratio"],
                example["n_pictures"],
            )
            == []
        )


@pytest.mark.parametrize(
    ("caption", "ratio", "n_pictures", "user_ratio", "expected_issue"),
    [
        (
            "A red cup is <Subject 1>. The target image shows a cup. <Subject 1> is fully preserved.",
            "1:1",
            1,
            "adaptive",
            "def_without_picture",
        ),
        (
            "A red cup in <Picture 1> is <Subject 1>. The target image shows the cup removed. <Subject 1> is absent.",
            "1:1",
            1,
            "adaptive",
            "subject_defined_only_to_remove",
        ),
        (
            "A red cup in <Picture 1> is <Subject 1>. The target image shows a cup. "
            "<Subject 1> is fully preserved with no changes.",
            "1:1",
            1,
            "adaptive",
            "fully_preserved_not_standalone",
        ),
        (
            "A 红 bowl in <Picture 1> is <Subject 1>. The target image shows the bowl. <Subject 1> is fully preserved.",
            "1:1",
            1,
            "adaptive",
            "non_english_prose",
        ),
        (
            'A sign in <Picture 1> is <Subject 1>. The target image shows "Open". <Subject 1> is fully preserved.',
            "1:1",
            1,
            "adaptive",
            "double_quoted_literal",
        ),
        (
            "A red cup in <Picture 1> is <Subject 1>. <Subject 1> is fully preserved.",
            "1:1",
            1,
            "adaptive",
            "no_target_overview",
        ),
        (
            "A red cup in <Picture 1> is <Subject 1>. The target image shows <Subject 2>. "
            "<Subject 1> is fully preserved.",
            "1:1",
            1,
            "adaptive",
            "undefined_subject",
        ),
        (_valid_edit_caption(), "1:1", 1, "4:3", "ratio_ignores_user_selection"),
        (_valid_edit_caption().replace("table. ", "table.\n"), "1:1", 1, "adaptive", "line_breaks"),
    ],
)
def test_check_edit_caption_rejects_each_spec_violation(caption, ratio, n_pictures, user_ratio, expected_issue):
    assert any(
        issue == expected_issue or issue.startswith(f"{expected_issue}:")
        for issue in check_edit_caption(caption, ratio, n_pictures, user_ratio)
    )


def test_check_t2i_checks_ratio_english_newlines_and_exact_keys():
    valid = json.dumps(
        {"expanded_caption": 'The image is a sign reading "北京".', "aspect_ratio": "16:9"},
        ensure_ascii=False,
    )
    assert check_t2i(valid, T2I_RATIOS) == []
    assert "ratio_enum" in check_t2i(
        json.dumps({"expanded_caption": "The image is a landscape.", "aspect_ratio": "16:10"}),
        T2I_RATIOS,
    )
    assert "non_english_prose" in check_t2i(
        json.dumps({"expanded_caption": "The image has a 红 roof.", "aspect_ratio": "1:1"}, ensure_ascii=False),
        T2I_RATIOS,
    )
    assert "line_breaks" in check_t2i(
        json.dumps({"expanded_caption": "The image is\n a landscape.", "aspect_ratio": "1:1"}),
        T2I_RATIOS,
    )
    assert any(
        issue.startswith("keys=")
        for issue in check_t2i(
            json.dumps({"expanded_caption": "The image is a landscape.", "aspect_ratio": "1:1", "extra": True}),
            T2I_RATIOS,
        )
    )


def test_parse_json_object_recovers_fenced_object_and_describes_issues():
    parsed, issues = parse_json_object('```json\n{"a": 1}\n```', ("a",))

    assert parsed == {"a": 1}
    assert issues == ["not_pure_json"]
    assert describe_issues(issues) == ["the answer is not a bare JSON object"]


def test_enhance_repairs_t2i_response_and_passes_request_parameters(tmp_path):
    prompts = _write_fake_prompts(tmp_path / "model")
    chat = FakeChat(
        json.dumps({"expanded_caption": "The image has a 红 roof.", "aspect_ratio": "1:1"}, ensure_ascii=False),
        json.dumps(
            {"expanded_caption": 'The image is a red roof labeled "北京".', "aspect_ratio": "1:1"}, ensure_ascii=False
        ),
    )

    result = enhance(chat, model="deepseek-v4-1-flash", prompts=prompts, prompt="a red roof")

    assert result == {
        "revised_prompt": 'The image is a red roof labeled "北京".',
        "aspect_ratio": "1:1",
        "issues": [],
        "attempts": 2,
    }
    assert len(chat.calls) == 2
    first_messages, params = chat.calls[0]
    assert first_messages[0]["role"] == "system"
    assert params == {
        "model": "deepseek-v4-1-flash",
        "max_tokens": 8192,
        "reasoning_effort": "none",
        "temperature": 1.0,
        "top_p": 0.95,
        "response_format": {"type": "json_object"},
    }
    second_messages, _ = chat.calls[1]
    assert second_messages[-2] == {
        "role": "assistant",
        "content": json.dumps(
            {"expanded_caption": "The image has a 红 roof.", "aspect_ratio": "1:1"}, ensure_ascii=False
        ),
    }
    repair_message = second_messages[-1]["content"]
    assert isinstance(repair_message, str)
    assert "non_english_prose" in repair_message


def test_enhance_retries_empty_content_and_returns_user_selected_edit_ratio(tmp_path):
    prompts = _write_fake_prompts(tmp_path / "model")
    content = json.dumps({"expanded_caption": _valid_edit_caption(), "target_aspect_ratio": "4:3"})
    chat = FakeChat("", content)

    result = enhance(
        chat,
        model="unlisted-model",
        prompts=prompts,
        prompt="change the cup",
        images=["https://example.test/reference.png"],
        aspect_ratio="4:3",
    )

    assert result == {
        "revised_prompt": _valid_edit_caption(),
        "aspect_ratio": None,
        "issues": [],
        "attempts": 2,
    }
    assert chat.calls[0][0] == chat.calls[1][0]
    assert chat.calls[0][1]["reasoning_effort"] == "low"


def test_enhance_returns_last_parseable_response_with_issues(tmp_path):
    prompts = _write_fake_prompts(tmp_path / "model")
    content = json.dumps({"expanded_caption": "The image has a 红 roof.", "aspect_ratio": "1:1"}, ensure_ascii=False)
    chat = FakeChat(content)

    result = enhance(chat, model="qwen3.8-flash-fp8", prompts=prompts, prompt="a red roof", max_repairs=0)

    assert result["revised_prompt"] == "The image has a 红 roof."
    assert result["issues"] == ["non_english_prose"]
    assert result["attempts"] == 1


def test_enhance_raises_when_no_attempt_has_parseable_json(tmp_path):
    prompts = _write_fake_prompts(tmp_path / "model")

    with pytest.raises(ContextIRError, match="No parseable Context-IR response"):
        enhance(FakeChat("not json"), model="qwen3.8-flash-fp8", prompts=prompts, prompt="a scene", max_repairs=0)


def test_model_profiles_match_recommended_effort_levels():
    assert MODEL_PROFILES == {
        "qwen3.8-flash-fp8": {"edit": "none", "t2i": "low"},
        "deepseek-v4-1-flash": {"edit": "none", "t2i": "none"},
        "glm-5.3-flash-w4a8": {"edit": "low", "t2i": "low"},
        "minimax-m3.1-flash": {"edit": "low", "t2i": "low"},
    }
