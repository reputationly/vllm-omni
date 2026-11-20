# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Local prompt enhancement for MiniMax-H3-Image."""

from __future__ import annotations

import binascii
import io
import json
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import regex as re
from PIL import Image, ImageOps

EDIT_APPENDIX = (
    "Compliance checklist (apply silently before answering; these restate the rules above):\n"
    "1. Coverage first. Part 1 is a complete inventory of the source-derived content that appears in the target, not "
    "only the edited element. Define the overall scene/background/canvas of each base picture as a Subject, define the "
    "edited element(s), and define every clearly readable text element (signs, labels, titles, slogans, captions, UI "
    "strings, menu items, prices, chart titles, axis ticks, legend entries, data values, table headers and cells, "
    "handwriting) as a Subject or as part of a text-block Subject, transcribing every line verbatim inside single "
    "quotes in its original language and script. Unchanged elements are still defined and then marked fully preserved. "
    "Never write that readable text is unreadable or omit a readable line because it is unchanged.\n"
    "2. Every definition sentence names its source with an explicit '<Picture N>' and follows the pattern "
    "'... in <Picture N> is <Subject N>.' (or '... are <Subject N>.'). Number Subjects 1, 2, 3, ... in order of "
    "first definition.\n"
    "3. Do not define a Subject for something that is removed or wholly replaced; mention it once in the target "
    "overview with its <Picture N> and describe the filled-in result in Part 3. Every defined Subject must end up "
    "fully preserved, partially preserved, transferred, or weakly referenced.\n"
    "4. Part 2 starts with exactly one sentence beginning 'The target image'. Then one separate sentence per Subject, "
    "in Subject order: '<Subject N> is fully preserved.' exactly, with nothing appended, or '<Subject N> is partially "
    "preserved; <retained categories> remain, while <changed categories> change.', or an explicit transfer / "
    "weak-reference clause.\n"
    "5. Part 3 describes only changed or new content, richly and concretely, and gives the complete final wording of "
    "every changed or new text in single quotes.\n"
    "6. All prose is English. Non-English characters may appear only inside single-quoted literals. Never use double "
    "quotation marks for literals. Grids are described as 'R rows and C columns'.\n"
    "7. target_aspect_ratio: an explicit user selection wins; otherwise an explicit ratio/orientation in the "
    "instruction; otherwise copy the main edited picture's supplied aspect_ratio exactly, even when it is not a "
    "common ratio."
)

T2I_APPENDIX = (
    "Compliance checklist (apply silently; these restate the rules above): write every word of expanded_caption in "
    "English except the exact text that must appear in the image; keep every requested literal (titles, labels, "
    "numbers, units, dates, names) verbatim in its original language and script, wrapped in escaped double quotes, "
    "and never translate, transliterate, or paraphrase it; state where each text appears; keep requested counts, "
    "left/right and top/bottom relations, and colors exactly; begin with 'The image is'; choose aspect_ratio from the "
    "allowed list by the deliverable and composition rules above; output only the two-key JSON object."
)

T2I_RATIOS = ("1:1", "5:4", "4:3", "3:2", "16:9", "21:9", "4:5", "3:4", "2:3", "9:16")
SUBJECT_PATTERN = re.compile(r"<Subject (\d+)>")
PICTURE_PATTERN = re.compile(r"<Picture (\d+)>")
CJK_PATTERN = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]")
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z<'\"(])")

MODEL_PROFILES: dict[str, dict[str, str]] = {
    "qwen3.8-flash-fp8": {"edit": "none", "t2i": "low"},
    "deepseek-v4-1-flash": {"edit": "none", "t2i": "none"},
    "glm-5.3-flash-w4a8": {"edit": "low", "t2i": "low"},
    "minimax-m3.1-flash": {"edit": "low", "t2i": "low"},
}


class ContextIRError(RuntimeError):
    """Raised when Context-IR cannot produce a parseable response."""


@dataclass(frozen=True)
class PEPrompts:
    edit_task: str
    t2i_system: str
    t2i_task: str
    t2i_params: dict[str, object]


def load_pe_prompts(pe_dir: str | Path) -> PEPrompts:
    """Load the locally supplied MiniMax-H3-Image-PE prompt files."""
    prompt_dir = Path(pe_dir) / "prompt"
    files = {
        "edit_task": prompt_dir / "edit" / "task_prompt.txt",
        "t2i_system": prompt_dir / "t2i" / "system_prompt.txt",
        "t2i_task": prompt_dir / "t2i" / "task_prompt.txt",
        "t2i_params": prompt_dir / "t2i" / "request_params.json",
    }
    if not Path(pe_dir).is_dir():
        raise FileNotFoundError(
            f"PE prompt directory not found: {pe_dir}. Download it with: "
            "huggingface-cli download MiniMaxAI/MiniMax-H3-Image-PE --include 'prompt/*'"
        )
    missing = [path for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing PE prompt file(s): {', '.join(str(path) for path in missing)}. "
            "Download them with: huggingface-cli download MiniMaxAI/MiniMax-H3-Image-PE --include 'prompt/*'"
        )
    params = json.loads(files["t2i_params"].read_text(encoding="utf-8"))
    if not isinstance(params, dict):
        raise ValueError(f"Expected a JSON object in {files['t2i_params']}")
    return PEPrompts(
        edit_task=files["edit_task"].read_text(encoding="utf-8"),
        t2i_system=files["t2i_system"].read_text(encoding="utf-8").rstrip("\n"),
        t2i_task=files["t2i_task"].read_text(encoding="utf-8"),
        t2i_params=params,
    )


def reference_media(image: str | Path | bytes | Image.Image) -> tuple[str, str | None]:
    """Return a media URI and reduced EXIF-corrected source aspect ratio."""
    if isinstance(image, str) and image.startswith(("data:", "http://", "https://")):
        return image, None

    if isinstance(image, Image.Image):
        source = ImageOps.exif_transpose(image).convert("RGB")
        return _encode_preview(source)
    if isinstance(image, bytes):
        with Image.open(io.BytesIO(image)) as opened:
            source = ImageOps.exif_transpose(opened).convert("RGB")
            return _encode_preview(source)
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            source = ImageOps.exif_transpose(opened).convert("RGB")
            return _encode_preview(source)
    raise TypeError(f"Unsupported image input: {type(image).__name__}")


def _encode_preview(image: Image.Image, max_side: int = 1568) -> tuple[str, str]:
    width, height = image.size
    divisor = gcd(width, height)
    ratio = f"{width // divisor}:{height // divisor}"
    if max(width, height) > max_side:
        scale = max_side / max(width, height)
        image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=92)
    encoded = binascii.b2a_base64(buffer.getvalue(), newline=False).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}", ratio


def build_edit_messages(
    prompts: PEPrompts,
    instruction: str,
    images: Sequence[str | Path | bytes | Image.Image],
    aspect_ratio: str = "adaptive",
    appendix: bool = True,
) -> list[dict[str, object]]:
    content: list[dict[str, object]] = [
        {"type": "text", "text": prompts.edit_task},
        {
            "type": "text",
            "text": f"Reference Image Count:\n{len(images)} input image(s), labeled "
            f"<Picture 1> through <Picture {len(images)}>.",
        },
    ]
    for index, image in enumerate(images, 1):
        uri, ratio = reference_media(image)
        ratio_label = f"aspect_ratio {ratio}" if ratio is not None else "aspect_ratio unavailable"
        content.extend(
            [
                {"type": "text", "text": f"Reference Media:\n- <Picture {index}> [{ratio_label}]"},
                {"type": "image_url", "image_url": {"url": uri}},
            ]
        )
    final_text = (
        "\nUser Editing Instruction (JSON string):\n"
        f"{json.dumps(instruction, ensure_ascii=False)}\n\n"
        f"User Aspect Ratio Selection:\n{aspect_ratio}"
    )
    if appendix:
        final_text += f"\n\n{EDIT_APPENDIX}"
    final_text += (
        "\n\nApply the instructions above to the supplied inputs and reference images. "
        "Output only the required two-field JSON object."
    )
    content.append({"type": "text", "text": final_text})
    return [{"role": "user", "content": content}]


def build_t2i_messages(prompts: PEPrompts, prompt: str, appendix: bool = True) -> list[dict[str, object]]:
    user_content = prompts.t2i_task.format(prompt)
    if appendix:
        user_content += f"\n\n{T2I_APPENDIX}"
    return [
        {"role": "system", "content": prompts.t2i_system},
        {"role": "user", "content": user_content},
    ]


def _strip_fence(content: str) -> str:
    stripped = content.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.S)
    return match.group(1) if match else stripped


def parse_json_object(content: object, keys: Sequence[str]) -> tuple[dict[str, object] | None, list[str]]:
    """Parse one JSON object with exactly the supplied keys."""
    issues: list[str] = []
    if not isinstance(content, str) or not content.strip():
        return None, ["empty_content"]
    raw = content.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        issues.append("not_pure_json")
        stripped = _strip_fence(raw)
        start, end = stripped.find("{"), stripped.rfind("}")
        try:
            parsed = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None, [*issues, "json_unrecoverable"]
    if not isinstance(parsed, dict):
        return None, [*issues, "not_object"]
    if set(parsed) != set(keys):
        issues.append(f"keys={sorted(parsed)}")
    return parsed, issues


def _unquoted(text: str) -> str:
    return re.sub(r"(?<![A-Za-z0-9])'[^']*'(?![A-Za-z])", "''", text)


def _split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in SENTENCE_BOUNDARY.split(text) if sentence.strip()]


def check_edit_caption(caption: object, ratio: object, n_pictures: int, user_ratio: str = "adaptive") -> list[str]:
    """Return issue codes for an edit caption and its target ratio."""
    issues: list[str] = []
    if not isinstance(caption, str) or not caption.strip():
        return ["empty_caption"]
    if "\n" in caption.strip():
        issues.append("line_breaks")
    if re.search(r"```|\*\*|^#+ |^\s*[-*] ", caption, re.M):
        issues.append("markdown")
    if not isinstance(ratio, str) or not re.fullmatch(r"[1-9]\d*:[1-9]\d*", ratio or ""):
        issues.append("bad_ratio")
    elif user_ratio != "adaptive" and ratio != user_ratio:
        issues.append("ratio_ignores_user_selection")
    for picture in set(int(value) for value in PICTURE_PATTERN.findall(caption)):
        if not 1 <= picture <= n_pictures:
            issues.append(f"picture_out_of_range:{picture}")
    if re.search(r"<(?:Subject|Picture)\s*\d+\s*(?:-|–|to|through)\s*", caption) or re.search(
        r"<Pictures? \d+>\s*(?:-|–|to|through)\s*<Picture", caption
    ):
        issues.append("range_reference")
    if re.search(r"<(?!Subject \d+>|Picture \d+>)[A-Za-z][^<>]{0,30}>", caption):
        issues.append("malformed_token")
    target_index = caption.find("The target image")
    if target_index < 0:
        return [*issues, "no_target_overview"]
    if caption.count("The target image") > 1:
        issues.append("multiple_target_overview")
    part_one, rest = caption[:target_index], caption[target_index:]
    definitions: list[int] = []
    for sentence in _split_sentences(part_one):
        subject_ids = [int(value) for value in SUBJECT_PATTERN.findall(sentence)]
        new_subjects = [subject for subject in subject_ids if subject not in definitions]
        if not new_subjects:
            issues.append("part1_sentence_without_new_subject")
            continue
        if not PICTURE_PATTERN.search(sentence):
            issues.append(f"def_without_picture:{new_subjects[0]}")
        if not re.search(r"\b(?:is|are)\b[^.]*<Subject \d+>", sentence) and not re.search(
            r"<Subject \d+>[^.]*\b(?:is|are)\b", sentence
        ):
            issues.append(f"def_not_natural:{new_subjects[0]}")
        definitions.extend(new_subjects)
    if not definitions:
        issues.append("no_subjects")
    if definitions != list(range(1, len(definitions) + 1)):
        issues.append("subject_numbering")
    referenced_subjects = set(int(value) for value in SUBJECT_PATTERN.findall(rest))
    for subject in sorted(referenced_subjects - set(definitions)):
        issues.append(f"undefined_subject:{subject}")
    sentences = _split_sentences(rest)
    relationships: dict[int, list[str]] = {}
    for sentence in sentences[1:]:
        match = re.match(r"<Subject (\d+)>(?:'s)? ", sentence)
        if not match:
            continue
        subject = int(match.group(1))
        kind = None
        if re.fullmatch(r"<Subject \d+> is fully preserved\.", sentence):
            kind = "full"
        elif "fully preserved" in sentence:
            kind = "full_dirty"
        elif re.match(r"<Subject \d+> (?:is|are) partially preserved[;,:]", sentence):
            kind = "partial"
        elif re.search(r"\b(?:suppl(?:y|ies)|transferred|transfers?|provides?|donates?)\b", sentence):
            kind = "transfer"
        elif "weak" in sentence and "reference" in sentence:
            kind = "weak"
        elif re.search(r"\b(?:removed|absent|deleted|no longer (?:appears|present)|not used|unused)\b", sentence):
            kind = "removed"
        if kind:
            relationships.setdefault(subject, []).append(kind)
    for subject in definitions:
        kinds = relationships.get(subject, [])
        if not kinds:
            issues.append(f"no_relationship:{subject}")
        elif len(kinds) > 1 and kinds.count("full") + kinds.count("partial") + kinds.count("full_dirty") > 1:
            issues.append(f"multiple_relationships:{subject}")
        if "full_dirty" in kinds:
            issues.append(f"fully_preserved_not_standalone:{subject}")
        if "removed" in kinds and len(kinds) == 1:
            issues.append(f"subject_defined_only_to_remove:{subject}")
    if re.search(r"<Subject \d+>(?:,| and)[^.]*\b(?:are|is) (?:all |both |each )?fully preserved", rest):
        issues.append("merged_fully_preserved")
    if CJK_PATTERN.search(_unquoted(caption)):
        issues.append("non_english_prose")
    if '"' in caption or "“" in caption or "”" in caption:
        issues.append("double_quoted_literal")
    if re.search(r"\b\d+\s*[x×]\s*\d+\b(?! (?:px|pixel))", _unquoted(caption)) and not re.search(
        r"\d+ rows? and \d+ columns?|(?:one|two|three|four|five|six|seven|eight|nine|ten) rows? and "
        r"(?:one|two|three|four|five|six|seven|eight|nine|ten) columns?",
        _unquoted(caption),
    ):
        issues.append("unlabeled_grid")
    return issues


def check_t2i(content: object, allowed_ratios: Sequence[str]) -> list[str]:
    """Return issue codes for a text-to-image response."""
    parsed, issues = parse_json_object(content, ("expanded_caption", "aspect_ratio"))
    if parsed is None or not isinstance(parsed.get("expanded_caption"), str):
        return issues
    caption = parsed["expanded_caption"]
    if parsed.get("aspect_ratio") not in allowed_ratios:
        issues.append("ratio_enum")
    if "\n" in caption.strip():
        issues.append("line_breaks")
    unquoted = re.sub(r'"[^"]*"|\'[^\']*\'|“[^”]*”', "", caption)
    if CJK_PATTERN.search(unquoted):
        issues.append("non_english_prose")
    return issues


def describe_issues(codes: Sequence[str]) -> list[str]:
    descriptions = {
        "def_without_picture": "a Subject definition lacks an explicit <Picture N> source",
        "subject_defined_only_to_remove": (
            "a Subject is defined for an element that is removed; drop that Subject and renumber"
        ),
        "no_relationship": "a defined Subject has no relationship sentence",
        "non_english_prose": "non-English characters appear outside single-quoted literals",
        "double_quoted_literal": "literals must use single quotes, not double quotes",
        "no_target_overview": "missing the sentence beginning 'The target image'",
        "no_subjects": "no <Subject N> definitions before 'The target image'",
        "fully_preserved_not_standalone": (
            "a fully-preserved clause has extra words; it must be exactly '<Subject N> is fully preserved.'"
        ),
        "part1_sentence_without_new_subject": "a sentence before 'The target image' does not define a new Subject",
        "undefined_subject": "a <Subject N> is used without being defined",
        "multiple_relationships": "a Subject receives more than one relationship",
        "merged_fully_preserved": "several Subjects share one fully-preserved clause",
        "line_breaks": "expanded_caption contains line breaks",
        "keys": "the object must have exactly expanded_caption and target_aspect_ratio",
        "not_pure_json": "the answer is not a bare JSON object",
        "ratio_ignores_user_selection": "target_aspect_ratio must equal the user aspect ratio selection",
        "bad_ratio": "target_aspect_ratio must be W:H with positive integers",
        "subject_numbering": "Subjects must be numbered 1, 2, 3, ... in definition order",
        "def_not_natural": "each definition must read '... in <Picture N> is <Subject N>.'",
        "unlabeled_grid": "describe grids as 'R rows and C columns'",
    }
    result: list[str] = []
    for issue in codes:
        code = issue.split(":")[0].split("=")[0]
        description = descriptions.get(code, issue)
        if description not in result:
            detail = f" ({issue.split(':', 1)[1]})" if ":" in issue else ""
            result.append(description + detail)
    return result


def repair_prompt(issues: Sequence[str], _context: object | None = None) -> str:
    return (
        "Your JSON violates the output contract: "
        + "; ".join(issues)
        + ". Rewrite the complete answer so it satisfies every rule above, keeping the correct content. "
        "Output only the required two-field JSON object."
    )


def _response_content(response: dict[str, object]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _validate_response(
    content: str,
    *,
    edit: bool,
    image_count: int,
    user_ratio: str,
) -> tuple[dict[str, object] | None, list[str]]:
    keys = ("expanded_caption", "target_aspect_ratio") if edit else ("expanded_caption", "aspect_ratio")
    parsed, issues = parse_json_object(content, keys)
    if parsed is None:
        return None, issues
    if edit:
        issues.extend(
            check_edit_caption(
                parsed.get("expanded_caption"), parsed.get("target_aspect_ratio"), image_count, user_ratio
            )
        )
    else:
        issues = check_t2i(content, T2I_RATIOS)
    return parsed, issues


def enhance(
    chat: Callable[..., dict[str, object]],
    *,
    model: str,
    prompts: PEPrompts,
    prompt: str,
    images: Sequence[str | Path | bytes | Image.Image] | None = None,
    aspect_ratio: str | None = "auto",
    max_repairs: int = 2,
    reasoning_effort: str | None = None,
) -> dict[str, object]:
    """Enhance an image prompt and repair responses that violate the PE contract."""
    edit = bool(images)
    image_inputs = images or []
    user_ratio = "adaptive" if aspect_ratio in (None, "auto") else aspect_ratio
    messages = (
        build_edit_messages(prompts, prompt, image_inputs, user_ratio) if edit else build_t2i_messages(prompts, prompt)
    )
    profile = MODEL_PROFILES.get(model, {"edit": "low", "t2i": "low"})
    effort = reasoning_effort if reasoning_effort is not None else profile["edit" if edit else "t2i"]
    # Without reasoning_effort, these models spend all 8192 tokens on reasoning and return an empty body.
    params: dict[str, object] = {"model": model, "max_tokens": 8192}
    if effort:
        params["reasoning_effort"] = effort
    if not edit:
        params.update(
            {
                "temperature": prompts.t2i_params["temperature"],
                "top_p": prompts.t2i_params["top_p"],
                "response_format": prompts.t2i_params["response_format"],
            }
        )

    last_parsed: dict[str, object] | None = None
    last_issues: list[str] = []
    repair_issues: list[str] = []
    attempts = 0
    content = ""
    for repair in range(max_repairs + 1):
        request_messages = messages
        if repair and content:
            issues_for_prompt = describe_issues(repair_issues)
            repair_text = (
                repair_prompt(issues_for_prompt)
                if edit
                else "Your JSON violates the output contract: "
                + ", ".join(repair_issues or ["invalid JSON"])
                + ". Prose must be English; non-English text may appear only inside double-quoted literals. "
                "Rewrite the complete answer; output only the two-key JSON object."
            )
            request_messages = [
                *messages,
                {"role": "assistant", "content": content},
                {"role": "user", "content": repair_text},
            ]
        response = chat(request_messages, **params)
        attempts += 1
        content = _response_content(response)
        parsed, issues = _validate_response(
            content,
            edit=edit,
            image_count=len(image_inputs),
            user_ratio=user_ratio,
        )
        repair_issues = issues
        if parsed is not None:
            last_parsed, last_issues = parsed, issues
            if not issues:
                break

    if last_parsed is None:
        raise ContextIRError(f"No parseable Context-IR response after {attempts} attempt(s)")
    if edit:
        output_ratio = None if user_ratio != "adaptive" else last_parsed.get("target_aspect_ratio")
    else:
        output_ratio = last_parsed.get("aspect_ratio")
    return {
        "revised_prompt": last_parsed.get("expanded_caption"),
        "aspect_ratio": output_ratio,
        "issues": last_issues,
        "attempts": attempts,
    }


def openai_chat(base_url: str, api_key: str, timeout: int = 600) -> Callable[..., dict[str, object]]:
    """Create an OpenAI-compatible chat-completions function using urllib."""
    url = f"{base_url.rstrip('/')}/chat/completions"

    def chat(messages: list[dict[str, object]], **params: object) -> dict[str, object]:
        body = {"messages": messages, **params}
        request = urllib.request.Request(
            url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                retryable = error.code == 429 or 500 <= error.code < 600
                error.close()
                if retryable and attempt < 2:
                    continue
                raise ContextIRError(f"Chat completion failed with HTTP {error.code}") from error
            except urllib.error.URLError as error:
                raise ContextIRError("Chat completion request failed") from error
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ContextIRError("Chat completion returned invalid JSON") from error
            if not isinstance(result, dict):
                raise ContextIRError("Chat completion returned a non-object JSON response")
            return result
        raise ContextIRError("Chat completion retries exhausted")

    return chat
