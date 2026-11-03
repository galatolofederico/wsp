#!/usr/bin/env python3
"""Ask an LLM what a safetensors checkpoint does by showing it the weights.

The whole checkpoint is rendered as text and sent to an OpenAI-compatible
chat completions endpoint. Inspired by:
https://rikiyat.github.io/blog/weight-space-programming/

Example:
    python wsp_read.py \
        --base-url http://localhost:8000/v1 \
        --api-key sk-... \
        --model-name my-llm \
        --model ../rl/cartpole_ppo.safetensors
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import TextIO

from safetensors import safe_open

SYSTEM_PROMPT = (
    "You are an expert in neural network architectures and mechanistic "
    "interpretability. You can infer what a network does by reading its weights."
)

DEFAULT_CHOICES = ("CartPole", "Flappy", "BallBeam", "Catch")

USER_PROMPT = """\
Below is a PyTorch neural network checkpoint rendered as text. You cannot run
the network. Use only the weights, tensor names, and shapes.

Format facts:
- nn.Linear weights are stored as [out_features, in_features], and biases as
  [out_features].
- Tensor order in the file is not necessarily forward-pass order.
- The checkpoint may contain a shared trunk and multiple heads. Do not assume
  every tensor belongs to one sequential stack.
- A [1] bias paired with a [1, N] weight can be a scalar value head, not an
  input normalization.
- Hidden activations are not stored.

{task_instruction}

{choices}{description}Explain:
1. the architecture and data flow, pairing each weight with the bias of the
   layer it belongs to,
2. the concrete task or behavior the network implements, including the
   observations and actions if it is a policy,
3. which tensor names, shapes, and standout values support your conclusion.

Be concrete: cite the weights you rely on.
End with exactly these two lines:
{task_line}
Confidence: <0-100>%

{metadata}=== CHECKPOINT ===
{summary}

=== WEIGHTS ===
{weights}
=== END OF CHECKPOINT ===
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ask an LLM what a .safetensors model does from its weights.",
    )
    p.add_argument(
        "--base-url",
        required=True,
        help="OpenAI-compatible base URL, e.g. http://localhost:8000/v1",
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="API key (default: $OPENAI_API_KEY; omit for local servers)",
    )
    p.add_argument("--model-name", required=True, help="name of the LLM to query")
    p.add_argument("--model", required=True, type=Path, help="path to .safetensors")
    p.add_argument(
        "--max-chars",
        type=int,
        default=0,
        help="truncate the weight text after this many characters (0 = no limit)",
    )
    p.add_argument(
        "--metadata",
        action="store_true",
        help="also send the safetensors metadata (may reveal the answer)",
    )
    p.add_argument(
        "--description-file",
        type=Path,
        default=None,
        help="optional neutral description of the environment and network to "
             "prepend to the prompt (see descriptions/)",
    )
    choice_group = p.add_mutually_exclusive_group()
    choice_group.add_argument(
        "--choices",
        nargs="+",
        metavar="TASK",
        default=DEFAULT_CHOICES,
        help="candidate tasks; the model must answer with exactly one of them "
             "(default: CartPole Flappy BallBeam Catch)",
    )
    choice_group.add_argument(
        "--no-choices",
        action="store_true",
        help="ask for an open-ended task instead of the default candidates",
    )
    p.add_argument(
        "--save",
        type=Path,
        default=None,
        help="write the full prompt, reasoning and answer to this JSON file",
    )
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument(
        "--reasoning-effort",
        "--thinking-effort",
        dest="reasoning_effort",
        default=None,
        help="provider-specific thinking effort, sent as top-level "
             "reasoning_effort (DeepSeek: low, high, max; also accepts "
             "minimal, medium, xhigh, none)",
    )
    p.add_argument(
        "--thinking",
        choices=("enabled", "disabled"),
        default=None,
        help="DeepSeek thinking-mode toggle, sent as thinking.type "
             "(omitted by default)",
    )
    p.add_argument("--timeout", type=float, default=600.0)
    return p.parse_args()


def _fmt_number(value: float | int) -> str:
    if isinstance(value, float):
        if value != value:
            return "nan"
        if value == float("inf"):
            return "inf"
        if value == float("-inf"):
            return "-inf"
        return f"{value:.6g}"
    return str(value)


def _fmt_sequence(values: list, indent: int = 0) -> str:
    """Format nested lists of numbers compactly, one matrix row per line."""
    pad = " " * indent

    # Flat sequence: wrap at 16 values per line.
    if all(not isinstance(v, list) for v in values):
        if len(values) <= 16:
            return "[" + ", ".join(_fmt_number(v) for v in values) + "]"
        lines = [
            pad + "  " + ", ".join(_fmt_number(v) for v in values[i:i + 16]) + ","
            for i in range(0, len(values), 16)
        ]
        return "[\n" + "\n".join(lines) + "\n" + pad + "]"

    # Matrix (or batch of flat rows): one row per line.
    if all(
        isinstance(v, list) and all(not isinstance(x, list) for x in v)
        for v in values
    ):
        rows = [
            pad + "  [" + ", ".join(_fmt_number(x) for x in row) + "],"
            for row in values
        ]
        return "[\n" + "\n".join(rows) + "\n" + pad + "]"

    # Higher-dimensional tensor: recurse.
    lines = [pad + "  " + _fmt_sequence(v, indent + 2) + "," for v in values]
    return "[\n" + "\n".join(lines) + "\n" + pad + "]"


def format_tensor(name: str, tensor) -> str:
    dtype = str(tensor.dtype).removeprefix("torch.")
    header = f"### {name} | shape={list(tensor.shape)} | dtype={dtype}"

    if tensor.is_floating_point() and tensor.numel() > 0:
        flat = tensor.float()
        stats = [
            f"min={_fmt_number(flat.min().item())}",
            f"max={_fmt_number(flat.max().item())}",
            f"mean={_fmt_number(flat.mean().item())}",
        ]
        if flat.numel() > 1:
            stats.append(f"std={_fmt_number(flat.std().item())}")
        header += "\n" + " | ".join(stats)

    values = tensor.tolist()
    values_text = (
        _fmt_sequence(values) if isinstance(values, list) else _fmt_number(values)
    )
    return header + "\n" + values_text


def _natural_key(name: str):
    """Sort ``layer.2`` before ``layer.10`` instead of lexicographically."""
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"(\d+)", name)
    )


def build_checkpoint(path: Path, max_chars: int, include_metadata: bool):
    """Return (summary, metadata_text, weight_text)."""
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = sorted(f.keys(), key=_natural_key)
        metadata = f.metadata() or {}

        # Read only tensor headers up front, so the total is accurate even
        # when the weight text is truncated.
        total_params = sum(
            math.prod(f.get_slice(key).get_shape()) for key in keys
        )

        sections: list[str] = []
        used = 0
        truncated_at: str | None = None

        for key in keys:
            section = format_tensor(key, f.get_tensor(key))
            if max_chars and used + len(section) > max_chars:
                remaining = max_chars - used
                if remaining > 0:
                    sections.append(
                        section[:remaining].rstrip() + "\n... [truncated]"
                    )
                truncated_at = key
                break
            sections.append(section)
            used += len(section) + 2

    summary_lines = [
        f"tensors: {len(keys)}",
        f"parameters: {total_params:,}",
    ]
    if truncated_at is not None:
        summary_lines.append(
            f"NOTE: weight text was truncated at tensor {truncated_at!r}; "
            "the remaining tensors are missing."
        )
    summary = "\n".join(summary_lines)

    metadata_text = ""
    if include_metadata and metadata:
        metadata_text = (
            "=== METADATA ===\n"
            + json.dumps(metadata, indent=2, sort_keys=True)
            + "\n\n"
        )

    return summary, metadata_text, "\n\n".join(sections)


def chat_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return url + "/chat/completions"


def _text_from(value) -> str:
    """Extract text from a string or an OpenAI-style content-part list."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for key in ("text", "summary", "content"):
                    text = item.get(key)
                    if isinstance(text, str):
                        parts.append(text)
                        break
        return "".join(parts)
    return ""


def _reasoning_from(message: dict) -> str:
    """Read whichever reasoning field the endpoint decided to use."""
    if not isinstance(message, dict):
        return ""
    for key in ("reasoning_content", "reasoning"):
        text = _text_from(message.get(key))
        if text:
            return text
    return _text_from(message.get("reasoning_details"))


def _task_from_answer(text: str) -> str | None:
    """Extract the value from the final ``Task:`` line, if there is one."""
    for line in reversed(text.splitlines()):
        cleaned = line.strip().lstrip("- ").strip("*`_ ")
        if cleaned.lower().startswith("task:"):
            value = cleaned.split(":", 1)[1].strip().strip("*`_ ").strip()
            if value:
                return value
    return None


class StreamPrinter:
    """Print reasoning and answer chunks as they arrive, keeping newlines."""

    HEADER = "\x1b[1;36m"
    DIM = "\x1b[2m"
    RESET = "\x1b[0m"

    def __init__(self, stream: TextIO | None = None, color: bool | None = None):
        self.stream = stream if stream is not None else sys.stdout
        if color is None:
            color = self.stream.isatty() and "NO_COLOR" not in os.environ
        self.color = color
        self._style = ""
        self._at_line_start = True
        self._anything = False
        self._reasoning = False
        self._answer = False
        self._reasoning_parts: list[str] = []
        self._answer_parts: list[str] = []

    @property
    def reasoning_text(self) -> str:
        return "".join(self._reasoning_parts)

    @property
    def answer_text(self) -> str:
        return "".join(self._answer_parts)

    def _set_style(self, style: str) -> None:
        if not self.color or style == self._style:
            return
        self.stream.write(self.RESET if style == "" else style)
        self._style = style

    def _newline(self) -> None:
        if not self._at_line_start:
            self.stream.write("\n")
            self._at_line_start = True

    def _header(self, label: str) -> None:
        self._set_style("")
        self._newline()
        if self._anything:
            self.stream.write("\n")
        if self.color:
            self.stream.write(f"{self.HEADER}{label}{self.RESET}\n")
        else:
            self.stream.write(f"{label}\n")
        self.stream.flush()
        self._at_line_start = True
        self._anything = True

    def _write(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()
        self._at_line_start = text.endswith("\n")
        self._anything = True

    def reasoning(self, text: str) -> None:
        if not text:
            return
        if not self._reasoning:
            self._header("Reasoning")
            self._reasoning = True
        self._set_style(self.DIM)
        self._reasoning_parts.append(text)
        self._write(text)

    def answer(self, text: str) -> None:
        if not text:
            return
        if not self._answer:
            self._header("Answer")
            self._answer = True
        self._set_style("")
        self._answer_parts.append(text)
        self._write(text)

    def finish(self) -> None:
        self._set_style("")
        self._newline()
        self.stream.flush()


def _iter_sse(response):
    """Yield the data payload of each server-sent event."""
    data_lines: list[str] = []
    for raw_line in response:
        line = raw_line.decode("utf-8", "replace").rstrip("\r\n")
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines.clear()
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if field == "data":
            data_lines.append(value[1:] if value.startswith(" ") else value)
    if data_lines:
        yield "\n".join(data_lines)


def _consume_full(body: str, printer: StreamPrinter) -> dict | None:
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        raise SystemExit(f"API returned non-JSON:\n{body[:4000]}")

    if isinstance(data, dict) and data.get("error"):
        raise SystemExit(
            "API error:\n" + json.dumps(data["error"], indent=2)[:4000]
        )

    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        raise SystemExit(
            "Unexpected API response:\n" + json.dumps(data, indent=2)[:4000]
        )
    if not isinstance(message, dict):
        raise SystemExit(
            "Unexpected API response:\n" + json.dumps(data, indent=2)[:4000]
        )

    reasoning = _reasoning_from(message)
    content = _text_from(message.get("content"))
    if reasoning:
        printer.reasoning(reasoning)
    if content:
        printer.answer(content)
    if not reasoning and not content:
        raise SystemExit("The API returned an empty message.")

    return data.get("usage")


def _consume_stream(response, printer: StreamPrinter) -> dict | None:
    usage: dict | None = None
    saw_text = False

    for payload in _iter_sse(response):
        if payload.strip() == "[DONE]":
            break

        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            raise SystemExit(f"Invalid stream event:\n{payload[:4000]}")

        if isinstance(event, dict) and event.get("error"):
            raise SystemExit(
                "API error:\n" + json.dumps(event["error"], indent=2)[:4000]
            )
        if not isinstance(event, dict):
            continue

        if event.get("usage"):
            usage = event["usage"]

        choices = event.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                continue
            reasoning = _reasoning_from(delta)
            content = _text_from(delta.get("content"))
            if reasoning:
                printer.reasoning(reasoning)
                saw_text = True
            if content:
                printer.answer(content)
                saw_text = True

    if not saw_text:
        raise SystemExit("The API returned an empty response.")
    return usage


def ask_llm(
    args: argparse.Namespace, prompt: str, printer: StreamPrinter
) -> dict | None:
    payload: dict = {
        "model": args.model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
    }
    if args.max_tokens is not None:
        payload["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if args.reasoning_effort is not None:
        payload["reasoning_effort"] = args.reasoning_effort
    if args.thinking is not None:
        payload["thinking"] = {"type": args.thinking}

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "wsp-read-harness/0.1",
    }
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    url = chat_url(args.base_url)
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            content_type = response.headers.get_content_type()
            if content_type == "application/json" or content_type.endswith("+json"):
                body = response.read().decode("utf-8", "replace")
                return _consume_full(body, printer)
            return _consume_stream(response, printer)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"API request failed ({exc.code}):\n{detail[:4000]}")
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not reach {url}: {exc.reason}")


def main() -> None:
    args = parse_args()

    if not args.model.is_file():
        raise SystemExit(f"No such file: {args.model}")
    if args.save and args.save.resolve() == args.model.resolve():
        raise SystemExit("--save must not point to the input model")

    description_text = ""
    if args.description_file:
        if not args.description_file.is_file():
            raise SystemExit(f"No such file: {args.description_file}")
        description = args.description_file.read_text(encoding="utf-8").strip()
        if description:
            description_text = (
                "=== RELEASE DESCRIPTION ===\n"
                f"{description}\n\n"
            )

    choices: list[str] = []
    if args.choices and not args.no_choices:
        for item in args.choices:
            for part in item.split(","):
                name = part.strip()
                if name and name not in choices:
                    choices.append(name)

    if choices:
        joined = ", ".join(choices)
        task_instruction = (
            "Identify which of the candidate tasks this network solves. "
            "Choose exactly one candidate, and do not invent another name."
        )
        choices_text = (
            "=== CANDIDATE TASKS ===\n"
            + "\n".join(f"- {name}" for name in choices)
            + "\n\n"
        )
        task_line = f"Task: <exactly one of: {joined}>"
    else:
        task_instruction = (
            "Identify the specific problem this network solves, not just its "
            "architecture family. If it is a policy, name the concrete "
            'environment or task. Do not answer only "classifier", '
            '"regression", or "MLP".'
        )
        choices_text = ""
        task_line = "Task: <specific task or environment>"

    print(f"Reading {args.model} ...", file=sys.stderr)
    summary, metadata_text, weights = build_checkpoint(
        args.model, args.max_chars, args.metadata
    )
    prompt = USER_PROMPT.format(
        task_instruction=task_instruction,
        choices=choices_text,
        task_line=task_line,
        description=description_text,
        metadata=metadata_text,
        summary=summary,
        weights=weights,
    )

    print(
        f"Prompt: {len(prompt):,} chars "
        f"({', '.join(summary.splitlines()[:2])})",
        file=sys.stderr,
    )
    if len(prompt) > 1_000_000:
        print(
            "warning: this prompt is large; use --max-chars to truncate it",
            file=sys.stderr,
        )

    printer = StreamPrinter()
    try:
        usage = ask_llm(args, prompt, printer)
    finally:
        printer.finish()

    chosen_task = _task_from_answer(printer.answer_text) if choices else None
    if choices:
        canonical = {name.lower(): name for name in choices}
        if chosen_task is None:
            print("warning: no Task: line found in the answer", file=sys.stderr)
        else:
            normalized = canonical.get(chosen_task.lower())
            if normalized is None:
                print(
                    f'warning: Task "{chosen_task}" is not one of: '
                    f'{", ".join(choices)}',
                    file=sys.stderr,
                )
            else:
                chosen_task = normalized

    if usage:
        print(
            f"\n[tokens: prompt={usage.get('prompt_tokens', '?')}, "
            f"completion={usage.get('completion_tokens', '?')}]",
            file=sys.stderr,
        )

    if args.save:
        record = {
            "base_url": args.base_url,
            "model": args.model_name,
            "checkpoint": str(args.model),
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "reasoning": printer.reasoning_text,
            "output": printer.answer_text,
            "task": chosen_task,
            "reasoning_effort": args.reasoning_effort,
            "thinking": args.thinking,
            "usage": usage,
        }
        try:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            args.save.write_text(
                json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            raise SystemExit(f"Could not write {args.save}: {exc}")
        print(f"Saved run to {args.save}", file=sys.stderr)


if __name__ == "__main__":
    main()
