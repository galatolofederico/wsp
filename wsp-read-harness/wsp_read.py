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
        --model ../cartpole/cartpole_ppo.safetensors
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

from safetensors import safe_open

SYSTEM_PROMPT = (
    "You are an expert in neural network architectures and mechanistic "
    "interpretability. You can infer what a network does by reading its weights."
)

USER_PROMPT = """\
Below is a neural network checkpoint from a safetensors file, rendered as text.
You cannot run the network. Use only the weights and the tensor names/shapes.

Infer what this model was trained to do. Explain:
1. the architecture and how data flows through it,
2. the task or behavior the network implements,
3. which tensors and weight patterns support your conclusion.

Be concrete: cite tensor names, shapes, and standout values.

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
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
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
        f"file: {path}",
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


def ask_llm(args: argparse.Namespace, prompt: str) -> dict:
    payload: dict = {
        "model": args.model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    if args.max_tokens is not None:
        payload["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        payload["temperature"] = args.temperature

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
            body = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"API request failed ({exc.code}):\n{detail[:4000]}")
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not reach {url}: {exc.reason}")

    try:
        return json.loads(body)
    except json.JSONDecodeError:
        raise SystemExit(f"API returned non-JSON:\n{body[:4000]}")


def main() -> None:
    args = parse_args()

    if not args.model.is_file():
        raise SystemExit(f"No such file: {args.model}")

    print(f"Reading {args.model} ...", file=sys.stderr)
    summary, metadata_text, weights = build_checkpoint(
        args.model, args.max_chars, args.metadata
    )
    prompt = USER_PROMPT.format(
        metadata=metadata_text, summary=summary, weights=weights
    )

    print(
        f"Prompt: {len(prompt):,} chars "
        f"({summary.splitlines()[1]}, {summary.splitlines()[2]})",
        file=sys.stderr,
    )
    if len(prompt) > 1_000_000:
        print(
            "warning: this prompt is large; use --max-chars to truncate it",
            file=sys.stderr,
        )

    response = ask_llm(args, prompt)

    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        raise SystemExit(
            "Unexpected API response:\n" + json.dumps(response, indent=2)[:4000]
        )

    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    if not content:
        content = message.get("reasoning_content") or ""
    if not content:
        raise SystemExit("The API returned an empty message.")

    print(content)

    usage = response.get("usage")
    if usage:
        print(
            f"\n[tokens: prompt={usage.get('prompt_tokens', '?')}, "
            f"completion={usage.get('completion_tokens', '?')}]",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
