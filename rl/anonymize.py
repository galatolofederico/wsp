#!/usr/bin/env python3
"""Make an anonymized copy of a safetensors checkpoint.

Safetensors stores an optional ``__metadata__`` dictionary in its JSON header,
and the tensor names themselves are visible too. For blind weight-space
experiments both can give the task away (here: ``format=cartpole-ppo`` and
``actor`` / ``critic`` / ``trunk`` names).

This script removes the metadata and, by default, renames every tensor to a
neutral identifier. Tensor values, dtypes and shapes are unchanged.

Examples:
    python anonymize.py model.safetensors ../anonymous/policy.safetensors \
        --mapping ../anonymous/policy_mapping.json
    python anonymize.py model.safetensors ../anonymous/policy_nometa.safetensors \
        --keep-names
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--keep-names", action="store_true",
                   help="strip metadata only; keep the original tensor names")
    p.add_argument("--mapping", type=Path,
                   help="write the original -> anonymized name mapping here "
                        "(do not include this file in a blind prompt)")
    p.add_argument("--prefix", default="tensor",
                   help="prefix for anonymized tensor names (default: tensor)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with safe_open(str(args.input), framework="pt", device="cpu") as f:
        metadata = f.metadata()
        state = {key: f.get_tensor(key) for key in f.keys()}

    if args.keep_names:
        mapping = {key: key for key in state}
    else:
        # Order by size and shape only, so the new names do not encode the
        # original semantic names.
        keys = sorted(state, key=lambda k: (state[k].numel(), tuple(state[k].shape)))
        width = max(2, len(str(max(len(keys) - 1, 0))))
        mapping = {
            key: f"{args.prefix}_{i:0{width}d}"
            for i, key in enumerate(keys)
        }

    anonymized = {
        mapping[key]: tensor.contiguous() for key, tensor in state.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(anonymized, str(args.output), metadata=None)

    print(f"Wrote {args.output}")
    if metadata:
        print(f"Removed {len(metadata)} metadata entries:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    else:
        print("No metadata was present.")

    if mapping == {k: k for k in state}:
        print("Tensor names preserved.")
    else:
        print("Tensor name mapping (original -> anonymized):")
        for original, anonymous in mapping.items():
            print(f"  {original:16s} -> {anonymous}")

    if args.mapping:
        args.mapping.parent.mkdir(parents=True, exist_ok=True)
        args.mapping.write_text(json.dumps({
            "input": str(args.input),
            "output": str(args.output),
            "removed_metadata": metadata,
            "tensor_map": mapping,
        }, indent=2) + "\n")
        print(f"Saved name mapping to {args.mapping}")


if __name__ == "__main__":
    main()
