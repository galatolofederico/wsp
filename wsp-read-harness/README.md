# WSP read harness

Ask an LLM what a `.safetensors` checkpoint does by sending it the raw weights
as text. Inspired by
[Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/).

The script talks to any OpenAI-compatible `/chat/completions` endpoint and
does not need an SDK.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python wsp_read.py \
  --base-url http://localhost:8000/v1 \
  --api-key "$OPENAI_API_KEY" \
  --model-name my-llm \
  --model ../cartpole/cartpole_ppo.safetensors
```

`--base-url` is the part before `/chat/completions` (for example
`https://api.openai.com/v1`). If you pass the full endpoint URL, it is used
as-is. `--api-key` may be omitted for local servers; it defaults to
`$OPENAI_API_KEY`.

The prompt contains the tensor names, shapes, dtype, basic statistics and all
weight values, so the LLM can reason about the network without running it.

Useful options:

```bash
--max-chars 400000   # truncate the weight text for smaller contexts
--metadata           # also send the safetensors metadata (may give the answer away)
--max-tokens 2048
--temperature 0
```

For large checkpoints the full weight dump will not fit in a context window.
Either raise `--max-chars` to whatever the model supports, or use a harness
that lets the model inspect tensors with tools.
