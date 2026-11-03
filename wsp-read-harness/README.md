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
  --model ../rl/cartpole_ppo.safetensors
```

`--base-url` is the part before `/chat/completions` (for example
`https://api.openai.com/v1`). If you pass the full endpoint URL, it is used
as-is. `--api-key` may be omitted for local servers; it defaults to
`$OPENAI_API_KEY`.

The prompt contains the tensor names, shapes, dtype, basic statistics and all
weight values, so the LLM can reason about the network without running it.

Responses are streamed as server-sent events. Reasoning is printed under a
`Reasoning` heading (dimmed when stdout is a terminal), then the reply under
`Answer`; newlines are preserved. ANSI styling is disabled automatically when
output is piped or `NO_COLOR` is set. If the endpoint ignores streaming and
returns a normal JSON response, that is handled too.

Useful options:

```bash
--max-chars 400000   # truncate the weight text for smaller contexts
--metadata           # also send the safetensors metadata (may give the answer away)
--description-file descriptions/policy_4d_2a.txt  # neutral task/interface description
--reasoning-effort high  # thinking effort (alias: --thinking-effort)
--thinking disabled      # DeepSeek thinking-mode toggle
--save run.json      # save the full input, reasoning and answer as JSON
--max-tokens 2048
--temperature 0
```

For DeepSeek V4, `--reasoning-effort` is sent as the top-level
`reasoning_effort` field (`low`, `high`, `max`; compatibility values such as
`medium` and `xhigh` are also accepted). `--thinking` maps to
`thinking.type`. Thinking is enabled by default at high effort, and
`temperature` has no effect while it is enabled.

For policy checkpoints, `--description-file` prepends a neutral release
description: observation, action and network interface, but no task name. This
mirrors the setup in the paper and prevents the model from misreading a
multi-head actor-critic as a single sequential classifier.
`descriptions/policy_4d_2a.txt` matches the anonymized CartPole/Flappy pair.
The prompt also states PyTorch's `[out_features, in_features]` weight layout
and requires a final `Task:` line.

For large checkpoints the full weight dump will not fit in a context window.
Either raise `--max-chars` to whatever the model supports, or use a harness
that lets the model inspect tensors with tools.

`--save` writes a JSON object containing the endpoint and model names, the
checkpoint path, thinking settings, the full messages sent, the accumulated
reasoning, the final answer, and token usage when the endpoint reports it.

## Two-task blind comparison

`../anonymous/` contains two anonymized checkpoints with identical tensor
names, shapes and file size:

- `policy.safetensors` — trained on CartPole
- `policy_flappy.safetensors` — trained on the 4-D / 2-action Flappy task

Run the harness on both with the same prompt and options. Because the
architecture and header are identical, the model must distinguish the tasks
from the weight values alone. Do not pass `--metadata`, and do not include the
`*_mapping.json` files in the prompt.
