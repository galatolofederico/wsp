# wsp

Weight-space programming experiments: can a language model identify what a
neural network does by reading its weights?

The repo contains four small PPO control tasks with an identical
`4 -> 256 -> 192` actor-critic (**51,203 parameters**), anonymized copies of the
trained checkpoints, and a harness that sends the raw weights to an
OpenAI-compatible chat endpoint.

## Layout

| Path | Purpose |
| --- | --- |
| `rl/` | PPO training, evaluation, the four environments, and checkpoint anonymization |
| `anonymous/` | Blind checkpoints with neutral tensor names and no metadata |
| `wsp-read-harness/` | LLM harness: renders a checkpoint as text and asks it to identify the task |
| `outputs/` | Raw JSON records from model runs (prompt, reasoning, answer, usage) |
| `reports/` | Experiment write-ups and analysis |

Each subproject has its own README with more detail:
[`rl/README.md`](rl/README.md) and
[`wsp-read-harness/README.md`](wsp-read-harness/README.md).

## Tasks

All four tasks expose four scalar observations, two discrete actions, reward
`+1` per surviving step, and truncation at 500 steps.

| Task | Observations | Actions |
| --- | --- | --- |
| CartPole | cart position/velocity, pole angle/angular velocity | push left/right |
| Flappy | bird height/velocity, next gap centre/distance | glide/flap |
| BallBeam | ball position/velocity, beam angle/angular velocity | torque left/right |
| Catch | paddle x, ball x/y, ball horizontal velocity | move left/right |

The shared network is an actor-critic MLP with two `tanh` hidden layers, a
2-logit policy head, and a 1-scalar value head.

## Quick start

Python 3.10+ is recommended.

### Train and evaluate

```bash
cd rl
pip install -r requirements.txt

python train.py --env flappy --out flappy_ppo.safetensors
python eval.py --env flappy --model flappy_ppo.safetensors --headless --episodes 20
```

Use `--env cartpole`, `flappy`, `ballbeam`, or `catch`. Omit `--headless` to
open the Tkinter viewer.

### Anonymize a checkpoint

```bash
cd rl
python anonymize.py flappy_ppo.safetensors ../anonymous/policy_flappy.safetensors \
  --mapping ../anonymous/policy_flappy_mapping.json
```

The anonymized files have identical tensor names, shapes, and size. The
mapping files contain the ground-truth task and must be kept out of prompts.

### Ask a model to read a checkpoint

```bash
cd wsp-read-harness
pip install -r requirements.txt

python wsp_read.py \
  --base-url https://api.deepseek.com \
  --api-key "$DEEPSEEK_API_KEY" \
  --model-name deepseek-flash \
  --model ../anonymous/policy_flappy.safetensors \
  --description-file descriptions/policy_4d_2a.txt \
  --thinking enabled \
  --reasoning-effort max \
  --save ../outputs/run.json
```

For blind runs, do not pass `--metadata`: safetensors metadata or the mapping
files can reveal the answer. Any OpenAI-compatible `/chat/completions`
endpoint works; the harness streams reasoning and answer text and saves the
full record with `--save`.

## Ground truth and results

| Checkpoint | Task |
| --- | --- |
| `anonymous/policy.safetensors` | CartPole |
| `anonymous/policy_flappy.safetensors` | Flappy |
| `anonymous/policy_ballbeam.safetensors` | BallBeam |
| `anonymous/policy_catch.safetensors` | Catch |

The DeepSeek V4.1 Flash 1017 run identified 3/4 correctly, confusing BallBeam
with CartPole. See [`reports/dsv4.1-flash-1017.md`](reports/dsv4.1-flash-1017.md)
for the full results and per-run analysis.
