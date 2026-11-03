# RL environments: CartPole and Flappy

A minimal, dependency-light implementation of PPO that trains a ~51k-parameter
network to balance the CartPole. The same network is also trained on a second
4-observation / 2-action task, a one-dimensional Flappy gate-navigation game,
which gives a matching checkpoint for blind weight-space experiments. The
CartPole physics is a NumPy port of the classic Gymnasium CartPole-v1 dynamics,
so no Gym installation is required. The trained weights are stored as a
`.safetensors` file, and a Tkinter viewer lets you watch either policy.

## Setup

```bash
pip install -r requirements.txt
```

Only PyTorch, NumPy and safetensors are needed. The viewer uses Python's
standard-library Tkinter (no pygame/matplotlib).

## Train

```bash
python train.py
```

- 16 parallel environments, 128-step rollouts, PPO with GAE.
- Network: `4 -> 256 -> 192 -> {2 policy logits, 1 value}` = **51,203 params**.
- Trains for at most 400k env steps and stops early once the deterministic
  policy averages at least 475/500 over 10 evaluation episodes.
- Saves the best checkpoint to `cartpole_ppo.safetensors` (with architecture
  metadata such as hidden sizes inside the file).
- `--env flappy` trains the second task described below; no network or PPO
  changes are involved.

On a CPU this typically takes a couple of minutes. Useful flags:

```bash
python train.py --timesteps 200000
python train.py --seed 2 --out my_policy.safetensors
python train.py --env flappy --out flappy_ppo.safetensors
```

## Watch it balance

```bash
python eval.py                       # 5 episodes at 1x speed
python eval.py --episodes 0          # loop until you close the window
python eval.py --episodes 20 --speed 2
python eval.py --model my_policy.safetensors
```

The viewer shows the episode, step count, return and pole angle in real time.
Close the window to quit. For a quick no-window check:

```bash
python eval.py --headless --episodes 50
python eval.py --env flappy --headless --episodes 50
```

## Second 4-D / 2-action task: Flappy

`model.py` is shaped only by `obs_dim=4` and `n_actions=2`, so it is not
CartPole-specific. `flappy.py` implements a one-dimensional gate-navigation
task with the same interface:

- observation `[y, vy / 2, gap_y, gap_dx]`: bird height, vertical velocity,
  centre of the next gate gap, and distance to the gate
- actions `0 = glide`, `1 = flap`
- reward `+1` per surviving step; terminate on the ground, the ceiling or a
  gate edge; truncate at 500 steps
- `FlappyVec` provides the same `reset`/`step` contract as `CartPoleVec`,
  including `boot_obs` and `episode_returns`, so the PPO/GAE code is reused
  unchanged

Train and evaluate it with:

```bash
python train.py --env flappy --seed 1 --out flappy_ppo.safetensors
python eval.py --env flappy --model flappy_ppo.safetensors --headless --episodes 50
python eval.py --env flappy --model flappy_ppo.safetensors
```

The checkpoint has exactly the same 8 tensors and 51,203 parameters as the
CartPole one. Check that with:

```python
from safetensors.torch import load_file

a = load_file("cartpole_ppo.safetensors")
b = load_file("flappy_ppo.safetensors")
assert a.keys() == b.keys()
assert all(a[k].shape == b[k].shape for k in a)
```

### Blind pair

Anonymize the Flappy checkpoint with the existing script:

```bash
python anonymize.py flappy_ppo.safetensors ../anonymous/policy_flappy.safetensors \
    --mapping ../anonymous/policy_flappy_mapping.json
```

`policy.safetensors` and `policy_flappy.safetensors` then have identical
tensor names, shapes and file size; only the values differ. Keep both mapping
files out of the prompt sent to the model under test.

## Anonymized checkpoint (blind weight-space experiments)

Safetensors files carry a JSON header with optional metadata, and that metadata
is visible to anyone who opens the file. The checkpoints store `format`,
`obs_dim`, `n_actions`, `hidden`, `parameters`, `timesteps` and `eval_return`.

Metadata is not needed for loading. To make a blind copy, also rename the
semantic tensor keys (`actor`, `critic`, `trunk`):

```bash
python anonymize.py cartpole_ppo.safetensors ../anonymous/policy.safetensors \
    --mapping ../anonymous/policy_mapping.json
```

This writes a copy with no `__metadata__` block and neutral tensor names.
The mapping file contains the original names — keep it out of any prompt sent
to the model being tested. Use `--keep-names` to strip metadata but preserve
tensor names (that copy stays loadable by `eval.py`).

Note that even a fully anonymized file still reveals the architecture from its
shapes (`4 -> 256 -> 192`, two heads of size 2 and 1), and its weights may
still reveal the task. Only the explicit labels are removed. The Flappy
checkpoint above has exactly the same shapes, so an anonymized CartPole/Flappy
pair reveals nothing from its structure and the model must reason from the
weights alone.

## Files

| File | Purpose |
| --- | --- |
| `cartpole.py` | CartPole physics (single + vectorised) and Tkinter viewer |
| `flappy.py` | Flappy gate environment (single + vectorised) and Tkinter viewer |
| `model.py` | Actor-critic MLP and safetensors save/load helpers |
| `train.py` | PPO training loop (`--env cartpole` or `--env flappy`) |
| `eval.py` | Loads a checkpoint and renders episodes |
| `anonymize.py` | Strips safetensors metadata and/or renames tensors |
