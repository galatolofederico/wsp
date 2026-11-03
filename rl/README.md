# RL environments: CartPole, Flappy, BallBeam and Catch

A minimal, dependency-light implementation of PPO that trains a ~51k-parameter
network on four low-dimensional control tasks with an identical interface:
four scalar observations, two discrete actions, reward +1 per surviving step,
and truncation at 500 steps. The tasks are CartPole (balance a pole), Flappy
(flap through gates), BallBeam (stabilise a ball on a tilting beam) and Catch
(move a paddle under falling balls). The CartPole physics is a NumPy port of
the classic Gymnasium CartPole-v1 dynamics, so no Gym installation is required.
Trained weights are stored as `.safetensors` files, and a Tkinter viewer lets
you watch any policy.

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
- Saves the best checkpoint to a `.safetensors` file (with architecture
  metadata such as hidden sizes inside the file).
- `--env` selects the task; no network or PPO changes are involved.

On a CPU this typically takes a couple of minutes. Useful flags:

```bash
python train.py --timesteps 200000
python train.py --seed 2 --out my_policy.safetensors
python train.py --env flappy --out flappy_ppo.safetensors
python train.py --env ballbeam --out ballbeam_ppo.safetensors
python train.py --env catch --out catch_ppo.safetensors
```

## Watch it run

```bash
python eval.py                       # 5 episodes at 1x speed
python eval.py --episodes 0          # loop until you close the window
python eval.py --episodes 20 --speed 2
python eval.py --model my_policy.safetensors
```

The viewer shows the episode, step count, return and task-specific state in
real time. Close the window to quit. For a quick no-window check:

```bash
python eval.py --headless --episodes 50
python eval.py --env flappy --headless --episodes 50
python eval.py --env ballbeam --headless --episodes 50
python eval.py --env catch --headless --episodes 50
```

## Tasks with the same 4-D observation / 2-action interface

`model.py` is shaped only by `obs_dim=4` and `n_actions=2`, so it is not
CartPole-specific. All four environments expose the same `reset`/`step`
contract, including `boot_obs` and `episode_returns`, so the PPO/GAE code is
reused unchanged.

| Task | Observations | Actions | Episode ends when |
| --- | --- | --- | --- |
| CartPole | cart position/velocity, pole angle/angular velocity | push left/right | pole tilts >12 deg or cart leaves the track |
| Flappy | bird height/velocity, next gap centre/distance | glide/flap | ground, ceiling or gate edge |
| BallBeam | ball position/velocity, beam angle/angular velocity | torque left/right | ball rolls off the beam |
| Catch | paddle x, ball x/y, ball horizontal velocity | move left/right | ball reaches the floor outside the paddle |

Reward is `+1` for every surviving step in all four tasks.

### Flappy

`flappy.py` implements a one-dimensional gate-navigation task:

- observation `[y, vy / 2, gap_y, gap_dx]`: bird height, vertical velocity,
  centre of the next gate gap, and distance to the gate
- actions `0 = glide`, `1 = flap`
- reward `+1` per surviving step; terminate on the ground, the ceiling or a
  gate edge; truncate at 500 steps

### BallBeam

`ballbeam.py` implements a ball rolling on a beam controlled by two torques:

- observation `[x / 0.8, vx / 2, theta / 0.5, omega / 2]`: ball position along
  the beam, ball velocity, beam angle, and beam angular velocity
- actions `0 = negative torque`, `1 = positive torque`
- reward `+1` per surviving step; terminate when the ball rolls past either
  end of the beam; truncate at 500 steps

### Catch

`catch.py` implements a one-paddle falling-ball game:

- observation `[paddle_x - 0.5, ball_x - 0.5, ball_y, ball_vx / 0.4]`: paddle
  position, ball position, and ball horizontal velocity
- actions `0 = move left`, `1 = move right`
- reward `+1` per surviving step; a successful catch launches a new ball from
  the top, while a miss terminates the episode; truncate at 500 steps

Train and evaluate the new tasks with:

```bash
python train.py --env ballbeam --seed 1 --out ballbeam_ppo.safetensors
python train.py --env catch --seed 1 --out catch_ppo.safetensors
python eval.py --env ballbeam --model ballbeam_ppo.safetensors --headless --episodes 50
python eval.py --env catch --model catch_ppo.safetensors --headless --episodes 50
python eval.py --env ballbeam --model ballbeam_ppo.safetensors
python eval.py --env catch --model catch_ppo.safetensors
```

All four checkpoints have exactly the same 8 tensors and 51,203 parameters,
and the anonymized copies have identical file sizes. Check that with:

```python
from safetensors.torch import load_file

models = [
    load_file(path)
    for path in [
        "cartpole_ppo.safetensors",
        "flappy_ppo.safetensors",
        "ballbeam_ppo.safetensors",
        "catch_ppo.safetensors",
    ]
]
assert all(m.keys() == models[0].keys() for m in models)
assert all(
    m[k].shape == models[0][k].shape for m in models for k in models[0]
)
```

### Blind set

Anonymize the new checkpoints with the existing script:

```bash
python anonymize.py ballbeam_ppo.safetensors ../anonymous/policy_ballbeam.safetensors \
    --mapping ../anonymous/policy_ballbeam_mapping.json
python anonymize.py catch_ppo.safetensors ../anonymous/policy_catch.safetensors \
    --mapping ../anonymous/policy_catch_mapping.json
```

All anonymized policy files then have identical tensor names, shapes and file
size; only the values differ. Keep the mapping files out of the prompt sent to
the model under test.

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
still reveal the task. Only the explicit labels are removed. CartPole, Flappy,
BallBeam and Catch have exactly the same shapes, so an anonymized four-task set
reveals nothing from its structure and the model must reason from the weights
alone.

## Files

| File | Purpose |
| --- | --- |
| `cartpole.py` | CartPole physics (single + vectorised) and Tkinter viewer |
| `flappy.py` | Flappy gate environment (single + vectorised) and Tkinter viewer |
| `ballbeam.py` | Ball-beam environment (single + vectorised) and Tkinter viewer |
| `catch.py` | Falling-ball catch environment (single + vectorised) and Tkinter viewer |
| `model.py` | Actor-critic MLP and safetensors save/load helpers |
| `train.py` | PPO training loop (`--env cartpole`, `flappy`, `ballbeam`, `catch`) |
| `eval.py` | Loads a checkpoint and renders episodes for any task |
| `anonymize.py` | Strips safetensors metadata and/or renames tensors |
