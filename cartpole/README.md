# CartPole PPO

A minimal, dependency-light implementation of PPO that trains a ~51k-parameter
network to balance the CartPole. The physics is a NumPy port of the classic
Gymnasium CartPole-v1 dynamics, so no Gym installation is required. The trained
weights are stored as a `.safetensors` file, and a Tkinter viewer lets you watch
the trained policy.

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

- 16 parallel CartPole environments, 128-step rollouts, PPO with GAE.
- Network: `4 -> 256 -> 192 -> {2 policy logits, 1 value}` = **51,203 params**.
- Trains for at most 400k env steps and stops early once the deterministic
  policy averages at least 475/500 over 10 evaluation episodes.
- Saves the best checkpoint to `cartpole_ppo.safetensors` (with architecture
  metadata such as hidden sizes inside the file).

On a CPU this typically takes a couple of minutes. Useful flags:

```bash
python train.py --timesteps 200000
python train.py --seed 2 --out my_policy.safetensors
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
```

## Anonymized checkpoint (blind weight-space experiments)

Safetensors files carry a JSON header with optional metadata, and that metadata
is visible to anyone who opens the file. This checkpoint stores `format`,
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
still reveal the task. Only the explicit labels are removed.

## Files

| File | Purpose |
| --- | --- |
| `cartpole.py` | CartPole physics (single + vectorised) and Tkinter viewer |
| `model.py` | Actor-critic MLP and safetensors save/load helpers |
| `train.py` | PPO training loop |
| `eval.py` | Loads the checkpoint and renders episodes |
| `anonymize.py` | Strips safetensors metadata and/or renames tensors |
