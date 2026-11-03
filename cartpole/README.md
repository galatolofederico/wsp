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

## Files

| File | Purpose |
| --- | --- |
| `cartpole.py` | CartPole physics (single + vectorised) and Tkinter viewer |
| `model.py` | Actor-critic MLP and safetensors save/load helpers |
| `train.py` | PPO training loop |
| `eval.py` | Loads the checkpoint and renders episodes |
