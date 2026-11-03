#!/usr/bin/env python3
"""Load a trained CartPole PPO policy and watch it balance the pole.

Examples:
    python eval.py                          # 5 episodes, 1x speed
    python eval.py --episodes 0             # run until you close the window
    python eval.py --episodes 20 --speed 2  # 20 episodes at 2x speed
    python eval.py --headless --episodes 50 # no window, just print returns
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch

from cartpole import CartPoleEnv, CartPoleViewer
from model import load_model

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path,
                   default=SCRIPT_DIR / "cartpole_ppo.safetensors")
    p.add_argument("--episodes", type=int, default=5,
                   help="number of episodes to show (0 = forever)")
    p.add_argument("--speed", type=float, default=1.0,
                   help="playback speed multiplier (default: 1.0)")
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions instead of using the greedy policy")
    p.add_argument("--headless", action="store_true",
                   help="no window; only print episode returns")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def run_headless(model, args: argparse.Namespace) -> None:
    env = CartPoleEnv(seed=args.seed)
    returns = []
    episodes = args.episodes or 10
    for episode in range(1, episodes + 1):
        obs = env.reset()
        total = 0.0
        while True:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                action, _ = model.act(obs_t, deterministic=not args.stochastic)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total += reward
            if terminated or truncated:
                break
        returns.append(total)
        print(f"episode {episode:3d}: return {total:5.0f}")

    print(f"\nmean return: {statistics.mean(returns):.1f} "
          f"(min {min(returns):.0f}, max {max(returns):.0f})")


def run_viewer(model, args: argparse.Namespace) -> None:
    env = CartPoleEnv(seed=args.seed)
    viewer = CartPoleViewer(
        title=f"CartPole - PPO ({args.model.name})"
    )
    frame_ms = 20  # 50 Hz simulation, matching TAU = 0.02 s
    state = {
        "phase": "run",       # run | reset | finished
        "frame": 0,
        "next_reset": 0,
        "episode": 0,
        "steps": 0,
        "episode_return": 0.0,
        "last_status": "running",
        "step_accumulator": 0.0,
        "returns": [],
        "obs": env.reset(),
    }

    def start_episode() -> None:
        state["obs"] = env.reset()
        state["steps"] = 0
        state["episode_return"] = 0.0
        state["step_accumulator"] = 0.0
        state["phase"] = "run"
        state["last_status"] = "running"

    def tick() -> None:
        if not viewer.root.winfo_exists():
            return
        state["frame"] += 1

        if state["phase"] == "reset" and state["frame"] >= state["next_reset"]:
            state["episode"] += 1
            start_episode()

        if state["phase"] == "run":
            state["step_accumulator"] += args.speed
            n_sim_steps = int(state["step_accumulator"])
            state["step_accumulator"] -= n_sim_steps

            for _ in range(n_sim_steps):
                obs_t = torch.from_numpy(state["obs"]).float().unsqueeze(0)
                with torch.no_grad():
                    action, _ = model.act(
                        obs_t, deterministic=not args.stochastic
                    )
                obs, reward, terminated, truncated, _ = env.step(int(action))
                state["obs"] = obs
                state["episode_return"] += reward
                state["steps"] += 1

                if terminated or truncated:
                    state["phase"] = "reset"
                    state["last_status"] = (
                        "fell over - resetting" if terminated
                        else "time limit - resetting"
                    )
                    state["returns"].append(state["episode_return"])
                    print(
                        f"episode {state['episode'] + 1:3d}: "
                        f"return {state['episode_return']:5.0f}"
                    )
                    if args.episodes and len(state["returns"]) >= args.episodes:
                        state["phase"] = "finished"
                        state["next_reset"] = state["frame"] + 50
                    else:
                        state["next_reset"] = state["frame"] + 40
                    break

        if state["phase"] == "finished":
            if state["frame"] >= state["next_reset"]:
                viewer.close()
                return
            status = "all episodes done"
        else:
            status = state["last_status"]

        viewer.draw(
            state["obs"],
            episode=state["episode"] + 1,
            steps=state["steps"],
            episode_return=state["episode_return"],
            status=status,
            subtitle="q / close window to quit",
        )
        viewer.root.after(frame_ms, tick)

    viewer.root.after(frame_ms, tick)
    viewer.root.mainloop()

    if state["returns"]:
        print(f"\nmean return: {statistics.mean(state['returns']):.1f} "
              f"over {len(state['returns'])} episodes")


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(
            f"Model not found: {args.model}\nRun `python train.py` first."
        )

    model = load_model(args.model)
    label = "stochastic" if args.stochastic else "greedy"
    print(f"Loaded {args.model} ({label} policy)")

    if args.headless:
        run_headless(model, args)
    else:
        run_viewer(model, args)


if __name__ == "__main__":
    main()
