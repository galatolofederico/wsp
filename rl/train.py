#!/usr/bin/env python3
"""Train a tiny PPO agent on CartPole or Flappy and save the network.

Examples:
    python train.py                      # trains CartPole until solved (or 400k steps)
    python train.py --timesteps 200000   # shorter budget
    python train.py --env flappy         # second 4-D / 2-action task
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from cartpole import CartPoleVec
from flappy import FlappyVec
from model import ActorCritic, count_parameters, save_model

SCRIPT_DIR = Path(__file__).resolve().parent

ENVS = {
    "cartpole": CartPoleVec,
    "flappy": FlappyVec,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", choices=sorted(ENVS), default="cartpole",
                   help="environment to train on (default: cartpole)")
    p.add_argument("--timesteps", type=int, default=400_000,
                   help="maximum environment steps (default: 400k)")
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--rollout-steps", type=int, default=128,
                   help="steps per environment per PPO update")
    p.add_argument("--epochs", type=int, default=10,
                   help="optimisation epochs per rollout")
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-return", type=float, default=475.0,
                   help="stop when the deterministic eval reaches this mean")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=Path,
                   default=SCRIPT_DIR / "cartpole_ppo.safetensors")
    return p.parse_args()


@torch.no_grad()
def evaluate(
    model: ActorCritic,
    n_episodes: int,
    seed: int = 10_000,
    env_cls=CartPoleVec,
) -> float:
    """Mean return of the deterministic policy over ``n_episodes`` episodes."""
    env = env_cls(n_episodes, seed=seed)
    obs = env.reset()
    returns = np.zeros(n_episodes, dtype=np.float64)
    active = np.ones(n_episodes, dtype=bool)

    for _ in range(env.max_steps):
        if not active.any():
            break
        actions, _ = model.act(torch.from_numpy(obs).float(), deterministic=True)
        obs, _, rewards, terminated, truncated, _ = env.step(actions.numpy())
        returns += rewards * active
        active &= ~(terminated | truncated)

    return float(returns.mean())


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = ActorCritic()
    n_params = count_parameters(model)
    print(f"ActorCritic: {n_params:,} parameters "
          f"(obs {model.obs_dim} -> {' -> '.join(map(str, model.hidden))} -> "
          f"{model.n_actions} actions + value)")
    print(f"Environment: {args.env}")

    env_cls = ENVS[args.env]
    optimizer = Adam(model.parameters(), lr=args.lr, eps=1e-5)
    env = env_cls(args.n_envs, seed=args.seed)
    obs = env.reset()

    n_steps = args.rollout_steps
    n_envs = args.n_envs
    batch_size = n_steps * n_envs
    updates = max(1, args.timesteps // batch_size)

    ep_returns: list[float] = []
    best_eval = -float("inf")
    saved = False
    steps_done = 0
    start_time = time.time()

    print(f"Training for up to {updates} updates "
          f"({batch_size:,} steps each, {n_envs} parallel envs)\n")

    for update in range(1, updates + 1):
        # ----- collect a rollout ------------------------------------------
        mb_obs = torch.zeros((n_steps, n_envs, model.obs_dim))
        mb_actions = torch.zeros((n_steps, n_envs), dtype=torch.long)
        mb_logp = torch.zeros((n_steps, n_envs))
        mb_rewards = torch.zeros((n_steps, n_envs))
        mb_terminated = torch.zeros((n_steps, n_envs))
        mb_truncated = torch.zeros((n_steps, n_envs))
        mb_boot_obs = torch.zeros((n_steps, n_envs, model.obs_dim))

        for t in range(n_steps):
            obs_t = torch.from_numpy(obs).float()
            with torch.no_grad():
                actions, logp = model.act(obs_t)

            next_obs, boot_obs, rewards, terminated, truncated, info = env.step(
                actions.numpy()
            )

            mb_obs[t] = obs_t
            mb_actions[t] = actions
            mb_logp[t] = logp
            mb_rewards[t] = torch.from_numpy(rewards)
            mb_terminated[t] = torch.from_numpy(terminated.astype(np.float32))
            mb_truncated[t] = torch.from_numpy(truncated.astype(np.float32))
            mb_boot_obs[t] = torch.from_numpy(boot_obs)

            ep_returns.extend(info["episode_returns"])
            obs = next_obs

        steps_done += batch_size

        # ----- GAE and returns --------------------------------------------
        with torch.no_grad():
            values = model.value(mb_obs.reshape(-1, model.obs_dim)).reshape(n_steps, n_envs)
            boot_values = model.value(
                mb_boot_obs.reshape(-1, model.obs_dim)
            ).reshape(n_steps, n_envs)

            advantages = torch.zeros_like(values)
            last_gae = torch.zeros(n_envs)
            for t in reversed(range(n_steps)):
                # Bootstrap on time-limit truncation, but not on termination
                # (fallen pole / off-track cart).
                nonterminal = 1.0 - mb_terminated[t]
                delta = (
                    mb_rewards[t]
                    + args.gamma * boot_values[t] * nonterminal
                    - values[t]
                )
                done = torch.clamp(mb_terminated[t] + mb_truncated[t], 0.0, 1.0)
                last_gae = delta + args.gamma * args.gae_lambda * (1.0 - done) * last_gae
                advantages[t] = last_gae
            returns = advantages + values

        flat_obs = mb_obs.reshape(-1, model.obs_dim)
        flat_actions = mb_actions.reshape(-1)
        flat_logp = mb_logp.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_adv = advantages.reshape(-1)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        # ----- PPO update -------------------------------------------------
        for _ in range(args.epochs):
            perm = torch.randperm(batch_size)
            for start in range(0, batch_size, args.minibatch_size):
                idx = perm[start:start + args.minibatch_size]
                logp, entropy, value = model.evaluate(flat_obs[idx], flat_actions[idx])

                ratio = (logp - flat_logp[idx]).exp()
                pg_loss = -torch.min(
                    ratio * flat_adv[idx],
                    torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * flat_adv[idx],
                ).mean()
                v_loss = 0.5 * ((value - flat_returns[idx]) ** 2).mean()
                loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

        # ----- logging and evaluation -------------------------------------
        if update % args.log_every == 0 or update == updates:
            recent = ep_returns[-20:]
            mean_return = float(np.mean(recent)) if recent else float("nan")
            sps = steps_done / max(time.time() - start_time, 1e-9)
            print(
                f"update {update:4d}/{updates} | steps {steps_done:7d} | "
                f"mean return (last {len(recent):2d}) {mean_return:6.1f} | "
                f"{sps:5.0f} steps/s",
                flush=True,
            )

        if update >= args.eval_every and update % args.eval_every == 0:
            eval_return = evaluate(
                model, args.eval_episodes, env_cls=env_cls
            )
            marker = ""
            if eval_return > best_eval:
                best_eval = eval_return
                save_model(model, args.out, {
                    "format": f"{args.env}-ppo",
                    "eval_return": f"{eval_return:.1f}",
                    "timesteps": steps_done,
                    "parameters": count_parameters(model),
                })
                saved = True
                marker = "  <- saved"
            print(f"  deterministic eval: {eval_return:6.1f}{marker}", flush=True)

            if eval_return >= args.target_return:
                print(f"\nSolved: eval mean {eval_return:.1f} >= {args.target_return:.1f}")
                break

    if not saved:
        save_model(model, args.out, {
            "format": f"{args.env}-ppo",
            "eval_return": "n/a",
            "timesteps": steps_done,
            "parameters": count_parameters(model),
        })
        best_eval = float("nan")

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s ({steps_done:,} steps)")
    print(f"Best eval return: {best_eval:.1f}")
    print(f"Saved weights to: {args.out}")


if __name__ == "__main__":
    main()
