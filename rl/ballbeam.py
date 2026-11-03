"""Ball-beam balancing environment and a small Tkinter viewer.

The state is ``[x, vx, theta, omega]``: the ball position along the beam, its
velocity, the beam angle, and the beam angular velocity. Action 0 applies a
negative torque and action 1 applies a positive torque. The task is to keep
the rolling ball on the beam by tilting it; episodes end when the ball rolls
off the end and are truncated at 500 steps.

The interface intentionally matches ``CartPoleVec`` and ``FlappyVec``: four
scalar observations, two discrete actions, reward +1 per surviving step, and
the same reset/step contract including ``boot_obs`` and ``episode_returns``.

Only numpy and tkinter (stdlib) are required.
"""

from __future__ import annotations

import numpy as np
import tkinter as tk

# --- Physics constants ------------------------------------------------------
DT = 0.02
GRAVITY = 9.8
TORQUE = 6.0
DAMPING = 1.0
BALL_FRICTION = 0.4
THETA_MAX = 0.5  # beam angle limit (~28.6 degrees)
X_LIMIT = 0.8  # half the usable beam length
VX_SCALE = 2.0
OMEGA_SCALE = 2.0
MAX_STEPS = 500

STATE_DIM = 4


def _initial_state(rng: np.random.Generator, n: int = 1) -> np.ndarray:
    state = np.zeros((n, STATE_DIM), dtype=np.float64)
    state[:, 0] = rng.uniform(-0.25, 0.25, size=n)
    state[:, 1] = rng.uniform(-0.4, 0.4, size=n)
    state[:, 2] = rng.uniform(-0.15, 0.15, size=n)
    state[:, 3] = rng.uniform(-0.3, 0.3, size=n)
    return state


def _dynamics(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """One Euler step of the ball-beam dynamics.

    ``state`` is (..., 4) and ``action`` is (...) with 0 = torque left and
    1 = torque right. Works for scalars and batches.
    """
    x, vx, theta, omega = (state[..., i] for i in range(4))
    torque = np.where(action == 1, TORQUE, -TORQUE)

    omega = omega + torque * DT
    omega = omega * (1.0 - DAMPING * DT)
    theta = theta + omega * DT

    at_limit = np.abs(theta) > THETA_MAX
    theta = np.clip(theta, -THETA_MAX, THETA_MAX)
    omega = np.where(at_limit, 0.0, omega)

    # A solid sphere rolling on the beam: a = (5/7) g sin(theta), plus a small
    # rolling friction term that keeps the ball from oscillating forever.
    vx = vx + ((5.0 / 7.0) * GRAVITY * np.sin(theta) - BALL_FRICTION * vx) * DT
    x = x + vx * DT
    return np.stack([x, vx, theta, omega], axis=-1)


def _obs(state: np.ndarray) -> np.ndarray:
    """Normalise the raw state for the network."""
    obs = np.array(state, dtype=np.float64, copy=True)
    obs[..., 0] /= X_LIMIT
    obs[..., 1] /= VX_SCALE
    obs[..., 2] /= THETA_MAX
    obs[..., 3] /= OMEGA_SCALE
    return obs


class BallBeamEnv:
    """Single environment (no auto-reset) used for evaluation."""

    def __init__(self, seed: int | None = None, max_steps: int = MAX_STEPS):
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.state = _initial_state(self.rng)[0]
        self.steps = 0
        self.reset()

    def reset(self) -> np.ndarray:
        self.state = _initial_state(self.rng)[0]
        self.steps = 0
        return _obs(self.state)

    def step(self, action: int):
        self.state = _dynamics(self.state, np.asarray(action))
        self.steps += 1
        terminated = bool(abs(self.state[0]) > X_LIMIT)
        truncated = (not terminated) and self.steps >= self.max_steps
        return _obs(self.state), 1.0, terminated, truncated, {}


class BallBeamVec:
    """Vectorised ball-beam environments with automatic reset.

    ``step`` returns the post-step observation, the observation to bootstrap
    from (identical except on truncation, where it is the final observation),
    rewards, terminated/truncated masks and finished-episode returns.
    """

    def __init__(
        self,
        n_envs: int,
        seed: int | None = None,
        max_steps: int = MAX_STEPS,
    ):
        self.n_envs = n_envs
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.state = _initial_state(self.rng, n_envs)
        self.steps = np.zeros(n_envs, dtype=np.int64)
        self.ep_return = np.zeros(n_envs, dtype=np.float64)
        self.reset()

    def reset(self) -> np.ndarray:
        self.state[:] = _initial_state(self.rng, self.n_envs)
        self.steps[:] = 0
        self.ep_return[:] = 0.0
        return _obs(self.state)

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.int64)
        self.state = _dynamics(self.state, actions)
        self.steps += 1
        self.ep_return += 1.0

        terminated = np.abs(self.state[:, 0]) > X_LIMIT
        truncated = (self.steps >= self.max_steps) & ~terminated
        done = terminated | truncated

        rewards = np.ones(self.n_envs, dtype=np.float32)
        boot_obs = _obs(self.state)
        finished_returns = self.ep_return[done].tolist()

        if done.any():
            self.state[done] = _initial_state(self.rng, int(done.sum()))
            self.steps[done] = 0
            self.ep_return[done] = 0.0

        return (
            _obs(self.state),
            boot_obs,
            rewards,
            terminated,
            truncated,
            {"episode_returns": finished_returns},
        )


class BallBeamViewer:
    """Minimal Tkinter renderer for a single ball-beam episode."""

    WIDTH = 760
    HEIGHT = 480
    PIVOT_X = WIDTH / 2
    PIVOT_Y = HEIGHT / 2 + 40
    BEAM_PX = 560.0
    BALL_R = 13

    BG = "#0f1720"
    BEAM = "#38bdf8"
    BEAM_EDGE = "#0ea5e9"
    PIVOT = "#64748b"
    BALL = "#fbbf24"
    BALL_EDGE = "#f59e0b"
    TEXT = "#e2e8f0"
    MUTED = "#94a3b8"

    def __init__(self, title: str = "BallBeam - PPO"):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.resizable(False, False)
        self.canvas = tk.Canvas(
            self.root,
            width=self.WIDTH,
            height=self.HEIGHT,
            bg=self.BG,
            highlightthickness=0,
        )
        self.canvas.pack()

        c = self.canvas
        self.beam = c.create_line(
            0, 0, 0, 0, fill=self.BEAM, width=12, capstyle=tk.ROUND
        )
        self.pivot = c.create_oval(0, 0, 0, 0, fill=self.PIVOT, outline="")
        self.ball = c.create_oval(
            0, 0, 0, 0, fill=self.BALL, outline=self.BALL_EDGE, width=2
        )

        self.hud = c.create_text(
            24, 20, anchor="nw", text="", fill=self.TEXT,
            font=("Courier", 15, "bold"),
        )
        self.hint = c.create_text(
            self.WIDTH - 24, 20, anchor="ne", text="", fill=self.MUTED,
            font=("Courier", 11),
        )

    def draw(
        self,
        state: np.ndarray,
        episode: int,
        steps: int,
        episode_return: float,
        status: str,
        subtitle: str = "",
    ) -> None:
        """Redraw the scene. ``state`` is ``[x, vx, theta, omega]``."""
        x, _, theta, _ = state
        half = self.BEAM_PX / 2
        dx = np.cos(theta) * half
        dy = np.sin(theta) * half

        # Physics y points up; screen y points down.
        self.canvas.coords(
            self.beam,
            self.PIVOT_X - dx, self.PIVOT_Y + dy,
            self.PIVOT_X + dx, self.PIVOT_Y - dy,
        )
        pivot_r = 10
        self.canvas.coords(
            self.pivot,
            self.PIVOT_X - pivot_r, self.PIVOT_Y - pivot_r,
            self.PIVOT_X + pivot_r, self.PIVOT_Y + pivot_r,
        )

        ball_x = self.PIVOT_X + float(x) / X_LIMIT * half * np.cos(theta)
        ball_y = self.PIVOT_Y - float(x) / X_LIMIT * half * np.sin(theta)
        r = self.BALL_R
        self.canvas.coords(
            self.ball, ball_x - r, ball_y - r, ball_x + r, ball_y + r
        )

        text = (
            f"episode {episode}\n"
            f"step    {steps:3d} / 500\n"
            f"return  {episode_return:5.0f}\n"
            f"ball    {x:+5.2f}\n"
            f"angle   {np.degrees(theta):+6.1f} deg\n"
            f"status  {status}"
        )
        self.canvas.itemconfigure(self.hud, text=text)
        self.canvas.itemconfigure(self.hint, text=subtitle)
        self.canvas.update_idletasks()

    def close(self) -> None:
        try:
            self.root.destroy()
        except tk.TclError:
            pass
