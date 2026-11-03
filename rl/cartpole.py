"""CartPole environment and a small Tkinter viewer.

The physics is a NumPy port of the classic CartPole equations used by
Gymnasium's CartPole-v1, so the task behaves exactly like the usual benchmark:
reward +1 per step, episode ends when the pole tilts more than 12 degrees or
the cart leaves the +/-2.4 m track, and episodes are truncated at 500 steps.

Only numpy and tkinter (stdlib) are required; no Gym installation.
"""

from __future__ import annotations

import numpy as np
import tkinter as tk

# --- Physics constants (same as Gymnasium CartPole-v1) ----------------------
GRAVITY = 9.8
MASSCART = 1.0
MASSPOLE = 0.1
TOTAL_MASS = MASSCART + MASSPOLE
LENGTH = 0.5  # half the pole length
POLEMASS_LENGTH = MASSPOLE * LENGTH
FORCE_MAG = 10.0
TAU = 0.02
THETA_LIMIT = 12 * 2 * np.pi / 360  # ~0.2095 rad
X_LIMIT = 2.4
MAX_STEPS = 500


def _dynamics(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """One Euler step of the CartPole dynamics.

    ``state`` is (..., 4) = [x, x_dot, theta, theta_dot] and ``action`` is
    (...) with 0 = push left, 1 = push right. Works for scalars and batches.
    """
    x, x_dot, theta, theta_dot = (state[..., i] for i in range(4))
    force = np.where(action == 1, FORCE_MAG, -FORCE_MAG)

    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    temp = (force + POLEMASS_LENGTH * theta_dot**2 * sintheta) / TOTAL_MASS
    thetaacc = (GRAVITY * sintheta - costheta * temp) / (
        LENGTH * (4.0 / 3.0 - MASSPOLE * costheta**2 / TOTAL_MASS)
    )
    xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS

    x = x + TAU * x_dot
    x_dot = x_dot + TAU * xacc
    theta = theta + TAU * theta_dot
    theta_dot = theta_dot + TAU * thetaacc
    return np.stack([x, x_dot, theta, theta_dot], axis=-1)


class CartPoleEnv:
    """Single-environment CartPole (no auto-reset) used for evaluation."""

    def __init__(self, seed: int | None = None, max_steps: int = MAX_STEPS):
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.state = np.zeros(4, dtype=np.float64)
        self.steps = 0
        self.reset()

    def reset(self) -> np.ndarray:
        self.state = self.rng.uniform(-0.05, 0.05, size=4)
        self.steps = 0
        return self.state.copy()

    def step(self, action: int):
        self.state = _dynamics(self.state, np.asarray(action))
        self.steps += 1
        x, _, theta, _ = self.state
        terminated = bool(abs(x) > X_LIMIT or abs(theta) > THETA_LIMIT)
        truncated = (not terminated) and self.steps >= self.max_steps
        return self.state.copy(), 1.0, terminated, truncated, {}


class CartPoleVec:
    """Vectorised CartPole environments with automatic reset.

    ``step`` returns the post-step observation, the observation to bootstrap
    from (identical except on truncation, where it is the final observation),
    rewards, terminated/truncated masks and finished-episode returns. The
    separation is what lets PPO bootstrap correctly on the time limit without
    bootstrapping through a terminal (fallen pole) state.
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
        self.state = np.zeros((n_envs, 4), dtype=np.float64)
        self.steps = np.zeros(n_envs, dtype=np.int64)
        self.ep_return = np.zeros(n_envs, dtype=np.float64)
        self.reset()

    def reset(self) -> np.ndarray:
        self.state[:] = self.rng.uniform(-0.05, 0.05, size=(self.n_envs, 4))
        self.steps[:] = 0
        self.ep_return[:] = 0.0
        return self.state.copy()

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.int64)
        self.state = _dynamics(self.state, actions)
        self.steps += 1
        self.ep_return += 1.0

        terminated = (np.abs(self.state[:, 0]) > X_LIMIT) | (
            np.abs(self.state[:, 2]) > THETA_LIMIT
        )
        truncated = (self.steps >= self.max_steps) & ~terminated
        done = terminated | truncated

        rewards = np.ones(self.n_envs, dtype=np.float32)
        boot_obs = self.state.copy()
        finished_returns = self.ep_return[done].tolist()

        if done.any():
            self.state[done] = self.rng.uniform(-0.05, 0.05, size=(done.sum(), 4))
            self.steps[done] = 0
            self.ep_return[done] = 0.0

        return (
            self.state.copy(),
            boot_obs,
            rewards,
            terminated,
            truncated,
            {"episode_returns": finished_returns},
        )


class CartPoleViewer:
    """Minimal Tkinter renderer for a single CartPole episode."""

    WIDTH = 760
    HEIGHT = 480
    TRACK_Y = 400
    PX_PER_M = 125.0
    POLE_PX = 170.0
    CART_W = 96.0
    CART_H = 40.0

    BG = "#0f1720"
    TRACK = "#334155"
    CART = "#38bdf8"
    WHEEL = "#0ea5e9"
    POLE = "#fbbf24"
    TIP = "#f59e0b"
    TEXT = "#e2e8f0"
    MUTED = "#94a3b8"

    def __init__(self, title: str = "CartPole - PPO"):
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
        c.create_line(24, self.TRACK_Y, self.WIDTH - 24, self.TRACK_Y,
                      fill=self.TRACK, width=3)
        # Track tick marks every half metre.
        for x_m in np.arange(-2.4, 2.41, 0.6):
            px = self.WIDTH / 2 + x_m * self.PX_PER_M
            c.create_line(px, self.TRACK_Y, px, self.TRACK_Y + 8,
                          fill=self.TRACK, width=2)

        pivot_y = self.TRACK_Y - self.CART_H - 4
        self.pole = c.create_line(
            self.WIDTH / 2, pivot_y, self.WIDTH / 2, pivot_y - self.POLE_PX,
            fill=self.POLE, width=9, capstyle=tk.ROUND,
        )
        self.tip = c.create_oval(0, 0, 0, 0, fill=self.TIP, outline="")
        self.cart = c.create_rectangle(0, 0, 0, 0, fill=self.CART, outline="")
        self.wheel_l = c.create_oval(0, 0, 0, 0, fill=self.WHEEL, outline="")
        self.wheel_r = c.create_oval(0, 0, 0, 0, fill=self.WHEEL, outline="")

        self.hud = c.create_text(
            24, 20, anchor="nw", text="", fill=self.TEXT,
            font=("Courier", 15, "bold"),
        )
        self.hint = c.create_text(
            self.WIDTH - 24, 20, anchor="ne", text="", fill=self.MUTED,
            font=("Courier", 11),
        )
        self._last_draw: tuple | None = None

    def draw(
        self,
        state: np.ndarray,
        episode: int,
        steps: int,
        episode_return: float,
        status: str,
        subtitle: str = "",
    ) -> None:
        """Redraw the scene. ``state`` is [x, x_dot, theta, theta_dot]."""
        x, _, theta, _ = state
        cx = self.WIDTH / 2 + float(x) * self.PX_PER_M
        pivot_y = self.TRACK_Y - self.CART_H - 4

        left = cx - self.CART_W / 2
        right = cx + self.CART_W / 2
        top = self.TRACK_Y - self.CART_H
        self.canvas.coords(self.cart, left, top, right, self.TRACK_Y)
        r = 7
        self.canvas.coords(self.wheel_l, left + 20 - r, self.TRACK_Y - r,
                           left + 20 + r, self.TRACK_Y + r)
        self.canvas.coords(self.wheel_r, right - 20 - r, self.TRACK_Y - r,
                           right - 20 + r, self.TRACK_Y + r)

        tip_x = cx + np.sin(theta) * self.POLE_PX
        tip_y = pivot_y - np.cos(theta) * self.POLE_PX
        self.canvas.coords(self.pole, cx, pivot_y, tip_x, tip_y)
        tip_r = 9
        self.canvas.coords(self.tip, tip_x - tip_r, tip_y - tip_r,
                           tip_x + tip_r, tip_y + tip_r)

        text = (
            f"episode {episode}    step {steps:3d} / 500\n"
            f"return  {episode_return:5.0f}     angle {np.degrees(theta):+6.1f} deg\n"
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
