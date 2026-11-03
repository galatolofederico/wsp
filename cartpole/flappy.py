"""One-dimensional Flappy gate-navigation environment and a Tkinter viewer.

The state is ``[y, vy, gap_y, gap_dx]``: bird height, vertical velocity, the
centre of the next gate gap, and the distance to that gate. Action 0 glides
and action 1 flaps. The layout intentionally mirrors CartPole: reward +1 per
surviving step, termination on a crash, truncation at 500 steps, and four
scalar observations with two discrete actions. That makes the task a drop-in
fit for the same actor-critic network and PPO trainer.

Only numpy and tkinter (stdlib) are required.
"""

from __future__ import annotations

import numpy as np
import tkinter as tk

# --- Physics constants ------------------------------------------------------
DT = 0.02
GRAVITY = 2.0
FLAP_VELOCITY = 0.8
GATE_SPEED = 0.8
GATE_DX = 1.0  # distance at which a new gate appears
GAP_HALF = 0.13
GAP_MIN = 0.2
GAP_MAX = 0.8
BIRD_RADIUS = 0.02
VY_SCALE = 2.0  # observation velocity is divided by this
MAX_STEPS = 500

STATE_DIM = 4


def _initial_state(rng: np.random.Generator, n: int = 1) -> np.ndarray:
    state = np.zeros((n, STATE_DIM), dtype=np.float64)
    state[:, 0] = 0.5
    state[:, 1] = 0.0
    state[:, 2] = rng.uniform(GAP_MIN, GAP_MAX, size=n)
    state[:, 3] = GATE_DX
    return state


def _obs(state: np.ndarray) -> np.ndarray:
    """Normalise the raw state for the network (velocity is scaled only)."""
    obs = np.array(state, dtype=np.float64, copy=True)
    obs[..., 1] /= VY_SCALE
    return obs


class FlappyEnv:
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
        y, vy, gap_y, gap_dx = self.state
        vy -= GRAVITY * DT
        if int(action) == 1:
            vy = FLAP_VELOCITY
        y += vy * DT
        gap_dx -= GATE_SPEED * DT

        terminated = bool(y < BIRD_RADIUS or y > 1.0 - BIRD_RADIUS)
        if gap_dx <= 0.0:
            if abs(y - gap_y) > GAP_HALF - BIRD_RADIUS:
                terminated = True
            else:
                gap_dx += GATE_DX
                gap_y = float(self.rng.uniform(GAP_MIN, GAP_MAX))

        self.state = np.array([y, vy, gap_y, gap_dx], dtype=np.float64)
        self.steps += 1
        truncated = (not terminated) and self.steps >= self.max_steps
        return _obs(self.state), 1.0, terminated, truncated, {}


class FlappyVec:
    """Vectorised Flappy environments with automatic reset.

    The API matches ``CartPoleVec``: ``step`` returns the post-step
    observation, the observation to bootstrap from (identical except on
    truncation, where it is the final observation), rewards,
    terminated/truncated masks and finished-episode returns.
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
        y = self.state[:, 0].copy()
        vy = self.state[:, 1].copy()
        gap_y = self.state[:, 2].copy()
        gap_dx = self.state[:, 3].copy()

        vy -= GRAVITY * DT
        vy = np.where(actions == 1, FLAP_VELOCITY, vy)
        y += vy * DT
        gap_dx -= GATE_SPEED * DT

        terminated = (y < BIRD_RADIUS) | (y > 1.0 - BIRD_RADIUS)
        passed = gap_dx <= 0.0
        missed = passed & (np.abs(y - gap_y) > GAP_HALF - BIRD_RADIUS)
        terminated |= missed
        cleared = passed & ~missed
        gap_dx[cleared] += GATE_DX
        gap_y[cleared] = self.rng.uniform(
            GAP_MIN, GAP_MAX, size=int(cleared.sum())
        )

        self.state = np.stack([y, vy, gap_y, gap_dx], axis=-1)
        self.steps += 1
        self.ep_return += 1.0

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


class FlappyViewer:
    """Minimal Tkinter renderer for a single Flappy episode."""

    WIDTH = 760
    HEIGHT = 480
    TOP = 48
    GROUND = 432
    BIRD_X = 190.0
    BIRD_R = 13
    GATE_W = 54.0
    GATE_MARGIN = 40.0

    BG = "#0f1720"
    SKY = "#15212e"
    GROUND_COLOR = "#334155"
    GATE = "#22c55e"
    GATE_EDGE = "#16a34a"
    BIRD = "#fbbf24"
    BIRD_EDGE = "#f59e0b"
    EYE = "#0f1720"
    TEXT = "#e2e8f0"
    MUTED = "#94a3b8"

    def __init__(self, title: str = "Flappy - PPO"):
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
        c.create_rectangle(
            0, self.TOP - 6, self.WIDTH, self.GROUND,
            fill=self.SKY, outline="",
        )
        c.create_line(
            24, self.GROUND, self.WIDTH - 24, self.GROUND,
            fill=self.GROUND_COLOR, width=4,
        )

        self.gate_top = c.create_rectangle(
            0, 0, 0, 0, fill=self.GATE, outline=self.GATE_EDGE, width=2
        )
        self.gate_bottom = c.create_rectangle(
            0, 0, 0, 0, fill=self.GATE, outline=self.GATE_EDGE, width=2
        )
        self.bird = c.create_oval(
            0, 0, 0, 0, fill=self.BIRD, outline=self.BIRD_EDGE, width=2
        )
        self.eye = c.create_oval(0, 0, 0, 0, fill=self.EYE, outline="")

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
        """Redraw the scene. ``state`` is ``[y, vy, gap_y, gap_dx]``."""
        y, _, gap_y, gap_dx = state
        play_h = self.GROUND - self.TOP

        bird_y = self.TOP + (1.0 - float(y)) * play_h
        r = self.BIRD_R
        self.canvas.coords(
            self.bird,
            self.BIRD_X - r, bird_y - r,
            self.BIRD_X + r, bird_y + r,
        )
        eye_r = 3
        self.canvas.coords(
            self.eye,
            self.BIRD_X + 3 - eye_r, bird_y - 4 - eye_r,
            self.BIRD_X + 3 + eye_r, bird_y - 4 + eye_r,
        )

        gate_x = self.BIRD_X + float(gap_dx) * (
            self.WIDTH - self.BIRD_X - self.GATE_MARGIN
        )
        gap_center = self.TOP + (1.0 - float(gap_y)) * play_h
        gap_half = GAP_HALF * play_h
        left = gate_x - self.GATE_W / 2
        right = gate_x + self.GATE_W / 2
        self.canvas.coords(
            self.gate_top,
            left, self.TOP, right, gap_center - gap_half,
        )
        self.canvas.coords(
            self.gate_bottom,
            left, gap_center + gap_half, right, self.GROUND,
        )

        text = (
            f"episode {episode}\n"
            f"step    {steps:3d} / 500\n"
            f"return  {episode_return:5.0f}\n"
            f"gap     {gap_dx:4.2f}\n"
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
