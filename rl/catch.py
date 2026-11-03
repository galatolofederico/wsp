"""Falling-ball catch environment and a small Tkinter viewer.

The state is ``[paddle_x, ball_x, ball_y, ball_vx]``: the paddle position, the
ball position, and the ball's horizontal velocity. Action 0 moves the paddle
left and action 1 moves it right. A ball falls at a constant vertical speed;
the paddle must intercept it before it reaches the floor. A successful catch
immediately launches a new ball from the top, while a miss terminates the
episode. Episodes are truncated at 500 steps.

Like CartPole, Flappy and BallBeam, every observation is four scalars, there
are two discrete actions, and survival pays +1 per step. The reset/step
contract, including ``boot_obs`` and ``episode_returns``, matches the other
vectorised environments so the PPO trainer is reused unchanged.

Only numpy and tkinter (stdlib) are required.
"""

from __future__ import annotations

import numpy as np
import tkinter as tk

# --- Physics constants ------------------------------------------------------
DT = 0.02
PADDLE_HALF = 0.12
PADDLE_SPEED = 1.1
BALL_RADIUS = 0.02
BALL_SPEED_Y = 0.55
BALL_VX_MIN = 0.12
BALL_VX_MAX = 0.30
BALL_VX_SCALE = 0.4
PADDLE_TOP = 0.04
CATCH_Y = PADDLE_TOP + BALL_RADIUS
MAX_STEPS = 500

STATE_DIM = 4


def _reset_ball(
    rng: np.random.Generator, n: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Return a new ball at the top with a random horizontal drift."""
    ball_x = rng.uniform(0.12, 0.88, size=n)
    sign = rng.choice((-1.0, 1.0), size=n)
    ball_vx = sign * rng.uniform(BALL_VX_MIN, BALL_VX_MAX, size=n)
    return ball_x, ball_vx


def _initial_state(rng: np.random.Generator, n: int = 1) -> np.ndarray:
    state = np.zeros((n, STATE_DIM), dtype=np.float64)
    state[:, 0] = rng.uniform(
        PADDLE_HALF + 0.05, 1.0 - PADDLE_HALF - 0.05, size=n
    )
    state[:, 1], state[:, 3] = _reset_ball(rng, n)
    state[:, 2] = 1.0
    return state


def _obs(state: np.ndarray) -> np.ndarray:
    """Centre the positions and scale the horizontal velocity."""
    obs = np.array(state, dtype=np.float64, copy=True)
    obs[..., 0] -= 0.5
    obs[..., 1] -= 0.5
    obs[..., 3] /= BALL_VX_SCALE
    return obs


def _move_paddle(paddle_x: np.ndarray, action: np.ndarray) -> np.ndarray:
    direction = np.where(action == 1, 1.0, -1.0)
    return np.clip(
        paddle_x + direction * PADDLE_SPEED * DT,
        PADDLE_HALF,
        1.0 - PADDLE_HALF,
    )


class CatchEnv:
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
        paddle_x, ball_x, ball_y, ball_vx = self.state

        paddle_x = float(_move_paddle(paddle_x, int(action)))
        ball_x = float(ball_x + ball_vx * DT)
        ball_y = float(ball_y - BALL_SPEED_Y * DT)

        if ball_x < BALL_RADIUS:
            ball_x = BALL_RADIUS
            ball_vx = abs(ball_vx)
        elif ball_x > 1.0 - BALL_RADIUS:
            ball_x = 1.0 - BALL_RADIUS
            ball_vx = -abs(ball_vx)

        terminated = False
        if ball_y <= CATCH_Y:
            if abs(ball_x - paddle_x) <= PADDLE_HALF + BALL_RADIUS:
                new_x, new_vx = _reset_ball(self.rng)
                ball_x = float(new_x[0])
                ball_vx = float(new_vx[0])
                ball_y = 1.0
            else:
                terminated = True
                ball_y = CATCH_Y

        self.state = np.array(
            [paddle_x, ball_x, ball_y, ball_vx], dtype=np.float64
        )
        self.steps += 1
        truncated = (not terminated) and self.steps >= self.max_steps
        return _obs(self.state), 1.0, terminated, truncated, {}


class CatchVec:
    """Vectorised catch environments with automatic reset.

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
        paddle_x = self.state[:, 0].copy()
        ball_x = self.state[:, 1].copy()
        ball_y = self.state[:, 2].copy()
        ball_vx = self.state[:, 3].copy()

        paddle_x = _move_paddle(paddle_x, actions)
        ball_x = ball_x + ball_vx * DT
        ball_y = ball_y - BALL_SPEED_Y * DT

        left = ball_x < BALL_RADIUS
        right = ball_x > 1.0 - BALL_RADIUS
        ball_x = np.where(left, BALL_RADIUS, ball_x)
        ball_x = np.where(right, 1.0 - BALL_RADIUS, ball_x)
        ball_vx = np.where(left, np.abs(ball_vx), ball_vx)
        ball_vx = np.where(right, -np.abs(ball_vx), ball_vx)

        at_floor = ball_y <= CATCH_Y
        caught = at_floor & (
            np.abs(ball_x - paddle_x) <= PADDLE_HALF + BALL_RADIUS
        )
        terminated = at_floor & ~caught

        if caught.any():
            n_caught = int(caught.sum())
            ball_x[caught], ball_vx[caught] = _reset_ball(
                self.rng, n_caught
            )
            ball_y[caught] = 1.0
        ball_y[terminated] = CATCH_Y

        self.state = np.stack([paddle_x, ball_x, ball_y, ball_vx], axis=-1)
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


class CatchViewer:
    """Minimal Tkinter renderer for a single catch episode."""

    WIDTH = 760
    HEIGHT = 480
    TOP = 48
    FLOOR = 432
    MARGIN = 24
    PADDLE_H = 14
    BALL_R = 11

    BG = "#0f1720"
    COURT = "#15212e"
    FLOOR_COLOR = "#334155"
    PADDLE = "#38bdf8"
    PADDLE_EDGE = "#0ea5e9"
    BALL = "#fbbf24"
    BALL_EDGE = "#f59e0b"
    TEXT = "#e2e8f0"
    MUTED = "#94a3b8"

    def __init__(self, title: str = "Catch - PPO"):
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
            self.MARGIN, self.TOP, self.WIDTH - self.MARGIN, self.FLOOR,
            fill=self.COURT, outline=self.FLOOR_COLOR, width=2,
        )
        c.create_line(
            self.MARGIN, self.FLOOR, self.WIDTH - self.MARGIN, self.FLOOR,
            fill=self.FLOOR_COLOR, width=4,
        )

        self.paddle = c.create_rectangle(
            0, 0, 0, 0, fill=self.PADDLE, outline=self.PADDLE_EDGE, width=2
        )
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
        """Redraw the scene. ``state`` is ``[paddle_x, ball_x, ball_y, ball_vx]``."""
        paddle_x, ball_x, ball_y, ball_vx = state
        play_w = self.WIDTH - 2 * self.MARGIN
        play_h = self.FLOOR - self.TOP

        px = self.MARGIN + float(paddle_x) * play_w
        paddle_half = PADDLE_HALF * play_w
        paddle_y = self.FLOOR - self.PADDLE_H
        self.canvas.coords(
            self.paddle,
            px - paddle_half, paddle_y,
            px + paddle_half, self.FLOOR,
        )

        bx = self.MARGIN + float(ball_x) * play_w
        by = self.FLOOR - float(ball_y) * play_h
        r = self.BALL_R
        self.canvas.coords(self.ball, bx - r, by - r, bx + r, by + r)

        text = (
            f"episode {episode}\n"
            f"step    {steps:3d} / 500\n"
            f"return  {episode_return:5.0f}\n"
            f"ball    {ball_x:4.2f}, {ball_y:4.2f}\n"
            f"vx      {ball_vx:+5.2f}\n"
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
