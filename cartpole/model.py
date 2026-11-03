"""Tiny actor-critic network for CartPole, with safetensors save/load."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from torch.distributions import Categorical

DEFAULT_HIDDEN = (256, 192)


def _orthogonal(layer: nn.Linear, std: float, bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    """Shared MLP trunk with separate policy and value heads.

    The default 4 -> 256 -> 192 trunk has ~51k parameters:
      4*256 + 256 = 1,280
      256*192 + 192 = 49,344
      policy 192*2 + 2 = 386
      value  192*1 + 1 = 193
    """

    def __init__(
        self,
        obs_dim: int = 4,
        n_actions: int = 2,
        hidden: tuple[int, int] = DEFAULT_HIDDEN,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden = tuple(hidden)

        h1, h2 = self.hidden
        self.trunk = nn.Sequential(
            _orthogonal(nn.Linear(obs_dim, h1), std=2.0),
            nn.Tanh(),
            _orthogonal(nn.Linear(h1, h2), std=2.0),
            nn.Tanh(),
        )
        self.actor = _orthogonal(nn.Linear(h2, n_actions), std=0.01)
        self.critic = _orthogonal(nn.Linear(h2, 1), std=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(obs)
        return self.actor(features), self.critic(features).squeeze(-1)

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, _ = self.forward(obs)
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action)

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values

    @torch.no_grad()
    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.trunk(obs)).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_model(
    model: ActorCritic, path: str | Path, extra_metadata: dict | None = None
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "cartpole-ppo",
        "obs_dim": str(model.obs_dim),
        "n_actions": str(model.n_actions),
        "hidden": ",".join(str(h) for h in model.hidden),
    }
    if extra_metadata:
        metadata.update({k: str(v) for k, v in extra_metadata.items()})
    save_file(model.state_dict(), str(path), metadata=metadata)


def load_model(path: str | Path) -> ActorCritic:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        state = {key: f.get_tensor(key) for key in f.keys()}

    hidden = tuple(
        int(h) for h in metadata.get("hidden", ",".join(map(str, DEFAULT_HIDDEN))).split(",")
    )
    model = ActorCritic(
        obs_dim=int(metadata.get("obs_dim", 4)),
        n_actions=int(metadata.get("n_actions", 2)),
        hidden=hidden,
    )
    model.load_state_dict(state)
    model.eval()
    return model
