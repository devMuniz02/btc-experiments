from __future__ import annotations

import torch
from torch import nn

from src.btc_direction_learning.models import MambaBlock, SequenceBackbone


class MLPContinuationActorCriticPolicy(nn.Module):
    def __init__(self, sequence_length: int, feature_dim: int, action_dim: int = 2, hidden_dim: int = 256) -> None:
        super().__init__()
        self.backbone = SequenceBackbone(sequence_length=sequence_length, feature_dim=feature_dim, hidden_dim=hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(observations)
        logits = self.policy_head(hidden)
        values = self.value_head(hidden).squeeze(-1)
        return logits, values


class LSTMContinuationActorCriticPolicy(nn.Module):
    def __init__(self, sequence_length: int, feature_dim: int, action_dim: int = 2, hidden_dim: int = 128) -> None:
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.input_proj(observations)
        outputs, _ = self.lstm(projected)
        hidden = self.norm(outputs[:, -1, :])
        logits = self.policy_head(hidden)
        values = self.value_head(hidden).squeeze(-1)
        return logits, values


class TransformerContinuationActorCriticPolicy(nn.Module):
    def __init__(self, sequence_length: int, feature_dim: int, action_dim: int = 2, hidden_dim: int = 128) -> None:
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, sequence_length, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.norm = nn.LayerNorm(hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.input_proj(observations) + self.position_embedding[:, : observations.shape[1], :]
        encoded = self.encoder(hidden)
        pooled = self.norm(encoded.mean(dim=1))
        logits = self.policy_head(pooled)
        values = self.value_head(pooled).squeeze(-1)
        return logits, values


class MambaContinuationActorCriticPolicy(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        feature_dim: int,
        action_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.action_dim = int(action_dim)
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([MambaBlock(hidden_dim, dropout=dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def extract_features(self, observations: torch.Tensor) -> torch.Tensor:
        hidden = self.input_dropout(self.input_proj(observations))
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden[:, -1, :])

    def forward_from_features(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        return logits, values

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_from_features(self.extract_features(observations))

    def freeze_backbone(self) -> None:
        for module in (self.input_proj, self.input_dropout, self.layers, self.norm):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def unfreeze_all(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(True)


class MambaBanditPolicy(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        feature_dim: int,
        action_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.0,
        bandit_strategy: str = "ts",
        ucb_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.action_dim = int(action_dim)
        self.bandit_strategy = str(bandit_strategy).strip().lower() or "ts"
        self.ucb_alpha = float(ucb_alpha)
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([MambaBlock(hidden_dim, dropout=dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, action_dim)
        self.register_buffer("posterior_weights", torch.zeros(hidden_dim, action_dim))
        self.register_buffer("posterior_bias", torch.zeros(action_dim))
        self.register_buffer("sampled_weights", torch.zeros(hidden_dim, action_dim))
        self.register_buffer("covariance", torch.eye(hidden_dim).repeat(action_dim, 1, 1))

    def extract_features(self, observations: torch.Tensor) -> torch.Tensor:
        hidden = self.input_dropout(self.input_proj(observations))
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden[:, -1, :])

    def set_posterior(
        self,
        *,
        weights: torch.Tensor,
        bias: torch.Tensor,
        covariance: torch.Tensor,
        sampled_weights: torch.Tensor | None = None,
    ) -> None:
        self.posterior_weights.copy_(weights)
        self.posterior_bias.copy_(bias)
        self.covariance.copy_(covariance)
        if sampled_weights is None:
            self.sampled_weights.copy_(weights)
        else:
            self.sampled_weights.copy_(sampled_weights)
        self.head.weight.data.copy_(self.sampled_weights.T)
        self.head.bias.data.copy_(self.posterior_bias)

    def freeze_backbone(self) -> None:
        for module in (self.input_proj, self.input_dropout, self.layers, self.norm):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(observations)
        if self.bandit_strategy == "ucb":
            mean_logits = features @ self.posterior_weights + self.posterior_bias
            bonuses = []
            for action_index in range(self.action_dim):
                covariance = self.covariance[action_index]
                variance = torch.sum((features @ covariance) * features, dim=-1).clamp_min(1e-8)
                bonuses.append(torch.sqrt(variance) * self.ucb_alpha)
            stacked_bonus = torch.stack(bonuses, dim=-1)
            return mean_logits + stacked_bonus
        return features @ self.sampled_weights + self.posterior_bias
