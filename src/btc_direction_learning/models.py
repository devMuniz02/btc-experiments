from __future__ import annotations

import torch
from torch import nn

from src.btc_direction_learning.env import ENV_VERSION_INTENSITY11, ENV_VERSION_TERNARY, NONE_ACTION, intensity11_action_from_probability


class SequenceBackbone(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        feature_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        flat_dim = sequence_length * feature_dim
        self.network = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        flat = observations.reshape(observations.shape[0], -1)
        return self.network(flat)


class LSTMClassificationPolicy(nn.Module):
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
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        projected = self.input_proj(observations)
        outputs, _ = self.lstm(projected)
        hidden = self.norm(outputs[:, -1, :])
        return self.head(hidden)


class LSTMRegressionPolicy(nn.Module):
    def __init__(self, sequence_length: int, feature_dim: int, hidden_dim: int = 128) -> None:
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
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        projected = self.input_proj(observations)
        outputs, _ = self.lstm(projected)
        hidden = self.norm(outputs[:, -1, :])
        return self.head(hidden).squeeze(-1)


class TransformerClassificationPolicy(nn.Module):
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
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(observations) + self.position_embedding[:, : observations.shape[1], :]
        encoded = self.encoder(hidden)
        pooled = self.norm(encoded.mean(dim=1))
        return self.head(pooled)


class TransformerClassificationPolicyV2(nn.Module):
    def __init__(self, sequence_length: int, feature_dim: int, action_dim: int = 2, hidden_dim: int = 256) -> None:
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, sequence_length, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(observations) + self.position_embedding[:, : observations.shape[1], :]
        encoded = self.encoder(hidden)
        # Use the most recent timestep representation instead of mean pooling so
        # directional decisions stay anchored to the latest market context.
        pooled = self.norm(encoded[:, -1, :])
        return self.head(pooled)


class MambaBlock(nn.Module):
    def __init__(self, hidden_dim: int, conv_kernel_size: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.depthwise_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=conv_kernel_size,
            groups=hidden_dim,
            padding=conv_kernel_size - 1,
        )
        self.dt_proj = nn.Linear(hidden_dim, hidden_dim)
        self.b_proj = nn.Linear(hidden_dim, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, hidden_dim)
        self.a_log = nn.Parameter(torch.linspace(-1.0, 0.0, hidden_dim))
        self.d = nn.Parameter(torch.ones(hidden_dim))
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        hidden = self.norm(inputs)
        x_branch, gate = self.in_proj(hidden).chunk(2, dim=-1)
        sequence_length = x_branch.shape[1]
        mixed = self.depthwise_conv(x_branch.transpose(1, 2))[:, :, :sequence_length].transpose(1, 2)
        mixed = torch.nn.functional.silu(mixed)
        mixed = self.dropout(mixed)

        delta = torch.nn.functional.softplus(self.dt_proj(mixed))
        b_term = torch.tanh(self.b_proj(mixed))
        c_term = torch.tanh(self.c_proj(mixed))
        a_term = -torch.exp(self.a_log).view(1, 1, -1)
        decay = torch.exp(a_term * delta).clamp_min(1e-12)
        input_term = delta * b_term * mixed

        # Closed-form parallel scan for the scalar-per-channel recurrence:
        # h_t = decay_t * h_{t-1} + input_t, h_0 = 0
        prefix = torch.cumprod(decay, dim=1)
        state = prefix * torch.cumsum(input_term / prefix.clamp_min(1e-12), dim=1)

        stacked = c_term * state + self.d.view(1, 1, -1) * mixed
        gated = stacked * torch.sigmoid(gate)
        return residual + self.dropout(self.out_proj(gated))


class MambaClassificationPolicy(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        feature_dim: int,
        action_dim: int = 2,
        hidden_dim: int = 192,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([MambaBlock(hidden_dim, dropout=dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        hidden = self.input_dropout(self.input_proj(observations))
        for layer in self.layers:
            hidden = layer(hidden)
        pooled = self.norm(hidden[:, -1, :])
        return self.head(pooled)


class ClassificationPolicy(nn.Module):
    def __init__(self, sequence_length: int, feature_dim: int, action_dim: int = 2, hidden_dim: int = 256) -> None:
        super().__init__()
        self.backbone = SequenceBackbone(sequence_length=sequence_length, feature_dim=feature_dim, hidden_dim=hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(observations)
        return self.policy_head(hidden)


class ActorCriticPolicy(nn.Module):
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


def extract_policy_logits(policy: nn.Module, observations: torch.Tensor) -> torch.Tensor:
    output = policy(observations)
    if isinstance(output, tuple):
        return output[0]
    return output


def select_actions_from_logits(
    logits: torch.Tensor,
    env_version: str,
    ternary_confidence_threshold: float = 0.0,
) -> torch.Tensor:
    if env_version == ENV_VERSION_INTENSITY11 and logits.shape[-1] == 2:
        directional_probs = torch.softmax(logits, dim=-1)
        probability_up = directional_probs[..., 1]
        actions = [
            intensity11_action_from_probability(float(prob))
            for prob in probability_up.detach().cpu().reshape(-1).tolist()
        ]
        return torch.tensor(actions, dtype=torch.long, device=logits.device)

    if env_version == ENV_VERSION_TERNARY and logits.shape[-1] >= 2 and ternary_confidence_threshold > 0.0:
        directional_logits = logits[..., :2]
        directional_probs = torch.softmax(directional_logits, dim=-1)
        best_probs, best_actions = torch.max(directional_probs, dim=-1)
        none_actions = torch.full_like(best_actions, fill_value=NONE_ACTION)
        return torch.where(best_probs >= ternary_confidence_threshold, best_actions, none_actions)

    return torch.argmax(logits, dim=-1)
