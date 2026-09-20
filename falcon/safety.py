"""Structured Safety Adapter (SSA) with binary presence supervision."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .capabilities import ALL_SAFETY_HEADS, SafetyCapabilities

# These indices are the canonical Falcon-X annotation order.  The paper's
# "main charge" is represented by the dataset label ``explosive``.
COMPONENT_TYPES = ("detonator", "explosive", "battery")
LINK_PAIRS = ((2, 0), (2, 1), (0, 1))
SAFETY_TOKEN_TYPES = (
    "scene_risk",
    "presence_detonator",
    "presence_explosive",
    "presence_battery",
    "link_battery_detonator",
    "link_battery_explosive",
    "link_detonator_explosive",
)


@dataclass
class SafetyOutput:
    slots: Tensor
    attention: Tensor
    presence_logits: Tensor
    presence_probabilities: Tensor
    link_logits: Tensor
    link_probabilities: Tensor
    risk_logit: Tensor
    risk_probability: Tensor
    tokens: Tensor
    capabilities: SafetyCapabilities = SafetyCapabilities()

    def prediction(self, batch_index: int = 0) -> dict:
        """Serialize unavailable heads as null, never plausible-looking numbers."""
        return {
            "risk": float(self.risk_probability[batch_index].detach().cpu())
            if self.capabilities.risk
            else None,
            "presence": [
                float(value.detach().cpu()) if enabled else None
                for value, enabled in zip(
                    self.presence_probabilities[batch_index],
                    self.capabilities.presence,
                    strict=True,
                )
            ],
            "links": [
                float(value.detach().cpu()) if enabled else None
                for value, enabled in zip(
                    self.link_probabilities[batch_index], self.capabilities.links, strict=True
                )
            ],
            "capabilities": self.capabilities.as_dict(),
        }

    @property
    def scalar_values(self) -> Tensor:
        """Values in the seven-token paper order: risk, presence, links."""

        return torch.cat(
            (
                self.risk_probability.unsqueeze(-1),
                self.presence_probabilities,
                self.link_probabilities,
            ),
            dim=-1,
        )


class StructuredSafetyAdapter(nn.Module):
    """Map a variable proposal set to three typed component slots and seven tokens."""

    def __init__(
        self,
        model_dim: int,
        token_dim: int,
        *,
        head_hidden_dim: int | None = None,
        capabilities: SafetyCapabilities = ALL_SAFETY_HEADS,
    ) -> None:
        super().__init__()
        hidden_dim = head_hidden_dim or model_dim
        self.model_dim = model_dim
        self.token_dim = token_dim
        self.capabilities = capabilities
        self.component_queries = nn.Parameter(torch.empty(3, model_dim))
        nn.init.normal_(self.component_queries, std=0.02)

        self.presence_head = nn.Linear(model_dim, 1)
        self.link_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * model_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in LINK_PAIRS
            ]
        )
        self.risk_head = nn.Sequential(
            nn.Linear(3 * model_dim + len(LINK_PAIRS), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_projection = nn.Linear(1, token_dim)
        self.type_embedding = nn.Embedding(len(SAFETY_TOKEN_TYPES), token_dim)
        self.freeze_unavailable_heads()

    def freeze_unavailable_heads(self) -> None:
        """Reapply after a training-stage change has enabled the shared adapter."""
        modules = [
            (self.presence_head, any(self.capabilities.presence)),
            (self.risk_head, self.capabilities.risk),
        ]
        modules.extend(zip(self.link_heads, self.capabilities.links, strict=True))
        for module, enabled in modules:
            if not enabled:
                for parameter in module.parameters():
                    parameter.requires_grad = False

    def _component_slots(
        self,
        region_embeddings: Tensor,
        valid: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, proposals, channels = region_embeddings.shape
        if channels != self.model_dim:
            raise ValueError(f"expected region dimension {self.model_dim}, received {channels}")
        if valid.shape != (batch, proposals):
            raise ValueError("valid must have shape [batch, proposals]")
        if proposals == 0:
            slots = region_embeddings.new_zeros((batch, 3, channels))
            attention = region_embeddings.new_zeros((batch, 3, 0))
            return slots, attention

        scores = torch.einsum("cd,bnd->bcn", self.component_queries, region_embeddings)
        scores = scores / math.sqrt(float(channels))
        expanded_valid = valid[:, None, :].to(dtype=torch.bool)
        scores = scores.masked_fill(~expanded_valid, torch.finfo(scores.dtype).min)
        attention = scores.softmax(dim=-1) * expanded_valid.to(dtype=scores.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        slots = torch.einsum("bcn,bnd->bcd", attention, region_embeddings)
        return slots, attention

    def forward(
        self,
        region_embeddings: Tensor,
        valid: Tensor | None = None,
    ) -> SafetyOutput:
        if region_embeddings.ndim != 3:
            raise ValueError("region_embeddings must have shape [batch, proposals, channels]")
        if valid is None:
            valid = torch.ones(
                region_embeddings.shape[:2],
                dtype=torch.bool,
                device=region_embeddings.device,
            )
        else:
            valid = valid.to(device=region_embeddings.device, dtype=torch.bool)

        slots, attention = self._component_slots(region_embeddings, valid)
        presence_logits = (
            self.presence_head(slots).squeeze(-1)
            if any(self.capabilities.presence)
            else slots.new_zeros(slots.shape[:2])
        )
        link_logits = torch.stack(
            [
                head(torch.cat((slots[:, left], slots[:, right]), dim=-1)).squeeze(-1)
                if enabled
                else slots.new_zeros(slots.shape[0])
                for head, (left, right), enabled in zip(
                    self.link_heads, LINK_PAIRS, self.capabilities.links, strict=True
                )
            ],
            dim=-1,
        )
        link_probabilities = link_logits.sigmoid() * slots.new_tensor(self.capabilities.links)

        # Risk is explicitly conditioned on every slot and the three inferred links.
        risk_input = torch.cat((slots.flatten(start_dim=1), link_probabilities), dim=-1)
        risk_logit = (
            self.risk_head(risk_input).squeeze(-1)
            if self.capabilities.risk
            else slots.new_zeros(slots.shape[0])
        )
        risk_probability = risk_logit.sigmoid() * int(self.capabilities.risk)
        presence_probabilities = presence_logits.sigmoid() * slots.new_tensor(
            self.capabilities.presence
        )

        scalar_values = torch.cat(
            (
                risk_probability.unsqueeze(-1),
                presence_probabilities,
                link_probabilities,
            ),
            dim=-1,
        )
        type_ids = torch.tensor(
            self.capabilities.token_indices, device=slots.device, dtype=torch.long
        )
        tokens = self.value_projection(scalar_values[:, type_ids].unsqueeze(-1))
        tokens = tokens + self.type_embedding(type_ids).unsqueeze(0)
        return SafetyOutput(
            slots=slots,
            attention=attention,
            presence_logits=presence_logits,
            presence_probabilities=presence_probabilities,
            link_logits=link_logits,
            link_probabilities=link_probabilities,
            risk_logit=risk_logit,
            risk_probability=risk_probability,
            tokens=tokens,
            capabilities=self.capabilities,
        )

    @staticmethod
    def loss(
        output: SafetyOutput,
        *,
        risk: Tensor,
        presence: Tensor,
        links: Tensor,
    ) -> dict[str, Tensor]:
        """Stage-2/3 objective: presence BCE with logits and risk/link mean L1."""

        risk_target = risk.to(output.risk_probability).reshape_as(output.risk_probability)
        presence_logits = output.presence_logits.float()
        presence_target = presence.to(
            device=presence_logits.device, dtype=torch.float32
        ).reshape_as(presence_logits)
        link_target = links.to(output.link_probabilities).reshape_as(output.link_probabilities)
        if not output.capabilities.risk:
            risk_target = torch.full_like(risk_target, float("nan"))
        presence_target = presence_target.masked_fill(
            ~torch.tensor(output.capabilities.presence, device=presence_target.device), float("nan")
        )
        link_target = link_target.masked_fill(
            ~torch.tensor(output.capabilities.links, device=link_target.device), float("nan")
        )

        def finite_l1(prediction: Tensor, target: Tensor) -> Tensor:
            finite = torch.isfinite(target)
            if finite.any():
                return F.l1_loss(prediction[finite], target[finite])
            # Keep a differentiable zero when a sample has no annotated links.
            return prediction.sum() * 0.0

        risk_loss = finite_l1(output.risk_probability, risk_target)
        observed_presence = torch.isfinite(presence_target)
        if observed_presence.any():
            presence_loss = F.binary_cross_entropy_with_logits(
                presence_logits[observed_presence], presence_target[observed_presence]
            )
        else:
            presence_loss = presence_logits.sum() * 0.0
        link_loss = finite_l1(output.link_probabilities, link_target)
        return {
            "risk": risk_loss,
            "presence": presence_loss,
            "links": link_loss,
        }
