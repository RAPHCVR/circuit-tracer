from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from engine import FeatureRef, format_chat_prompt

# circuit-tracer intervention format: (layer, pos, feature_idx, value)
Intervention = tuple[int, int, int, float]


@dataclass(frozen=True)
class InterventionSpec:
    feature: FeatureRef
    value: float

    def as_tuple(self) -> Intervention:
        return (
            int(self.feature.layer),
            int(self.feature.pos),
            int(self.feature.feature_idx),
            float(self.value),
        )


def next_token_logits_with_interventions(
    model: Any,
    user_prompt: str,
    interventions: list[InterventionSpec],
    *,
    freeze_attention: bool = True,
    constrained_layers: range | None = None,
    apply_activation_function: bool = True,
) -> torch.Tensor:
    """Recommended intervention entrypoint for demos.

    This uses the official `ReplacementModel.feature_intervention()` implementation, which
    matches the circuit-tracing tooling more closely than ad-hoc residual-stream hooks.
    """
    prompt = format_chat_prompt(model, user_prompt)
    tokens = model.to_tokens(prompt)
    tuples: list[Intervention] = [i.as_tuple() for i in interventions]
    logits, _acts = model.feature_intervention(  # type: ignore[attr-defined]
        tokens,
        tuples,
        constrained_layers=constrained_layers,
        freeze_attention=freeze_attention,
        apply_activation_function=apply_activation_function,
        return_activations=False,
    )
    return logits[0, -1]
