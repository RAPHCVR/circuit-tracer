from __future__ import annotations

from typing import Any

import torch

from engine import format_chat_prompt


def sampling_conflict_proxy(
    model: Any,
    user_prompt: str,
    *,
    temperature: float = 0.7,
    top_p: float | None = 0.9,
    seed: int | None = None,
) -> dict[str, Any]:
    """A lightweight *sampling* diagnostic (not true mechanistic introspection).

    Returns whether the sampled token differs from the argmax token under the specified
    sampling distribution, plus basic probabilities.

    This can be useful to explain demo variability, but should not be presented as
    "introspection" in the Lindsey (2025) sense.
    """
    prompt = format_chat_prompt(model, user_prompt)
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    with torch.inference_mode():
        tokens = model.to_tokens(prompt)
        logits = model(tokens)
        last_logits = logits[0, -1]

        temp = max(float(temperature), 1e-6)
        logits_f = last_logits.float()
        scaled = (logits_f - logits_f.max()) / temp

        if top_p is not None and 0 < float(top_p) < 1:
            probs = torch.softmax(scaled, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            keep = cum <= float(top_p)
            keep[0] = True
            filtered = torch.zeros_like(probs)
            filtered[sorted_idx[keep]] = probs[sorted_idx[keep]]
            probs = filtered / filtered.sum().clamp(min=1e-20)
        else:
            probs = torch.softmax(scaled, dim=-1)

        top_id = int(torch.argmax(probs).item())
        sampled_id = int(torch.multinomial(probs, num_samples=1).item())

        top_prob = float(probs[top_id].item())
        sampled_prob = float(probs[sampled_id].item())

        coherence = 1.0 if top_prob <= 0 else float(sampled_prob / top_prob)

        return {
            "top_token_id": top_id,
            "sampled_token_id": sampled_id,
            "top_token": model.tokenizer.decode([top_id]),
            "sampled_token": model.tokenizer.decode([sampled_id]),
            "top_prob": top_prob,
            "sampled_prob": sampled_prob,
            "coherence": coherence,
            "conflict": sampled_id != top_id,
        }
