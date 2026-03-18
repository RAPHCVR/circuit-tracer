from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import streamlit as st
import torch
import warnings
import re

from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.graph import (
    Graph,
    compute_graph_scores,
    compute_node_influence,
    find_threshold,
)


def _coerce_constrained_layers(constrained_layers: range | None, *, n_layers: int) -> range | None:
    if constrained_layers is None:
        return None
    start = int(constrained_layers.start)
    stop = int(constrained_layers.stop)
    step = int(constrained_layers.step)
    if step != 1:
        raise ValueError("constrained_layers must be a contiguous range (step=1).")
    start = max(0, min(n_layers, start))
    stop = max(0, min(n_layers, stop))
    if stop < start:
        start, stop = stop, start
    if start == stop:
        return None
    return range(start, stop)


def _activation_value_lookup(
    activation_cache: torch.Tensor | None,
    feature: FeatureRef,
    *,
    default: float = 0.0,
) -> float:
    if activation_cache is None:
        return float(default)

    layer = int(feature.layer)
    pos = int(feature.pos)
    feature_idx = int(feature.feature_idx)

    try:
        if activation_cache.is_sparse:
            acts = activation_cache.coalesce()
            indices = acts.indices()
            values = acts.values()
            mask = (indices[0] == layer) & (indices[1] == pos) & (indices[2] == feature_idx)
            if not bool(mask.any().item()):
                return float(default)
            # There should only be one match, but sum() is safe.
            return float(values[mask].sum().item())

        # Dense case.
        return float(activation_cache[layer, pos, feature_idx].item())
    except Exception:
        return float(default)


def _sparse_value_map(activation_cache: torch.Tensor) -> dict[tuple[int, int, int], float]:
    """Build a CPU-side lookup map for sparse activations.

    This is much faster than scanning sparse indices repeatedly when we need values for many
    (layer, pos, feature_idx) tuples.
    """
    if not activation_cache.is_sparse:
        raise TypeError("Expected a sparse tensor.")

    acts = activation_cache.coalesce()
    indices = acts.indices().detach().cpu()
    values = acts.values().detach().cpu()

    out: dict[tuple[int, int, int], float] = {}
    for (layer, pos, feat), val in zip(indices.t().tolist(), values.tolist(), strict=False):
        out[(int(layer), int(pos), int(feat))] = float(val)
    return out


def activation_change_report(
    model: Any,
    user_prompt: str,
    interventions: list[tuple[int, int, int, float]],
    tracked_features: list[FeatureRef],
    *,
    freeze_attention: bool = True,
    constrained_layers: range | None = None,
    apply_activation_function: bool = True,
    sparse: bool = True,
    baseline_match_freeze: bool = False,
) -> dict[str, Any]:
    """Compute baseline vs intervened activations for selected features.

    This is meant for demo-quality *causal validation*: after steering feature(s),
    show that other feature activations (often downstream in the attribution graph)
    move in a consistent direction, not just the logits.

    Notes:
    - `feature_intervention(..., freeze_attention=True)` freezes attention patterns and
      layernorms to their baseline values (see circuit-tracer docs).
    - If `baseline_match_freeze=True`, we run a "no-op" intervention to ensure the
      baseline activations were computed under the same frozen-attention codepath.
      This is slower but makes comparisons cleaner.
    """
    prompt = format_chat_prompt(model, user_prompt)
    tokens = model.to_tokens(prompt)

    n_layers = int(getattr(getattr(model, "cfg", None), "n_layers", 0) or 0)
    constrained_layers = (
        _coerce_constrained_layers(constrained_layers, n_layers=n_layers)
        if n_layers
        else constrained_layers
    )

    base_logits, base_acts = model.feature_intervention(  # type: ignore[attr-defined]
        tokens,
        [],
        constrained_layers=constrained_layers,
        freeze_attention=False,  # baseline forward pass
        apply_activation_function=apply_activation_function,
        sparse=sparse,
        return_activations=True,
    )

    base_acts = (
        base_acts.coalesce()
        if isinstance(base_acts, torch.Tensor) and base_acts.is_sparse
        else base_acts
    )
    base_map = (
        _sparse_value_map(base_acts)
        if isinstance(base_acts, torch.Tensor) and base_acts.is_sparse
        else None
    )

    baseline_mode = "normal"
    if baseline_match_freeze and freeze_attention and interventions:
        # Trigger the frozen-attention intervention codepath with a no-op patch for the
        # *intervened* features only.
        noop: list[tuple[int, int, int, float]] = []
        for layer, pos, fidx, _v in interventions:
            feat = FeatureRef(layer=int(layer), pos=int(pos), feature_idx=int(fidx))
            noop.append(
                (feat.layer, feat.pos, feat.feature_idx, _activation_value_lookup(base_acts, feat))
            )

        base_logits, base_acts = model.feature_intervention(  # type: ignore[attr-defined]
            tokens,
            noop,
            constrained_layers=constrained_layers,
            freeze_attention=True,
            apply_activation_function=apply_activation_function,
            sparse=sparse,
            return_activations=True,
        )
        base_acts = (
            base_acts.coalesce()
            if isinstance(base_acts, torch.Tensor) and base_acts.is_sparse
            else base_acts
        )
        base_map = (
            _sparse_value_map(base_acts)
            if isinstance(base_acts, torch.Tensor) and base_acts.is_sparse
            else None
        )
        baseline_mode = "frozen_attention_noop"

    new_logits, new_acts = model.feature_intervention(  # type: ignore[attr-defined]
        tokens,
        interventions,
        constrained_layers=constrained_layers,
        freeze_attention=freeze_attention,
        apply_activation_function=apply_activation_function,
        sparse=sparse,
        return_activations=True,
    )
    new_acts = (
        new_acts.coalesce()
        if isinstance(new_acts, torch.Tensor) and new_acts.is_sparse
        else new_acts
    )
    new_map = (
        _sparse_value_map(new_acts)
        if isinstance(new_acts, torch.Tensor) and new_acts.is_sparse
        else None
    )

    rows: list[dict[str, Any]] = []
    for feat in tracked_features:
        if base_map is not None:
            base = float(base_map.get((feat.layer, feat.pos, feat.feature_idx), 0.0))
        else:
            base = _activation_value_lookup(base_acts, feat)

        if new_map is not None:
            new = float(new_map.get((feat.layer, feat.pos, feat.feature_idx), 0.0))
        else:
            new = _activation_value_lookup(new_acts, feat)
        delta = new - base
        pct = None
        if abs(base) > 1e-9:
            pct = 100.0 * (delta / base)
        rows.append(
            {
                "feature": feat,
                "base": float(base),
                "new": float(new),
                "delta": float(delta),
                "pct_delta": float(pct) if pct is not None else None,
            }
        )

    rows.sort(key=lambda r: abs(float(r["delta"])), reverse=True)

    return {
        "baseline_mode": baseline_mode,
        "freeze_attention": bool(freeze_attention),
        "constrained_layers": None
        if constrained_layers is None
        else {"start": int(constrained_layers.start), "stop": int(constrained_layers.stop)},
        "apply_activation_function": bool(apply_activation_function),
        "sparse": bool(sparse),
        "n_tracked": int(len(tracked_features)),
        "rows": rows,
    }


@dataclass(frozen=True)
class FeatureRef:
    """Reference to a single transcoder feature instance."""

    layer: int
    pos: int
    feature_idx: int

    @classmethod
    def from_any(cls, obj: Any) -> "FeatureRef":
        if isinstance(obj, FeatureRef):
            return obj
        if isinstance(obj, (tuple, list)) and len(obj) == 3:
            return cls(layer=int(obj[0]), pos=int(obj[1]), feature_idx=int(obj[2]))
        if isinstance(obj, dict):
            return cls(
                layer=int(obj["layer"]), pos=int(obj["pos"]), feature_idx=int(obj["feature_idx"])
            )
        raise TypeError(f"Unsupported feature reference: {type(obj)!r}")


def _default_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        # Prefer bf16 on Ampere+; fall back to fp16 on older GPUs (e.g. Turing / RTX 20xx).
        try:
            major, _minor = torch.cuda.get_device_capability()
            if major >= 8:
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32


def _default_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def format_chat_prompt(model: Any, user_prompt: str) -> str:
    """Format an instruction-tuned prompt (Gemma-IT) when possible."""
    tok = getattr(model, "tokenizer", None)
    if tok is None:
        return user_prompt
    apply_chat_template = getattr(tok, "apply_chat_template", None)
    if apply_chat_template is None:
        return user_prompt
    try:
        return apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return user_prompt


@st.cache_resource(show_spinner="Loading Gemma + transcoders…")
def load_model(
    model_name: str = "google/gemma-2-2b-it",
    transcoder_set: str = "gemma",
    *,
    backend: Literal["transformerlens", "nnsight"] = "transformerlens",
    dtype: torch.dtype | None = None,
) -> Any:
    """Load a circuit-tracer ReplacementModel once per Streamlit session.

    Ref: Biology / Attribution Graphs papers — local replacement model + feature tracing.
    """
    device = _default_device()
    dtype = _default_dtype() if dtype is None else dtype

    try:
        model = ReplacementModel.from_pretrained(
            model_name=model_name,
            transcoder_set=transcoder_set,
            backend=backend,
            device=device,
            dtype=dtype,
        )
    except Exception:
        # Fallback: TL backend doesn't support every HF model (README notes this).
        if backend != "nnsight":
            model = ReplacementModel.from_pretrained(
                model_name=model_name,
                transcoder_set=transcoder_set,
                backend="nnsight",
                device=device,
                dtype=dtype,
            )
        else:
            raise

    model.eval()
    return model


def generate_text(
    model: Any,
    user_prompt: str,
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float | None = 0.9,
    seed: int | None = None,
) -> str:
    """Generate a continuation from the model (no circuit tracing)."""
    prompt = format_chat_prompt(model, user_prompt)
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    do_sample = temperature > 0
    try:
        with torch.inference_mode():
            input_tokens = model.to_tokens(prompt)
            out_tokens = model.generate(
                input_tokens,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 1e-6),
                top_p=top_p,
                do_sample=do_sample,
                verbose=False,
                return_type="tokens",
            )
            new_tokens = out_tokens[0, input_tokens.shape[1] :]
            return model.tokenizer.decode(new_tokens)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise MemoryError("GPU OOM during generation. Try fewer tokens / smaller batch.") from e
        raise


def _decode_token(tokenizer: Any, token_id: int) -> str:
    # Keep it readable in tables; preserve newlines explicitly.
    try:
        s = tokenizer.decode([int(token_id)])
    except Exception:
        s = str(token_id)
    return s.replace("\n", "\\n")


def _surface_key(token: str) -> str:
    # Group tokens that only differ by leading space / casing / newline-escaping.
    # This helps demos where "Austin", " Austin", "austin" are separate tokens.
    return str(token).replace("\\n", "\n").strip().lower()


def parse_manual_interventions(
    text: str,
    *,
    base_lookup: dict[tuple[int, int, int], float] | None = None,
) -> tuple[list[tuple[int, int, int, float]], list[str], list[str]]:
    """Parse manual interventions from free-form text.

    Supported line formats (whitespace / separators flexible):
    - ``L24 P28 #6044 = 0``
    - ``24:28:6044=0``
    - ``24 28 6044 0``

    Optional relative updates (requires `base_lookup` for that feature):
    - ``L24 P28 #6044 += 1.5``
    - ``24:28:6044 -= 2``

    Returns: (interventions, warnings, errors)
    """
    interventions: list[tuple[int, int, int, float]] = []
    warnings_out: list[str] = []
    errors: list[str] = []

    line_re = re.compile(
        r"^\s*(?:L\s*)?(?P<layer>\d+)\s*[,:\s]\s*(?:P\s*)?(?P<pos>\d+)\s*[,:\s]\s*(?:#\s*)?(?P<feat>\d+)\s*(?P<op>\+=|-=|=)?\s*(?P<val>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)?\s*$"
    )

    for i, raw in enumerate(str(text).splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("//"):
            continue

        # Allow "a b c d" without an explicit "=" operator.
        if "=" not in line and "+=" not in line and "-=" not in line:
            parts = re.split(r"\s+", line)
            if len(parts) == 4:
                layer_s, pos_s, feat_s, val_s = parts
                try:
                    interventions.append((int(layer_s), int(pos_s), int(feat_s), float(val_s)))
                    continue
                except Exception:
                    errors.append(f"Line {i}: could not parse '{raw}'")
                    continue

        m = line_re.match(line)
        if not m:
            errors.append(f"Line {i}: could not parse '{raw}'")
            continue

        layer = int(m.group("layer"))
        pos = int(m.group("pos"))
        feat = int(m.group("feat"))
        op = m.group("op") or "="
        val_s = m.group("val")
        if val_s is None:
            errors.append(f"Line {i}: missing value in '{raw}'")
            continue

        try:
            val = float(val_s)
        except Exception:
            errors.append(f"Line {i}: invalid float '{val_s}' in '{raw}'")
            continue

        if op == "=":
            interventions.append((layer, pos, feat, float(val)))
            continue

        if base_lookup is None or (layer, pos, feat) not in base_lookup:
            warnings_out.append(
                f"Line {i}: '{op}' needs baseline for ({layer},{pos},{feat}); skipping (not in current feature list)."
            )
            continue

        base = float(base_lookup[(layer, pos, feat)])
        if op == "+=":
            interventions.append((layer, pos, feat, float(base + val)))
        elif op == "-=":
            interventions.append((layer, pos, feat, float(base - val)))
        else:
            errors.append(f"Line {i}: unsupported operator '{op}' in '{raw}'")

    return interventions, warnings_out, errors


def summarize_tokens(model: Any, tokens: torch.Tensor) -> list[dict[str, Any]]:
    tokenizer = getattr(model, "tokenizer", None)
    out: list[dict[str, Any]] = []
    for pos, tok_id in enumerate(tokens.detach().cpu().tolist()):
        out.append(
            {
                "pos": int(pos),
                "token_id": int(tok_id),
                "token": _decode_token(tokenizer, int(tok_id))
                if tokenizer is not None
                else str(tok_id),
            }
        )
    return out


def summarize_logits(
    model: Any, logit_tokens: torch.Tensor, logit_probabilities: torch.Tensor
) -> list[dict[str, Any]]:
    tokenizer = getattr(model, "tokenizer", None)
    out: list[dict[str, Any]] = []
    ids = logit_tokens.detach().cpu().tolist()
    ps = logit_probabilities.detach().cpu().tolist()
    for i, (tok_id, p) in enumerate(zip(ids, ps, strict=False)):
        out.append(
            {
                "logit_index": int(i),
                "token_id": int(tok_id),
                "token": _decode_token(tokenizer, int(tok_id))
                if tokenizer is not None
                else str(tok_id),
                "prob": float(p),
            }
        )
    return out


def next_token_distribution(
    model: Any,
    user_prompt: str,
    *,
    temperature: float = 1.0,
    top_p: float | None = None,
    top_k: int = 20,
) -> dict[str, Any]:
    """Compute next-token distribution from the model (no tracing).

    This is a *diagnostic* helper for demos (entropy/margin + top-k tokens).
    """
    prompt = format_chat_prompt(model, user_prompt)
    with torch.inference_mode():
        tokens = model.to_tokens(prompt)
        logits = model(tokens)
        last_logits = logits[0, -1]

        temp = max(float(temperature), 1e-6)
        logits_f = last_logits.float()
        scaled = (logits_f - logits_f.max()) / temp
        probs = torch.softmax(scaled, dim=-1)

        if top_p is not None and 0 < float(top_p) < 1:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            keep = cum <= float(top_p)
            keep[0] = True
            filtered = torch.zeros_like(probs)
            filtered[sorted_idx[keep]] = probs[sorted_idx[keep]]
            probs = filtered / filtered.sum().clamp(min=1e-20)

        entropy = float(-(probs * torch.log(probs.clamp(min=1e-20))).sum().item())
        top2 = torch.topk(probs, k=2)
        margin = float((top2.values[0] - top2.values[1]).item())

        top = torch.topk(probs, k=min(int(top_k), probs.numel()))
        tokenizer = getattr(model, "tokenizer", None)
        rows = []
        for tok_id, p in zip(
            top.indices.detach().cpu().tolist(), top.values.detach().cpu().tolist(), strict=False
        ):
            rows.append(
                {
                    "token_id": int(tok_id),
                    "token": _decode_token(tokenizer, int(tok_id))
                    if tokenizer is not None
                    else str(tok_id),
                    "prob": float(p),
                }
            )

    groups: dict[str, float] = {}
    for r in rows:
        k = _surface_key(r["token"])
        groups[k] = float(groups.get(k, 0.0) + float(r["prob"]))
    top_groups = sorted(
        [{"surface": k, "prob": v} for k, v in groups.items()],
        key=lambda x: x["prob"],
        reverse=True,
    )

    return {
        "entropy": entropy,
        "margin_top1_top2": margin,
        "top": rows,
        "top_groups": top_groups[:10],
    }


def distribution_from_logits(
    model: Any,
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_p: float | None = None,
    top_k: int = 20,
) -> dict[str, Any]:
    """Same as `next_token_distribution`, but uses precomputed logits (e.g. post-intervention)."""
    last_logits = logits
    temp = max(float(temperature), 1e-6)
    logits_f = last_logits.float()
    scaled = (logits_f - logits_f.max()) / temp
    probs = torch.softmax(scaled, dim=-1)

    if top_p is not None and 0 < float(top_p) < 1:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        keep = cum <= float(top_p)
        keep[0] = True
        filtered = torch.zeros_like(probs)
        filtered[sorted_idx[keep]] = probs[sorted_idx[keep]]
        probs = filtered / filtered.sum().clamp(min=1e-20)

    entropy = float(-(probs * torch.log(probs.clamp(min=1e-20))).sum().item())
    top2 = torch.topk(probs, k=2)
    margin = float((top2.values[0] - top2.values[1]).item())

    top = torch.topk(probs, k=min(int(top_k), probs.numel()))
    tokenizer = getattr(model, "tokenizer", None)
    rows = []
    for tok_id, p in zip(
        top.indices.detach().cpu().tolist(), top.values.detach().cpu().tolist(), strict=False
    ):
        rows.append(
            {
                "token_id": int(tok_id),
                "token": _decode_token(tokenizer, int(tok_id))
                if tokenizer is not None
                else str(tok_id),
                "prob": float(p),
            }
        )

    groups: dict[str, float] = {}
    for r in rows:
        k = _surface_key(r["token"])
        groups[k] = float(groups.get(k, 0.0) + float(r["prob"]))
    top_groups = sorted(
        [{"surface": k, "prob": v} for k, v in groups.items()],
        key=lambda x: x["prob"],
        reverse=True,
    )

    return {
        "entropy": entropy,
        "margin_top1_top2": margin,
        "top": rows,
        "top_groups": top_groups[:10],
    }


def direct_logit_effects_for_feature(
    model: Any,
    user_prompt: str,
    feature: FeatureRef,
    *,
    base_value: float,
    delta: float,
    freeze_attention: bool = True,
    constrained_layers: range | None = None,
    apply_activation_function: bool = True,
    top_k: int = 15,
) -> dict[str, Any]:
    """Empirical "direct logit effects" for a feature via intervention.

    This is a demo-friendly way to show what a feature "pushes" in the next-token distribution,
    without requiring backend-specific access to transcoder weights.
    """
    top_k = int(max(1, top_k))
    prompt = format_chat_prompt(model, user_prompt)
    tokens = model.to_tokens(prompt)

    with torch.inference_mode():
        base_logits, _acts = model.feature_intervention(  # type: ignore[attr-defined]
            tokens,
            [],
            constrained_layers=constrained_layers,
            freeze_attention=freeze_attention,
            apply_activation_function=apply_activation_function,
            return_activations=False,
        )
        base_last = base_logits[0, -1]

        new_value = float(base_value) + float(delta)
        interventions = [
            (int(feature.layer), int(feature.pos), int(feature.feature_idx), float(new_value))
        ]
        new_logits, _acts2 = model.feature_intervention(  # type: ignore[attr-defined]
            tokens,
            interventions,
            constrained_layers=constrained_layers,
            freeze_attention=freeze_attention,
            apply_activation_function=apply_activation_function,
            return_activations=False,
        )
        new_last = new_logits[0, -1]

    delta_logits = (new_last.float() - base_last.float()).detach()
    delta_l2 = float(torch.linalg.vector_norm(delta_logits).item())
    delta_max_abs = float(delta_logits.abs().max().item())
    vocab = int(delta_logits.numel())
    k = min(int(top_k), vocab)
    top_pos = torch.topk(delta_logits, k=k)
    top_neg = torch.topk(-delta_logits, k=k)

    lse_base = torch.logsumexp(base_last.float(), dim=-1)
    lse_new = torch.logsumexp(new_last.float(), dim=-1)
    tokenizer = getattr(model, "tokenizer", None)
    tok_at_pos = "?"
    try:
        pos = int(feature.pos)
        if 0 <= pos < int(tokens.shape[1]):
            tok_at_pos = (
                _decode_token(tokenizer, int(tokens[0, pos].item()))
                if tokenizer is not None
                else str(pos)
            )
    except Exception:
        tok_at_pos = "?"

    def _rows(indices: torch.Tensor, values: torch.Tensor) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for tok_id_t, dlogit_t in zip(indices.tolist(), values.tolist(), strict=False):
            tok_id = int(tok_id_t)
            base_p = float(torch.exp(base_last.float()[tok_id] - lse_base).item())
            new_p = float(torch.exp(new_last.float()[tok_id] - lse_new).item())
            out.append(
                {
                    "token_id": tok_id,
                    "token": _decode_token(tokenizer, tok_id)
                    if tokenizer is not None
                    else str(tok_id),
                    "delta_logit": float(dlogit_t),
                    "p_before": base_p,
                    "p_after": new_p,
                    "delta_p": float(new_p - base_p),
                }
            )
        return out

    return {
        "feature": {
            "layer": int(feature.layer),
            "pos": int(feature.pos),
            "feature_idx": int(feature.feature_idx),
        },
        "token_at_pos": tok_at_pos,
        "base_value": float(base_value),
        "delta": float(delta),
        "new_value": float(new_value),
        "delta_logit_l2": float(delta_l2),
        "delta_logit_max_abs": float(delta_max_abs),
        "top_positive": _rows(top_pos.indices.detach().cpu(), top_pos.values.detach().cpu()),
        "top_negative": _rows(top_neg.indices.detach().cpu(), (-top_neg.values).detach().cpu()),
    }


def feature_intervention_next_token(
    model: Any,
    user_prompt: str,
    interventions: list[tuple[int, int, int, float]],
    *,
    freeze_attention: bool = True,
    constrained_layers: range | None = None,
    apply_activation_function: bool = True,
) -> torch.Tensor:
    """Run `feature_intervention` and return last-position logits.

    Interventions are (layer, pos, feature_idx, value).
    """
    prompt = format_chat_prompt(model, user_prompt)
    tokens = model.to_tokens(prompt)
    logits, _acts = model.feature_intervention(  # type: ignore[attr-defined]
        tokens,
        interventions,
        constrained_layers=constrained_layers,
        freeze_attention=freeze_attention,
        apply_activation_function=apply_activation_function,
        return_activations=False,
    )
    return logits[0, -1]


def generate_text_stepwise_with_interventions(
    model: Any,
    user_prompt: str,
    interventions: list[tuple[int, int, int, float]],
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float | None = 0.9,
    freeze_attention: bool = False,
    constrained_layers: range | None = None,
    apply_activation_function: bool = True,
) -> str:
    """Generate by re-running `feature_intervention` at every step (slow, but position-correct).

    This avoids the KV-cache position ambiguity for prompt-position interventions.
    """
    prompt = format_chat_prompt(model, user_prompt)
    tokens = model.to_tokens(prompt)

    do_sample = float(temperature) > 0
    max_new_tokens = int(max(0, max_new_tokens))

    for _ in range(max_new_tokens):
        logits, _acts = model.feature_intervention(  # type: ignore[attr-defined]
            tokens,
            interventions,
            constrained_layers=constrained_layers,
            freeze_attention=freeze_attention,
            apply_activation_function=apply_activation_function,
            return_activations=False,
        )
        last_logits = logits[0, -1]

        temp = max(float(temperature), 1e-6)
        logits_f = last_logits.float()
        scaled = (logits_f - logits_f.max()) / temp
        probs = torch.softmax(scaled, dim=-1)
        if top_p is not None and 0 < float(top_p) < 1:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            keep = cum <= float(top_p)
            keep[0] = True
            filtered = torch.zeros_like(probs)
            filtered[sorted_idx[keep]] = probs[sorted_idx[keep]]
            probs = filtered / filtered.sum().clamp(min=1e-20)

        if do_sample:
            next_id = int(torch.multinomial(probs, num_samples=1).item())
        else:
            next_id = int(torch.argmax(probs).item())

        next_tok = torch.tensor([[next_id]], device=tokens.device, dtype=tokens.dtype)
        tokens = torch.cat([tokens, next_tok], dim=1)

    new_tokens = tokens[0, -max_new_tokens:] if max_new_tokens > 0 else tokens[0, 0:0]
    return model.tokenizer.decode(new_tokens)


def prune_graph_for_target_logit(
    graph: Graph,
    *,
    target_logit_index: int,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prune graph for a specific target logit.

    Returns: (node_mask, edge_mask, cumulative_node_scores).
    """
    n_tokens = int(graph.input_tokens.numel())
    n_logits = int(graph.logit_tokens.numel())
    n_features = int(graph.selected_features.numel())
    n_nodes = int(graph.adjacency_matrix.shape[0])

    target_logit_index = int(max(0, min(n_logits - 1, int(target_logit_index))))

    logit_weights = torch.zeros(n_nodes, device=graph.adjacency_matrix.device)
    logit_weights[-n_logits + target_logit_index] = 1.0

    node_influence = compute_node_influence(graph.adjacency_matrix, logit_weights)
    node_mask = node_influence >= find_threshold(node_influence, float(node_threshold))
    node_mask[-n_logits - n_tokens :] = True

    pruned_matrix = graph.adjacency_matrix.clone()
    pruned_matrix[~node_mask] = 0
    pruned_matrix[:, ~node_mask] = 0

    from circuit_tracer.graph import compute_edge_influence  # local import to keep surface small

    edge_scores = compute_edge_influence(pruned_matrix, logit_weights)
    edge_mask = edge_scores >= find_threshold(edge_scores.flatten(), float(edge_threshold))

    old_node_mask = node_mask.clone()
    node_mask[: -n_logits - n_tokens] &= edge_mask[:, : -n_logits - n_tokens].any(0)
    node_mask[:n_features] &= edge_mask[:n_features].any(1)

    while not torch.all(node_mask == old_node_mask):
        old_node_mask[:] = node_mask
        edge_mask[~node_mask] = False
        edge_mask[:, ~node_mask] = False
        node_mask[: -n_logits - n_tokens] &= edge_mask[:, : -n_logits - n_tokens].any(0)
        node_mask[:n_features] &= edge_mask[:n_features].any(1)

    sorted_scores, sorted_indices = torch.sort(node_influence, descending=True)
    cumulative_scores = torch.cumsum(sorted_scores, dim=0) / torch.sum(sorted_scores).clamp(
        min=1e-20
    )
    cumulative_by_node = torch.zeros_like(node_influence)
    cumulative_by_node[sorted_indices] = cumulative_scores

    return node_mask, edge_mask, cumulative_by_node


def graph_node_label(model: Any, graph: Graph, node_i: int) -> str:
    tokenizer = getattr(model, "tokenizer", None)
    n_tokens = int(graph.input_tokens.numel())
    n_features = int(graph.selected_features.numel())
    n_errors = int(graph.cfg.n_layers) * n_tokens
    n_embeds = n_tokens

    def _pretty_token(token: str) -> str:
        return token.replace("\n", "\\n").replace("\t", "\\t")

    def _tok_str_at_pos(pos_idx: int) -> str:
        if pos_idx < 0 or pos_idx >= n_tokens:
            return "?"
        tok_id = int(graph.input_tokens[pos_idx].item())
        tok_str = _decode_token(tokenizer, tok_id) if tokenizer is not None else str(tok_id)
        return _pretty_token(tok_str)

    if node_i < n_features:
        feat_global = int(graph.selected_features[node_i].item())
        layer, pos, fidx = graph.active_features[feat_global].tolist()
        return f"L{layer} '{_tok_str_at_pos(int(pos))}' #{fidx}"

    if node_i < n_features + n_errors:
        j = node_i - n_features
        layer = j // n_tokens
        pos = j % n_tokens
        return f"Err L{layer} '{_tok_str_at_pos(int(pos))}'"

    if node_i < n_features + n_errors + n_embeds:
        pos = node_i - (n_features + n_errors)
        return f"Emb '{_tok_str_at_pos(int(pos))}'"

    logit_i = node_i - (n_features + n_errors + n_embeds)
    tok_id = int(graph.logit_tokens[logit_i].item())
    tok_str = _decode_token(tokenizer, tok_id) if tokenizer is not None else str(tok_id)
    return f"Logit '{_pretty_token(tok_str)}'"


def summarize_graph_for_target(
    model: Any,
    graph: Graph,
    *,
    target_logit_index: int,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    max_features: int = 200,
    max_edges: int = 300,
) -> dict[str, Any]:
    """Build UI-friendly summaries for a specific target logit, without re-running attribution."""
    logits_summary = summarize_logits(model, graph.logit_tokens, graph.logit_probabilities)
    tokens_summary = summarize_tokens(model, graph.input_tokens)

    target_logit_index = int(max(0, min(len(logits_summary) - 1, int(target_logit_index))))
    node_mask, edge_mask, cumulative_scores = prune_graph_for_target_logit(
        graph,
        target_logit_index=target_logit_index,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
    )

    n_logits = int(graph.logit_tokens.numel())
    n_features = int(graph.selected_features.numel())
    n_nodes = int(graph.adjacency_matrix.shape[0])

    logit_weights = torch.zeros(n_nodes, device=graph.adjacency_matrix.device)
    logit_weights[-n_logits + target_logit_index] = 1.0
    node_influence = compute_node_influence(graph.adjacency_matrix, logit_weights).detach().cpu()

    feature_rows: list[dict[str, Any]] = []
    for node_i in range(n_features):
        feat_global_idx = int(graph.selected_features[node_i].item())
        layer, pos, feature_idx = graph.active_features[feat_global_idx].tolist()
        act_val = float(graph.activation_values[feat_global_idx].item())
        feature_rows.append(
            {
                "node_index": int(node_i),
                "feature": FeatureRef(layer=int(layer), pos=int(pos), feature_idx=int(feature_idx)),
                "activation": float(act_val),
                "influence": float(node_influence[node_i].item()),
                "cumulative_influence_rank": float(cumulative_scores[node_i].detach().cpu().item()),
                "kept": bool(node_mask[node_i].detach().cpu().item()),
            }
        )
    feature_rows.sort(key=lambda r: abs(r["influence"]), reverse=True)

    src_idx, dst_idx = edge_mask.nonzero(as_tuple=True)
    weights = graph.adjacency_matrix[src_idx, dst_idx]
    if weights.numel() > 0:
        order = torch.argsort(weights.abs(), descending=True)[
            : min(int(max_edges), weights.numel())
        ]
        src_idx = src_idx[order].detach().cpu()
        dst_idx = dst_idx[order].detach().cpu()
        weights = weights[order].detach().cpu()

    edges: list[dict[str, Any]] = []
    for t, s, w in zip(src_idx.tolist(), dst_idx.tolist(), weights.tolist(), strict=False):
        edges.append(
            {
                "source": int(s),
                "target": int(t),
                "weight": float(w),
                "source_label": graph_node_label(model, graph, int(s)),
                "target_label": graph_node_label(model, graph, int(t)),
            }
        )

    replacement_score, completeness_score = compute_graph_scores(graph)

    return {
        "target_logit_index": target_logit_index,
        "tokens": tokens_summary,
        "logits": logits_summary,
        "feature_rows": feature_rows[: int(max_features)],
        "edges": edges,
        "graph_scores": {
            "replacement_score": float(replacement_score),
            "completeness_score": float(completeness_score),
        },
    }


def analyze_prompt(
    model: Any,
    user_prompt: str,
    *,
    max_feature_nodes: int = 7500,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    batch_size: int = 64,
    offload: Literal["auto", "cpu", "disk"] | None = "auto",
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    target_logit_index: int = 0,
    verbose: bool = False,
    return_graph: bool = False,
) -> dict[str, Any]:
    """Run circuit-tracer attribution and return summaries for Streamlit.

    Ref: "On the Biology of a Large Language Model" — extracting active features and
    attribution graphs for a concrete prompt.
    """
    prompt = format_chat_prompt(model, user_prompt)

    resolved_offload: Literal["cpu", "disk", None]
    if offload == "auto":
        resolved_offload = None
        if torch.cuda.is_available():
            # Heuristic: on consumer GPUs, attribution often needs offloading.
            try:
                total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if total_gb <= 10:
                    resolved_offload = "cpu"
            except Exception:
                resolved_offload = "cpu"
    else:
        resolved_offload = offload

    try:
        # Attribution relies on autograd (gradient injection + backward on residual stream).
        # Using inference_mode/no_grad will break hook registration inside circuit-tracer.
        with torch.enable_grad(), warnings.catch_warnings():
            # PyTorch warns about full backward hooks when no module inputs require grad.
            # circuit-tracer/TransformerLens uses backward hooks intentionally for attribution.
            warnings.filterwarnings(
                "ignore",
                message=r"Full backward hook is firing when gradients are computed with respect to module outputs.*",
                category=UserWarning,
            )
            graph: Graph = attribute(
                prompt=prompt,
                model=model,
                max_feature_nodes=max_feature_nodes,
                max_n_logits=max_n_logits,
                desired_logit_prob=desired_logit_prob,
                batch_size=batch_size,
                offload=resolved_offload,
                verbose=verbose,
            )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise MemoryError("GPU OOM during attribution. Try smaller max_feature_nodes.") from e
        raise
    finally:
        # Backward passes can populate parameter .grad buffers; clear them to limit RAM/VRAM growth
        # across Streamlit reruns.
        zero_grad = getattr(model, "zero_grad", None)
        if callable(zero_grad):
            try:
                zero_grad(set_to_none=True)
            except TypeError:
                zero_grad()

    summary = summarize_graph_for_target(
        model,
        graph,
        target_logit_index=target_logit_index,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
        max_features=200,
        max_edges=300,
    )

    out: dict[str, Any] = {
        "user_prompt": user_prompt,
        "prompt_formatted": prompt,
        **summary,
        "n_nodes": int(graph.adjacency_matrix.shape[0]),
        "n_selected_features": int(graph.selected_features.numel()),
        "n_active_features_total": int(graph.active_features.shape[0]),
        "n_prompt_tokens": int(graph.input_tokens.numel()),
    }

    if return_graph:
        out["graph"] = graph  # not JSON-serializable; can be very large

    return out
