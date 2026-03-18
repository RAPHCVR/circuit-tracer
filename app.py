from __future__ import annotations

import gc
import json
import re
from typing import Any, Literal, cast

import pandas as pd
import streamlit as st
import torch

from engine import (
    FeatureRef,
    activation_change_report,
    analyze_prompt,
    direct_logit_effects_for_feature,
    distribution_from_logits,
    feature_intervention_next_token,
    generate_text,
    generate_text_stepwise_with_interventions,
    load_model,
    next_token_distribution,
    parse_manual_interventions,
    summarize_graph_for_target,
)

st.set_page_config(page_title="circuit-tracer dashboard", layout="wide")


DEMO_PROMPTS: dict[str, str] = {
    "Capitale (France)": "La capitale de la France est",
    "Capital (France, EN)": "The capital of France is",
    "Antonymes": 'Le contraire de "petit" est "',
    "Antonyms (EN)": 'The opposite of "small" is "',
    "Addition": "calc: 36+59=",
    "Multi-step (Dallas→Texas→Austin)": "Fact: the capital of the state containing Dallas is",
    "Multi-step (Oakland→California→Sacramento)": "Fact: the capital of the state containing Oakland is",
    "Entités": "Qui est Michael Jordan ?",
}


def _safety_block_reason(user_prompt: str) -> str | None:
    text = user_prompt.lower()
    patterns = [
        r"\bhow to (make|build)\b.*\b(bomb|explosive)\b",
        r"\bmake (a )?bomb\b",
        r"\bexplosive(s)?\b.*\brecipe\b",
        r"\b(how to|best way to|guide to)\b.*\b(kill|murder|poison)\b",
        r"\b(comment|comment faire|fabriquer|construire)\b.*\b(bombe|explosif|explosifs)\b",
        r"\bfabriquer (une )?bombe\b|\bfaire (une )?bombe\b",
        r"\b(tuer|assassiner|empoisonner)\b.*\b(comment|meilleure façon|guide)\b",
        r"\bsuicide\b|\bself-harm\b|\bself harm\b",
        r"\bcredit card\b.*\bfraud\b|\bwire fraud\b|\bidentity theft\b",
        r"\bhack\b.*\b(password|account|wifi)\b|\bransomware\b",
    ]
    if any(re.search(p, text) for p in patterns):
        return "Blocked: prompt looks like it requests harmful / illegal instructions."
    return None


def _feature_key(feature: FeatureRef) -> str:
    return f"{feature.layer}:{feature.pos}:{feature.feature_idx}"


def _pretty_token(token: str | None) -> str:
    if not token:
        return "?"
    return token.replace("\n", "\\n").replace("\t", "\\t")


def _fmt_pct(x: Any) -> str:
    try:
        return f"{float(x):.1%}"
    except Exception:
        return "—"


def _neuronpedia_feature_url(
    *,
    model_name: str,
    transcoder_set: str,
    layer: int,
    feature_idx: int,
) -> str | None:
    name = str(model_name).lower()
    if "gemma-2-2b-it" in name:
        model_id = "gemma-2-2b-it"
    elif "gemma-2-2b" in name:
        model_id = "gemma-2-2b"
    else:
        return None

    tset = str(transcoder_set).lower()
    if not (tset in {"gemma", "mwhanna/gemma-scope-transcoders"} or "gemma-scope" in tset):
        return None

    # Basic width detection from the HF ref / shortcut (prevents generating dead links).
    if "1m" in tset:
        width = "1m"
        max_features = 1_000_000
    elif "65k" in tset:
        width = "65k"
        max_features = 65_536
    else:
        width = "16k"
        max_features = 16_384

    if int(feature_idx) < 0 or int(feature_idx) >= max_features:
        return None

    # GemmaScope naming convention on Neuronpedia.
    sae_id = f"{int(layer)}-gemmascope-mlp-{width}"

    return f"https://www.neuronpedia.org/{model_id}/{sae_id}/{int(feature_idx)}"


def _serialize_analysis(analysis: dict[str, Any]) -> str:
    export = {
        "prompt_formatted": analysis.get("prompt_formatted"),
        "graph_scores": analysis.get("graph_scores"),
        "target_logit_index": analysis.get("target_logit_index"),
        "tokens": analysis.get("tokens"),
        "logits": analysis.get("logits"),
        "features": [
            {
                "feature": {
                    "layer": int(r["feature"].layer),
                    "pos": int(r["feature"].pos),
                    "feature_idx": int(r["feature"].feature_idx),
                },
                "activation": float(r["activation"]),
                "influence": float(r["influence"]),
                "kept": bool(r["kept"]),
            }
            for r in (analysis.get("feature_rows") or [])
        ],
        "edges": analysis.get("edges"),
    }
    return json.dumps(export, ensure_ascii=False, indent=2)


def _serialize_activation_validation(report: dict[str, Any]) -> str:
    rows = report.get("rows") or []
    export = {
        "baseline_mode": report.get("baseline_mode"),
        "freeze_attention": report.get("freeze_attention"),
        "constrained_layers": report.get("constrained_layers"),
        "apply_activation_function": report.get("apply_activation_function"),
        "sparse": report.get("sparse"),
        "n_tracked": report.get("n_tracked"),
        "rows": [
            {
                "feature": {
                    "layer": int(r["feature"].layer),
                    "pos": int(r["feature"].pos),
                    "feature_idx": int(r["feature"].feature_idx),
                },
                "base": float(r["base"]),
                "new": float(r["new"]),
                "delta": float(r["delta"]),
                "pct_delta": r["pct_delta"],
            }
            for r in rows
        ],
    }
    return json.dumps(export, ensure_ascii=False, indent=2)


def _render_markdown_report(
    analysis: dict[str, Any],
    *,
    validation: dict[str, Any] | None,
    activation_validation: dict[str, Any] | None,
    feature_inspection: dict[str, Any] | None,
) -> str:
    prompt = analysis.get("user_prompt", "")
    target = analysis.get("target_logit_index", 0)

    lines = [
        "# circuit-tracer demo report",
        "",
        "## Prompt",
        "```text",
        str(prompt).strip(),
        "```",
        "",
        "## Target logit",
        f"- `target_logit_index`: `{int(target)}`",
        "",
    ]

    if validation is not None:
        interventions = validation.get("interventions") or []
        before = validation.get("before") or {}
        after = validation.get("after") or {}
        lines += [
            "## Next-Token Validation",
            f"- `n_interventions`: `{len(interventions)}`",
            f"- `entropy_before`: `{before.get('entropy')}`",
            f"- `entropy_after`: `{after.get('entropy')}`",
            f"- `margin_before`: `{before.get('margin_top1_top2')}`",
            f"- `margin_after`: `{after.get('margin_top1_top2')}`",
            "",
            "### Top surface groups (before)",
            "```json",
            json.dumps(before.get("top_groups", []), ensure_ascii=False, indent=2),
            "```",
            "",
            "### Top surface groups (after)",
            "```json",
            json.dumps(after.get("top_groups", []), ensure_ascii=False, indent=2),
            "```",
            "",
            "### Top logits (before)",
            "```json",
            json.dumps(before.get("top", []), ensure_ascii=False, indent=2),
            "```",
            "",
            "### Top logits (after)",
            "```json",
            json.dumps(after.get("top", []), ensure_ascii=False, indent=2),
            "```",
            "",
        ]

    if activation_validation is not None:
        rows = activation_validation.get("rows") or []
        lines += [
            "## Activation Validation",
            f"- `baseline_mode`: `{activation_validation.get('baseline_mode')}`",
            f"- `freeze_attention`: `{activation_validation.get('freeze_attention')}`",
            f"- `constrained_layers`: `{activation_validation.get('constrained_layers')}`",
            f"- `apply_activation_function`: `{activation_validation.get('apply_activation_function')}`",
            "",
            "### Top activation deltas",
            "```json",
            json.dumps(
                [
                    {
                        "feature": f"L{r['feature'].layer} P{r['feature'].pos} #{r['feature'].feature_idx}",
                        "base": r["base"],
                        "new": r["new"],
                        "delta": r["delta"],
                        "pct_delta": r["pct_delta"],
                    }
                    for r in rows[:30]
                ],
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
        ]

    if feature_inspection is not None:
        report = feature_inspection.get("report") or {}
        key = feature_inspection.get("key")
        lines += [
            "## Feature inspection (direct logit effects)",
            f"- `feature`: `{key}`",
            f"- `base_value`: `{report.get('base_value')}`",
            f"- `delta`: `{report.get('delta')}`",
            f"- `new_value`: `{report.get('new_value')}`",
            f"- `delta_logit_max_abs`: `{report.get('delta_logit_max_abs')}`",
            "",
            "### Top pushed tokens",
            "```json",
            json.dumps(report.get("top_positive", [])[:10], ensure_ascii=False, indent=2),
            "```",
            "",
            "### Top suppressed tokens",
            "```json",
            json.dumps(report.get("top_negative", [])[:10], ensure_ascii=False, indent=2),
            "```",
            "",
        ]

    return "\n".join(lines).strip() + "\n"


def _sample_token_id_from_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float | None,
) -> int:
    temp = max(float(temperature), 1e-6)
    logits_f = logits.float()
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
    return int(torch.multinomial(probs, num_samples=1).item())


st.title("circuit-tracer dashboard")
st.caption(
    "Attribution graphs, feature interventions, and next-token validation for local demos."
)


if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "last_generation" not in st.session_state:
    st.session_state.last_generation = ""
if "feature_controls" not in st.session_state:
    st.session_state.feature_controls = {}
if "validation" not in st.session_state:
    st.session_state.validation = None
if "activation_validation" not in st.session_state:
    st.session_state.activation_validation = None
if "feature_inspection" not in st.session_state:
    st.session_state.feature_inspection = None
if "model_name" not in st.session_state:
    st.session_state.model_name = "google/gemma-2-2b-it"
if "transcoder_set" not in st.session_state:
    st.session_state.transcoder_set = "gemma"
if "backend" not in st.session_state:
    st.session_state.backend = "transformerlens"


with st.sidebar:
    st.header("Controls")

    with st.expander("Model", expanded=False):
        model_name = st.text_input("Model name", value=str(st.session_state.model_name))
        transcoder_set = st.text_input(
            "Transcoder set (HF ref)", value=str(st.session_state.transcoder_set)
        )
        backend_choice = st.selectbox(
            "Backend",
            ["transformerlens", "nnsight"],
            index=0 if str(st.session_state.backend) == "transformerlens" else 1,
            help="TransformerLens is faster but supports fewer HF models; nnsight is broader but slower.",
        )
        if (
            model_name != st.session_state.model_name
            or transcoder_set != st.session_state.transcoder_set
            or backend_choice != st.session_state.backend
        ):
            st.session_state.model_name = str(model_name)
            st.session_state.transcoder_set = str(transcoder_set)
            st.session_state.backend = str(backend_choice)
            st.session_state.analysis = None
            st.session_state.validation = None
            st.session_state.activation_validation = None
            st.session_state.feature_inspection = None
            st.session_state.feature_controls = {}
            st.info("Model settings changed — analysis cleared.")

    preset = st.selectbox("Démos", ["(custom)"] + list(DEMO_PROMPTS.keys()))
    if preset == "(custom)":
        default_prompt = st.session_state.get("user_prompt", DEMO_PROMPTS["Capitale (France)"])
    else:
        default_prompt = DEMO_PROMPTS[preset]

    user_prompt = st.text_area("Prompt", value=default_prompt, height=140)
    st.session_state["user_prompt"] = user_prompt

    safe_mode = st.checkbox(
        "Safe mode", value=True, help="Bloque quelques prompts manifestement dangereux."
    )

    st.divider()
    st.subheader("Generation")
    max_new_tokens = st.slider("Max new tokens", 8, 256, 96, step=8)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, step=0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, step=0.05)

    st.divider()
    st.subheader("Attribution")
    attr_max_feature_nodes = st.slider("Max feature nodes", 500, 7500, 1500, step=250)
    attr_batch_size = st.slider("Attribution batch size", 8, 256, 32, step=8)
    attr_offload = st.selectbox(
        "Offload",
        ["auto", None, "cpu", "disk"],
        index=0,
        format_func=lambda v: {
            "auto": "auto (heuristic)",
            None: "none (keep on GPU/VRAM)",
            "cpu": "cpu (RAM)",
            "disk": "disk (slowest)",
        }.get(v, str(v)),
    )
    attr_max_n_logits = st.slider("Max logits", 1, 30, 10, step=1)
    # If the model is very confident in the top-1 next token, small desired_logit_prob values
    # can lead to attributing a single logit only. Allow going higher for demos.
    attr_desired_logit_prob = st.slider("Desired logit prob", 0.5, 0.999, 0.95, step=0.001)
    node_threshold = st.slider("Node threshold", 0.5, 0.99, 0.8, step=0.01)
    edge_threshold = st.slider("Edge threshold", 0.5, 0.999, 0.98, step=0.001)

    st.divider()
    run_attr = st.button("Analyze circuits", type="primary", width="stretch")
    clear_attr = st.button("Clear analysis", width="stretch")


try:
    backend_name = cast(Literal["transformerlens", "nnsight"], str(st.session_state.backend))
    model = load_model(
        model_name=str(st.session_state.model_name),
        transcoder_set=str(st.session_state.transcoder_set),
        backend=backend_name,
    )
except Exception as exc:
    st.error(
        "Model initialization failed. Check your local environment / model access, then retry."
    )
    st.exception(exc)
    st.stop()

st.caption(
    f"Using model `{st.session_state.model_name}` with transcoders `{st.session_state.transcoder_set}` "
    f"(backend `{st.session_state.backend}`)"
)

with st.sidebar:
    st.divider()
    st.subheader("Runtime")
    backend_runtime = getattr(model, "backend", "unknown")
    try:
        p0 = next(model.parameters())
        model_device = str(p0.device)
        model_dtype = str(p0.dtype)
    except Exception:
        model_device = str(getattr(getattr(model, "cfg", None), "device", "unknown"))
        model_dtype = "unknown"

    info: dict[str, Any] = {
        "backend": backend_runtime,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "model_device": model_device,
        "model_dtype": model_dtype,
    }
    if torch.cuda.is_available():
        try:
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["vram_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
            )
        except Exception:
            pass
    else:
        info["cuda_available"] = False
    st.write(info)

if safe_mode:
    block_reason = _safety_block_reason(user_prompt)
    if block_reason is not None:
        st.error(block_reason)
        st.stop()

if clear_attr:
    st.session_state.analysis = None
    st.session_state.validation = None
    st.session_state.activation_validation = None
    st.session_state.feature_inspection = None


if run_attr:
    st.session_state.analysis = None
    st.session_state.validation = None
    st.session_state.activation_validation = None
    st.session_state.feature_inspection = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with st.spinner("Computing attribution graph… (can take a while)"):
        try:
            offload_mode = cast(Literal["auto", "cpu", "disk"] | None, attr_offload)
            st.session_state.analysis = analyze_prompt(
                model,
                user_prompt,
                max_feature_nodes=attr_max_feature_nodes,
                max_n_logits=attr_max_n_logits,
                desired_logit_prob=attr_desired_logit_prob,
                batch_size=attr_batch_size,
                offload=offload_mode,
                node_threshold=node_threshold,
                edge_threshold=edge_threshold,
                target_logit_index=0,
                return_graph=True,
            )
            st.session_state.validation = None
            st.session_state.activation_validation = None
            st.session_state.feature_inspection = None
        except MemoryError as exc:
            st.error(str(exc))
            st.session_state.analysis = None
        except Exception as exc:
            st.exception(exc)
            st.session_state.analysis = None
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


analysis = st.session_state.analysis
if analysis is None:
    st.info("Run `Analyze circuits` to start.")
    st.stop()

if user_prompt != analysis.get("user_prompt"):
    st.warning(
        "Prompt has changed since the last attribution run. Re-run `Analyze circuits` for consistent results."
    )
    st.stop()


col_left, col_right = st.columns([1.2, 1.0], gap="large")

with col_left:
    st.subheader("Attribution (target logit)")
    st.caption(
        f"tokens={analysis.get('n_prompt_tokens')} • active_features={analysis.get('n_active_features_total')} • "
        f"graph_nodes={analysis.get('n_nodes')} • selected_features={analysis.get('n_selected_features')}"
    )

    with st.expander("Prompt (formatted)", expanded=False):
        st.code(analysis.get("prompt_formatted", ""), language="text")

    st.markdown("**Graph quality (heuristics)**")
    scores = analysis.get("graph_scores") or {}
    col_s1, col_s2 = st.columns(2)
    col_s1.metric(
        "Replacement Score",
        _fmt_pct(scores.get("replacement_score")),
        help="Fraction de l'influence totale qui passe par des features interprétables (vs. error nodes).",
    )
    col_s2.metric(
        "Completeness Score",
        _fmt_pct(scores.get("completeness_score")),
        help="Couverture globale des chemins explicatifs.",
    )
    replacement_score = scores.get("replacement_score")
    try:
        replacement = float(replacement_score) if replacement_score is not None else None
    except Exception:
        replacement = None
    if replacement is not None and replacement < 0.5:
        st.error(
            f"Low explainability confidence: replacement_score={replacement:.1%}. "
            "The circuit graph misses >50% of the causal flow (error nodes / reconstruction gaps)."
        )

    logits_df = pd.DataFrame(analysis.get("logits", []))
    if not logits_df.empty:
        logits_df["label"] = logits_df.apply(
            lambda r: f"[{int(r['logit_index'])}] {r['token']}  (p={float(r['prob']):.3f})",
            axis=1,
        )
        options = logits_df["label"].tolist()
        current_target = int(analysis.get("target_logit_index", 0))

        target_selected = st.radio(
            "Target logit",
            options,
            index=min(current_target, len(options) - 1),
        )
        new_target_idx = int(target_selected.split("]")[0].strip("[")) if target_selected else 0

        if new_target_idx != current_target:
            graph = analysis.get("graph")
            if graph is None:
                st.warning("No cached graph; re-run `Analyze circuits` to change target.")
            else:
                refreshed = summarize_graph_for_target(
                    model,
                    graph,
                    target_logit_index=new_target_idx,
                    node_threshold=node_threshold,
                    edge_threshold=edge_threshold,
                )
                st.session_state.analysis = {**analysis, **refreshed}
                analysis = st.session_state.analysis

        st.markdown("**Top logits (next token)**")
        st.dataframe(logits_df[["logit_index", "token", "prob"]], width="stretch", height=240)

    st.divider()
    st.markdown("**Tokenization (prompt)**")
    tokens_df = pd.DataFrame(analysis.get("tokens", []))
    st.dataframe(tokens_df, width="stretch", height=260)

    st.divider()
    st.markdown("**Top features (ranked by influence on target logit)**")
    feat_rows = analysis.get("feature_rows", [])
    if feat_rows:
        tokens_list = analysis.get("tokens") or []
        tok_map: dict[int, str] = {}
        for t in tokens_list:
            if "pos" not in t:
                continue
            tok = t.get("token")
            tok_map[int(t["pos"])] = _pretty_token(tok if isinstance(tok, str) else None)
        feat_df = pd.DataFrame(
            [
                {
                    "feature": (
                        f"L{r['feature'].layer} '{tok_map.get(int(r['feature'].pos), '?')}' "
                        f"#{r['feature'].feature_idx}"
                    ),
                    "neuronpedia": _neuronpedia_feature_url(
                        model_name=str(st.session_state.model_name),
                        transcoder_set=str(st.session_state.transcoder_set),
                        layer=int(r["feature"].layer),
                        feature_idx=int(r["feature"].feature_idx),
                    ),
                    "activation": float(r["activation"]),
                    "influence": float(r["influence"]),
                    "kept": bool(r["kept"]),
                }
                for r in feat_rows[:60]
            ]
        )
        st.dataframe(
            feat_df,
            width="stretch",
            height=320,
            column_config={
                "neuronpedia": st.column_config.LinkColumn(
                    "Neuronpedia",
                    help="External feature browser (when available for this model/transcoder set).",
                    display_text="open",
                )
            },
        )

    st.divider()
    st.markdown("**Pruned edges (top |weight|)**")
    edges = analysis.get("edges") or []
    if edges:
        edges_df = pd.DataFrame(edges)[["source_label", "target_label", "weight"]]
        st.dataframe(edges_df, width="stretch", height=320)

        try:
            import graphviz  # type: ignore[import-not-found]

            st.markdown("**Computation graph (top edges)**")
            g = graphviz.Digraph()
            g.attr(rankdir="LR", size="10,5")

            for edge in edges[:20]:
                w = float(edge["weight"])
                color = "green" if w > 0 else "red"
                penwidth = str(max(1.0, min(5.0, abs(w) * 10.0)))

                s = int(edge["source"])
                t = int(edge["target"])
                s_id = f"n{s}"
                t_id = f"n{t}"
                g.node(s_id, label=str(edge["source_label"]), shape="box", style="rounded")
                g.node(t_id, label=str(edge["target_label"]), shape="box", style="rounded")
                g.edge(s_id, t_id, color=color, penwidth=penwidth, label=f"{w:.2f}")

            st.graphviz_chart(g, width="stretch")
        except Exception as exc:
            st.info(
                "Graphviz visualization unavailable. Install `graphviz` (Python) and Graphviz system "
                "binaries to enable it."
            )
            st.caption(str(exc))
    else:
        st.info("No pruned edges (try adjusting thresholds).")


with col_right:
    st.subheader("Interventions (next-token) + Generation")

    st.markdown("**1) Configure interventions (from top-influence features)**")
    intervention_mode = st.selectbox(
        "Intervention mode",
        [
            "None",
            "Add delta (value = base + delta)",
            "Set value",
            "Ablate (set 0)",
            "Cap (min(value, cap))",
        ],
        index=1,
    )
    freeze_attention = st.checkbox(
        "Freeze attention",
        value=True,
        help="Recommandé pour une validation plus 'causale' et stable.",
    )
    baseline_match_freeze = st.checkbox(
        "Baseline uses frozen attention (slower)",
        value=False,
        help=(
            "Calcule le baseline via une intervention no-op pour matcher exactement le codepath "
            "des runs avec freeze_attention. Plus lent, mais comparaisons plus propres."
        ),
    )
    apply_activation_function = st.checkbox(
        "Apply activation fn (feature space)",
        value=True,
        help=(
            "Quand désactivé, les activations / interventions sont en pré-activation (utile pour certaines "
            "comparaisons, mais plus facile de se tromper)."
        ),
    )

    n_layers = int(getattr(getattr(model, "cfg", None), "n_layers", 0) or 0)
    use_constrained = st.checkbox(
        "Constrain layers (direct-effects style)",
        value=False,
        help="Option avancée: contraint la propagation des effets sur une plage de layers.",
    )
    constrained_layers: range | None = None
    if use_constrained and n_layers > 0:
        start, stop = st.slider(
            "Constrained layer range [start, stop)",
            0,
            n_layers,
            (0, min(n_layers, 8)),
            help="Applique les contraintes sur la plage. `stop` est exclusif.",
        )
        constrained_layers = range(int(start), int(stop))

    feat_rows = analysis.get("feature_rows", [])
    tokens_list = analysis.get("tokens") or []
    tok_map: dict[int, str] = {}
    for t in tokens_list:
        if "pos" not in t:
            continue
        tok = t.get("token")
        tok_map[int(t["pos"])] = _pretty_token(tok if isinstance(tok, str) else None)
    option_labels: list[str] = []
    row_by_key: dict[str, dict[str, Any]] = {}
    for row in feat_rows[:30]:
        feature: FeatureRef = row["feature"]
        key = _feature_key(feature)
        label = (
            f"{key} | tok='{tok_map.get(int(feature.pos), '?')}' | infl={float(row['influence']):.3e} "
            f"| act={float(row['activation']):.3f}"
        )
        option_labels.append(label)
        row_by_key[key] = row

    selected_options = st.multiselect(
        "Pick features",
        option_labels,
        default=option_labels[: min(5, len(option_labels))],
    )

    base_lookup: dict[tuple[int, int, int], float] = {}
    for r in feat_rows:
        fr: FeatureRef = r["feature"]
        base_lookup[(int(fr.layer), int(fr.pos), int(fr.feature_idx))] = float(
            r.get("activation", 0.0)
        )

    interventions_from_top: list[tuple[int, int, int, float]] = []
    for option in selected_options:
        key = option.split("|")[0].strip()
        row = row_by_key.get(key)
        if row is None:
            continue

        feature: FeatureRef = row["feature"]
        base = float(row["activation"])

        if intervention_mode == "None":
            continue

        if intervention_mode.startswith("Add delta"):
            delta = float(st.session_state.feature_controls.get(key, 0.0))
            delta = st.slider(f"Δ {key}", -10.0, 10.0, float(delta), 0.1)
            st.session_state.feature_controls[key] = float(delta)
            value = base + float(delta)
        elif intervention_mode == "Set value":
            lo = base - max(2.0, abs(base) * 4.0)
            hi = base + max(2.0, abs(base) * 4.0)
            value = float(st.session_state.feature_controls.get(key, base))
            value = st.slider(f"set {key}", float(lo), float(hi), float(value), 0.1)
            st.session_state.feature_controls[key] = float(value)
        elif intervention_mode.startswith("Ablate"):
            value = 0.0
        else:
            lo = base - max(2.0, abs(base) * 4.0)
            hi = base + max(2.0, abs(base) * 4.0)
            cap = float(st.session_state.feature_controls.get(key, base))
            cap = st.slider(f"cap {key}", float(lo), float(hi), float(cap), 0.1)
            st.session_state.feature_controls[key] = float(cap)
            value = min(base, float(cap))

        interventions_from_top.append(
            (int(feature.layer), int(feature.pos), int(feature.feature_idx), float(value))
        )

    with st.expander("Feature inspection (direct logit effects)", expanded=False):
        if not selected_options:
            st.info("Pick at least one feature above to inspect it.")
        else:
            inspect_option = st.selectbox("Inspect feature", selected_options, index=0)
            inspect_key = inspect_option.split("|")[0].strip()
            inspect_row = row_by_key.get(inspect_key)
            if inspect_row is None:
                st.info("Selected feature not found (re-run analysis).")
            else:
                inspect_feature: FeatureRef = inspect_row["feature"]
                inspect_base = float(inspect_row.get("activation", 0.0))
                inspect_infl = float(inspect_row.get("influence", 0.0))
                st.caption(
                    f"feature={inspect_key} • tok='{tok_map.get(int(inspect_feature.pos), '?')}' • "
                    f"activation={inspect_base:.3f} • influence={inspect_infl:.3e}"
                )
                np_url = _neuronpedia_feature_url(
                    model_name=str(st.session_state.model_name),
                    transcoder_set=str(st.session_state.transcoder_set),
                    layer=int(inspect_feature.layer),
                    feature_idx=int(inspect_feature.feature_idx),
                )
                if np_url:
                    st.markdown(f"[Open in Neuronpedia]({np_url})")

                inspect_delta = st.slider(
                    "Δ (sets value = base + Δ)",
                    -10.0,
                    10.0,
                    1.0,
                    step=0.1,
                )
                inspect_top_k = st.slider("Top-k tokens", 5, 50, 15, step=5)
                run_inspect = st.button("Compute direct logit effects", width="stretch")

                if run_inspect:
                    with st.spinner("Computing direct logit effects…"):
                        report = direct_logit_effects_for_feature(
                            model,
                            user_prompt,
                            inspect_feature,
                            base_value=inspect_base,
                            delta=float(inspect_delta),
                            freeze_attention=freeze_attention,
                            constrained_layers=constrained_layers,
                            apply_activation_function=apply_activation_function,
                            top_k=int(inspect_top_k),
                        )
                    st.session_state.feature_inspection = {"key": inspect_key, "report": report}

                current = st.session_state.feature_inspection
                if current is not None and current.get("key") == inspect_key:
                    report = current.get("report") or {}
                    st.caption(
                        f"value: {float(report.get('base_value', inspect_base)):.3f} → "
                        f"{float(report.get('new_value', inspect_base)):.3f}"
                    )
                    try:
                        max_abs = float(report.get("delta_logit_max_abs", 0.0))
                    except Exception:
                        max_abs = 0.0
                    if max_abs < 0.05:
                        st.info(
                            "Direct logit effects look small for this feature. This often happens for early-layer / "
                            "latent features whose influence is mostly indirect. Use activation validation and/or "
                            "follow downstream edges for stronger semantics."
                        )
                    col_p, col_n = st.columns(2)
                    with col_p:
                        st.markdown("**Top pushed tokens**")
                        st.dataframe(
                            pd.DataFrame(report.get("top_positive") or []),
                            width="stretch",
                            height=280,
                        )
                    with col_n:
                        st.markdown("**Top suppressed tokens**")
                        st.dataframe(
                            pd.DataFrame(report.get("top_negative") or []),
                            width="stretch",
                            height=280,
                        )

    st.markdown("**Manual interventions (optional)**")
    manual_text = st.text_area(
        "Paste interventions (one per line)",
        value=str(st.session_state.get("manual_interventions_text", "") or ""),
        height=120,
        help=(
            "Formats: `L24 P28 #6044 = 0`, `24:28:6044=0`, or `24 28 6044 0`. "
            "Relative ops: `+=` / `-=` only work for features present in the current feature table."
        ),
    )
    st.session_state["manual_interventions_text"] = manual_text
    manual_interventions, manual_warnings, manual_errors = parse_manual_interventions(
        manual_text,
        base_lookup=base_lookup,
    )
    if manual_errors:
        st.error("Manual intervention parse errors:\n- " + "\n- ".join(manual_errors))
    if manual_warnings:
        st.warning("Manual intervention warnings:\n- " + "\n- ".join(manual_warnings))

    # Merge (manual overrides top-feature sliders).
    merged: dict[tuple[int, int, int], float] = {}
    for layer, pos, feature_idx, value in interventions_from_top:
        merged[(int(layer), int(pos), int(feature_idx))] = float(value)
    for layer, pos, feature_idx, value in manual_interventions:
        merged[(int(layer), int(pos), int(feature_idx))] = float(value)

    interventions: list[tuple[int, int, int, float]] = [
        (int(layer), int(pos), int(feature_idx), float(value))
        for (layer, pos, feature_idx), value in merged.items()
    ]
    interventions.sort(key=lambda t: (t[0], t[1], t[2]))

    if interventions:
        with st.expander(f"Final interventions ({len(interventions)})", expanded=False):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "layer": int(layer),
                            "pos": int(pos),
                            "feature_idx": int(feature_idx),
                            "value": float(value),
                        }
                        for (layer, pos, feature_idx, value) in interventions
                    ]
                ),
                width="stretch",
                height=220,
            )

    st.divider()
    st.markdown("**2) Validate on next-token distribution (before/after)**")
    run_validate = st.button("Run validation", type="primary", width="stretch")
    if run_validate:
        with st.spinner("Computing distributions…"):
            try:
                before = next_token_distribution(
                    model,
                    user_prompt,
                    temperature=1.0,
                    top_p=None,
                    top_k=20,
                )
                if interventions:
                    after_logits = feature_intervention_next_token(
                        model,
                        user_prompt,
                        interventions,
                        freeze_attention=freeze_attention,
                        constrained_layers=constrained_layers,
                        apply_activation_function=apply_activation_function,
                    )
                    after = distribution_from_logits(
                        model,
                        after_logits,
                        temperature=1.0,
                        top_p=None,
                        top_k=20,
                    )
                else:
                    after = before
                st.session_state.validation = {
                    "before": before,
                    "after": after,
                    "interventions": interventions,
                }
                st.session_state.activation_validation = None
            except MemoryError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.exception(exc)

    validation = st.session_state.validation
    if validation is not None:
        before = validation["before"]
        after = validation["after"]
        st.write(
            {
                "entropy_before": round(before["entropy"], 3),
                "entropy_after": round(after["entropy"], 3),
            }
        )
        st.write(
            {
                "margin_before": round(before["margin_top1_top2"], 3),
                "margin_after": round(after["margin_top1_top2"], 3),
            }
        )
        col_b, col_a = st.columns(2)
        with col_b:
            st.markdown("**Before**")
            st.dataframe(pd.DataFrame(before["top"]), width="stretch", height=260)
            if before.get("top_groups"):
                st.caption("Top surface groups (merged variants)")
                st.dataframe(pd.DataFrame(before["top_groups"]), width="stretch", height=180)
        with col_a:
            st.markdown("**After**")
            st.dataframe(pd.DataFrame(after["top"]), width="stretch", height=260)
            if after.get("top_groups"):
                st.caption("Top surface groups (merged variants)")
                st.dataframe(pd.DataFrame(after["top_groups"]), width="stretch", height=180)

        if interventions:
            st.markdown("**Quick demo**")
            if st.button("Sample 1 token (after intervention)", width="stretch"):
                try:
                    after_logits = feature_intervention_next_token(
                        model,
                        user_prompt,
                        interventions,
                        freeze_attention=freeze_attention,
                        constrained_layers=constrained_layers,
                        apply_activation_function=apply_activation_function,
                    )
                    sampled_id = _sample_token_id_from_logits(
                        after_logits,
                        temperature=max(temperature, 1e-6),
                        top_p=top_p,
                    )
                    sampled = model.tokenizer.decode([sampled_id]).replace("\n", "\\n")
                    st.write({"sampled_token": sampled, "token_id": sampled_id})
                except Exception as exc:
                    st.exception(exc)

            st.divider()
            st.markdown("**Activation validation (feature activations before/after)**")
            track_mode = st.selectbox(
                "Track which features?",
                ["Intervened only", "Top influence (N)", "Kept nodes (N)"],
                index=1,
            )
            n_track = st.slider("N features to track", 5, 200, 40, step=5)
            run_act = st.button("Run activation validation", type="primary", width="stretch")
            if run_act:
                try:
                    if track_mode == "Intervened only":
                        tracked = [
                            FeatureRef(layer=int(layer), pos=int(pos), feature_idx=int(feature_idx))
                            for (layer, pos, feature_idx, _value) in interventions
                        ]
                    elif track_mode == "Kept nodes (N)":
                        tracked = [r["feature"] for r in feat_rows if bool(r.get("kept"))][
                            : int(n_track)
                        ]
                    else:
                        tracked = [r["feature"] for r in feat_rows][: int(n_track)]

                    with st.spinner("Computing activations (baseline vs intervened)…"):
                        report = activation_change_report(
                            model,
                            user_prompt,
                            interventions,
                            tracked,
                            freeze_attention=freeze_attention,
                            constrained_layers=constrained_layers,
                            apply_activation_function=apply_activation_function,
                            sparse=True,
                            baseline_match_freeze=baseline_match_freeze,
                        )
                    st.session_state.activation_validation = report
                except MemoryError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.exception(exc)

            act_report = st.session_state.activation_validation
            if act_report is not None:
                st.caption(
                    f"baseline_mode={act_report.get('baseline_mode')} • "
                    f"tracked={act_report.get('n_tracked')} • sparse={act_report.get('sparse')}"
                )
                rows = act_report.get("rows") or []
                token_by_pos = {
                    int(t["pos"]): str(t["token"]) for t in (analysis.get("tokens") or [])
                }
                df = pd.DataFrame(
                    [
                        {
                            "feature": f"L{r['feature'].layer} P{r['feature'].pos} #{r['feature'].feature_idx}",
                            "token@pos": token_by_pos.get(int(r["feature"].pos), ""),
                            "base": float(r["base"]),
                            "new": float(r["new"]),
                            "delta": float(r["delta"]),
                            "pct_delta": r["pct_delta"],
                        }
                        for r in rows[: min(200, len(rows))]
                    ]
                )
                st.dataframe(df, width="stretch", height=320)

            st.divider()
            st.markdown("**Intervention strength sweep (probability vs scale)**")
            sweep_enabled = st.checkbox(
                "Enable sweep",
                value=False,
                help="Rejoue plusieurs fois les interventions en les 'scalant' (alpha) autour du baseline.",
            )
            if sweep_enabled:
                token_opts = [
                    f"[{int(r['logit_index'])}] {r['token']}"
                    for r in (analysis.get("logits") or [])
                    if "logit_index" in r
                ]
                default_target = int(analysis.get("target_logit_index", 0))
                token_choice = st.selectbox(
                    "Token to track (probability)",
                    token_opts,
                    index=min(default_target, max(0, len(token_opts) - 1)) if token_opts else 0,
                )
                choice_idx = (
                    int(token_choice.split("]")[0].strip("[")) if token_choice else default_target
                )
                logits_rows = analysis.get("logits") or []
                tok_id = (
                    int(logits_rows[choice_idx]["token_id"])
                    if choice_idx < len(logits_rows)
                    else None
                )

                max_alpha = st.slider("Max alpha", 0.0, 10.0, 4.0, step=0.5)
                n_points = st.slider("Points", 3, 15, 7, step=1)
                run_sweep = st.button("Run sweep", type="primary", width="stretch")

                if run_sweep and tok_id is not None:
                    try:
                        alphas = torch.linspace(0.0, float(max_alpha), int(n_points)).tolist()
                        sweep_rows: list[dict[str, Any]] = []

                        # We interpret the current `interventions` as absolute feature values.
                        # Scale them around the baseline values (from the attribution summary).
                        base_lookup: dict[tuple[int, int, int], float] = {}
                        for r in feat_rows:
                            fr: FeatureRef = r["feature"]
                            base_lookup[(fr.layer, fr.pos, fr.feature_idx)] = float(r["activation"])

                        for a in alphas:
                            scaled: list[tuple[int, int, int, float]] = []
                            for layer, pos, feature_idx, value in interventions:
                                base_v = base_lookup.get(
                                    (int(layer), int(pos), int(feature_idx)), 0.0
                                )
                                scaled_v = float(base_v + a * (float(value) - float(base_v)))
                                scaled.append(
                                    (int(layer), int(pos), int(feature_idx), float(scaled_v))
                                )

                            logits = feature_intervention_next_token(
                                model,
                                user_prompt,
                                scaled,
                                freeze_attention=freeze_attention,
                                constrained_layers=constrained_layers,
                                apply_activation_function=apply_activation_function,
                            )
                            dist = distribution_from_logits(
                                model,
                                logits,
                                temperature=1.0,
                                top_p=None,
                                top_k=5,
                            )
                            # Compute exact prob for tok_id from logits.
                            probs = torch.softmax(logits, dim=-1)
                            p_tok = float(probs[int(tok_id)].item())
                            sweep_rows.append(
                                {
                                    "alpha": float(a),
                                    "p(token)": p_tok,
                                    "top1": dist["top"][0]["token"] if dist.get("top") else None,
                                    "top1_p": dist["top"][0]["prob"] if dist.get("top") else None,
                                }
                            )

                        sweep_df = pd.DataFrame(sweep_rows)
                        st.dataframe(sweep_df, width="stretch", height=220)
                        st.line_chart(sweep_df.set_index("alpha")["p(token)"], height=160)
                    except MemoryError as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        st.exception(exc)

    st.divider()
    st.markdown("**3) Generate (baseline)**")
    run_gen = st.button("Generate", width="stretch")
    if run_gen:
        with st.spinner("Generating…"):
            try:
                st.session_state.last_generation = generate_text(
                    model,
                    user_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            except MemoryError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.exception(exc)

    run_gen_int = st.button("Generate with interventions (slow)", width="stretch")
    if run_gen_int:
        with st.spinner("Generating with interventions…"):
            try:
                st.session_state.last_generation = generate_text_stepwise_with_interventions(
                    model,
                    user_prompt,
                    interventions,
                    max_new_tokens=min(max_new_tokens, 96),
                    temperature=temperature,
                    top_p=top_p,
                    freeze_attention=freeze_attention,
                    constrained_layers=constrained_layers,
                    apply_activation_function=apply_activation_function,
                )
            except MemoryError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.exception(exc)

    st.text_area(
        "Output", value=st.session_state.last_generation, height=380, label_visibility="collapsed"
    )

    st.divider()
    st.subheader("Export")
    st.download_button(
        "Download analysis JSON",
        data=_serialize_analysis(analysis),
        file_name="neural_dashboard_analysis.json",
        mime="application/json",
        width="stretch",
    )

    if st.session_state.validation is not None:
        st.download_button(
            "Download validation JSON",
            data=json.dumps(st.session_state.validation, ensure_ascii=False, indent=2),
            file_name="neural_dashboard_validation.json",
            mime="application/json",
            width="stretch",
        )

    if st.session_state.activation_validation is not None:
        st.download_button(
            "Download activation validation JSON",
            data=_serialize_activation_validation(st.session_state.activation_validation),
            file_name="neural_dashboard_activation_validation.json",
            mime="application/json",
            width="stretch",
        )

    if st.session_state.feature_inspection is not None:
        st.download_button(
            "Download feature inspection JSON",
            data=json.dumps(
                st.session_state.feature_inspection, ensure_ascii=False, indent=2, default=str
            ),
            file_name="neural_dashboard_feature_inspection.json",
            mime="application/json",
            width="stretch",
        )

    if (
        st.session_state.validation is not None
        or st.session_state.activation_validation is not None
        or st.session_state.feature_inspection is not None
    ):
        md = _render_markdown_report(
            analysis,
            validation=st.session_state.validation,
            activation_validation=st.session_state.activation_validation,
            feature_inspection=st.session_state.feature_inspection,
        )
        st.download_button(
            "Download demo report (Markdown)",
            data=md,
            file_name="neural_dashboard_report.md",
            mime="text/markdown",
            width="stretch",
        )
