from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal, cast

import torch

from circuit_tracer import ReplacementModel
from engine import (
    FeatureRef,
    activation_change_report,
    analyze_prompt,
    distribution_from_logits,
    feature_intervention_next_token,
    next_token_distribution,
)


DEMO_PRESETS: dict[str, str] = {
    "capitale_fr": "La capitale de la France est",
    "antonymes_fr": 'Le contraire de "petit" est "',
    "addition": "calc: 36+59=",
}

BackendName = Literal["transformerlens", "nnsight"]
OffloadMode = Literal["auto", "cpu", "disk"]


def _default_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        try:
            major, _minor = torch.cuda.get_device_capability()
            if major >= 8:
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32


def _load_model(
    *,
    model_name: str,
    transcoder_set: str,
    backend: BackendName,
    dtype: torch.dtype | None,
) -> Any:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _default_dtype() if dtype is None else dtype
    model = ReplacementModel.from_pretrained(
        model_name=model_name,
        transcoder_set=transcoder_set,
        backend=backend,
        device=device,
        dtype=dtype,
    )
    model.eval()
    return model


def _pick_target_logit_index(
    logits: list[dict[str, Any]],
    *,
    target_token_substring: str | None,
) -> int:
    if not logits:
        return 0
    if target_token_substring:
        needle = target_token_substring
        for r in logits:
            if needle in str(r.get("token", "")):
                return int(r["logit_index"])
    return 0


def _render_md(
    *,
    prompt: str,
    analysis: dict[str, Any],
    validation: dict[str, Any],
    activation_validation: dict[str, Any] | None,
) -> str:
    lines = [
        "# circuit-tracer CLI report",
        "",
        "## Prompt",
        "```text",
        prompt.strip(),
        "```",
        "",
        "## Target",
        f"- `target_logit_index`: `{analysis.get('target_logit_index')}`",
        "",
        "## Next-token validation",
        f"- `entropy_before`: `{validation['before'].get('entropy')}`",
        f"- `entropy_after`: `{validation['after'].get('entropy')}`",
        f"- `margin_before`: `{validation['before'].get('margin_top1_top2')}`",
        f"- `margin_after`: `{validation['after'].get('margin_top1_top2')}`",
        "",
        "### Top surface groups (before)",
        "```json",
        json.dumps(validation["before"].get("top_groups", []), ensure_ascii=False, indent=2),
        "```",
        "",
        "### Top surface groups (after)",
        "```json",
        json.dumps(validation["after"].get("top_groups", []), ensure_ascii=False, indent=2),
        "```",
        "",
        "### Top logits (before)",
        "```json",
        json.dumps(validation["before"].get("top", []), ensure_ascii=False, indent=2),
        "```",
        "",
        "### Top logits (after)",
        "```json",
        json.dumps(validation["after"].get("top", []), ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    if activation_validation is not None:
        rows = activation_validation.get("rows") or []
        lines += [
            "## Activation validation (top deltas)",
            "```json",
            json.dumps(
                [
                    {
                        "feature": f"L{r['feature'].layer} P{r['feature'].pos} #{r['feature'].feature_idx}",
                        "base": r["base"],
                        "new": r["new"],
                        "delta": r["delta"],
                    }
                    for r in rows[:30]
                ],
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
        ]
    return "\n".join(lines).strip() + "\n"


def run_one(
    model: Any,
    *,
    prompt: str,
    out_dir: Path,
    slug: str,
    max_feature_nodes: int,
    max_n_logits: int,
    desired_logit_prob: float,
    batch_size: int,
    offload: OffloadMode | None,
    node_threshold: float,
    edge_threshold: float,
    target_token_substring: str | None,
    n_track: int,
) -> None:
    analysis = analyze_prompt(
        model,
        prompt,
        max_feature_nodes=max_feature_nodes,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
        batch_size=batch_size,
        offload=offload,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
        target_logit_index=0,
        return_graph=True,
    )

    target_logit_index = _pick_target_logit_index(
        analysis.get("logits", []),
        target_token_substring=target_token_substring,
    )

    # Refresh summaries for the chosen target (without re-running attribution) when possible.
    graph = analysis.get("graph")
    if graph is not None:
        from engine import summarize_graph_for_target

        analysis.update(
            summarize_graph_for_target(
                model,
                graph,
                target_logit_index=target_logit_index,
                node_threshold=node_threshold,
                edge_threshold=edge_threshold,
            )
        )

    feat_rows = analysis.get("feature_rows") or []
    interventions: list[tuple[int, int, int, float]] = []
    if feat_rows:
        top_feat: FeatureRef = feat_rows[0]["feature"]
        interventions = [(top_feat.layer, top_feat.pos, top_feat.feature_idx, 0.0)]

    before = next_token_distribution(model, prompt, temperature=1.0, top_p=None, top_k=20)
    if interventions:
        after_logits = feature_intervention_next_token(
            model,
            prompt,
            interventions,
            freeze_attention=True,
            constrained_layers=None,
            apply_activation_function=True,
        )
        after = distribution_from_logits(model, after_logits, temperature=1.0, top_p=None, top_k=20)
    else:
        after = before

    validation = {"before": before, "after": after, "interventions": interventions}

    activation_validation = None
    if interventions and feat_rows:
        tracked = [r["feature"] for r in feat_rows[: int(n_track)]]
        activation_validation = activation_change_report(
            model,
            prompt,
            interventions,
            tracked,
            freeze_attention=True,
            constrained_layers=None,
            apply_activation_function=True,
            sparse=True,
            baseline_match_freeze=False,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{slug}.analysis.json").write_text(
        json.dumps(
            {
                "user_prompt": analysis.get("user_prompt"),
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
                        "activation": float(r.get("activation", 0.0)),
                        "influence": float(r.get("influence", 0.0)),
                        "kept": bool(r.get("kept", False)),
                    }
                    for r in (analysis.get("feature_rows") or [])
                ],
                "edges": analysis.get("edges"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / f"{slug}.validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if activation_validation is not None:
        # Serialize FeatureRef objects.
        act_rows = activation_validation.get("rows") or []
        activation_validation_export = {
            **{k: v for k, v in activation_validation.items() if k != "rows"},
            "rows": [
                {
                    "feature": {
                        "layer": int(r["feature"].layer),
                        "pos": int(r["feature"].pos),
                        "feature_idx": int(r["feature"].feature_idx),
                    },
                    "base": r["base"],
                    "new": r["new"],
                    "delta": r["delta"],
                    "pct_delta": r["pct_delta"],
                }
                for r in act_rows
            ],
        }
        (out_dir / f"{slug}.activations.json").write_text(
            json.dumps(activation_validation_export, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    md = _render_md(
        prompt=prompt,
        analysis=analysis,
        validation=validation,
        activation_validation=activation_validation,
    )
    (out_dir / f"{slug}.report.md").write_text(md, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Run circuit-tracer demo workflows from the CLI.")
    p.add_argument("--model", default="google/gemma-2-2b-it")
    p.add_argument("--transcoder-set", default="gemma")
    p.add_argument("--backend", default="transformerlens", choices=["transformerlens", "nnsight"])
    p.add_argument("--out", default="demo_reports")

    p.add_argument("--preset", choices=sorted(DEMO_PRESETS.keys()))
    p.add_argument("--prompt", help="Custom prompt (overrides --preset).")
    p.add_argument("--slug", default="demo")
    p.add_argument("--target-token-substring", default=None)

    p.add_argument("--max-feature-nodes", type=int, default=2500)
    p.add_argument("--max-n-logits", type=int, default=10)
    p.add_argument("--desired-logit-prob", type=float, default=0.95)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--offload", default="auto", choices=["auto", "cpu", "disk", "none"])
    p.add_argument("--node-threshold", type=float, default=0.8)
    p.add_argument("--edge-threshold", type=float, default=0.98)
    p.add_argument(
        "--track", type=int, default=40, help="Tracked features for activation validation."
    )

    args = p.parse_args()

    if args.prompt:
        prompt = args.prompt
    elif args.preset:
        prompt = DEMO_PRESETS[args.preset]
        if args.slug == "demo":
            args.slug = args.preset
    else:
        raise SystemExit("Provide --prompt or --preset.")

    offload: OffloadMode | None
    if args.offload == "none":
        offload = None
    else:
        offload = cast(OffloadMode, args.offload)

    model = _load_model(
        model_name=args.model,
        transcoder_set=args.transcoder_set,
        backend=cast(BackendName, args.backend),
        dtype=None,
    )

    run_one(
        model,
        prompt=prompt,
        out_dir=Path(args.out),
        slug=args.slug,
        max_feature_nodes=args.max_feature_nodes,
        max_n_logits=args.max_n_logits,
        desired_logit_prob=args.desired_logit_prob,
        batch_size=args.batch_size,
        offload=offload,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        target_token_substring=args.target_token_substring,
        n_track=args.track,
    )

    print(f"Wrote demo artifacts to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
