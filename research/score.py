from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreResult:
    track: str
    primary_metric_name: str
    primary_metric_value: float
    score: float


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def score_summary(track: str, summary: dict) -> ScoreResult:
    t = track.strip().lower()
    if t == "gpt_text":
        metric_name = "best_val_bpc"
        metric_value = _as_float(summary.get(metric_name), default=float("inf"))
        return ScoreResult(track=t, primary_metric_name=metric_name, primary_metric_value=metric_value, score=-metric_value)

    if t == "proof_factoring":
        metric_name = str(summary.get("best_primary_metric_name", "solve_rate"))
        metric_value = _as_float(summary.get("best_primary_metric"), default=0.0)
        return ScoreResult(track=t, primary_metric_name=metric_name, primary_metric_value=metric_value, score=metric_value)

    if t == "proof_affine":
        metric_name = str(summary.get("best_primary_metric_name", "val_policy_loss"))
        metric_value = _as_float(summary.get("best_primary_metric"), default=float("inf"))
        base_score = -metric_value
        rollout = _as_float(summary.get("quick_rollout_success"), default=0.0)
        return ScoreResult(
            track=t,
            primary_metric_name=metric_name,
            primary_metric_value=metric_value,
            score=base_score + 1e-3 * rollout,
        )

    raise ValueError(f"Unknown track: {track}")


def is_better(candidate_score: float, best_score: float, min_delta: float = 1e-12) -> bool:
    return candidate_score > (best_score + min_delta)
