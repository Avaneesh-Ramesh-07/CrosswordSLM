"""Verifiable reward for RLVR/GRPO — turns one generated program into a scalar.

The policy emits a `generate_crossword(topic, word_source, size)` program from a
bare "make an NxN vocabulary crossword" request. We reward it by running that
program through the EXISTING sandbox + deterministic scorer (untrusted output ->
`in_process=False`) and folding the scorer's criteria into a composite reward:

    R = (1 - binary_weight) * R_graded + binary_weight * R_binary

- R_graded  : dense shaping so partly-valid grids still get signal (validity rate,
              density vs target, vocab fraction, coverage, crossings, clean
              crossings/entries, fill quality, runtime).
- R_binary  : the user's yes/no gates (valid? no invalid crossings? >=X% vocab?
              black squares within target? connected (subsumed by valid)?
              crossings>0?), averaged.

Everything reuses `pipeline/eval_harness.py` + `pipeline/oe_evaluator.py`; nothing
in the verifier is modified. Reward-time constraints come from a CANONICAL per-size
Spec (not the per-row effective_spec), because identical prompts must not earn
contradictory rewards under GRPO's within-group advantage normalization.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.scorer import Spec
from pipeline.eval_harness import build_palette, extract_code
from pipeline.oe_evaluator import evaluate_code

# Modal per-size constraints from the training corpus (build_dataset effective_spec).
# The model never sees these in the prompt, so the reward pins one canonical Spec
# per size instead of using the row's effective_spec.
_DENSITY_BY_SIZE = {7: 0.80, 9: 0.76, 11: 0.74}
_TIME_BUDGET_BY_SIZE = {7: 3.0, 9: 5.0, 11: 5.0}


@dataclass
class RewardConfig:
    n_draws: int = 2                      # >=2 punishes lucky-seed nondeterminism
    cap: int | None = None                # None=full palette; int=subsample (word anti-hardcode)
    require_symmetry: bool = False        # symmetry curriculum: start off, anneal to True
    min_vocab_fraction: float = 0.70      # ">=X% real vocabulary" gate
    coverage_target: float = 0.15         # SAT-target coverage is hard; modest, tunable
    density_slack: float = 0.10           # binary black-square gate = density_target - slack
                                          # (density_target is aspirational/soft, not a hard floor)
    min_word_len: int = 3
    binary_weight: float = 0.30           # R = (1-bw)*graded + bw*binary
    floor_no_code: float = 0.0            # no parseable program
    floor_no_run: float = 0.05            # parsed but every draw failed to run
    max_workers: int = 8                  # reward is subprocess-I/O bound -> parallelize a batch
    graded_weights: dict = field(default_factory=lambda: {
        "valid": 0.25, "density": 0.15, "vocab": 0.15, "coverage": 0.10,
        "crossings": 0.10, "clean_cross": 0.10, "clean_entry": 0.05,
        "quality": 0.05, "runtime": 0.05,
    })


@lru_cache(maxsize=1)
def get_palette() -> dict:
    """{word_source, vocab_set, scores, targets} — spec-independent, built once."""
    return build_palette()


def canonical_eff(size: int, cfg: RewardConfig) -> dict:
    size = int(size)
    return {
        "size": size,
        "require_symmetry": cfg.require_symmetry,
        "min_word_len": cfg.min_word_len,
        "time_budget_s": _TIME_BUDGET_BY_SIZE.get(size, 5.0),
        "density_target": _DENSITY_BY_SIZE.get(size, 0.72),
        "topic": "vocabulary",
    }


def build_spec(eff: dict, palette: dict) -> Spec:
    return Spec(
        size=int(eff["size"]),
        topic_words=tuple(palette["targets"]),
        require_symmetry=eff["require_symmetry"],
        min_word_len=eff["min_word_len"],
        time_budget_s=eff["time_budget_s"],
        density_target=eff["density_target"],
    )


def _clamp(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def _num(m: dict, key: str) -> float:
    v = m.get(key)
    return float(v) if v is not None else 0.0


def compute_reward(metrics: dict, eff: dict, cfg: RewardConfig) -> tuple[float, dict]:
    """Fold scorer metrics into the composite reward. Returns (reward, breakdown)."""
    dt = eff["density_target"]
    size = eff["size"]

    valid = _num(metrics, "valid")                    # validity RATE across draws (0..1)
    fill_density = _num(metrics, "fill_density")
    vocab_fraction = 1.0 - _num(metrics, "filler_fraction")
    coverage = _num(metrics, "coverage")
    crossings = _num(metrics, "crossings")
    inv_cross = _num(metrics, "invalid_crossing_frac")
    inv_entry = _num(metrics, "invalid_entry_frac")
    # Gate "absence of bad" credits on the grid actually having content, so an empty
    # / near-empty grid can't farm reward from "no filler, no invalid crossings, fast
    # runtime" (a reward-hacking hole the dryrun surfaced).
    has_entries = _num(metrics, "n_entries") > 0
    has_cross = crossings > 0

    graded = {
        "valid": _clamp(valid),
        "density": _clamp(fill_density / dt) if dt else 0.0,
        "vocab": _clamp(vocab_fraction / cfg.min_vocab_fraction) if has_entries else 0.0,
        "coverage": _clamp(coverage / cfg.coverage_target) if cfg.coverage_target else 0.0,
        "crossings": _clamp(crossings / max(1, size)),
        "clean_cross": _clamp(1.0 - inv_cross) if has_cross else 0.0,
        "clean_entry": _clamp(1.0 - inv_entry) if has_entries else 0.0,
        "quality": _clamp(_num(metrics, "fill_quality")) if has_entries else 0.0,
        "runtime": _clamp(_num(metrics, "runtime_ok")) if has_entries else 0.0,
    }
    w = cfg.graded_weights
    wsum = sum(w.values()) or 1.0
    r_graded = sum(w[k] * graded[k] for k in graded) / wsum

    binary = {
        "valid": 1.0 if valid >= 0.999 else 0.0,
        "no_bad_cross": 1.0 if (has_cross and inv_cross == 0.0) else 0.0,
        "no_bad_entry": 1.0 if (has_entries and inv_entry == 0.0) else 0.0,
        "vocab": 1.0 if (has_entries and vocab_fraction >= cfg.min_vocab_fraction) else 0.0,
        "density": 1.0 if fill_density >= dt - cfg.density_slack else 0.0,  # not too many black squares
        "coverage": 1.0 if coverage >= cfg.coverage_target else 0.0,
        "crossings": 1.0 if crossings > 0 else 0.0,
    }
    r_binary = sum(binary.values()) / len(binary)

    reward = (1.0 - cfg.binary_weight) * r_graded + cfg.binary_weight * r_binary
    breakdown = {"reward": round(reward, 4), "r_graded": round(r_graded, 4),
                 "r_binary": round(r_binary, 4), "graded": graded, "binary": binary,
                 "vocab_fraction": round(vocab_fraction, 4)}
    return reward, breakdown


def score_code_multi(code: str, eff: dict, palette: dict, cfg: RewardConfig) -> dict:
    """Sandbox-run + score a program across n_draws. Returns evaluate_code output."""
    spec = build_spec(eff, palette)
    return evaluate_code(
        code, spec,
        word_source=palette["word_source"],
        scores=palette["scores"],
        n_draws=cfg.n_draws,
        cap=cfg.cap,
        vocab_set=palette["vocab_set"],
        quality_penalty=False,     # reward composes penalties itself
        in_process=False,          # UNTRUSTED model output -> subprocess sandbox
    )


def reward_from_text(text: str, size: int, palette: dict, cfg: RewardConfig) -> tuple[float, dict]:
    """Full path: completion text -> code -> sandbox -> scorer -> composite reward."""
    code = extract_code(text)
    if not code:
        return cfg.floor_no_code, {"reward": cfg.floor_no_code, "reason": "no_code"}
    eff = canonical_eff(size, cfg)
    try:
        out = score_code_multi(code, eff, palette, cfg)
    except Exception as exc:  # sandbox/scoring blew up -> treat as non-run
        return cfg.floor_no_run, {"reward": cfg.floor_no_run, "reason": f"error:{exc}"[:200]}
    ran_any = any(r.get("status") == "ok" for r in out["fuzz"]["results"])
    if not ran_any:
        return cfg.floor_no_run, {"reward": cfg.floor_no_run, "reason": "no_run",
                                  "statuses": [r.get("status") for r in out["fuzz"]["results"]]}
    return compute_reward(out["metrics"], eff, cfg)


def _completion_text(completion) -> str:
    """GRPO passes a str (standard) or a list of message dicts (conversational)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        return last.get("content", "") if isinstance(last, dict) else str(last)
    return ""


def make_reward_fn(cfg: RewardConfig | None = None):
    """Build a TRL GRPOTrainer-compatible reward function.

    Signature: reward_fn(prompts, completions, **cols) -> list[float]. The dataset's
    flat `size` column arrives in `cols`; the canonical Spec is derived from it. The
    batch is scored concurrently since each reward is subprocess-I/O bound.
    """
    cfg = cfg or RewardConfig()
    palette = get_palette()

    def reward_fn(prompts=None, completions=None, **cols):
        texts = [_completion_text(c) for c in (completions or [])]
        sizes = cols.get("size") or [7] * len(texts)

        def _one(i):
            r, _ = reward_from_text(texts[i], int(sizes[i]), palette, cfg)
            return r

        if len(texts) <= 1 or cfg.max_workers <= 1:
            return [_one(i) for i in range(len(texts))]
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
            return list(ex.map(_one, range(len(texts))))

    reward_fn.__name__ = "crossword_verifiable_reward"
    return reward_fn
