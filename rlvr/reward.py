"""Verifiable reward for RLVR/GRPO on the SELF-CONTAINED (hardcoded-words) SLM.

The SLM emits `generate_crossword(topic="vocabulary", word_source=None, size=N)` whose
body begins `word_source = word_source or _WORDS` -- called with no word_source it uses
its OWN embedded word list and returns a layout dict. So the reward:

  1. runs it with word_source=[]  -> the model's OWN crossword (this is what's graded)
  2. runs it with an injected palette -> ONLY to catch memorization
  3. requires the two grids to DIFFER: a program that returns one fixed literal grid is
     insensitive to word_source (same grid both runs) -> memorization penalty.

Words are self-chosen, so "real word" is validated against a broad English dictionary
(data/wordlists/words_alpha.txt); educational quality is `vocab_fraction` vs the curated
purified vocabulary list (data/wordlists/WORD_LIST_FULLY_PURIFIED.txt).
No coverage term (there is no injected target vocabulary). All execution is sandboxed
(untrusted model code). Reward = (1-bw)*graded + bw*binary, then * memo_penalty if the
program failed the distinctness check.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.sandbox import run_candidate
from harness.scorer import Spec, score
from pipeline.eval_harness import build_palette, extract_code

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DICT_PATH = os.path.join(_ROOT, "data", "wordlists", "words_alpha.txt")
# "Educational vocab" reference for vocab_fraction: the curated purified word list
# (NOT the derived build_clean_education_source intersection).
_VOCAB_LIST_PATH = os.path.join(_ROOT, "data", "wordlists", "WORD_LIST_FULLY_PURIFIED.txt")

# Per-size constraints (density from spec_generator's NYT-shaped targets) and sandbox
# timeouts (the hardcoded generator has an internal ~21s deadline at size 15).
_DENSITY_BY_SIZE = {7: 0.80, 9: 0.76, 11: 0.74, 15: 0.72}
_TIME_BUDGET_BY_SIZE = {7: 5.0, 9: 8.0, 11: 15.0, 15: 25.0}
_TIMEOUT_BY_SIZE = {7: 10.0, 9: 14.0, 11: 22.0, 15: 35.0}


@dataclass
class RewardConfig:
    require_symmetry: bool = False        # symmetry curriculum: start off, anneal to True
    min_vocab_fraction: float = 0.70      # ">=X% real vocabulary" gate
    density_slack: float = 0.10           # binary black-square gate = density_target - slack
    min_word_len: int = 3
    binary_weight: float = 0.30
    memo_penalty: float = 0.30            # reward *= this when the two runs are NOT distinct
    floor_no_code: float = 0.0            # no parseable program
    floor_no_run: float = 0.05            # parsed but its own run failed to produce a grid
    max_workers: int = 8
    graded_weights: dict = field(default_factory=lambda: {
        "valid": 0.30, "density": 0.15, "vocab": 0.20, "crossings": 0.15,
        "clean_cross": 0.10, "clean_entry": 0.05, "quality": 0.05,
    })


@lru_cache(maxsize=1)
def get_palette() -> dict:
    """{word_source, vocab_set, scores, targets} — built once."""
    return build_palette()


@lru_cache(maxsize=1)
def get_dictionary() -> frozenset:
    """Broad English dictionary for the real-word check (model self-sources words)."""
    with open(_DICT_PATH, encoding="utf-8") as fh:
        return frozenset(w.strip().upper() for w in fh if w.strip())


@lru_cache(maxsize=1)
def get_vocab_set() -> frozenset:
    """Educational-vocab reference for vocab_fraction: the curated WORD_LIST_FULLY_PURIFIED."""
    with open(_VOCAB_LIST_PATH, encoding="utf-8") as fh:
        return frozenset(w.strip().upper() for w in fh if w.strip() and w.strip().isalpha())


def canonical_eff(size: int, cfg: RewardConfig) -> dict:
    size = int(size)
    return {
        "size": size,
        "require_symmetry": cfg.require_symmetry,
        "min_word_len": cfg.min_word_len,
        "time_budget_s": _TIME_BUDGET_BY_SIZE.get(size, 8.0),
        "density_target": _DENSITY_BY_SIZE.get(size, 0.74),
    }


def build_spec(eff: dict) -> Spec:
    # topic_words=() -> coverage is 1.0/ignored (no injected target vocabulary)
    return Spec(
        size=int(eff["size"]),
        topic_words=(),
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

    valid = _num(metrics, "valid")
    fill_density = _num(metrics, "fill_density")
    vocab_fraction = 1.0 - _num(metrics, "filler_fraction")
    crossings = _num(metrics, "crossings")
    inv_cross = _num(metrics, "invalid_crossing_frac")
    inv_entry = _num(metrics, "invalid_entry_frac")
    has_entries = _num(metrics, "n_entries") > 0
    has_cross = crossings > 0

    graded = {
        "valid": _clamp(valid),
        "density": _clamp(fill_density / dt) if dt else 0.0,
        "vocab": _clamp(vocab_fraction / cfg.min_vocab_fraction) if has_entries else 0.0,
        "crossings": _clamp(crossings / max(1, size)),
        "clean_cross": _clamp(1.0 - inv_cross) if has_cross else 0.0,
        "clean_entry": _clamp(1.0 - inv_entry) if has_entries else 0.0,
        "quality": _clamp(_num(metrics, "fill_quality")) if has_entries else 0.0,
    }
    w = cfg.graded_weights
    wsum = sum(w.values()) or 1.0
    r_graded = sum(w[k] * graded[k] for k in graded) / wsum

    binary = {
        "valid": 1.0 if valid >= 0.999 else 0.0,
        "no_bad_cross": 1.0 if (has_cross and inv_cross == 0.0) else 0.0,
        "no_bad_entry": 1.0 if (has_entries and inv_entry == 0.0) else 0.0,
        "vocab": 1.0 if (has_entries and vocab_fraction >= cfg.min_vocab_fraction) else 0.0,
        "density": 1.0 if fill_density >= dt - cfg.density_slack else 0.0,
        "crossings": 1.0 if crossings > 0 else 0.0,
    }
    r_binary = sum(binary.values()) / len(binary)

    reward = (1.0 - cfg.binary_weight) * r_graded + cfg.binary_weight * r_binary
    breakdown = {"reward": round(reward, 4), "r_graded": round(r_graded, 4),
                 "r_binary": round(r_binary, 4), "graded": graded, "binary": binary,
                 "vocab_fraction": round(vocab_fraction, 4)}
    return reward, breakdown


def _run(code: str, size: int, word_source, cfg: RewardConfig) -> dict:
    """Sandbox-run generate_crossword(topic, word_source, size). word_source=[] (falsy)
    makes the model fall back to its own embedded _WORDS."""
    return run_candidate(
        code,
        {"topic": "vocabulary", "word_source": word_source, "size": int(size), "seed": 0},
        timeout_s=_TIMEOUT_BY_SIZE.get(int(size), 15.0), mem_mb=1536,
    )


def _score_layout(layout, eff: dict, scorer_word_source, runtime_s) -> dict:
    # real-word check vs the broad dictionary; educational-vocab % vs the purified list
    return score(layout, build_spec(eff), word_source=scorer_word_source,
                 dictionary=get_dictionary(), runtime_s=runtime_s,
                 vocab_set=get_vocab_set())


def _answers(layout) -> frozenset:
    if not isinstance(layout, dict):
        return frozenset()
    out = set()
    for key in ("across", "down"):
        for e in layout.get(key, []) or []:
            try:
                out.add((int(e["row"]), int(e["col"]), str(e["answer"]).upper()))
            except (KeyError, TypeError, ValueError):
                pass
    return frozenset(out)


def reward_from_text(text: str, size: int, palette: dict, cfg: RewardConfig) -> tuple[float, dict]:
    """completion -> code -> (own-words run graded) with a memorization penalty."""
    code = extract_code(text)
    if not code:
        return cfg.floor_no_code, {"reward": cfg.floor_no_code, "reason": "no_code"}
    eff = canonical_eff(size, cfg)

    # 1) the model's OWN crossword (word_source=[] -> uses embedded _WORDS)
    runA = _run(code, size, [], cfg)
    if runA.get("status") != "ok":
        return cfg.floor_no_run, {"reward": cfg.floor_no_run, "reason": f"own_run:{runA.get('status')}"}
    metricsA = _score_layout(runA["result"], eff, [], runA.get("runtime_s"))
    reward, bd = compute_reward(metricsA, eff, cfg)

    # 2) memorization check: run with an injected palette; a literal-returner yields the
    #    SAME grid -> penalize. (If it crashes/adapts, it's not a constant -> no penalty.)
    runB = _run(code, size, palette["word_source"], cfg)
    memorized = runB.get("status") == "ok" and _answers(runB["result"]) == _answers(runA["result"])
    if memorized:
        reward *= cfg.memo_penalty
    bd.update({"reward": round(reward, 4), "distinct": not memorized, "memorized": memorized})
    return reward, bd


def evaluate_text(text: str, size: int, palette: dict, cfg: RewardConfig) -> dict:
    """Score one completion for EVAL. Returns the reward AND the underlying verifier
    metrics (valid, vocab_fraction, crossings, invalid_*, density, black squares,
    n_entries, memorized) as a flat record for aggregation. Same scoring path as the
    training reward, so eval numbers are comparable to what GRPO optimized."""
    zero = {"reward": 0.0, "size": int(size), "ran": 0, "valid": 0, "vocab_fraction": 0.0,
            "crossings": 0, "n_entries": 0, "invalid_crossing_frac": 0.0,
            "invalid_entry_frac": 0.0, "fill_density": 0.0, "black_squares": None,
            "black_target": None, "memorized": 0, "status": "no_code"}
    code = extract_code(text)
    if not code:
        return zero
    eff = canonical_eff(size, cfg)
    runA = _run(code, size, [], cfg)
    if runA.get("status") != "ok":
        zero.update({"reward": cfg.floor_no_run, "status": f"own_run:{runA.get('status')}"})
        return zero
    m = _score_layout(runA["result"], eff, [], runA.get("runtime_s"))
    reward, _ = compute_reward(m, eff, cfg)
    runB = _run(code, size, palette["word_source"], cfg)
    memorized = runB.get("status") == "ok" and _answers(runB["result"]) == _answers(runA["result"])
    if memorized:
        reward *= cfg.memo_penalty

    n = int(size) * int(size)
    white = round(_num(m, "fill_density") * n)
    black_target = int(round(n * (1.0 - eff["density_target"])))
    return {
        "reward": round(reward, 4), "size": int(size), "ran": 1, "status": "ok",
        "valid": int(_num(m, "valid") >= 0.999),
        "vocab_fraction": round(1.0 - _num(m, "filler_fraction"), 4),
        "crossings": int(_num(m, "crossings")),
        "n_entries": int(_num(m, "n_entries")),
        "invalid_crossing_frac": round(_num(m, "invalid_crossing_frac"), 4),
        "invalid_entry_frac": round(_num(m, "invalid_entry_frac"), 4),
        "fill_density": round(_num(m, "fill_density"), 4),
        "black_squares": n - white, "black_target": black_target,
        "memorized": int(memorized),
    }


def _completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        return last.get("content", "") if isinstance(last, dict) else str(last)
    return ""


def make_reward_fn(cfg: RewardConfig | None = None):
    """TRL GRPOTrainer-compatible reward: reward_fn(prompts, completions, **cols)->list[float].
    Reads the flat `size` column; scores the batch concurrently (subprocess-I/O bound)."""
    cfg = cfg or RewardConfig()
    palette = get_palette()
    get_dictionary(); get_vocab_set()  # warm caches once

    def reward_fn(prompts=None, completions=None, **cols):
        texts = [_completion_text(c) for c in (completions or [])]
        sizes = cols.get("size") or [7] * len(texts)

        def _one(i):
            return reward_from_text(texts[i], int(sizes[i]), palette, cfg)[0]

        if len(texts) <= 1 or cfg.max_workers <= 1:
            return [_one(i) for i in range(len(texts))]
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
            return list(ex.map(_one, range(len(texts))))

    reward_fn.__name__ = "crossword_verifiable_reward"
    return reward_fn
