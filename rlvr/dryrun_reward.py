"""Phase-A local proof that the reward is wired correctly (no GPU needed).

Exercises the full path (extract_code -> sandbox subprocess -> scorer -> composite)
and asserts the ordering GOOD >> DEGENERATE >> non-parseable:
  - GOOD       : verified solutions from rlvr/dataset/train.jsonl (should score high).
  - DEGENERATE : synthetic broken programs (empty grid, no crossings, error, junk
                 words) -- the true low-reward cases the policy must be steered away
                 from (should score low / hit the no-run floor).
  - JUNK       : non-parseable text (must hit floor_no_code == 0).
  - NEGATIVES  : rlvr/dataset/negatives.jsonl, shown for INFO only (no assert): these
                 are near-misses that failed their ORIGINAL (often symmetry-required)
                 spec, so under the relaxed canonical RLVR Spec many are legitimately
                 decent grids -- they are not a clean "bad" proxy.

Run from the repo root:  python rlvr/dryrun_reward.py [--n 5]
Best on Linux/WSL (the sandbox mem cap is Linux-only); Windows works for the
wiring proof (cap skipped, numbers otherwise identical).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlvr.reward import RewardConfig, get_palette, reward_from_text

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRAIN = os.path.join(_ROOT, "rlvr", "dataset", "train.jsonl")
_NEG = os.path.join(_ROOT, "rlvr", "dataset", "negatives.jsonl")
CONSTRUCT_SIZES = (7, 9)


def _rows(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def good_samples(n):
    out = []
    for r in _rows(_TRAIN):
        m = r.get("meta") or {}
        eff = m.get("effective_spec") or {}
        if m.get("kind") == "fixed_template" or eff.get("size") not in CONSTRUCT_SIZES:
            continue
        out.append((r["messages"][2]["content"], int(eff["size"]), m.get("spec_id")))
        if len(out) >= n:
            break
    return out


def negatives_samples(n):
    out = []
    for r in _rows(_NEG):
        eff = r.get("effective_spec") or {}
        size = eff.get("size", 7)
        if size not in CONSTRUCT_SIZES or not r.get("code"):
            continue
        out.append((r["code"], int(size), r.get("spec_id")))
        if len(out) >= n:
            break
    return out


# Genuinely degenerate programs -- the low-reward cases the policy must avoid.
_EMPTY = (
    "def generate_crossword(topic, word_source, size):\n"
    "    return {'rows': size, 'cols': size, 'cells': [], 'across': [], 'down': []}\n")
_NO_CROSS = (
    "def generate_crossword(topic, word_source, size):\n"
    "    ws = word_source if not isinstance(word_source, dict) else "
    "list(word_source.get('theme', [])) + list(word_source.get('fill', []))\n"
    "    ws = [str(w).upper() for w in ws if str(w).isalpha() and len(str(w)) == size]\n"
    "    w = ws[0] if ws else 'A' * size\n"
    "    cells = [{'r': 0, 'c': c, 'letter': w[c]} for c in range(size)]\n"
    "    across = [{'number': 1, 'row': 0, 'col': 0, 'answer': w, 'len': size}]\n"
    "    return {'rows': size, 'cols': size, 'cells': cells, 'across': across, 'down': []}\n")
_ERROR = (
    "def generate_crossword(topic, word_source, size):\n"
    "    raise ValueError('boom')\n")
_JUNK_WORDS = (
    "def generate_crossword(topic, word_source, size):\n"
    "    w = 'ZQXJVK' * size\n"
    "    w = w[:size]\n"
    "    cells = [{'r': 0, 'c': c, 'letter': w[c]} for c in range(size)]\n"
    "    across = [{'number': 1, 'row': 0, 'col': 0, 'answer': w, 'len': size}]\n"
    "    return {'rows': size, 'cols': size, 'cells': cells, 'across': across, 'down': []}\n")
DEGENERATE = [(_EMPTY, 7, "empty"), (_NO_CROSS, 7, "no_cross"),
              (_ERROR, 7, "error"), (_JUNK_WORDS, 7, "junk_words")]


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _score_bucket(label, samples, palette, cfg):
    print(f"== {label} ==")
    rs = []
    for text, size, sid in samples:
        r, bd = reward_from_text(text, size, palette, cfg)
        rs.append(r)
        extra = bd.get("reason") or f"g={bd.get('r_graded')} b={bd.get('r_binary')} vocab={bd.get('vocab_fraction')}"
        print(f"  {sid!s:<12} size{size}  reward={r:.3f}  {extra}")
    print(f"  mean={_mean(rs):.3f}\n")
    return rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="samples per bucket")
    ap.add_argument("--n-draws", type=int, default=2)
    args = ap.parse_args()

    cfg = RewardConfig(n_draws=args.n_draws)
    palette = get_palette()
    print(f"palette: {len(palette['targets'])} targets, {len(palette['vocab_set'])} vocab; "
          f"n_draws={cfg.n_draws}\n")

    good = _score_bucket("GOOD (verified solutions)", good_samples(args.n), palette, cfg)
    degen = _score_bucket("DEGENERATE (synthetic)", DEGENERATE, palette, cfg)
    junk = _score_bucket("JUNK (non-parseable)",
                         [("I can't help with that.", 7, "junk1"),
                          ("Here is my plan, but no code follows.", 9, "junk2")], palette, cfg)
    _score_bucket("NEGATIVES (info only, near-misses)", negatives_samples(args.n), palette, cfg)

    g, d, j = _mean(good), _mean(degen), _mean(junk)
    error_r = next((reward_from_text(_ERROR, 7, palette, cfg)[0] for _ in [0]))
    print(f"SUMMARY  good={g:.3f}  degenerate={d:.3f}  junk={j:.3f}")

    checks = [
        ("good > degenerate", g > d),
        ("good >= 0.30", g >= 0.30),
        ("degenerate <= 0.50", d <= 0.50),
        ("junk == floor_no_code (0.0)", abs(j - cfg.floor_no_code) < 1e-9),
        ("error hits no-run floor", abs(error_r - cfg.floor_no_run) < 1e-9),
    ]
    ok = True
    for name, cond in checks:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and cond
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
