"""Word source for crossword generation.

Loads the Collaborative Word List (`data/collaborative-word-list/xwordlist.dict`,
format `WORD;score`, score 0-100) and exposes it as a scored fill dictionary
plus helpers a CSP filler needs (indexed by length).

At inference the model is *given* a `word_source` (it never selects words). A
`word_source` is two-tier: prioritized topic words + this scored fill list.
`Spec.topic_words` (the prioritized subset) drives the scorer's coverage metric.
"""

from __future__ import annotations

import os

_DEFAULT_DICT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "collaborative-word-list", "xwordlist.dict",
)


def load_scored_dict(path=_DEFAULT_DICT, min_score=55, min_len=3, max_len=21, alpha_only=True) -> dict:
    """Return {WORD: score} filtered by score, length, and (optionally) alpha-only.

    min_score=55 raises the bar above the Collaborative List's "acceptable"
    threshold (50) to keep the worst crosswordese out of the fill palette
    entirely (e.g. RCADOME=52, UIE/NUH/LKT=40 all drop). fill_quality then
    rewards using the better words among what remains.
    alpha_only drops entries with digits/punctuation (e.g. '100TH', '0AD').
    """
    scored: dict = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or ";" not in line:
                continue
            word, _, score = line.rpartition(";")
            try:
                sc = int(score)
            except ValueError:
                continue
            if sc < min_score or not (min_len <= len(word) <= max_len):
                continue
            word = word.upper()
            if alpha_only and not word.isalpha():
                continue
            scored[word] = sc
    return scored


def index_by_length(scored: dict) -> dict:
    """Return {length: [(word, score), ...]} sorted by score descending.

    This is the structure a fixed-grid CSP filler queries per slot length.
    """
    idx: dict = {}
    for word, sc in scored.items():
        idx.setdefault(len(word), []).append((word, sc))
    for length in idx:
        idx[length].sort(key=lambda ws: -ws[1])
    return idx


def build_word_source(topic_words, fill_dict: dict) -> list:
    """Combine prioritized topic words with the fill dictionary -> allowed words.

    Returns a sorted, de-duplicated, uppercased list of allowed words. The topic
    words are what `Spec.topic_words` should reference for coverage scoring.
    """
    words = {str(w).upper() for w in topic_words if str(w).isalpha()}
    words |= set(fill_dict.keys())
    return sorted(words)


# --- curated education word source (Insight #3) -----------------------------
#
# The Collaborative List scores CROSSWORD desirability, not learnability, and is
# full of proper nouns / abbreviations / phrases (RCADOME, LKT, LETITBE). For a
# vocabulary crossword we want a fill palette of *learnable, gettable words* and
# a set of *target vocabulary* to prioritize. We build that by intersecting three
# signals:
#
#   1. Collaborative List (score)      -> crossword-quality signal (already have)
#   2. Common-English frequency list   -> drops crosswordese/proper nouns; keeps
#                                         words a student would actually meet
#   3. SAT / academic vocabulary list  -> the words we WANT the puzzle to teach
#                                         (become Spec.topic_words for coverage)
#
# Data files to drop in (one word per line, or `WORD;...`), then this "just works":
#   data/wordlists/common_english.txt  e.g. github `first20hours/google-10000-english`
#                                      (public domain) or the `wordfreq` top-N export
#   data/wordlists/sat_words.txt       a published SAT list or the Academic Word
#                                      List (Coxhead AWL, ~570 families)
# Until present, it falls back to the plain scored dict and reports what it used.
#
# NOTE: true topic->words retrieval (e.g. "space" -> COMET, ORBIT) needs a
# semantic map (embeddings or a category dataset) and is a separate component;
# `topic` is accepted here but not yet used for filtering.

_COMMON_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "wordlists", "common_english.txt")
_SAT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "wordlists", "sat_words.txt")
_DICT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "wordlists", "words_alpha.txt")

# Proper nouns / acronyms that slip through the dictionary word list (they happen
# to appear in words_alpha) but are not vocabulary. Extend as leaks are spotted.
_NOT_VOCAB = {"CIA", "HORATIO", "NIELSEN", "FBI", "NASA", "NATO", "ROC"}


def _load_wordset(path):
    """Read a word list (one per line, or `WORD;...`) -> set of uppercase alpha words, or None."""
    if not path or not os.path.exists(path):
        return None
    out = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            token = line.strip().split(";")[0].strip().upper()
            if token.isalpha():
                out.add(token)
    return out


def build_education_source(min_score=55, min_len=3, max_len=15, sat_path=_SAT_PATH,
                           include_common_fill=False, common_path=_COMMON_PATH,
                           big_fill_per_len=None):
    """Vocabulary crossword source (Insight #3).

    By DEFAULT the palette is the PURE INTERSECTION of high-scoring crossword
    words (Collaborative List, score >= min_score) and SAT/high-school vocabulary:
    every word is both a real SAT-level vocabulary word AND rated good crossword
    fill. Maximally vocabulary-dense; `targets` == `allowed`.

    Set include_common_fill=True to also add common high-score crossword words as
    structural CONNECTORS (SAT lists have almost no 3-4 letter words, so pure-
    intersection grids are very hard to fill). `targets` always stays the
    intersection vocabulary, and `coverage` rewards placing it.

    Set big_fill_per_len=N for LARGE grids (11x11+): fill becomes the top-N
    highest-scored crossword words per length from the full scored list. This is
    the only tier with enough long words (9-11 letters) to fill a big grid; the
    common-English list (google-10000) is almost all short words. `targets` still
    stays the SAT vocabulary. Takes precedence over include_common_fill.
    """
    scored = load_scored_dict(min_score=min_score, min_len=min_len, max_len=max_len)
    sat = _load_wordset(sat_path)

    if sat:
        vocab = {w: scored[w] for w in sat if w in scored and min_len <= len(w) <= max_len}
    else:
        vocab = dict(scored)  # fallback: no SAT list -> plain scored dict

    allowed = dict(vocab)
    fill = {}
    used_big = False
    if big_fill_per_len:
        used_big = True
        for length, lst in index_by_length(scored).items():
            for word, sc in lst[:big_fill_per_len]:
                if word not in vocab:
                    fill[word] = sc
        allowed.update(fill)
    elif include_common_fill:
        common = _load_wordset(common_path)
        if common:
            fill = {w: s for w, s in scored.items() if w in common and w not in vocab}
            allowed.update(fill)

    return {
        "allowed": sorted(allowed),
        "scores": allowed,
        "targets": sorted(vocab),        # the SAT n crossword vocabulary to teach
        "fill_words": sorted(fill),      # empty unless a fill tier is enabled
        "n_allowed": len(allowed),
        "n_vocab": len(vocab),
        "n_fill": len(fill),
        "used_sat": sat is not None,
        "used_common": bool(fill) and not used_big,
        "used_big_fill": used_big,
    }


def build_clean_education_source(freq_n=50000, min_score=55, min_len=3, max_len=11,
                                 sat_path=_SAT_PATH, dict_path=_DICT_PATH):
    """Broadened CLEAN educational palette (the ">70% vocab n crossword-worthy" path).

    Every word in the palette is simultaneously:
      * a real dictionary word (words_alpha) -> no acronyms, ~no proper nouns
      * gettable: in the wordfreq top-`freq_n` English words, OR a SAT vocab word
      * crossword-worthy: Collaborative-List score >= min_score

    A grid filled ONLY from this palette is therefore ~100% "vocabulary that is also
    crossword-worthy" by construction (the crosswordese/abbreviation junk that a
    dense grid otherwise forces into short slots is gone). The strict SAT vocabulary
    stays as `targets` -- a tracked coverage sub-metric, not the acceptance gate.

    Unlike the pure/common education sources, this one is large enough at every
    length (incl. 9-11) to actually fill a dense 11x11. Requires the `wordfreq`
    package and the cached dictionary at dict_path (see scratchpad/fetch_dict.py).
    """
    import wordfreq

    scored = load_scored_dict(min_score=min_score, min_len=min_len, max_len=max_len)
    dict_words = _load_wordset(dict_path) or set()
    freq = {w.upper() for w in wordfreq.top_n_list("en", freq_n) if w.isalpha()}
    sat = _load_wordset(sat_path) or set()

    clean = {
        w: s for w, s in scored.items()
        if w in dict_words and w not in _NOT_VOCAB and (w in freq or w in sat)
    }
    vocab = {w: s for w, s in clean.items() if w in sat}
    return {
        "allowed": sorted(clean),
        "scores": clean,
        "clean_set": set(clean),          # THE vocab n crossword set (>70% criterion)
        "targets": sorted(vocab),         # strict SAT vocabulary (coverage sub-metric)
        "fill_words": sorted(set(clean) - set(vocab)),
        "n_allowed": len(clean),
        "n_vocab": len(vocab),
        "used_dict": bool(dict_words),
    }


if __name__ == "__main__":
    d = load_scored_dict()
    idx = index_by_length(d)
    print(f"loaded {len(d):,} alpha words (min_score=55, len 3-21)")
    print("count by length:")
    for L in range(3, 16):
        words = idx.get(L, [])
        top = words[0][0] if words else "-"
        print(f"  len {L:2d}: {len(words):>7,}   top-scored e.g. {top}")

    print("\neducation source (PURE intersection: SAT n high-score crossword):")
    edu = build_education_source()  # pure intersection (default)
    print(f"  palette = vocabulary targets: {edu['n_vocab']:,}  (allowed == targets: "
          f"{edu['n_allowed'] == edu['n_vocab']})")
    print(f"  used_sat={edu['used_sat']} include_common_fill=False")
    ev = index_by_length(edu["scores"])
    print("  by length:", {L: len(ev.get(L, [])) for L in range(3, 11)})
    print("  (set include_common_fill=True to add common connectors if grids won't fill)")
