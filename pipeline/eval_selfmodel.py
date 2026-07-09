"""Self-model crossword eval (English or Spanish).

Scores a submitted generate_crossword(topic, word_source, size) program -- e.g. one
written by a Claude Code agent acting as the "model" -- against held-out specs across
several grid sizes. Beyond structural validity it reports the graded metrics AND a
REAL DICTIONARY check on every entry, because palette membership is not enough:
raw frequency lists contain acronyms/proper nouns (AMLO, ONU, MISSISSIPPI...), so a
crossword can be "valid" against a dirty list yet full of non-words.

  English : palette = build_clean_education_source; dictionary = data/wordlists/words_alpha.txt
  Spanish : palette = wordfreq('es') INTERSECT pyspellchecker('es') dictionary (accent-normalized)

The submitted program is run in a timeout-protected in-process sandbox. It should be
self-contained (stdlib only) and use ONLY the word_source passed in.

    python pipeline/eval_selfmodel.py --lang en --submission submission_en.py
    python pipeline/eval_selfmodel.py --lang es --submission submission_es.py --sizes 7,9,11
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import unicodedata

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.sandbox import run_candidate_inprocess
from harness.scorer import Spec, score
from pipeline.word_source import build_clean_education_source

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUDGET = {5: 2.0, 7: 3.0, 9: 5.0, 11: 12.0, 13: 20.0, 15: 30.0}
EN_TOPICS = ["vocabulary", "words", "study", "learn", "review", "practice", "school", "language"]
ES_TOPICS = ["vocabulario", "palabras", "estudio", "aprender", "repaso", "escuela", "idioma", "examen"]
TOPICS = {"en": EN_TOPICS, "es": ES_TOPICS}
# small, non-exhaustive profanity stoplist so an educational ES palette stays classroom-safe
_ES_STOP = {"MIERDA", "PUTA", "PUTO", "JODER", "CONO", "POLLA", "CABRON", "MARICON",
            "EYACULACION", "PENE", "CULO", "TETAS", "COJONES"}


def _norm(w):
    w = unicodedata.normalize("NFKD", w)
    return "".join(c for c in w if not unicodedata.combining(c)).upper()


def english_palette(max_len):
    edu = build_clean_education_source(max_len=max_len)
    dpath = os.path.join(_ROOT, "data", "wordlists", "words_alpha.txt")
    if os.path.exists(dpath):
        DICT = {line.strip().upper() for line in open(dpath, encoding="utf-8") if line.strip()}
    else:
        DICT = set(edu["clean_set"])
        print("  WARN: words_alpha.txt missing; dictionary check falls back to palette")
    return {"ws": {"theme": edu["targets"], "fill": edu["fill_words"]},
            "allowed": edu["allowed"], "clean_set": edu["clean_set"],
            "targets": edu["targets"], "DICT": DICT}


def spanish_palette(max_len, freq_n=60000, min_len=3):
    try:
        import wordfreq
        from spellchecker import SpellChecker
    except ImportError as e:
        sys.exit(f"Spanish eval needs wordfreq + pyspellchecker: pip install pyspellchecker ({e})")
    sp = SpellChecker(language="es")
    DICT = {_norm(w) for w in sp.word_frequency.dictionary}
    seen, ordered = set(), []
    for w in wordfreq.top_n_list("es", freq_n):
        u = _norm(w)
        if (u.isalpha() and min_len <= len(u) <= max_len and u not in seen
                and u in DICT and u not in _ES_STOP):      # intersect real dictionary
            seen.add(u); ordered.append(u)
    targets = [w for w in ordered if len(w) >= 4]
    tset = set(targets)
    fill = [w for w in ordered if w not in tset]
    return {"ws": {"theme": targets, "fill": fill}, "allowed": ordered,
            "clean_set": set(ordered), "targets": targets, "DICT": DICT}


def eval_program(code, pal, sizes, per_size, topics):
    allowed, clean, DICT = pal["allowed"], pal["clean_set"], pal["DICT"]
    recs = []
    for size in sizes:
        budget = BUDGET.get(size, size * 2)
        spec = Spec(size=size, topic_words=tuple(pal["targets"]), require_symmetry=False,
                    min_word_len=3, time_budget_s=budget)
        for t in topics[:per_size]:
            spec_dict = {"topic": t, "word_source": pal["ws"], "size": size, "seed": 0}
            res = run_candidate_inprocess(code, spec_dict, timeout_s=budget * 2 + 3)
            rec = {"size": size, "topic": t, "status": res["status"]}
            if res["status"] != "ok" or not res.get("result"):
                rec.update(valid=0, within=0, fully_valid=0, dict_frac=0.0, coverage=0.0,
                           crossings=0, n_entries=0, filler=0.0, invalid_entry=0.0,
                           runtime=res.get("runtime_s", 0.0))
                recs.append(rec); continue
            lay = res["result"]
            m = score(lay, spec, allowed, runtime_s=res["runtime_s"], vocab_set=clean)
            entries = [e["answer"].upper() for e in lay.get("across", []) + lay.get("down", [])
                       if len(e.get("answer", "")) >= 3]
            in_dict = sum(1 for w in entries if w in DICT)
            dict_frac = in_dict / len(entries) if entries else 0.0
            valid = int(m["valid"] == 1)
            filler = m["filler_fraction"] or 0.0
            inv = (m["invalid_entry_frac"] or 0.0) + (m["invalid_crossing_frac"] or 0.0)
            within = int(valid and filler <= 0.30 and inv == 0.0 and res["runtime_s"] <= budget)
            rec.update(valid=valid, within=within,
                       fully_valid=int(valid and dict_frac >= 0.999),
                       dict_frac=round(dict_frac, 3), coverage=m["coverage"],
                       crossings=m["crossings"], n_entries=m["n_entries"],
                       filler=filler, invalid_entry=m["invalid_entry_frac"] or 0.0,
                       runtime=round(res["runtime_s"], 2))
            recs.append(rec)
    return recs


def _agg(recs):
    n = len(recs)
    f = lambda k: round(sum(r[k] for r in recs) / n, 3) if n else 0.0
    ran = [r for r in recs if r["valid"]]
    fr = lambda k: round(sum(r[k] for r in ran) / len(ran), 3) if ran else 0.0
    return {"n": n, "valid_pct": f("valid"), "fully_valid_pct": f("fully_valid"),
            "within_pct": f("within"), "mean_dict_frac": f("dict_frac"),
            "cov": fr("coverage"), "cross": fr("crossings"), "entries": fr("n_entries"),
            "filler_pct": fr("filler"), "rt": fr("runtime")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "es"], required=True)
    ap.add_argument("--submission", required=True, help="path to a .py with generate_crossword")
    ap.add_argument("--sizes", default=None, help="comma list; default en=7,9,11,15 es=7,9,11")
    ap.add_argument("--per-size", type=int, default=8, help="samples (topics) per size")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    sizes = ([int(s) for s in args.sizes.split(",")] if args.sizes
             else ([7, 9, 11, 15] if args.lang == "en" else [7, 9, 11]))
    topics = EN_TOPICS if args.lang == "en" else ES_TOPICS
    code = open(args.submission, encoding="utf-8").read()

    print(f"lang={args.lang}  submission={args.submission}  sizes={sizes}  per_size={args.per_size}")
    print("building palette + dictionary...")
    pal = english_palette(max(sizes)) if args.lang == "en" else spanish_palette(max(sizes))
    print(f"palette: {len(pal['allowed'])} words | dictionary: {len(pal['DICT'])} words\n")

    recs = eval_program(code, pal, sizes, args.per_size, topics)

    hdr = f"{'size':>5} {'n':>3} {'valid%':>7} {'fullyOK%':>9} {'within%':>8} {'dictOK':>7} {'cov':>5} {'cross':>6} {'entries':>8} {'filler%':>8} {'rt':>6}"
    print(hdr); print("-" * len(hdr))
    for size in sizes:
        a = _agg([r for r in recs if r["size"] == size])
        print(f"{size:>5} {a['n']:>3} {a['valid_pct']*100:>6.0f} {a['fully_valid_pct']*100:>8.0f} "
              f"{a['within_pct']*100:>7.0f} {a['mean_dict_frac']*100:>6.0f} {a['cov']:>5.2f} "
              f"{a['cross']:>6.0f} {a['entries']:>8.0f} {a['filler_pct']*100:>7.0f} {a['rt']:>6.2f}")
    ov = _agg(recs)
    print("-" * len(hdr))
    print(f"{'ALL':>5} {ov['n']:>3} {ov['valid_pct']*100:>6.0f} {ov['fully_valid_pct']*100:>8.0f} "
          f"{ov['within_pct']*100:>7.0f} {ov['mean_dict_frac']*100:>6.0f} {ov['cov']:>5.2f} "
          f"{ov['cross']:>6.0f} {ov['entries']:>8.0f} {ov['filler_pct']*100:>7.0f} {ov['rt']:>6.2f}")
    print("\nlegend: valid%=structurally valid | fullyOK%=valid AND every entry in dictionary | "
          "within%=valid+filler<=30%+in-budget | dictOK=mean fraction of entries that are real words")

    out = args.out or os.path.join(_ROOT, "runs", "eval", f"selfmodel_{args.lang}_{int(time.time())}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"lang": args.lang, "sizes": sizes, "overall": ov,
               "by_size": {s: _agg([r for r in recs if r["size"] == s]) for s in sizes},
               "records": recs}, open(out, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
