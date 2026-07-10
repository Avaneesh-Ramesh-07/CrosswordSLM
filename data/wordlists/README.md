# Word lists

## The `WORD_LIST_*` snapshots (cleaning progression)

Three frozen snapshots of the crossword **word palette** (the set of words a generated
`generate_crossword` program is allowed to fill from), captured at three successive
cleaning stages. Each is a subset of the previous one:

```
WORD_LIST_WORKS  ─(remove acronyms)→  WORD_LIST_ACRONYM_CLEANED  ─(WordNet cleanse)→  WORD_LIST_FULLY_PURIFIED
   27,232                                   27,213   (ACTIVE)                              24,542
```

| file | words | what it is |
|---|---|---|
| `WORD_LIST_WORKS.txt` | 27,232 | **Original baseline** — the palette before any acronym/proper-noun cleaning. This is the "normal" list that produced the original eval results; keep it as the **revert target**. Contains real short words *and* some acronyms/initialisms. |
| `WORD_LIST_ACRONYM_CLEANED.txt` | 27,213 | **ACTIVE palette.** `WORKS` minus **19 acronyms/initialisms** (ABC, AKA, AWOL, ETC, FAQ, GPS, IRA, IRS, LCD, MIA, MPH, MRS, MSG, PTA, RSVP, TNT, TPS, UFO, VIN). Real short words (RUE, FEZ, AYE, WRY, VEX, …) are intentionally **kept**. |
| `WORD_LIST_FULLY_PURIFIED.txt` | 24,542 | **Experimental / curiosity only** (not wired into the pipeline). `ACRONYM_CLEANED` further filtered to keep only words with a **WordNet common-word sense** — removes proper nouns (AARON, STYX, …), acronyms, and no-meaning tokens. Aggressive: it also drops some legitimate words (e.g. **AYE**), which is why it is *not* the active list. |

### Which one is actually used
The live palette is **generated at runtime** by
[`pipeline/word_source.py`](../../pipeline/word_source.py) → `build_clean_education_source()`,
which currently produces the **acronym-cleaned** set (via the `_NOT_VOCAB` exclusion in that
file). These `.txt` files are **reference/backup snapshots**, not loaded at runtime.

- To **revert** to the original baseline: remove the 19 acronyms from `_NOT_VOCAB` in
  `word_source.py` (the palette then matches `WORD_LIST_WORKS.txt`).
- The fully-purified variant is **not** wired into `word_source.py`; it exists only as this
  snapshot (built with an NLTK WordNet filter).

## Source lists (inputs the palette is built from)

The palette above is the intersection/union of these raw inputs (see
`build_clean_education_source`):

| file | role |
|---|---|
| `words_alpha.txt` | broad English dictionary (dwyl `words_alpha`, ~370k) — the "is it a real word" gate, and the dictionary used for the eval `dictOK`/`fullyOK` checks. |
| `sat_words.txt` | SAT / academic vocabulary — the prioritized **theme** tier (drives `coverage`). |
| `common_english.txt` | common-frequency English (optional connector tier). |

Note: the palette also intersects the Collaborative Word List crossword scores
(`../collaborative-word-list/xwordlist.dict`, score ≥ 55) and the `wordfreq` top-50k
frequency list, which are not stored here.
