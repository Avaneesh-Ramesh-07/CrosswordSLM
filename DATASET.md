# Dataset

The fine-tuning corpus for the crossword-generator SLM. Each example maps a
natural-language **SPEC** (a short, size-routed request) to a **verified Python
`generate_crossword(topic, word_source, size)` program**. This document describes every
`.jsonl` file, the record format of each, the train/dev/eval split semantics, and the
exact upsampling applied when building the consolidated training files.

## Layout

The corpus exists at two levels:

- **Per-section source dirs** — `runs/<section>/dataset/*.jsonl`, one set per generation
  of data. These are the raw, un-merged outputs of the harvest / template builds.
- **Consolidated training files** — `data/sft/*.jsonl`, produced by
  `pipeline/merge_dataset.py` (union across sections, deduped, size-upsampled). **This is
  what the trainer reads.**

### Sections (sources of `data/sft`)

| section (`runs/…/dataset/`) | approach | sizes | how produced |
|---|---|---|---|
| `bulk` | construct-from-scratch | 7, 9 | evolutionary-teacher harvest (`pipeline/harvest.py`) |
| `gen3` | construct-from-scratch | 7, 9 | later fusion generation, harvested |
| `templates15` | fixed-template (real NYT grids) | 15 | `build_template_dataset.py` (inlined NYT black-square patterns + fill) |
| `templates11` | fixed-template (minted grids) | 11 | `build_template_dataset.py` (minted fillable patterns + fill) |

## The `.jsonl` files

### Trained / evaluated (in `data/sft/`, and mirrored per-section)

| file | rows (`data/sft`) | trained on? | purpose |
|---|---|---|---|
| `train.jsonl` | 2,087 | **yes** | SFT training set (size-**upsampled**; see below). Model learns weights from these. |
| `dev.jsonl` | 236 | no (read during training) | **Validation** — watched each epoch (`eval_loss`) to pick the best checkpoint / detect overfitting. Never contributes gradients. |
| `eval.jsonl` | 203 | **never** | **Pristine held-out** test set. Touched once at the very end for the base-vs-tuned headline number. Never influences any training decision. |
| `negatives.jsonl` | 817 | not in SFT | Labeled **failed** programs (SOAR-style negatives) from train/dev specs. Reserved for optional preference training (DPO) and analysis. |
| `negatives_eval.jsonl` | 87 | never | Negatives derived from held-out `eval` specs, kept separate so nothing eval-derived can leak into training. |

Splits are disjoint by spec: an ~80/10/10 train/dev/eval partition of the specs, so the
same spec never appears in two splits.

### Source / reference (not trained directly)

| file | rows | purpose |
|---|---|---|
| `data/scraped/nyt_vocab.jsonl` | 160 | Real published **NYT 15×15 crosswords** (doshea corpus). Not `(spec → program)` pairs — they are `(spec → solved layout)`. Used only as a source of **grid geometry**: their 155 distinct black-square patterns seed the 15×15 fixed-template library. Never trained on directly. |
| `data/templates_11.json`, `data/templates_15.json` | 46 / 95 | **JSON, not JSONL.** The libraries of known-fillable black-square patterns baked into the fixed-template generators (11×11 minted; 15×15 from the NYT scrape). |

## Record formats

### 1. SFT record — `train.jsonl`, `dev.jsonl`, `eval.jsonl`

A 3-turn chat example plus non-trained curation metadata.

```json
{
  "messages": [
    {"role": "system",    "content": "<fixed task contract: rules + output schema + technique menu>"},
    {"role": "user",      "content": "Create a 9x9 fixed-grid (non-free-form) crossword about vocabulary."},
    {"role": "assistant", "content": "```python\n<verified generate_crossword program>\n```"}
  ],
  "meta": {
    "spec_id": "s00010",
    "kind": "solution",              // solution | hindsight_density | hindsight_symmetry | fixed_template
    "combined_score": 0.71,
    "program_hash": "…",             // dedup key (with spec_id)
    "effective_spec": {              // the constraints this example was verified against
      "spec_id": "s00010", "size": 9, "require_symmetry": true, "min_word_len": 3,
      "time_budget_s": 5, "density_target": 0.76, "topic": "vocabulary",
      "difficulty": "medium", "heuristic_hints": [...], "split": "dev"
    },
    "split": "train"
  }
}
```

- **Trained turn:** only the `assistant` message (response-only loss); `system` + `user`
  are masked.
- **`system`** is a fixed contract (identical across all rows): output schema + hard
  validity rules + a technique menu. **`user`** is the minimal, size-routed request — the
  topic is always "vocabulary"; only the size (and phrasing) varies, so the model must
  infer which construction/fill technique to apply from the size.
- **`meta`** is curation info, not fed to the model. `fixed_template` rows additionally
  carry `engine`, `selection`, and `subset` keys recording how that program was emitted.

### 2. Negative record — `negatives.jsonl`, `negatives_eval.jsonl`

A program that was generated and **failed** the verifier, kept as a labeled negative.

```json
{
  "spec_id": "s00042",
  "spec": "Make a 9x9 non-free-form vocabulary crossword.",   // rendered prompt
  "effective_spec": { … same shape as above … },
  "code": "<the failing generate_crossword program>",
  "kind": "negative",
  "split": "train",
  "program_hash": "…",
  "combined_score": 0.0,
  "metrics": {
    "valid": 0.0, "fill_density": …, "coverage": …, "filler_fraction": …,
    "invalid_crossing_frac": …, "invalid_entry_frac": …, "runtime_s": …,
    "combined_score": 0.0
  },
  "reasons": ["…scorer rejection reasons…"],
  "failure_category": "timeout | exception | oom | malformed | nonword | crossing_conflict | disconnected | declared_mismatch | low_coverage | other"
}
```

### 3. NYT scrape record — `data/scraped/nyt_vocab.jsonl`

A real published puzzle: a verbose SPEC plus its **solved** grid (no program).

```json
{
  "spec": "<verbose 15x15 SAT-vocabulary spec>",
  "resulting crossword": {
    "size": 15, "grid": [[...]], "black": [[r,c],…], "n_black": 38,
    "white_fraction": 0.83, "black_fraction": 0.17, "symmetric": true, "min_word_len": 3,
    "across": [{"number","row","col","answer","len"}], "down": [ …same… ],
    "words": [...], "n_words": 78,
    "vocab_words": [...], "n_vocab_words": 64, "vocab_fraction": 0.82,
    "vocab_set": "clean_educational_palette(27245)",
    "source": {"corpus": "doshea/nyt_crosswords", "publisher": "The New York Times",
               "date": "12/6/1999", "dow": "Monday", "author": "…", "title": "…"}
  }
}
```

## Upsampling (applied when building `data/sft/train.jsonl`)

`pipeline/merge_dataset.py --upsample 11=3,15=3` was run. Upsampling **duplicates**
existing 11×11 and 15×15 **train** rows (each ×3) so those sizes aren't drowned out by
7/9 during training. Only `train` is upsampled — `dev` and `eval` are left as-is.

- Command: `python pipeline/merge_dataset.py --upsample 11=3,15=3`
- Added **+416 duplicate rows** (11×11: 107 → 321; 15×15: 101 → 303).
- Duplicates are exact copies (a training-time balance knob, not new data), so the factor
  is kept modest to limit memorization risk.

### Resulting size distribution

| size | train (before) | train (after ×3 upsample) | dev | eval |
|---|---|---|---|---|
| 7×7 | 871 | 871 (42%) | 126 | 120 |
| 9×9 | 592 | 592 (28%) | 90 | 65 |
| 11×11 | 107 | **321 (15%)** | 11 | 8 |
| 15×15 | 101 | **303 (15%)** | 9 | 10 |
| **total** | **1,671** | **2,087** | **236** | **203** |

(Train totals are the deduped union across sections — `merge_dataset.py` drops repeated
`(spec_id, program_hash)` pairs before upsampling.)

## Rebuilding

```bash
python pipeline/merge_dataset.py --upsample 11=3,15=3   # -> data/sft/{train,dev,eval}.jsonl (+negatives)
python pipeline/dataset_stats.py --runs runs/bulk runs/gen3 runs/templates15 runs/templates11
```
