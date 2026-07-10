# `data/sft_hardcoded_words/` — SFT dataset (hardcoded-vocabulary variant)

Same as [`data/sft/`](../sft/README.md), **except each positive program has its curated
vocabulary baked in**. Every assistant program gains a `_WORDS = [...]` constant and uses it
by default:

```python
def generate_crossword(topic: str = "vocabulary", word_source=None, size: int = 11) -> dict:
    word_source = word_source or _WORDS   # self-contained; word_source still overrides
    ...
```

So a user query "make an *N*×*N* vocabulary crossword" maps to an expected output that carries
**both** the word list (`_WORDS`) *and* the algorithm — the model learns the words into its
weights. `word_source` remains an optional override/fallback.

## Files

| file | rows | role |
|---|---|---|
| `train.jsonl` | 1,865 | SFT training set |
| `dev.jsonl` | 230 | validation |
| `eval.jsonl` | 199 | pristine held-out |
| `negatives.jsonl` | 817 | copied unchanged from baseline — **not hardcoded, not used by SFT** |
| `negatives_eval.jsonl` | 87 | held-out negatives |

By grid size (train+dev+eval): **7×7: 1,117 · 9×9: 747 · 11×11: 124 · 15×15: 306**.

## How it was built
For each of the 50 distinct baseline programs, it was run many times on the curated palette;
the answer words from every **valid** run were unioned and baked in as `_WORDS`.
- **36 programs** hardcoded successfully, all verified to produce a valid crossword
  self-contained (with no `word_source`). Their `_WORDS` are **purified-only** (real
  dictionary words — no acronyms/proper nouns).
- **14 dense 11×11/15×15 templates were dropped** — they can only fill using the broader
  (proper-noun-bearing) palette, so they couldn't be baked with a clean word list. This is why
  **11×11 is thin (124)** here vs 340 in the baseline.

## Same caveat as the baseline: assistant programs are NOT reference output
Evaluation reads only the prompts (`messages[0]`, `messages[1]`) + size and **runs the model's
own emitted program** — it never compares against or executes the `assistant` program in these
records as expected output.

## Usage
The training notebook (`train/colab_train_qlora.ipynb`) trains on this folder when
`HARDCODED_WORDS = True` (its default); set it `False` to train on the baseline `data/sft/`.
This is an **SFT-only** dataset — the `negatives.jsonl` files are carried over for
completeness but are **not** consumed by the SFT loop (no negative reinforcement / DPO here).

Derived from `data/sft/` (identical prompts, splits, and `meta`); only the assistant programs
differ. 14 programs dropped → 2,294 records vs the baseline's 2,526.
