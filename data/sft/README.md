# `data/sft/` — SFT dataset (baseline)

Chat-JSONL supervised fine-tuning corpus for the crossword-generator SLM
(Qwen3-4B QLoRA). A natural-language request → a self-contained
`generate_crossword(topic, word_source, size)` Python program.

## Files

| file | rows | role |
|---|---|---|
| `train.jsonl` | 2,087 | SFT training set (response-only loss) |
| `dev.jsonl` | 236 | validation / early-stopping (not trained on) |
| `eval.jsonl` | 203 | **pristine held-out** — never trained or tuned on; the base-vs-tuned comparator |
| `negatives.jsonl` | 817 | labeled *bad* programs (SOAR-style). **Reserved for DPO / analysis — NOT used by SFT.** |
| `negatives_eval.jsonl` | 87 | held-out negatives (kept out of the DPO pool so nothing from `eval` leaks) |

By grid size (train+dev+eval): **7×7: 1,117 · 9×9: 747 · 11×11: 340 · 15×15: 322**.
Splits are 80/10/10 by a hash of `spec_id`; 11/15 are upsampled.

## Record schema

Each line is one chat example:

```json
{"messages": [
   {"role": "system",    "content": "You are an expert Python programmer."},
   {"role": "user",      "content": "Create a 11x11 fixed-grid (non-free-form) crossword about vocabulary."},
   {"role": "assistant", "content": "```python\n# === TASK CONTRACT ===\n...\ndef generate_crossword(...): ...\n```"}],
 "meta": {"spec_id": "...", "kind": "solution|fixed_template|...",
          "program_hash": "...", "effective_spec": {"size": 11, ...}, "split": "train"}}
```

- **system** is minimal; the full task contract lives as a comment header *inside* the
  assistant program, so the knowledge is trained into the weights and a bare user request suffices.
- **user** is one of a few phrasings of "make an *N*×*N* vocabulary crossword" (only the size varies).
- **assistant** is a verified generator program. `word_source` (the curated word palette) is
  **injected at runtime**; the program fills from whatever list it's handed.
- Training uses **response-only loss** (system + user are masked; loss is only on the assistant program).

## Important: the assistant programs are NOT used as reference output

Evaluation is **execution-based**, not exact-match. The eval harness reads **only**
`messages[0]` (system) + `messages[1]` (user) + `meta.effective_spec.size` from these records —
it feeds those prompts to a model, then **runs the model's *own* emitted program** in the
sandbox and scores the crossword it produces. The `assistant` program in `eval.jsonl` (and
`train`/`dev`) is **never** compared against, executed as a reference, or used as expected
output. It only serves to (a) define which held-out specs/prompts exist and (b) mark the split
as verified-positive.

## See also
- `data/sft_hardcoded_words/` — a variant where each program's vocabulary is baked into a
  `_WORDS` constant (self-contained). Same prompts/splits; only the assistant programs differ.
- `data/wordlists/README.md` — the word palettes the programs fill from.
