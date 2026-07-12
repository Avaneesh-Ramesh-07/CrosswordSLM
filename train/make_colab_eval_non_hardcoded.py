"""Generate train/colab_eval_non_hardcoded.ipynb — evaluate the tuned NON-HARDCODED (QLoRA SFT)
model on the size-specific prompt, scored through the same harness as Claude.

The base-vs-tuned headline for the SFT model. The tuned model is fed the SAME size-specific contract
prompt it was trained on (`user_contract(N)` = "Write Python code to generate a NxN … crossword"),
and every emitted `generate_crossword` program is scored by the identical sandbox + scorer +
real-dictionary check as Claude (`pipeline.eval_opus_fleet.score_one`) on the purified palette.

The matched Claude baseline is EVAL 3 (identical size-specific prompt + purified palette): 0/150
valid (GAP_ANALYSIS). So any non-zero here is the tuned model clearing an unaugmented-Opus floor of 0.

Adapted from the user-supplied "EVAL 2 tuned" spec, corrected for THIS model:
  - held-out `data/sft_non_hardcoded_enhanced/eval.jsonl` only holds sizes 7/15, and after the
    dataset transform every record of a size shares the identical prompt `user_contract(size)`, so we
    build the prompt directly per size and sample fresh (temp 1.0) — the tuned analog of EVAL 3.
  - sizes 7/9/11 (the trained sizes; 15 excluded, as in training).
  - purified palette (WORD_LIST_FULLY_PURIFIED) — the exact condition of Claude EVAL 3.
  - MAX_NEW_TOKENS raised to 8192 (non-hardcoded 11x11 programs embed the grid template, ~7k tokens).

    python train/make_colab_eval_non_hardcoded.py
"""

import json
import os

MD, CO = "markdown", "code"

REPO_URL = "https://github.com/Avaneesh-Ramesh-07/CrosswordSLM.git"

cells = [
 (MD, "# EVAL — tuned Qwen3-4B (non-hardcoded QLoRA SFT), harness-scored\n\n"
      "The base-vs-tuned headline for the SFT model. The tuned model is fed the **same size-specific "
      "contract prompt it was trained on** (`user_contract(N)` — *\"Write Python code to generate a "
      "NxN, fixed-grid, American-style crossword…\"*), and every emitted `generate_crossword` program "
      "is scored through the **identical sandbox + scorer + real-dictionary check** as Claude "
      "(`pipeline.eval_opus_fleet.score_one`) on the **purified palette** (`WORD_LIST_FULLY_PURIFIED`, "
      "24,542 words).\n\n"
      "**Matched Claude baseline — EVAL 3** (identical size-specific prompt + purified palette): "
      "**0 / 150 valid** (GAP_ANALYSIS; unaugmented Opus times out on the full palette). So any "
      "non-zero here is the tuned model clearing an unaugmented-Opus floor of **0%**.\n\n"
      "Sizes **7/9/11** (the trained sizes; 15 excluded, as in training). Every record of a size "
      "shares the identical prompt, so we build it directly per size and sample fresh programs at "
      "temperature 1.0 (the tuned analog of EVAL 3). **Runtime:** GPU, L4 (24 GB) / A100 (40 GB). "
      "**Order:** run top to bottom."),

 (MD, "## 1. Get the code\n"
      "Clone the repo (set your URL) or upload the folder and set `PROJECT_DIR`. The clone contains "
      "the word lists in `data/wordlists/` (incl. `WORD_LIST_FULLY_PURIFIED.txt`) and the "
      "`pipeline/` + `harness/` scoring code. **Push your latest local changes first** — the eval "
      "reflects whatever is committed here."),
 (CO, 'REPO_URL = "%s"\n'
      'import os\n'
      '!git clone -q $REPO_URL slm || echo "clone skipped/failed — upload the folder instead"\n'
      'PROJECT_DIR = "/content/slm"   # adjust if you uploaded elsewhere\n'
      'assert os.path.isdir(os.path.join(PROJECT_DIR, "pipeline")), "Set PROJECT_DIR to the repo root"\n'
      '%%cd $PROJECT_DIR' % REPO_URL),

 (MD, "## 2. GPU + install deps\n"
      "Colab ships torch. We add `transformers`/`accelerate` (pinned to the training snapshot) to "
      "load the merged model, and `wordfreq` (only needed if you switch to the clean English "
      "palette). No `bitsandbytes` — the merged model loads in 16-bit directly."),
 (CO, 'import torch\n'
      'assert torch.cuda.is_available(), "No GPU — Runtime > Change runtime type > GPU (L4/A100 recommended)"\n'
      'print("GPU:", torch.cuda.get_device_name(0))\n'
      '!pip install -q "transformers==4.53.*" "accelerate==1.8.*" wordfreq'),

 (MD, "> **Expected pip warning — safe to ignore.** Colab's pre-installed `gradio` wants a newer "
      "`huggingface-hub` than `transformers 4.53` pins. `gradio` is unused here; **do not upgrade "
      "`huggingface-hub`** (it would break `transformers`)."),

 (MD, "## 3. Point to the merged tuned model (on Drive)\n"
      "Mount Drive and set `MODEL_DIR` to your **merged** model folder "
      "(`…-merged`) — the full standalone model, not the LoRA adapter. The training notebook "
      "(`colab_train_qlora.ipynb`, cell 9) saves it under `MyDrive/slm_ckpt/`."),
 (CO, 'from google.colab import drive\n'
      'drive.mount("/content/drive")\n'
      '# EDIT to your merged-model path on Drive (training notebook saves here):\n'
      'MODEL_DIR = "/content/drive/MyDrive/slm_ckpt/qwen3-4b-crossword-qlora-merged"\n'
      'import os\n'
      'assert os.path.isdir(MODEL_DIR), f"MODEL_DIR not found: {MODEL_DIR}"\n'
      'assert os.path.exists(os.path.join(MODEL_DIR, "config.json")), \\\n'
      '    "not a full model dir (need config.json + model-*.safetensors, i.e. the -merged folder, not the adapter)"\n'
      'print("model dir OK:", MODEL_DIR)'),

 (MD, "## 4. Load the tuned model"),
 (CO, 'import torch\n'
      'from transformers import AutoModelForCausalLM, AutoTokenizer\n'
      'tok = AutoTokenizer.from_pretrained(MODEL_DIR)\n'
      'if tok.pad_token_id is None:\n'
      '    tok.pad_token = tok.eos_token\n'
      'tok.padding_side = "left"   # left-pad so batched generation aligns at the prompt end\n'
      'model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype="auto", device_map="auto")\n'
      'model.eval()\n'
      'print("loaded:", model.config.model_type, "| dtype", next(model.parameters()).dtype, "| device", model.device)'),

 (MD, "## 5. Generation settings + batched helper\n"
      "`GEN_TEMP = 1.0` matches Claude EVAL 3 (use 0.0 for greedy/deterministic — the model's single "
      "best output). `MAX_NEW_TOKENS = 8192` because a non-hardcoded 11×11 program embeds its grid "
      "template (~7k tokens) — **do not lower it below ~7000** or those programs truncate. KV cache "
      "grows with `BATCH × MAX_NEW_TOKENS`, so drop `BATCH` on a smaller GPU."),
 (CO, 'GEN_TEMP       = 1.0     # match Claude EVAL 3; use 0.0 for greedy/deterministic\n'
      'MAX_NEW_TOKENS = 8192    # 11x11 programs embed the template (~7k tok); do NOT lower below ~7000\n'
      'BATCH          = 4       # KV cache ~ BATCH x MAX_NEW_TOKENS; lower to 1-2 on a 24 GB GPU\n\n'
      'import torch\n'
      '@torch.no_grad()\n'
      'def generate_batch(pairs):\n'
      '    """pairs: list of (system, user) -> list of completion strings (assistant turn only)."""\n'
      '    outs = []\n'
      '    for i in range(0, len(pairs), BATCH):\n'
      '        chunk = pairs[i:i + BATCH]\n'
      '        texts = [tok.apply_chat_template(\n'
      '                    [{"role": "system", "content": s}, {"role": "user", "content": u}],\n'
      '                    tokenize=False, add_generation_prompt=True)\n'
      '                 for (s, u) in chunk]\n'
      '        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,\n'
      '                  max_length=2048).to(model.device)\n'
      '        gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS,\n'
      '                             do_sample=GEN_TEMP > 0, temperature=max(GEN_TEMP, 1e-5),\n'
      '                             top_p=0.95, pad_token_id=tok.pad_token_id)\n'
      '        new = gen[:, enc["input_ids"].shape[1]:]\n'
      '        outs.extend(tok.batch_decode(new, skip_special_tokens=True))\n'
      '        print(f"  generated {min(i + BATCH, len(pairs))}/{len(pairs)}", flush=True)\n'
      '    return outs'),

 (MD, "## 6. Generate on the prompt, score through the harness\n"
      "For each size in **7/9/11**, build the prompt and sample `PER_SIZE` fresh programs at `GEN_TEMP`, "
      "then score each with `score_one` on the purified palette (real-dictionary check on every entry). "
      "**`BARE_PROMPT` toggle** (set at the top of the cell): `False` uses the size-specific **contract** "
      "prompt the model trained on (matches Claude **EVAL 3** / the T2 condition); `True` uses the **bare "
      "`eval.jsonl` deployment prompt** — *\"Create a NxN vocabulary crossword…\"* with no contract or "
      "schema (the EVAL 2-style condition). The model is unchanged; only the prompt differs. "
      "`GEN_TIMEOUT = 60 s` gives each emitted program the same execution budget as the Claude evals."),
 (CO, 'import os, json, time\n'
      '# eval_opus_fleet reads ANTHROPIC_* at import; set dummies so nothing complains (no API calls here)\n'
      'os.environ.setdefault("ANTHROPIC_BASE_URL", ""); os.environ.setdefault("ANTHROPIC_AUTH_TOKEN", "")\n'
      'import pipeline.eval_opus_fleet as F\n'
      'from pipeline.eval_opus_fleet import score_one, agg, table, purified_palette\n'
      'from pipeline.contract_prompt import SYSTEM, user_contract\n'
      'from pipeline.eval_harness import extract_code\n\n'
      'SIZES    = [7, 9, 11]    # trained sizes (15 excluded, as in training)\n'
      'PER_SIZE = 25            # samples per size at GEN_TEMP; 25 x 3 = 75 total\n'
      'F.GEN_TIMEOUT = 60       # match EVAL 3: up to 60 s execution per emitted program\n'
      'pal = purified_palette() # WORD_LIST_FULLY_PURIFIED (24,542 words) -- the EVAL 3 condition\n\n'
      '# --- PROMPT STYLE TOGGLE --------------------------------------------------------------------\n'
      '# False (default): the size-specific CONTRACT prompt the model was trained on -> matches Claude\n'
      '#                  EVAL 3 (the T2 condition).\n'
      '# True:            the BARE eval.jsonl deployment prompt ("Create a NxN vocabulary crossword ...",\n'
      '#                  NO contract, NO schema, NO word list) -> the harder EVAL 2-style condition\n'
      '#                  (what Claude EVAL 2 / the T1 eval used). SAME model either way; only the\n'
      '#                  prompt changes, so this measures how well the tuning generalizes off-contract.\n'
      'BARE_PROMPT  = False\n'
      '_BARE_SYSTEM = "You are an expert Python programmer."\n'
      '_BARE_TEMPLATES = [   # the five eval.jsonl phrasings (data/sft/eval.jsonl), verbatim\n'
      '    "I need a {N}x{N} fixed-grid crossword for practicing vocabulary (non-free-form).",\n'
      '    "Generate a {N}x{N} fixed-grid crossword to teach vocabulary (not free-form).",\n'
      '    "Make a {N}x{N} non-free-form vocabulary crossword.",\n'
      '    "Create a {N}x{N} fixed-grid (non-free-form) crossword about vocabulary.",\n'
      '    "Build me a {N}x{N} vocabulary crossword on a fixed grid, not free-form.",\n'
      ']\n'
      'def _make_prompt(s, i):\n'
      '    if BARE_PROMPT:   # cycle the phrasings, like the held-out eval.jsonl distribution\n'
      '        return (_BARE_SYSTEM, _BARE_TEMPLATES[i % len(_BARE_TEMPLATES)].replace("{N}", str(s)), s)\n'
      '    return (SYSTEM, user_contract(s), s)\n'
      'STYLE = "bare eval.jsonl" if BARE_PROMPT else "size-specific contract"\n\n'
      'prompts = [_make_prompt(s, i) for s in SIZES for i in range(PER_SIZE)]\n'
      'print(f"{len(prompts)} prompts ({PER_SIZE}/size {SIZES}) | style: {STYLE}")\n'
      'print("  example user prompt:", prompts[0][1].splitlines()[0])\n\n'
      'print("generating...", flush=True)\n'
      'comps = generate_batch([(s, u) for (s, u, sz) in prompts])\n\n'
      'print("scoring on purified palette...", flush=True)\n'
      'rows = []\n'
      'for (s, u, sz), txt in zip(prompts, comps):\n'
      '    code = extract_code(txt)\n'
      '    if not code:\n'
      '        rec = {"valid": 0, "fully": 0, "within": 0, "dict_frac": 0.0, "coverage": 0.0,\n'
      '               "crossings": 0, "entries": 0, "filler": 0.0, "parsed": 0}\n'
      '    else:\n'
      '        rec = score_one(code, pal, sz, "vocabulary"); rec["parsed"] = 1\n'
      '    rec["size"] = sz; rows.append(rec)\n\n'
      'parse_rate = sum(r["parsed"] for r in rows) / len(rows)\n'
      'print(f"\\nparse rate (emitted a code block): {parse_rate*100:.0f}%")\n'
      'ov = table(f"TUNED Qwen3-4B (non-hardcoded SFT), {STYLE} prompt, purified "\n'
      '           f"(n={len(rows)}, temp={GEN_TEMP})", rows, SIZES)'),

 (MD, "## 7. Save results + base-vs-tuned comparison"),
 (CO, 'import os, json, time\n'
      'os.makedirs("runs/eval", exist_ok=True)\n'
      'out = f"runs/eval/tuned_nonhardcoded_{int(time.time())}.json"\n'
      'summary = {"model": "qwen3-4b-crossword-qlora-merged (non-hardcoded SFT)",\n'
      '           "condition": f"{STYLE} prompt, purified palette",\n'
      '           "n": len(rows), "parse_rate": parse_rate, "gen_temp": GEN_TEMP, "sizes": SIZES,\n'
      '           "overall": ov, "by_size": {s: agg([r for r in rows if r["size"] == s]) for s in SIZES}}\n'
      'json.dump(summary, open(out, "w", encoding="utf-8"), indent=2)\n'
      'print("wrote", out)\n'
      '!mkdir -p /content/drive/MyDrive/slm_runs/eval && cp "$out" /content/drive/MyDrive/slm_runs/eval/ 2>/dev/null || true\n\n'
      'print("\\n===== base vs tuned (size-specific prompt, purified palette) =====")\n'
      'print(f"  Claude Opus 4.8 (EVAL 3, n=150): valid   0%   fullyOK   0%   within   0%   [GAP_ANALYSIS]")\n'
      'print(f"  Tuned Qwen3-4B  (this run, n={len(rows)}): valid {ov[\'valid\']*100:3.0f}%   "\n'
      '      f"fullyOK {ov[\'fully\']*100:3.0f}%   within {ov[\'within\']*100:3.0f}%   (temp={GEN_TEMP})")'),

 (MD, "## 8. (Optional) dump the raw programs for inspection\n"
      "Writes each emitted program to `runs/eval/tuned_nonhardcoded_progs/` so you can eyeball the "
      "actual generators the model produced."),
 (CO, 'import os\n'
      'os.makedirs("runs/eval/tuned_nonhardcoded_progs", exist_ok=True)\n'
      'for k, ((s, u, sz), txt) in enumerate(zip(prompts, comps)):\n'
      '    code = extract_code(txt) or txt\n'
      '    open(f"runs/eval/tuned_nonhardcoded_progs/prog_{k:03d}_s{sz}.py", "w", encoding="utf-8").write(code)\n'
      'print("saved", len(prompts), "programs under runs/eval/tuned_nonhardcoded_progs/")'),

 (MD, "## Next\n"
      "If the tuned model clears a meaningful bar here (vs Claude's **0%** at EVAL 3), record the "
      "numbers in `GAP_ANALYSIS.md` as the **tuned column** of the base-vs-tuned story. To also test "
      "the clean English palette (Claude EVAL 2's condition), swap `purified_palette()` for "
      "`english_palette(max(SIZES))` from `pipeline.eval_selfmodel` (add `wordfreq`, already installed)."),
]


def build():
    nb_cells = []
    for kind, src in cells:
        cell = {"cell_type": kind, "metadata": {}, "source": src}
        if kind == CO:
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)
    nb = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colab_eval_non_hardcoded.ipynb")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1)
    return out


if __name__ == "__main__":
    path = build()
    with open(path, encoding="utf-8") as fh:
        nb = json.load(fh)
    print(f"wrote {path}")
    print(f"cells: {len(nb['cells'])} "
          f"({sum(c['cell_type']=='code' for c in nb['cells'])} code, "
          f"{sum(c['cell_type']=='markdown' for c in nb['cells'])} md)")
