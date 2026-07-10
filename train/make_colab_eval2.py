"""Generate train/colab_eval_tuned.ipynb — EVAL 2 for the tuned crossword SLM.

Runs EVAL 2 (held-out eval.jsonl, BARE deployment prompt) on the merged fine-tuned
model, scored the SAME way as Claude's EVAL 2 (pipeline.eval_opus_fleet.score_one on the
clean English palette + real-dictionary check). This is the base-vs-tuned headline:
Claude Opus scored 0/100 on these identical bare prompts (GAP_ANALYSIS EVAL 2).

Why no separate "as-is" run (Claude EVAL 2 had one): Claude didn't conform to our
generate_crossword contract, so we ran its programs on their own interface to rule out an
API-mismatch artifact. The tuned model is trained to emit a conforming
generate_crossword(topic, word_source, size) FUNCTION, so calling it and scoring the
returned layout (what score_one does) already IS its own-terms test.

    python train/make_colab_eval2.py
"""

import json
import os

MD, CO = "markdown", "code"

REPO_URL = "https://github.com/Avaneesh-Ramesh-07/CrosswordSLM.git"

cells = [
 (MD, "# EVAL 2 — tuned Qwen3-4B (hardcoded-words) on held-out `eval.jsonl` (bare deployment prompt)\n\n"
      "The base-vs-tuned **headline**. Each of the held-out `eval.jsonl` specs is fed to the "
      "tuned model as the **bare** system+user prompt it was trained on (no contract in the "
      "prompt — the contract lives in the weights). Every emitted `generate_crossword` program "
      "is scored through the **same** sandbox + scorer + real-dictionary check as Claude's "
      "EVAL 2 (`pipeline.eval_opus_fleet.score_one`).\n\n"
      "**Model under test:** the **hardcoded-words** variant "
      "(`qwen3-4b-crossword-qlora-hardcoded-merged`) — trained on programs that carry their "
      "vocabulary baked into a `_WORDS` constant (`word_source = word_source or _WORDS`). This "
      "eval still **supplies `word_source`** (the clean palette), exactly like the baseline "
      "tuned run and Claude's EVAL 2, so the numbers are directly comparable — it measures grid "
      "**construction + fill** on identical terms (the supplied palette overrides the baked "
      "`_WORDS`). Testing the self-contained *\"generate me a crossword, no word list given\"* "
      "behavior is a separate variant — see the final cell.\n\n"
      "**Comparison:** unaugmented Claude Opus scored **0/100** on these identical bare prompts "
      "(GAP_ANALYSIS EVAL 2).\n\n"
      "> **No separate \"as-is\" run.** Claude's EVAL 2 also ran each program on its own "
      "interface to rule out an API-mismatch artifact — needed only because Claude didn't "
      "conform to our contract. The tuned model emits a conforming `generate_crossword` "
      "*function*, so calling it and scoring the layout (below) already is the own-terms test. "
      "(Cell 8 optionally dumps the raw programs for manual inspection.)\n\n"
      "**Runtime:** GPU. L4 (24 GB) / A100 (40 GB) recommended; T4 works but generation is slower. "
      "**Order:** run top to bottom."),

 (MD, "## 1. Get the code\n"
      "Clone the repo (set your URL) or upload the folder and set `PROJECT_DIR`. The clone "
      "already contains everything the eval needs: `data/sft/eval.jsonl`, the word lists in "
      "`data/wordlists/` (incl. `words_alpha.txt`), and the `pipeline/` + `harness/` scoring "
      "code.\n\n"
      "> Push your latest local changes first — the eval reflects whatever is committed here."),
 (CO, 'REPO_URL = "%s"\n'
      'import os\n'
      '!git clone -q $REPO_URL slm || echo "clone skipped/failed — upload the folder instead"\n'
      'PROJECT_DIR = "/content/slm"   # adjust if you uploaded elsewhere\n'
      'assert os.path.isdir(os.path.join(PROJECT_DIR, "pipeline")), "Set PROJECT_DIR to the repo root"\n'
      '%%cd $PROJECT_DIR' % REPO_URL),

 (MD, "## 2. GPU + install deps\n"
      "Colab already ships torch. We add `transformers`/`accelerate` (pinned to the training "
      "snapshot) to load the merged model, and `wordfreq` (the English palette intersects the "
      "wordfreq top-N). No bitsandbytes — the merged model loads in 16-bit directly."),
 (CO, 'import torch\n'
      'assert torch.cuda.is_available(), "No GPU — Runtime > Change runtime type > GPU (L4/A100 recommended)"\n'
      'print("GPU:", torch.cuda.get_device_name(0))\n'
      '!pip install -q "transformers==4.53.*" "accelerate==1.8.*" wordfreq'),

 (MD, "> **Expected pip warning — safe to ignore.** Colab's pre-installed `gradio` wants a "
      "newer `huggingface-hub` than `transformers 4.53` pins. `gradio` is unused here; do **not** "
      "upgrade `huggingface-hub` (it would break `transformers`)."),

 (MD, "## 3. Point to the merged tuned model\n"
      "Set `MODEL_DIR` to the **hardcoded** merged model folder "
      "(`…/qwen3-4b-crossword-qlora-hardcoded-merged`) — the full standalone model, not the "
      "adapter. Default is the Drive copy you trained to; if you committed the model into the "
      "repo, use the in-repo path instead (shown commented)."),
 (CO, 'import os\n'
      '# --- Option A (default): merged model on Google Drive ---\n'
      'from google.colab import drive\n'
      'drive.mount("/content/drive")\n'
      'MODEL_DIR = "/content/drive/MyDrive/qwen3-4b-crossword-qlora-hardcoded-merged"\n'
      '# --- Option B: model committed inside the cloned repo (no Drive needed) ---\n'
      '# MODEL_DIR = os.path.join(PROJECT_DIR, "finetuned-models", "hardcoded",\n'
      '#                          "qwen3-4b-crossword-qlora-hardcoded-merged")\n'
      'assert os.path.isdir(MODEL_DIR), f"MODEL_DIR not found: {MODEL_DIR}"\n'
      'assert os.path.exists(os.path.join(MODEL_DIR, "config.json")), \\\n'
      '    "not a full model dir (need config.json + model-*.safetensors, i.e. the -merged folder, not the adapter)"\n'
      'print("model dir OK:", MODEL_DIR)'),

 (MD, "## 4. Load the tuned model\n"
      "First it **repairs shard filenames** if needed: some browsers append a `-NNN` dedup "
      "suffix on download (e.g. `model-00001-of-00002-002.safetensors`), which no longer matches "
      "`model.safetensors.index.json` and makes `from_pretrained` fail. This renames them back."),
 (CO, 'import os, re, json\n'
      '# --- repair browser-suffixed shard names so they match the index (idempotent) ---\n'
      '_idx = os.path.join(MODEL_DIR, "model.safetensors.index.json")\n'
      'if os.path.exists(_idx):\n'
      '    _expected = set(json.load(open(_idx))["weight_map"].values())\n'
      '    for _fn in os.listdir(MODEL_DIR):\n'
      '        _m = re.match(r"(model-\\d+-of-\\d+)-\\d+\\.safetensors$", _fn)\n'
      '        if _m:\n'
      '            _clean = _m.group(1) + ".safetensors"\n'
      '            if _clean in _expected and not os.path.exists(os.path.join(MODEL_DIR, _clean)):\n'
      '                os.rename(os.path.join(MODEL_DIR, _fn), os.path.join(MODEL_DIR, _clean))\n'
      '                print("repaired shard name:", _fn, "->", _clean)\n\n'
      'import torch\n'
      'from transformers import AutoModelForCausalLM, AutoTokenizer\n'
      'tok = AutoTokenizer.from_pretrained(MODEL_DIR)\n'
      'if tok.pad_token_id is None:\n'
      '    tok.pad_token = tok.eos_token\n'
      'tok.padding_side = "left"   # left-pad so batched generation aligns at the prompt end\n'
      'model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype="auto", device_map="auto")\n'
      'model.eval()\n'
      'print("loaded:", model.config.model_type, "| dtype", next(model.parameters()).dtype, "| device", model.device)'),

 (MD, "## 5. Generation settings + batched helper\n"
      "`GEN_TEMP = 1.0` matches Claude's EVAL 2 (temperature 1.0). Set it to `0.0` for greedy / "
      "deterministic decoding (the tuned model's single best output). `MAX_NEW_TOKENS` is generous; "
      "the tuned programs are compact — raise it only if you see truncated code."),
 (CO, 'GEN_TEMP       = 1.0     # match Claude EVAL 2; use 0.0 for greedy/deterministic\n'
      'MAX_NEW_TOKENS = 3072   # tuned programs are compact; raise if any get truncated\n'
      'BATCH          = 8      # lower if you OOM on a small GPU\n\n'
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

 (MD, "## 6. EVAL 2 — generate on bare eval.jsonl prompts, score through the harness\n"
      "Identical to Claude's EVAL 2: sizes 7/9/11/15, 25 prompts per size (n=100, drawn from "
      "`eval.jsonl` with the same seed), scored by `score_one` on the clean English palette with "
      "a real-dictionary check on every entry."),
 (CO, 'import os, json, time\n'
      '# these modules read ANTHROPIC_* via os.environ.get; set dummies so nothing complains\n'
      'os.environ.setdefault("ANTHROPIC_BASE_URL", ""); os.environ.setdefault("ANTHROPIC_AUTH_TOKEN", "")\n'
      'from pipeline.eval_opus_evalset import load_prompts\n'
      'from pipeline.eval_opus_fleet import score_one, agg, table\n'
      'from pipeline.eval_selfmodel import english_palette\n'
      'from pipeline.eval_harness import extract_code\n\n'
      'SIZES = [7, 9, 11, 15]; PER_SIZE = 25\n'
      'prompts = load_prompts("data/sft/eval.jsonl", SIZES, PER_SIZE)   # (system, user, size), BARE\n'
      'print(f"{len(prompts)} bare prompts")\n'
      'print(f"  example -> system={prompts[0][0]!r}\\n             user={prompts[0][1]!r}")\n'
      'pal = english_palette(max(SIZES))\n\n'
      'print("generating...", flush=True)\n'
      'comps = generate_batch([(s, u) for (s, u, sz) in prompts])\n\n'
      'print("scoring...", flush=True)\n'
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
      'ov = table(f"TUNED Qwen3-4B (hardcoded) on eval.jsonl BARE prompts (n={len(rows)}, temp={GEN_TEMP})", rows, SIZES)'),

 (MD, "## 7. Save results + base-vs-tuned comparison"),
 (CO, 'import os, json, time\n'
      'os.makedirs("runs/eval", exist_ok=True)\n'
      'out = f"runs/eval/tuned_evalset_{int(time.time())}.json"\n'
      'summary = {"model": "qwen3-4b-crossword-qlora-hardcoded-merged", "condition": "bare eval.jsonl prompts (word_source supplied)",\n'
      '           "n": len(rows), "parse_rate": parse_rate, "gen_temp": GEN_TEMP, "overall": ov,\n'
      '           "by_size": {s: agg([r for r in rows if r["size"] == s]) for s in SIZES}}\n'
      'json.dump(summary, open(out, "w", encoding="utf-8"), indent=2)\n'
      'print("wrote", out)\n'
      '!mkdir -p /content/drive/MyDrive/slm_runs/eval && cp "$out" /content/drive/MyDrive/slm_runs/eval/ 2>/dev/null || true\n\n'
      'print("\\n===== EVAL 2 (bare eval.jsonl, harness-scored) — base vs tuned =====")\n'
      'print(f"  Claude Opus 4.8   : valid  0%   fullyOK  0%   within  0%    [GAP_ANALYSIS EVAL 2, n=100]")\n'
      'print(f"  Tuned (hardcoded) : valid {ov[\'valid\']*100:3.0f}%   fullyOK {ov[\'fully\']*100:3.0f}%   "\n'
      '      f"within {ov[\'within\']*100:3.0f}%   (n={len(rows)}, temp={GEN_TEMP})")'),

 (MD, "## 8. (Optional) dump the raw programs for inspection\n"
      "Parallels Claude's saved as-is programs. Writes each emitted program to "
      "`runs/eval/tuned_progs/` so you can eyeball the actual generators the model produced."),
 (CO, 'import os\n'
      'os.makedirs("runs/eval/tuned_progs", exist_ok=True)\n'
      'for k, ((s, u, sz), txt) in enumerate(zip(prompts, comps)):\n'
      '    code = extract_code(txt) or txt\n'
      '    open(f"runs/eval/tuned_progs/prog_{k:03d}_s{sz}.py", "w", encoding="utf-8").write(code)\n'
      'print("saved", len(prompts), "programs under runs/eval/tuned_progs/")'),

 (MD, "## (Optional) Self-contained eval — does the baked `_WORDS` actually work?\n"
      "The eval above **supplies `word_source`**, so it never exercises the hardcoding — it's the "
      "same construction test as the baseline model. To test the point of the hardcoded variant "
      "(a user says *\"generate me a crossword\"* with **no word list**), re-score the same "
      "emitted programs but call them with `word_source=None` so the program must fall back to "
      "its own baked `_WORDS`. The layout is still scored against the clean palette's dictionary, "
      "so a valid grid means the model both **wrote a working generator and baked in real, "
      "placeable words**.\n\n"
      "```python\n"
      "from harness.sandbox import run_candidate\n"
      "from harness.scorer import Spec, score\n"
      "from pipeline.eval_selfmodel import BUDGET\n"
      "from pipeline.eval_opus_fleet import agg, table\n"
      "from pipeline.eval_selfmodel import _norm as norm\n"
      "rows_sc = []\n"
      "for (s, u, sz), txt in zip(prompts, comps):\n"
      "    code = extract_code(txt)\n"
      "    z = {'valid':0,'fully':0,'within':0,'dict_frac':0.0,'coverage':0.0,'crossings':0,'entries':0,'filler':0.0}\n"
      "    if code:\n"
      "        budget = BUDGET.get(sz, sz*2)\n"
      "        res = run_candidate(code, {'topic':'vocabulary','word_source':None,'size':sz,'seed':0},\n"
      "                            timeout_s=budget, mem_mb=1024)          # word_source=None -> uses baked _WORDS\n"
      "        if res['status']=='ok' and res.get('result'):\n"
      "            lay = res['result']\n"
      "            spec = Spec(size=sz, topic_words=tuple(pal['targets']), require_symmetry=False,\n"
      "                        min_word_len=3, time_budget_s=budget)\n"
      "            try:\n"
      "                m = score(lay, spec, pal['allowed'], runtime_s=res['runtime_s'], vocab_set=pal['clean_set'])\n"
      "                ents = [e['answer'] for e in (lay.get('across') or [])+(lay.get('down') or [])\n"
      "                        if len(str(e.get('answer','')))>=3]\n"
      "                df = (sum(1 for w in ents if norm(w) in pal['DICT'])/len(ents)) if ents else 0.0\n"
      "                v = int(m['valid']==1); fl = m['filler_fraction'] or 0.0\n"
      "                z = {'valid':v,'fully':int(v and df>=0.999),\n"
      "                     'within':int(v and fl<=0.30 and res['runtime_s']<=budget),\n"
      "                     'dict_frac':df,'coverage':m['coverage'],'crossings':m['crossings'],\n"
      "                     'entries':m['n_entries'],'filler':fl}\n"
      "            except Exception:\n"
      "                pass\n"
      "    z['size'] = sz; rows_sc.append(z)\n"
      "table(f'SELF-CONTAINED (word_source=None, baked _WORDS) n={len(rows_sc)}', rows_sc, SIZES)\n"
      "```\n\n"
      "If self-contained validity is far below the supplied-`word_source` numbers, the model "
      "learned the algorithm but not to emit a usable `_WORDS`; if they're close, the hardcoding "
      "took. Record the winning numbers in `GAP_ANALYSIS.md` as the tuned column of EVAL 2.\n\n"
      "To also run EVAL 1 (English clean-room) or the Extra Spanish eval, reuse "
      "`english_palette`/`spanish_palette` + `score_one` with the clean-room fleet prompt the same way."),
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
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colab_eval_tuned.ipynb")
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
