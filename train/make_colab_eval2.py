"""Generate train/colab_eval_tuned.ipynb — query the tuned SLM and SAVE its programs.

This notebook does ONE job: feed the held-out `eval.jsonl` deployment prompts to the tuned
**hardcoded-words** model and **save every emitted `generate_crossword` program together with
its input spec** into a zip you download. It does NOT score anything.

Gauging happens locally (outside Colab): each saved program is run **self-contained** —
`generate_crossword(topic, word_source=None, size)` so it fills from its OWN baked `_WORDS`
(exercising the hardcoding) — and the returned crossword is checked by a standalone validator
(structure + real-dictionary words), exactly the way the 36 hardcoded dataset programs were
gauged. So Colab only needs a GPU + the model; no palette / wordfreq / scoring code runs here.

    python train/make_colab_eval2.py
"""

import json
import os

MD, CO = "markdown", "code"

REPO_URL = "https://github.com/Avaneesh-Ramesh-07/CrosswordSLM.git"

cells = [
 (MD, "# Generate + save tuned-SLM programs (hardcoded-words model)\n\n"
      "One job: feed the held-out `eval.jsonl` **bare deployment prompts** to the tuned "
      "**hardcoded-words** model (`qwen3-4b-crossword-qlora-hardcoded-merged`) and **save every "
      "emitted `generate_crossword` program + its input spec** into a zip you download. "
      "**No scoring happens here.**\n\n"
      "You then hand the zip to Claude, who **gauges each program locally**: it runs every "
      "program **self-contained** — `generate_crossword(topic, word_source=None, size)` so the "
      "program fills from its OWN baked `_WORDS` (this is what *exercises the hardcoding*) — and "
      "validates the crossword it produces (structure + every entry a real dictionary word). "
      "That's the identical method already verified on the 36 hardcoded dataset programs "
      "(36/36 valid).\n\n"
      "**Runtime:** GPU. L4 (24 GB) / A100 (40 GB) recommended; T4 works but generation is slower. "
      "**Order:** run top to bottom, then download the zip from the last cell."),

 (MD, "## 1. Get the code\n"
      "Clone the repo (set your URL) or upload the folder and set `PROJECT_DIR`. The clone gives "
      "us `data/sft/eval.jsonl` (the prompts) and the tiny prompt/extract helpers in "
      "`pipeline/` — nothing else is needed (no scoring)."),
 (CO, 'REPO_URL = "%s"\n'
      'import os\n'
      '!git clone -q $REPO_URL slm || echo "clone skipped/failed — upload the folder instead"\n'
      'PROJECT_DIR = "/content/slm"   # adjust if you uploaded elsewhere\n'
      'assert os.path.isdir(os.path.join(PROJECT_DIR, "pipeline")), "Set PROJECT_DIR to the repo root"\n'
      '%%cd $PROJECT_DIR' % REPO_URL),

 (MD, "## 2. GPU + install deps\n"
      "Colab already ships torch. We add only `transformers`/`accelerate` (pinned to the training "
      "snapshot) to load the merged model. No `wordfreq`, no `bitsandbytes` — there's no palette "
      "and no scoring here, and the merged model loads in 16-bit directly."),
 (CO, 'import torch\n'
      'assert torch.cuda.is_available(), "No GPU — Runtime > Change runtime type > GPU (L4/A100 recommended)"\n'
      'print("GPU:", torch.cuda.get_device_name(0))\n'
      '!pip install -q "transformers==4.53.*" "accelerate==1.8.*"'),

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
      "`model.safetensors.index.json` and makes `from_pretrained` fail. This renames them back.\n\n"
      "**Load with `attn_implementation=\"sdpa\"` and `.to(\"cuda\")`, *not* `device_map=\"auto\"`.** "
      "Eager attention + accelerate's dispatch hooks make even an A100 crawl (~18 tok/s); SDPA "
      "on-device runs ~60+ tok/s. The 8 GB fp16 model fits on any single target GPU, so there's "
      "no need for `device_map`."),
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
      '# SDPA + on-device (NOT device_map="auto"): eager attention + accelerate dispatch hooks\n'
      '# make an A100 crawl at ~18 tok/s; SDPA on-device gives ~60+.\n'
      'model = AutoModelForCausalLM.from_pretrained(\n'
      '    MODEL_DIR, torch_dtype=torch.float16, attn_implementation="sdpa").to("cuda")\n'
      'model.eval()\n'
      '# Defensive: training sets use_cache=False (for gradient checkpointing); if that flag ever\n'
      '# survived into the merged config it would kill speed (no KV cache -> O(n^2) decode). It is\n'
      '# normally already True in a merged model; force it on for inference to be safe:\n'
      'model.config.use_cache = True\n'
      'print("loaded:", model.config.model_type, "| attn", model.config._attn_implementation,\n'
      '      "| dtype", next(model.parameters()).dtype, "| device", model.device,\n'
      '      "| use_cache", model.config.use_cache, "(must be True for fast generation)")'),

 (MD, "## 5. Generation settings + batched helper\n"
      "`GEN_TEMP = 1.0` gives varied samples per prompt (the prompts of a given size are nearly "
      "identical, so temperature is what produces distinct programs). Set `0.0` for greedy / "
      "deterministic.\n\n"
      "**Two things decide speed — get both right or a single batch can take >10 min:**\n\n"
      "1. **Stop token.** `generate()` stops on the model's `generation_config.eos_token_id`, "
      "which for a merged Qwen model often does **not** include `<|im_end|>` (the token that ends "
      "the assistant turn). If it's missing, every sequence runs to the full `MAX_NEW_TOKENS`. "
      "We pass the stop ids explicitly (`EOS_IDS`, includes `<|im_end|>`) so generation ends as "
      "soon as the program is done.\n"
      "2. **`MAX_NEW_TOKENS`.** Measured on the hardcoded dataset with the model's own "
      "tokenizer, full programs are **7×7 / 9×9 ≤ ~4.1k tokens** and **11×11 up to ~5.9k in the "
      "eval set (~8.85k across all of training)** — the baked word list dominates the length. "
      "The model is fine-tuned at **`max_seq_length = 10240`**, so we set **`MAX_NEW_TOKENS = "
      "9728`**: comfortably above the largest 11×11 program (~8.85k) so no valid generation is "
      "truncated, while prompt+completion stays under the 10240 training ceiling. `BATCH` is "
      "auto-tuned to the GPU (8 on an A100, 4 elsewhere).\n\n"
      "> **15×15 is not evaluated.** It was dropped from training entirely (its ~12–14k-token "
      "programs exceeded the sequence budget), so the model is trained and evaluated on **7/9/11 "
      "only** — matching the EVAL 2 comparison restricted to those sizes (25 each = 75)."),
 (CO, 'GEN_TEMP       = 1.0     # varied samples; use 0.0 for greedy/deterministic\n'
      'MAX_NEW_TOKENS = 9728    # < 10240 training ceiling; > largest 11x11 program (~8.85k tok) so nothing truncates\n\n'
      'import torch, time\n'
      '_vram = torch.cuda.get_device_properties(0).total_memory / 1e9\n'
      '# BATCH is the real speed lever: single-stream decode is kernel-launch/overhead-bound\n'
      '# (~18 tok/s, ~same on T4 and A100), and ONE batched decode step emits BATCH tokens for\n'
      '# ~the same overhead -> aggregate tok/s scales ~linearly with BATCH. Prompts are grouped\n'
      '# by size so each batch is uniform-length (efficient). Bigger = faster until you OOM.\n'
      'BATCH = 16 if _vram >= 38 else (8 if _vram >= 22 else 4)   # A100 -> 16, L4 -> 8, T4 -> 4 (halve if OOM)\n'
      'print(f"VRAM ~{_vram:.0f}GB -> BATCH={BATCH}")\n'
      '# generate() stops on generation_config.eos_token_id, which may NOT include <|im_end|>\n'
      '# (Qwen ends the assistant turn with it). Without it every sequence runs to the cap\n'
      '# (minutes per batch). Pass the stop ids explicitly.\n'
      'EOS_IDS = sorted({x for x in [tok.eos_token_id,\n'
      '                              (tok.convert_tokens_to_ids("<|im_end|>")\n'
      '                               if "<|im_end|>" in tok.get_vocab() else None)] if x is not None})\n'
      'print("stop token ids:", EOS_IDS)\n\n'
      '# --- within-batch progress: generate() is silent until it returns, so a long batch looks\n'
      '# hung. A StoppingCriteria is called EVERY decode step; ours never stops generation, it just\n'
      '# prints tokens-so-far every PROGRESS_EVERY steps. Works for any batch size (TextStreamer\n'
      '# only supports batch=1). Real stopping is still handled by eos_token_id / max_new_tokens.\n'
      'from transformers import StoppingCriteria, StoppingCriteriaList\n'
      'PROGRESS_EVERY = 128   # print progress every N new tokens per sequence (0 = off)\n'
      'class _Progress(StoppingCriteria):\n'
      '    def __init__(self, prompt_len, bsz, every):\n'
      '        self.prompt_len, self.bsz, self.every, self.t0, self.last = prompt_len, bsz, every, time.time(), 0\n'
      '    def __call__(self, input_ids, scores, **kw):\n'
      '        n = input_ids.shape[1] - self.prompt_len\n'
      '        if self.every and n - self.last >= self.every:\n'
      '            self.last = n; dt = time.time() - self.t0\n'
      '            print(f"      +{n} tok/seq  ({n/max(dt,1e-9):.0f}/s per seq, ~{self.bsz*n/max(dt,1e-9):.0f}/s aggregate)", flush=True)\n'
      '        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)\n\n'
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
      '        t = time.time()\n'
      '        _crit = StoppingCriteriaList([_Progress(enc["input_ids"].shape[1], len(chunk), PROGRESS_EVERY)])\n'
      '        gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, use_cache=True,\n'
      '                             do_sample=GEN_TEMP > 0, temperature=max(GEN_TEMP, 1e-5),\n'
      '                             top_p=0.95, pad_token_id=tok.pad_token_id, eos_token_id=EOS_IDS,\n'
      '                             stopping_criteria=_crit)\n'
      '        new = gen[:, enc["input_ids"].shape[1]:]\n'
      '        dt = time.time() - t\n'
      '        outs.extend(tok.batch_decode(new, skip_special_tokens=True))\n'
      '        print(f"  {min(i + BATCH, len(pairs))}/{len(pairs)}  "\n'
      '              f"[{dt:.0f}s  {new.shape[1]} tok/seq  {new.numel()/max(dt,1e-9):.0f} tok/s]", flush=True)\n'
      '    return outs'),

 (MD, "## 5b. Smoke test — confirm generation STOPS early (run this before the full cell)\n"
      "One short generation on a 7×7 prompt. If `stopped_early` is **False**, generation is "
      "running to the cap (the `<|im_end|>` stop isn't taking) and the full run will crawl — fix "
      "that before continuing. Also prints tok/s so you can estimate the full run."),
 (CO, 'import time, torch\n'
      'from pipeline.eval_opus_evalset import load_prompts\n'
      '_s, _u, _sz = load_prompts("data/sft/eval.jsonl", [7], 1)[0]\n'
      '_txt = tok.apply_chat_template([{"role": "system", "content": _s}, {"role": "user", "content": _u}],\n'
      '                               tokenize=False, add_generation_prompt=True)\n'
      '_enc = tok(_txt, return_tensors="pt").to(model.device)\n'
      '_t = time.time()\n'
      'with torch.no_grad():\n'
      '    _g = model.generate(**_enc, max_new_tokens=MAX_NEW_TOKENS, use_cache=True, do_sample=GEN_TEMP > 0,\n'
      '                        temperature=max(GEN_TEMP, 1e-5), top_p=0.95,\n'
      '                        pad_token_id=tok.pad_token_id, eos_token_id=EOS_IDS,\n'
      '                        stopping_criteria=StoppingCriteriaList([_Progress(_enc["input_ids"].shape[1], 1, PROGRESS_EVERY)]))\n'
      '_n = _g.shape[1] - _enc["input_ids"].shape[1]; _dt = time.time() - _t\n'
      '_stopped = _n < MAX_NEW_TOKENS\n'
      'print(f"smoke: size {_sz} -> {_n} new tokens in {_dt:.1f}s ({_n/max(_dt,1e-9):.0f} tok/s) | stopped_early={_stopped}")\n'
      'if not _stopped:\n'
      '    print("\\nWARNING: did NOT stop at <|im_end|> -> the full run will be very slow.")\n'
      '    print("Fix: check EOS_IDS above (must include the <|im_end|> id), or lower MAX_NEW_TOKENS.")\n'
      'else:\n'
      '    print("OK: generation stops on its own. With the KV cache on (use_cache=True), 7/9/11 are fast;")\n'
      '    print("    if this is still ~18 tok/s, check the use_cache line printed at load.")'),

 (MD, "## 6. Generate on the bare `eval.jsonl` prompts, then SAVE programs + specs\n"
      "Sizes 7/9/11 at **25 prompts each = 75** — the EVAL 2 protocol restricted to the trained "
      "sizes (15×15 dropped). Same file + seed as Claude's EVAL 2, so these are the identical "
      "7/9/11 bare prompts. For each prompt we save the extracted program "
      "to `progs/prog_<i>_s<NN>.py` (the size is in the filename so it can be run at the right "
      "size later), the raw completion to `raw/`, and one row per prompt to `specs.jsonl` "
      "(`idx, size, prog_file, parsed, system, user`). **No scoring** — that's done locally."),
 (CO, 'import os, json\n'
      'from pipeline.eval_opus_evalset import load_prompts\n'
      'from pipeline.eval_harness import extract_code\n\n'
      'SIZES = [7, 9, 11]; PER_SIZE = 25   # EVAL 2 protocol on trained sizes: 25/size x [7,9,11] = 75 (15x15 dropped)\n'
      'prompts = load_prompts("data/sft/eval.jsonl", SIZES, PER_SIZE)   # (system, user, size), BARE\n'
      'print(f"{len(prompts)} bare prompts")\n'
      'print(f"  example -> system={prompts[0][0]!r}\\n             user={prompts[0][1]!r}")\n\n'
      'print("generating...", flush=True)\n'
      'comps = generate_batch([(s, u) for (s, u, sz) in prompts])\n\n'
      'OUT = "runs/eval/slm_gen"\n'
      'os.makedirs(os.path.join(OUT, "progs"), exist_ok=True)\n'
      'os.makedirs(os.path.join(OUT, "raw"), exist_ok=True)\n'
      'specs, n_parsed = [], 0\n'
      'for i, ((s, u, sz), txt) in enumerate(zip(prompts, comps)):\n'
      '    code = extract_code(txt)\n'
      '    parsed = bool(code); n_parsed += parsed\n'
      '    # no closing ``` fence -> generation almost certainly hit MAX_NEW_TOKENS mid-code\n'
      '    looks_truncated = txt.count("```") < 2 or "def generate_crossword" not in (code or "")\n'
      '    prog_file = f"progs/prog_{i:03d}_s{sz:02d}.py"\n'
      '    open(os.path.join(OUT, prog_file), "w", encoding="utf-8").write(code or "")\n'
      '    open(os.path.join(OUT, f"raw/comp_{i:03d}.txt"), "w", encoding="utf-8").write(txt)\n'
      '    specs.append({"idx": i, "size": sz, "prog_file": prog_file,\n'
      '                  "parsed": parsed, "looks_truncated": looks_truncated, "system": s, "user": u})\n'
      'with open(os.path.join(OUT, "specs.jsonl"), "w", encoding="utf-8") as fh:\n'
      '    for rec in specs:\n'
      '        fh.write(json.dumps(rec) + "\\n")\n'
      'json.dump({"model": "qwen3-4b-crossword-qlora-hardcoded-merged", "n": len(prompts),\n'
      '           "parsed": n_parsed, "sizes": SIZES, "per_size": PER_SIZE, "gen_temp": GEN_TEMP,\n'
      '           "gauge": "run each prog self-contained: generate_crossword(topic, word_source=None, size)"},\n'
      '          open(os.path.join(OUT, "meta.json"), "w"), indent=2)\n'
      'by_size = {sz: sum(1 for r in specs if r["size"] == sz) for sz in SIZES}\n'
      'n_trunc = sum(r["looks_truncated"] for r in specs)\n'
      'print(f"\\nparsed {n_parsed}/{len(prompts)} as code | by size: {by_size}")\n'
      'print(f"looks-truncated (hit MAX_NEW_TOKENS): {n_trunc}"'
      ' + ("  <-- raise MAX_NEW_TOKENS or expect these to fail" if n_trunc else ""))\n'
      'trunc_by_size = {sz: sum(1 for r in specs if r["size"] == sz and r["looks_truncated"]) for sz in SIZES}\n'
      'print(f"  truncated by size: {trunc_by_size}")\n'
      'print(f"saved programs + specs under {OUT}/")'),

 (MD, "## 7. Package for download\n"
      "Zips `runs/eval/slm_gen/` (programs + `specs.jsonl` + raw completions), copies it to Drive, "
      "and triggers a browser download. **Hand this zip to Claude** — it has everything needed to "
      "gauge the run locally."),
 (CO, 'import shutil, os\n'
      'zip_path = shutil.make_archive("/content/slm_gen", "zip", "runs/eval/slm_gen")\n'
      'print("zip:", zip_path, f"({os.path.getsize(zip_path)/1e6:.1f} MB)")\n'
      '# copy to Drive (so you have a durable copy even if the browser download is flaky)\n'
      'try:\n'
      '    dst = "/content/drive/MyDrive/slm_runs/eval"; os.makedirs(dst, exist_ok=True)\n'
      '    shutil.copy(zip_path, dst); print("copied to", dst)\n'
      'except Exception as e:\n'
      '    print("Drive copy skipped:", e)\n'
      '# direct browser download\n'
      'try:\n'
      '    from google.colab import files; files.download(zip_path)\n'
      'except Exception as e:\n'
      '    print("auto-download unavailable; grab it from the Files panel or Drive:", e)'),

 (MD, "## Next — hand the zip to Claude\n"
      "Download `slm_gen.zip` and give it to Claude. It will, for every `progs/*.py`:\n\n"
      "1. run it **self-contained** — `generate_crossword(\"vocabulary\", word_source=None, "
      "size=<from filename/specs>)` so the program must fill from its own baked `_WORDS`;\n"
      "2. validate the returned crossword with the standalone checker (exactly `size×size`, all "
      "runs ≥ 3, declared entries == actual runs, single connected white region, **every entry a "
      "real dictionary word**);\n"
      "3. report valid% / dict% / crossings / density by size — the same gauge run on the 36 "
      "hardcoded dataset programs (36/36).\n\n"
      "Programs that emitted no `_WORDS` (or expect a supplied `word_source`) will fail when run "
      "with `word_source=None` — that's the signal that the hardcoding didn't take for that sample."),
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
