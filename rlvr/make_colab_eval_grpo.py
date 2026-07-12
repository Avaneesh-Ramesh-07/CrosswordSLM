"""Generate rlvr/colab_eval_grpo.ipynb — in-process eval of the GRPO (and SFT) adapter.

Base + LoRA adapter (NO merge needed) on the pristine held-out data/sft_hardcoded_words/
eval.jsonl prompts. Each generation is scored with the SAME verifier the reward used
(rlvr.reward.evaluate_text), so eval numbers are comparable to what GRPO optimized.
Evaluates every adapter in ADAPTERS so you get SFT-vs-GRPO side by side.

    python rlvr/make_colab_eval_grpo.py
"""

import json
import os

MD, CO = "markdown", "code"
REPO_URL = "https://github.com/Avaneesh-Ramesh-07/CrosswordSLM.git"

cells = [
 (MD, "# Eval: GRPO (and SFT) adapter on held-out `eval.jsonl`\n\n"
      "Loads **base + LoRA adapter** (no merge) and generates on the pristine held-out prompts, "
      "then scores each program with the training verifier (`rlvr.reward.evaluate_text`): valid, "
      "vocab %, crossings, invalid-crossing/entry, black-square delta, memorized. Evaluates the "
      "GRPO adapter by default; add an `sft` entry to `ADAPTERS` to print SFT vs GRPO side by side. "
      "HF generation (no vLLM). **L4/A100.** Run top to bottom."),

 (MD, "## 1. Install (no -U; remove the two packages that break peft/import)"),
 (CO, "!pip uninstall -y -q vllm torchao   # torchao 0.10 breaks recent peft; vllm not needed for eval\n"
      "!pip install -q wordfreq             # reward palette; keep Colab's torch/transformers/peft\n"
      "import os\n"
      "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n"
      "import torch, transformers, peft\n"
      "print(f'torch {torch.__version__} | transformers {transformers.__version__} | peft {peft.__version__}')"),

 (MD, "## 2. Get the repo + mount Drive\n"
      "Needs `rlvr/`, `harness/`, `pipeline/`, `data/sft_hardcoded_words/`, `data/wordlists/`."),
 (CO, f'REPO_URL = "{REPO_URL}"\n' + r'''import os, sys, torch
if os.path.exists("/content/slm"):
    !git -C /content/slm pull -q            # refresh: reruns pick up latest rlvr/ (e.g. evaluate_text)
else:
    !git clone -q $REPO_URL /content/slm
os.chdir("/content/slm"); sys.path.insert(0, "/content/slm")
assert torch.cuda.is_available(), "No GPU -- Runtime > Change runtime type > L4/A100"
from google.colab import drive
drive.mount("/content/drive")
print("GPU:", torch.cuda.get_device_name(0))'''),

 (MD, "## 3. Config — which adapters, how many samples\n"
      "`ADAPTERS` maps a label -> the LoRA adapter dir on Drive. Comment one out to eval just the "
      "other. `SAMPLES` completions per prompt (pass@k + averaged metrics). Generation of ~3-4k-token "
      "programs is slow in HF, so keep SAMPLES modest."),
 (CO, r'''CKPT = "/content/drive/MyDrive/slm_ckpt"
ADAPTERS = {
    "grpo": f"{CKPT}/qwen3-4b-crossword-grpo",
    # to also compare the SFT baseline, add:
    # "sft": f"{CKPT}/qwen3-4b-crossword-qlora-hardcoded",
}
SIZES          = (7, 9, 11)   # 15 is slow (long template fills); add if you want
SAMPLES        = 2            # completions per prompt
TEMPERATURE    = 0.7
MAX_NEW_TOKENS = 4096         # programs are ~3-4k tokens; don't truncate
for name, path in ADAPTERS.items():
    assert os.path.exists(os.path.join(path, "adapter_config.json")), f"{name}: no adapter at {path}"
print("evaluating:", list(ADAPTERS), "| sizes", SIZES, "| samples", SAMPLES)'''),

 (MD, "## 4. Held-out prompts + reward palette"),
 (CO, r'''from rlvr.prompts import held_out_eval_prompts
from rlvr.reward import RewardConfig, get_palette, get_dictionary, get_vocab_set, evaluate_text

cfg = RewardConfig()                       # same scoring config as training (symmetry off, etc.)
get_palette(); get_dictionary(); get_vocab_set()   # warm caches
prompts = [p for p in held_out_eval_prompts(sizes=SIZES)]
print(f"{len(prompts)} held-out eval prompts (never trained on):",
      {s: sum(1 for p in prompts if p["size"] == s) for s in SIZES})'''),

 (MD, "## 5. Model loader (base + adapter, bf16, no merge) + batched generation"),
 (CO, r'''import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load(adapter_dir):
    base_id = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))["base_model_name_or_path"]
    tok = AutoTokenizer.from_pretrained(adapter_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, adapter_dir).eval()
    return model, tok

@torch.no_grad()
def generate(model, tok, messages, n, temp, max_new):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**inp, do_sample=temp > 0, temperature=max(temp, 1e-5), top_p=0.95,
                         num_return_sequences=n, max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
    return [tok.decode(g[inp["input_ids"].shape[1]:], skip_special_tokens=True) for g in out]'''),

 (MD, "## 6. Run the eval (generate -> verifier-score every sample)"),
 (CO, r'''import gc, time
records = {}
for name, adapter in ADAPTERS.items():
    print(f"\n=== {name}: loading {adapter} ===", flush=True)
    model, tok = load(adapter)
    recs, t0 = [], time.time()
    for i, p in enumerate(prompts):
        comps = generate(model, tok, p["prompt"], SAMPLES, TEMPERATURE, MAX_NEW_TOKENS)
        for c in comps:
            r = evaluate_text(c, p["size"], get_palette(), cfg)
            r["pid"] = i
            recs.append(r)
        print(f"  [{name} {i+1}/{len(prompts)}] size{p['size']} "
              f"valid={[x['valid'] for x in recs[-SAMPLES:]]} "
              f"reward={[x['reward'] for x in recs[-SAMPLES:]]} ({time.time()-t0:.0f}s)", flush=True)
    records[name] = recs
    del model; gc.collect(); torch.cuda.empty_cache()'''),

 (MD, "## 7. Aggregate + SFT-vs-GRPO table"),
 (CO, r'''def agg(recs):
    n = len(recs); ran = [r for r in recs if r["ran"]]
    def mean(k, rows=ran): return round(sum(r[k] for r in rows) / len(rows), 3) if rows else 0.0
    by_pid = {}
    for r in recs: by_pid.setdefault(r["pid"], []).append(r)
    passk = round(sum(1 for v in by_pid.values() if any(x["valid"] for x in v)) / len(by_pid), 3) if by_pid else 0.0
    return {"n": n, "valid_rate": round(sum(r["valid"] for r in recs) / n, 3) if n else 0.0,
            "pass@k": passk, "mean_reward": round(sum(r["reward"] for r in recs) / n, 3) if n else 0.0,
            "vocab_frac": mean("vocab_fraction"), "crossings": mean("crossings"),
            "inv_cross": mean("invalid_crossing_frac"), "inv_entry": mean("invalid_entry_frac"),
            "black_gap": (round(sum(abs(r["black_squares"] - r["black_target"]) for r in ran) / len(ran), 2)
                          if ran else 0.0),
            "memorized": round(sum(r["memorized"] for r in recs) / n, 3) if n else 0.0}

cols = ["n", "valid_rate", "pass@k", "mean_reward", "vocab_frac", "crossings", "inv_cross", "inv_entry", "black_gap", "memorized"]
print(f"\n{'model':<8}" + "".join(f"{c:>12}" for c in cols))
print("-" * (8 + 12 * len(cols)))
summary = {}
for name, recs in records.items():
    a = agg(recs); summary[name] = a
    print(f"{name:<8}" + "".join(f"{a[c]:>12}" for c in cols))
    for s in SIZES:
        sub = [r for r in recs if r["size"] == s]
        if sub:
            asub = agg(sub)
            print(f"  size{s:<3}" + "".join(f"{asub[c]:>12}" for c in cols))'''),

 (MD, "## 8. Save results to Drive"),
 (CO, r'''import json, time, os
os.makedirs("runs/eval", exist_ok=True)
out = f"runs/eval/grpo_eval_{int(time.time())}.json"
json.dump({"summary": summary, "config": {"sizes": list(SIZES), "samples": SAMPLES,
           "temperature": TEMPERATURE}, "records": records}, open(out, "w"), indent=1)
!mkdir -p /content/drive/MyDrive/slm_runs/eval
!cp {out} /content/drive/MyDrive/slm_runs/eval/ 2>/dev/null; echo "saved {out} to Drive"'''),

 (MD, "## Reading it\n"
      "**valid_rate / pass@k** = fully-valid crosswords (strict, symmetry off). **vocab_frac** = share "
      "of answers in the purified list. **crossings** = interlock count. **inv_cross/inv_entry** should "
      "be ~0. **black_gap** = distance from the size's black-square target. **memorized** should stay "
      "low. Held-out prompts were never trained on, so this is a fair generalization test. Add an "
      "`sft` entry to `ADAPTERS` to get the SFT-vs-GRPO delta (RLVR should lift valid_rate / crossings "
      "/ lower inv_* without tanking vocab_frac)."),
]


def build():
    nb_cells = []
    for kind, src in cells:
        cell = {"cell_type": kind, "metadata": {}, "source": src}
        if kind == CO:
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)
    nb = {"cells": nb_cells,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}, "accelerator": "GPU",
                       "colab": {"provenance": []}},
          "nbformat": 4, "nbformat_minor": 5}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colab_eval_grpo.ipynb")
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
