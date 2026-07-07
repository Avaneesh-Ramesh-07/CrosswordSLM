"""Query Qwen3-4B (OpenAI-compatible endpoint) to generate a vocabulary crossword.

Dependency-free: stdlib only (urllib), so it runs on Python 3.14 with no pip installs.
Works against any OpenAI-compatible server:
  - vLLM   : --base-url http://localhost:8000/v1  --model Qwen/Qwen3-4B
  - Ollama : --base-url http://localhost:11434/v1 --model qwen3:4b
  - Colab  : --base-url https://<tunnel>/v1        --model Qwen/Qwen3-4B

The prompt asks the model to render an N x N crossword GRID plus numbered
Across/Down clues whose answers are SAT-level vocabulary, seeded from a word
bank sampled out of data/wordlists/sat_words.txt. Prints Qwen's raw output.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import urllib.error
import urllib.request

_ROOT = os.path.dirname(os.path.abspath(__file__))
_SAT = os.path.join(_ROOT, "data", "wordlists", "sat_words.txt")


def load_word_bank(size, n_sample, seed):
    """Sample SAT words that could plausibly fit an N x N grid (len 3..size)."""
    words = []
    if os.path.exists(_SAT):
        with open(_SAT, encoding="utf-8") as fh:
            for line in fh:
                w = line.strip().upper()
                if w.isalpha() and 3 <= len(w) <= size:
                    words.append(w)
    rng = random.Random(seed)
    rng.shuffle(words)
    return sorted(words[:n_sample])


def build_messages(size, word_bank, think):
    system = (
        "You are an expert crossword constructor specializing in educational "
        "vocabulary puzzles. You build valid, fully-interlocking American-style "
        "crossword grids and write precise, dictionary-style clues. You follow "
        "the requested output format exactly."
    )
    bank = ", ".join(word_bank)
    user = f"""Construct a {size}x{size} crossword puzzle whose answers are SAT-level vocabulary words.

RULES
- The grid is exactly {size} rows by {size} columns.
- Use '#' for a black (blocked) square and a single UPPERCASE letter for a filled square.
- Every horizontal run of 2+ letters (Across) and every vertical run of 2+ letters (Down) must spell a real word. Minimum answer length is 3.
- Answers must interlock: crossing entries share the letter at the cell where they cross. No isolated single letters.
- Aim for 10-16 answers total. Draw answers primarily from the SAT WORD BANK below; you may add a few common connector words only if a valid fill requires it.
- Clues are concise dictionary-style definitions (NOT fill-in-the-blank).

OUTPUT FORMAT (exactly these three sections, in order):
GRID:
<{size} lines, each with {size} tokens separated by single spaces; each token is one UPPERCASE letter or '#'>

ACROSS:
<one line per entry, formatted "N. clue (ANSWER)">

DOWN:
<one line per entry, formatted "N. clue (ANSWER)">

NUMBERING: number the cells in standard crossword order (scan left-to-right, top-to-bottom); a white cell gets the next number if it begins an Across entry, a Down entry, or both.

SAT WORD BANK (prefer interlocking words from this list):
{bank}
"""
    if not think:
        user += "\n/no_think"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_chat(base_url, api_key, model, messages, temperature, top_p, max_tokens, timeout):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser(description="Generate a SAT-vocab crossword with Qwen3-4B.")
    ap.add_argument("--base-url", default=os.environ.get("QWEN_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("QWEN_MODEL", "Qwen/Qwen3-4B"))
    ap.add_argument("--api-key", default=os.environ.get("QWEN_API_KEY", "EMPTY"))
    ap.add_argument("--size", type=int, default=11, help="grid dimension (NxN)")
    ap.add_argument("--bank-size", type=int, default=120, help="how many SAT words to offer the model")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-think", action="store_true", help="disable Qwen3 thinking (faster on CPU)")
    ap.add_argument("--show-prompt", action="store_true", help="print the constructed prompt and exit")
    args = ap.parse_args()

    bank = load_word_bank(args.size, args.bank_size, args.seed)
    if not bank:
        print(f"WARNING: no SAT words loaded from {_SAT}", file=sys.stderr)
    messages = build_messages(args.size, bank, think=not args.no_think)

    if args.show_prompt:
        print(messages[0]["content"], "\n\n---\n", messages[1]["content"], sep="")
        return

    print(f"[querying {args.model} at {args.base_url} | {args.size}x{args.size} | "
          f"{len(bank)} SAT words | think={not args.no_think}]", file=sys.stderr)
    try:
        out = call_chat(args.base_url, args.api_key, args.model, messages,
                        args.temperature, args.top_p, args.max_tokens, args.timeout)
    except urllib.error.URLError as e:
        print(f"\nERROR: could not reach {args.base_url} ({e}).\n"
              "No model server is running. Start one, e.g.:\n"
              "  Ollama : ollama serve && ollama pull qwen3:4b   "
              "then --base-url http://localhost:11434/v1 --model qwen3:4b\n"
              "  vLLM   : python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-4B\n"
              "  Colab  : pass the tunnel URL via --base-url https://<tunnel>/v1",
              file=sys.stderr)
        sys.exit(1)

    print("\n===== QWEN OUTPUT =====\n")
    print(out)


if __name__ == "__main__":
    main()
