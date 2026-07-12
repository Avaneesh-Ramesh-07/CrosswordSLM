"""Single source of truth for the crossword task-contract prompt.

Used identically by the SFT dataset targets (`pipeline/build_dataset.py`) and the
Claude EVAL 1 clean-room fleet (`pipeline/eval_opus_fleet.py`), so the tuned model is
trained on exactly the prompt it is evaluated with.

No symmetry requirement: the scorer does not gate `valid` on 180-degree rotational
symmetry (`harness/scorer.py` defaults `require_symmetry=False` and every caller passes
False), so the contract no longer asks for it.
"""

SYSTEM = "You are an expert Python programmer. When asked for code, output only code."

# Size-AGNOSTIC form (kept for the historical EVAL 1 which used it).
USER_CONTRACT = '''Write Python code to generate a fixed-grid, American-style crossword. Output ONLY the code in a single response. The main function in your code MUST BE: "def generate_crossword(topic: str, word_source, size: int) -> dict". This is the only one that works with our testing harness

Requirements:
- Python standard library ONLY.
- word_source is a dict {"theme": [...], "fill": [...]} of UPPERCASE words. Use ONLY these words; never invent or hardcode answer words.
- CONSTRUCT and FILL the grid, then return:
  {"rows": int, "cols": int,
   "cells": [{"r","c","letter","number"(optional)}],
   "across": [{"number","row","col","answer","len"}], "down": [ ...same... ]}
- Satisfy ALL: exactly size x size; every white run (across and down) >= 3 letters; every white cell part of BOTH an across and a down entry; all white cells form one connected region; every entry a real word from word_source; high white-square density.
- Handle sizes 7, 9, 11, and 15.

Output only the Python code.'''

# Size-SPECIFIC form (used by the SFT dataset targets and the EVAL 3 baseline): the concrete
# N x N is baked into the request. `{N}` is the only substituted token -- the schema's literal
# braces are left intact by a plain .replace (NOT str.format / f-string).
_SIZE_TEMPLATE = '''Write Python code to generate a {N}x{N}, fixed-grid, American-style crossword. Output ONLY the code in a single response. The main function in your code MUST BE: "def generate_crossword(topic: str, word_source, size: int) -> dict". This is the only one that works with our testing harness

Requirements:
- Python standard library ONLY.
- word_source is a dict {"theme": [...], "fill": [...]} of UPPERCASE words. Use ONLY these words; never invent or hardcode answer words.
- CONSTRUCT and FILL the grid, then return:
  {"rows": int, "cols": int,
   "cells": [{"r","c","letter","number"(optional)}],
   "across": [{"number","row","col","answer","len"}], "down": [ ...same... ]}
- Satisfy ALL: exactly {N}x{N}; every white run (across and down) >= 3 letters; every white cell part of BOTH an across and a down entry; all white cells form one connected region; every entry a real word from word_source; high white-square density.

Output only the Python code.'''


def user_contract(size) -> str:
    """The size-specific task contract for an N x N grid (N = 7, 9, 11, ...)."""
    return _SIZE_TEMPLATE.replace("{N}", str(int(size)))
