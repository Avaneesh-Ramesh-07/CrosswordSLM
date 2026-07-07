"""Exercise the sandbox failure taxonomy. Run on Linux (WSL/Colab) for the full
memory-cap + process-group-kill behavior:  python3 tests/test_sandbox.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.sandbox import run_candidate  # noqa: E402

PASS = 0

SPEC = {"topic": "animals", "word_source": ["CAT", "DOG"], "size": 5, "seed": 1}

GOOD = """
def generate_crossword(topic, word_source, size):
    return {"rows": size, "cols": size, "cells": [], "across": [], "down": []}
"""

LOOP = """
def generate_crossword(topic, word_source, size):
    while True:
        pass
"""

RAISES = """
def generate_crossword(topic, word_source, size):
    raise ValueError("boom")
"""

BANNED = """
import socket
def generate_crossword(topic, word_source, size):
    return {}
"""

NO_FN = """
def something_else():
    return 1
"""

SYNTAX = """
def generate_crossword(topic, word_source, size)
    return {}
"""

FILE_WRITE = """
def generate_crossword(topic, word_source, size):
    open("/tmp/should_not_write.txt", "w").write("x")
    return {}
"""

OOM = """
def generate_crossword(topic, word_source, size):
    b = bytearray(3 * 1024 * 1024 * 1024)
    return {"rows": size}
"""


def check(name, cond, detail=""):
    global PASS
    mark = "PASS" if cond else "FAIL"
    print(f"[{mark}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        raise AssertionError(f"{name}: {detail}")
    PASS += 1


def main():
    r = run_candidate(GOOD, SPEC, timeout_s=5)
    check("good -> ok", r["status"] == "ok", str(r))
    check("good -> returns dict", isinstance(r["result"], dict))
    check("good -> runtime measured", r["runtime_s"] >= 0)

    r = run_candidate(LOOP, SPEC, timeout_s=1)
    check("infinite loop -> timeout", r["status"] == "timeout", str(r))

    r = run_candidate(RAISES, SPEC, timeout_s=5)
    check("raise -> exception", r["status"] == "exception", str(r))
    check("raise -> stderr captured", "boom" in r["stderr"])

    r = run_candidate(BANNED, SPEC, timeout_s=5)
    check("import socket -> banned_import", r["status"] == "banned_import", str(r))

    r = run_candidate(NO_FN, SPEC, timeout_s=5)
    check("no generate_crossword -> no_function", r["status"] == "no_function", str(r))

    r = run_candidate(SYNTAX, SPEC, timeout_s=5)
    check("syntax error -> syntax_error", r["status"] == "syntax_error", str(r))

    r = run_candidate(FILE_WRITE, SPEC, timeout_s=5)
    check("file write blocked -> exception", r["status"] == "exception", str(r))

    if os.name == "posix":
        r = run_candidate(OOM, SPEC, timeout_s=10, mem_mb=512)
        check("huge alloc -> oom", r["status"] == "oom", str(r))
    else:
        print("[SKIP] oom (needs POSIX RLIMIT_AS)")

    print(f"\nAll {PASS} sandbox checks passed.")


if __name__ == "__main__":
    main()
