"""Safe execution of untrusted (model-generated) crossword generators.

Runs each candidate in a fresh isolated subprocess with a wall-clock timeout, a
memory cap (Linux), and an audit-hook denylist (no network / subprocess / file
write). Returns a structured result whose `status` follows a fixed taxonomy so
the pipeline can bucket failures:

    ok | timeout | exception | oom | bad_schema | no_function
       | banned_import | syntax_error

Threat model: the code is model-written crossword logic, not a determined
adversary. The real hazards are infinite loops, runaway memory, and stray file
writes — all covered. This is not a hardened security boundary.

Linux (Colab/WSL) gets the full treatment (RLIMIT_AS, process-group kill).
On Windows the resource cap is skipped and the child is killed directly, so the
module still runs for development; run real scoring on Linux.
"""

from __future__ import annotations

import ast
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

_RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sandbox_runner.py")
_IS_POSIX = os.name == "posix"

BANNED_MODULES = {
    "socket", "subprocess", "urllib", "requests", "shutil", "http",
    "ftplib", "ctypes", "multiprocessing", "asyncio", "pickle",
}


def static_gate(code: str):
    """Cheap AST pre-screen before we ever execute. Returns (ok, status, detail)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, "syntax_error", f"{exc.msg} (line {exc.lineno})"

    has_fn = any(isinstance(n, ast.FunctionDef) and n.name == "generate_crossword" for n in ast.walk(tree))
    if not has_fn:
        return False, "no_function", "generate_crossword() not defined"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in BANNED_MODULES:
                    return False, "banned_import", alias.name
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in BANNED_MODULES:
                return False, "banned_import", node.module
    return True, "ok", ""


def _fail(status, detail):
    return {"status": status, "result": None, "runtime_s": 0.0, "detail": detail, "stderr": ""}


def run_candidate(code: str, spec: dict, timeout_s: float = 5.0, mem_mb: int = 1536) -> dict:
    """Execute one candidate program against `spec`.

    `spec` is a dict with keys: topic, word_source, size, seed(optional).
    Returns {status, result, runtime_s, detail, stderr}. `result` is the layout
    dict the generator returned (only when status == 'ok').
    """
    ok, status, detail = static_gate(code)
    if not ok:
        return _fail(status, detail)

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
        fh.write(code)
        candidate_path = fh.name

    preexec = None
    if _IS_POSIX:
        def preexec():  # noqa: E306 - set memory cap + new session for group kill
            import resource

            soft = mem_mb * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (soft, soft))
            except (ValueError, OSError):
                pass
            os.setsid()

    start = time.perf_counter()
    try:
        proc = subprocess.Popen(
            [sys.executable, "-I", _RUNNER, candidate_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=preexec,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        try:
            out, err = proc.communicate(input=json.dumps(spec), timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill(proc)
            proc.communicate()
            return {"status": "timeout", "result": None, "runtime_s": timeout_s, "detail": "", "stderr": ""}
    finally:
        runtime_s = time.perf_counter() - start
        try:
            os.unlink(candidate_path)
        except OSError:
            pass

    if proc.returncode != 0:
        status = "oom" if "MemoryError" in (err or "") else "exception"
        return {"status": status, "result": None, "runtime_s": round(runtime_s, 4), "detail": "", "stderr": err[-2000:]}

    try:
        payload = json.loads(out)
    except (json.JSONDecodeError, ValueError):
        return {"status": "bad_schema", "result": None, "runtime_s": round(runtime_s, 4), "detail": "non-JSON stdout", "stderr": err[-2000:]}

    return {"status": "ok", "result": payload.get("result"), "runtime_s": round(runtime_s, 4), "detail": "", "stderr": err[-2000:]}


def _kill(proc):
    try:
        if _IS_POSIX:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, OSError):
        pass
