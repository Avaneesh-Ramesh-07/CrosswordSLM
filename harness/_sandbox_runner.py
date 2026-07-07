"""Child process that executes one candidate crossword generator.

Invoked as:  python -I _sandbox_runner.py <candidate_path>
Reads a JSON spec {topic, word_source, size, seed} on stdin, calls the
candidate's generate_crossword(topic, word_source, size), and writes
{"ok": true, "result": <layout>} to stdout — or {"ok": false, ...} on error.

An audit hook blocks network / subprocess / file-write before the candidate
runs, so untrusted generator code cannot touch the system.
"""

import importlib.util
import json
import random
import sys
import traceback


def _install_audit_hook():
    def hook(event, args):
        if event in ("socket.socket", "socket.bind", "socket.connect", "subprocess.Popen", "os.system", "os.exec"):
            raise PermissionError(f"blocked: {event}")
        if event == "open":
            mode = args[1] if len(args) > 1 and args[1] else "r"
            if any(m in str(mode) for m in ("w", "a", "x", "+")):
                raise PermissionError("blocked: file write")

    sys.addaudithook(hook)


def main():
    candidate_path = sys.argv[1]
    spec = json.loads(sys.stdin.read())
    random.seed(spec.get("seed", 0))

    _install_audit_hook()
    try:
        module_spec = importlib.util.spec_from_file_location("candidate", candidate_path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        fn = getattr(module, "generate_crossword")
        result = fn(spec["topic"], spec["word_source"], spec["size"])
    except MemoryError:
        sys.stderr.write("MemoryError\n")
        sys.exit(4)
    except Exception:
        sys.stderr.write(traceback.format_exc())
        sys.exit(3)

    try:
        sys.stdout.write(json.dumps({"ok": True, "result": result}))
    except (TypeError, ValueError):
        sys.stderr.write("result not JSON-serializable\n")
        sys.exit(5)


if __name__ == "__main__":
    main()
