"""Task accuracy for T12: GSM8K exact match, MBPP / HumanEval pass@1.

The T12 driver (``scripts/run_t12.py``) writes completions only; this module
turns one completion into a verdict, and ``scripts/score_t12.py`` aggregates
the verdicts per configuration.

* GSM8K: the reference is the number after ``####``.  The prediction is the
  number after the model's own ``####`` if it wrote one, else inside the last
  ``\\boxed{}``, else after the last "answer is", else the last number in the
  text.  Both sides are parsed as exact fractions (commas, ``$`` and ``%``
  dropped), so ``18``, ``18.0`` and ``$18`` all match ``18``.  A ``-`` counts
  as a sign only when it does not follow a word character, so ``10-18``
  extracts ``18``.  Each verdict also carries lm-eval's two ``gsm8k`` metrics,
  ``lmeval_strict`` and ``lmeval_flexible``, reproduced from its
  ``gsm8k.yaml`` (filters and ``exact_match`` normalisation), for comparing
  with published numbers.  Those keep lm-eval's own conventions, ``10-18``
  -> ``-18`` included.
* MBPP: the code is the last fenced block that defines a function.  Without a
  fence it is the completion, cut to the first ``def``/``class``/``import``
  line and with trailing lines dropped until it compiles, when the raw text
  does not compile.  It is followed by the record's ``test_setup_code`` and its
  three ``test_list`` asserts, which are the asserts the prompt showed.
  ``challenge_test_list`` is not used.
* HumanEval: the prompt is a signature plus docstring, so a completion either
  rewrites the whole function or gives only its body.  If the extracted code
  defines ``entry_point`` it is run after the prompt (keeping the prompt's
  imports and helpers), otherwise it is appended to the prompt.  Then the
  record's ``check(entry_point)`` runs.

Code runs in a fresh ``python -I`` subprocess in its own session and temporary
directory, with a wall-clock timeout and (on Linux) CPU, memory and file-size
limits.  A small runner ``exec``s the program with ``__name__`` not
``"__main__"`` (so ``unittest.main()`` and input-reading ``__main__`` blocks
stay inert), counts any exception, ``SystemExit`` included, as a failure, and
writes a per-call nonce only after the last assert.  A pass is exit status 0
*and* the nonce, so ``exit()``, ``sys.exit(0)`` and ``os._exit(0)`` before the
asserts fail.  The whole process group is killed afterwards, stdout is
discarded and only the tail of stderr is read.  That stops runaway or crashing
code; it is not a security sandbox, so only score completions from our own
models.
"""

from __future__ import annotations

import os
import re
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import warnings
from fractions import Fraction

TIMEOUT_S = 10.0
_MEM_BYTES = 2 << 30
_FSIZE_BYTES = 16 << 20
_STDERR_TAIL = 4096

_NUM = r"(?:(?<![\w.])-)?\$?\s*\d[\d,]*(?:\.\d+)?"
_NUM_RE = re.compile(_NUM)
_HASH_RE = re.compile(r"####\s*(" + _NUM + ")")
_ANSWER_IS_RE = re.compile(r"answer\s*(?:is|:)\s*:?\s*\**\s*(" + _NUM + ")", re.IGNORECASE)
_FENCE_RE = re.compile(r"```[ \t]*([\w+-]*)[^\n]*\n(.*?)(?:```|\Z)", re.DOTALL)
_CODE_START_RE = re.compile(r"^(def|class|import|from)\s")

# lm-eval gsm8k.yaml: strict-match / flexible-extract filters, exact_match settings
_LM_STRICT = re.compile(r"#### (\-?[0-9\.\,]+)")
_LM_FLEX = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
_LM_IGNORE = [re.compile(p) for p in (",", r"\$", r"(?s).*#### ", r"\.$")]

_COMPILE_LOCK = threading.Lock()

# Runs in the child: limits, then the program under a non-__main__ name, then the nonce.
_RUNNER = r'''
import os, sys
ok_path, cpu, mem, fsize = sys.argv[1], *map(int, sys.argv[2:5])
nonce = sys.stdin.read()
try:
    import resource
    for lim, v in ((resource.RLIMIT_CPU, cpu), (resource.RLIMIT_AS, mem),
                   (resource.RLIMIT_FSIZE, fsize)):
        resource.setrlimit(lim, (v, v))
except (ImportError, ValueError, OSError):
    pass
sys.argv = ["solution.py"]
try:
    with open("solution.py", encoding="utf-8") as fh:
        src = fh.read()
    exec(compile(src, "solution.py", "exec"),
         {"__name__": "c3_solution", "__builtins__": __builtins__})
except BaseException as exc:
    sys.stderr.write(f"\n{type(exc).__name__}: {exc}"[:300] + "\n")
    sys.stderr.flush()
    os._exit(1)
with open(ok_path, "w") as fh:
    fh.write(nonce)
os._exit(0)
'''


def parse_number(text: str | None) -> Fraction | None:
    """An exact value for ``-1,234.50`` / ``$18`` / ``75%``; None if there is none."""
    if text is None:
        return None
    m = _NUM_RE.search(text)
    if m is None:
        return None
    s = m.group(0).replace(",", "").replace("$", "").replace(" ", "")
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def gsm8k_reference(answer: str) -> Fraction:
    """The gold number of a GSM8K ``answer`` field (the text after ``####``)."""
    m = _HASH_RE.search(answer)
    if m is None:
        raise ValueError(f"GSM8K answer has no '####' line: {answer[-80:]!r}")
    return parse_number(m.group(1))


def _boxed(text: str) -> list[str]:
    """Contents of every ``\\boxed{...}``, nested braces included."""
    out, i = [], text.find("\\boxed{")
    while i >= 0:
        start = j = i + len("\\boxed{")
        depth = 1
        while j < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        if depth == 0:
            out.append(text[start:j - 1])
        i = text.find("\\boxed{", j)
    return out


def gsm8k_extract(completion: str) -> str | None:
    """The span of a completion that holds its final answer (see module docstring)."""
    for hits in (_HASH_RE.findall(completion), _boxed(completion),
                 _ANSWER_IS_RE.findall(completion)):
        hits = [h for h in hits if _NUM_RE.search(h)]
        if hits:
            return hits[-1].strip()
    nums = _NUM_RE.findall(completion)
    return nums[-1].strip() if nums else None


def _lm_filter(regex: re.Pattern, group_select: int, text: str) -> str:
    """lm-eval's ``RegexFilter`` for one response."""
    match = regex.findall(text)
    if not match:
        return "[invalid]"
    match = match[group_select]
    if isinstance(match, tuple):
        match = [m for m in match if m]
        match = match[0] if match else "[invalid]"
    return match.strip()


def _lm_norm(s: str) -> str:
    for rx in _LM_IGNORE:
        s = rx.sub("", s)
    return s.lower()


def lmeval_gsm8k(completion: str, answer: str) -> tuple[bool, bool]:
    """lm-eval ``gsm8k`` (strict-match, flexible-extract) exact_match for one completion."""
    ref = _lm_norm(answer)
    strict = _lm_norm(_lm_filter(_LM_STRICT, 0, completion)) == ref
    flexible = _lm_norm(_lm_filter(_LM_FLEX, -1, completion)) == ref
    return strict, flexible


def _compiles(src: str) -> bool:
    with _COMPILE_LOCK, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            compile(src, "<completion>", "exec")
            return True
        except (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError):
            return False


def _trim_to_code(text: str, prefix: str) -> str:
    """Unfenced code with prose around it: see the module docstring."""
    if _compiles(prefix + text):
        return text
    lines = text.splitlines()
    starts = [0] + [next((i for i, ln in enumerate(lines) if _CODE_START_RE.match(ln)), 0)]
    for start in dict.fromkeys(starts):
        for end in range(len(lines), start, -1):
            cand = "\n".join(lines[start:end])
            if _compiles(prefix + cand):
                return cand
    return "\n".join(lines[starts[-1]:])


def extract_code(completion: str, prefix: str = "") -> str:
    """The code in a completion: the last fenced block that defines something.

    Falls back to the last fenced block, then to the unfenced completion
    trimmed to code (``prefix`` is prepended for the compile check, e.g. the
    HumanEval prompt).  An unclosed final fence (a completion cut at the
    token limit) still counts.
    """
    blocks = [body for lang, body in _FENCE_RE.findall(completion)
              if lang.lower() in ("", "python", "py", "python3")]
    for body in reversed(blocks):
        if re.search(r"^\s*(def|class)\s", body, re.MULTILINE):
            return body.strip("\n")
    if blocks:
        return blocks[-1].strip("\n")
    return _trim_to_code(completion.strip("\n"), prefix)


def _stderr_tail(path: str) -> str | None:
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        fh.seek(max(0, fh.tell() - _STDERR_TAIL))
        lines = fh.read().decode("utf-8", "replace").strip().splitlines()
    return lines[-1][-300:] if lines else None


def run_program(source: str, timeout: float = TIMEOUT_S) -> tuple[bool, str | None]:
    """Run ``source`` in a fresh interpreter; (passed, error tail or None).

    Passed means exit status 0 and the runner's nonce written after the source
    finished (see the module docstring).
    """
    nonce = secrets.token_hex(16)
    with tempfile.TemporaryDirectory(prefix="c3score_", ignore_cleanup_errors=True) as tmp:
        for name, text in (("solution.py", source), ("runner.py", _RUNNER)):
            with open(os.path.join(tmp, name), "w", encoding="utf-8") as fh:
                fh.write(text)
        ok_path, err_path = os.path.join(tmp, "ok"), os.path.join(tmp, "stderr")
        cmd = [sys.executable, "-I", os.path.join(tmp, "runner.py"), ok_path,
               str(int(timeout) + 1), str(_MEM_BYTES), str(_FSIZE_BYTES)]
        timed_out = False
        with open(err_path, "wb") as ferr:
            p = subprocess.Popen(cmd, cwd=tmp, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                 stderr=ferr, start_new_session=True)
            try:
                try:
                    p.stdin.write(nonce.encode())
                    p.stdin.close()
                except OSError:
                    pass
                p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                p.wait()
        if timed_out:
            return False, "timeout"
        try:
            with open(ok_path, encoding="utf-8") as fh:
                marked = fh.read() == nonce
        except OSError:
            marked = False
        if p.returncode == 0 and marked:
            return True, None
        tail = _stderr_tail(err_path)
        if tail is None:
            tail = f"exit {p.returncode}" if p.returncode else "exited before the tests finished"
        return False, tail


def mbpp_program(rec: dict, code: str) -> str:
    setup = rec.get("test_setup_code") or ""
    # setup goes after the solution: some records' setup builds the solution's classes
    return "\n\n".join(s for s in (code, setup, "\n".join(rec["test_list"])) if s) + "\n"


def humaneval_program(rec: dict, code: str) -> str:
    entry = rec["entry_point"]
    if re.search(rf"^\s*def\s+{re.escape(entry)}\s*\(", code, re.MULTILINE):
        body = rec["prompt"] + "\n    pass\n\n" + code
    else:
        body = rec["prompt"] + code
    return f"{body}\n\n{rec['test']}\n\ncheck({entry})\n"


def score(task: str, rec: dict, completion: str, timeout: float = TIMEOUT_S) -> dict:
    """Score one completion against its ``load_task`` record.

    Returns ``{"correct": bool, "extracted": str | None, "error": str | None}``;
    ``extracted`` is the answer span (GSM8K) or the code that was run.  GSM8K
    verdicts add ``lmeval_strict`` and ``lmeval_flexible`` (bool).
    """
    if task == "gsm8k":
        strict, flexible = lmeval_gsm8k(completion, rec["answer"])
        lm = {"lmeval_strict": strict, "lmeval_flexible": flexible}
        pred = gsm8k_extract(completion)
        value = parse_number(pred)
        if value is None:
            return {"correct": False, "extracted": pred, "error": "no number", **lm}
        return {"correct": value == gsm8k_reference(rec["answer"]), "extracted": pred,
                "error": None, **lm}
    if task in ("mbpp", "humaneval"):
        prefix = rec["prompt"] if task == "humaneval" else ""
        code = extract_code(completion, prefix)
        program = mbpp_program(rec, code) if task == "mbpp" else humaneval_program(rec, code)
        ok, err = run_program(program, timeout)
        return {"correct": ok, "extracted": code, "error": err}
    raise KeyError(f"unknown task {task!r}")


def reference_completion(task: str, rec: dict) -> str:
    """The dataset's own solution, written the way a model would answer."""
    if task == "gsm8k":
        return rec["answer"]
    if task == "mbpp":
        return "```python\n" + rec["code"].replace("\r\n", "\n") + "\n```"
    if task == "humaneval":
        return rec["canonical_solution"]
    raise KeyError(f"unknown task {task!r}")
