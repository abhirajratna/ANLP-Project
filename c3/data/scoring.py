"""Task accuracy for T12: GSM8K exact match, MBPP / HumanEval pass@1.

The T12 driver (``scripts/run_t12.py``) writes completions only; this module
turns one completion into a verdict, and ``scripts/score_t12.py`` aggregates
the verdicts per configuration.

* GSM8K: the reference is the number after ``####``.  The prediction is the
  number after the model's own ``####`` if it wrote one, else inside the last
  ``\\boxed{}``, else after the last "answer is", else the last number in the
  text.  Both sides are parsed as exact fractions (commas, ``$`` and ``%``
  dropped), so ``18``, ``18.0`` and ``$18`` all match ``18``.
* MBPP: the code is the last fenced block that defines a function (or the whole
  completion when there is no fence), followed by the record's
  ``test_setup_code`` and its three ``test_list`` asserts, which are the asserts
  the prompt showed.  ``challenge_test_list`` is not used.
* HumanEval: the prompt is a signature plus docstring, so a completion either
  rewrites the whole function or gives only its body.  If the extracted code
  defines ``entry_point`` it is run after the prompt (keeping the prompt's
  imports and helpers), otherwise it is appended to the prompt.  Then the
  record's ``check(entry_point)`` runs.

Code runs in a fresh ``python -I`` subprocess, in a temporary directory, with a
wall-clock timeout and (on Linux) CPU, memory and file-size limits.  That stops
runaway or crashing code; it is not a security sandbox, so only score
completions from our own models.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from fractions import Fraction

TIMEOUT_S = 10.0
_MEM_BYTES = 2 << 30
_FSIZE_BYTES = 16 << 20

_NUM = r"-?\$?\s*\d[\d,]*(?:\.\d+)?"
_NUM_RE = re.compile(_NUM)
_HASH_RE = re.compile(r"####\s*(" + _NUM + ")")
_ANSWER_IS_RE = re.compile(r"answer\s*(?:is|:)\s*:?\s*\**\s*(" + _NUM + ")", re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_FENCE_RE = re.compile(r"```[ \t]*([\w+-]*)[^\n]*\n(.*?)(?:```|\Z)", re.DOTALL)


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


def gsm8k_extract(completion: str) -> str | None:
    """The span of a completion that holds its final answer (see module docstring)."""
    for pattern in (_HASH_RE, _BOXED_RE, _ANSWER_IS_RE):
        hits = pattern.findall(completion)
        hits = [h for h in hits if _NUM_RE.search(h)]
        if hits:
            return hits[-1].strip()
    nums = _NUM_RE.findall(completion)
    return nums[-1].strip() if nums else None


def extract_code(completion: str) -> str:
    """The code in a completion: the last fenced block that defines something.

    Falls back to the last fenced block, then to the raw completion.  An
    unclosed final fence (a completion cut at the token limit) still counts.
    """
    blocks = [body for lang, body in _FENCE_RE.findall(completion)
              if lang.lower() in ("", "python", "py", "python3")]
    for body in reversed(blocks):
        if re.search(r"^\s*(def|class)\s", body, re.MULTILINE):
            return body.strip("\n")
    if blocks:
        return blocks[-1].strip("\n")
    return completion.strip("\n")


def _limits() -> None:  # runs in the child before exec
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (int(TIMEOUT_S) + 1, int(TIMEOUT_S) + 1))
    resource.setrlimit(resource.RLIMIT_AS, (_MEM_BYTES, _MEM_BYTES))
    resource.setrlimit(resource.RLIMIT_FSIZE, (_FSIZE_BYTES, _FSIZE_BYTES))
    os.setsid()


def run_program(source: str, timeout: float = TIMEOUT_S) -> tuple[bool, str | None]:
    """Run ``source`` in a fresh interpreter; (exit status 0, error tail or None)."""
    with tempfile.TemporaryDirectory(prefix="c3score_") as tmp:
        path = os.path.join(tmp, "prog.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(source)
        try:
            p = subprocess.run([sys.executable, "-I", path], cwd=tmp, stdin=subprocess.DEVNULL,
                               capture_output=True, text=True, errors="replace", timeout=timeout,
                               preexec_fn=_limits if sys.platform == "linux" else None)
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if p.returncode == 0:
        return True, None
    tail = (p.stderr.strip().splitlines() or [f"exit {p.returncode}"])[-1]
    return False, tail[-300:]


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
    ``extracted`` is the answer span (GSM8K) or the code that was run.
    """
    if task == "gsm8k":
        pred = gsm8k_extract(completion)
        value = parse_number(pred)
        if value is None:
            return {"correct": False, "extracted": pred, "error": "no number"}
        return {"correct": value == gsm8k_reference(rec["answer"]), "extracted": pred,
                "error": None}
    if task in ("mbpp", "humaneval"):
        code = extract_code(completion)
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
