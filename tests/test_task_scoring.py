"""T12 task scoring: answer extraction, code execution, CLI aggregation.

The full gate (every reference solution scores 100%) needs the task files:
set C3_TASK_DIR to a directory holding gsm8k.jsonl / mbpp.jsonl / humaneval.jsonl
written by ``c3.data.tasks.save_task``; otherwise it is skipped.
"""
import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from c3.data.scoring import (extract_code, gsm8k_extract, parse_number, reference_completion,
                             score)

ROOT = Path(__file__).resolve().parents[1]
GSM = {"doc_id": 0, "question": "q", "answer": "6 * 3 = <<6*3=18>>18\n#### 18"}
MBPP = {"doc_id": 0, "text": "Add two numbers.", "code": "def add(a, b):\r\n  return a + b",
        "test_setup_code": "", "test_list": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]}
HE = {"doc_id": 0, "entry_point": "inc",
      "prompt": "import math\n\ndef inc(x):\n    \"\"\"x + 1\"\"\"\n",
      "canonical_solution": "    return x + 1\n",
      "test": "def check(f):\n    assert f(1) == 2\n    assert f(-1) == 0\n"}


@pytest.mark.parametrize("text,want", [
    ("The answer is 18.", "18"), ("so $1,200 in total. #### 1200", "1200"),
    ("\\boxed{18} and later 7 steps", "18"), ("She has 3 apples then 5.", "5"),
    ("Answer: **-4.5**", "-4.5")])
def test_gsm8k_extract(text, want):
    assert parse_number(gsm8k_extract(text)) == Fraction(want)


def test_gsm8k_score():
    assert score("gsm8k", GSM, "6 times 3 is 18.0")["correct"]
    assert score("gsm8k", GSM, "The total is $18.")["correct"]
    assert not score("gsm8k", GSM, "The answer is 17. Check: 18 = 18")["correct"]
    v = score("gsm8k", GSM, "I do not know")
    assert not v["correct"] and v["error"] == "no number"


def test_extract_code_picks_defining_fence():
    c = "Here:\n```python\ndef f():\n    return 1\n```\nUsage:\n```\nprint(f())\n```"
    assert extract_code(c) == "def f():\n    return 1"
    assert extract_code("```python\ndef g(): pass") == "def g(): pass"  # cut at token limit
    assert extract_code("def h(): pass") == "def h(): pass"


def test_mbpp_score():
    assert score("mbpp", MBPP, reference_completion("mbpp", MBPP))["correct"]
    v = score("mbpp", MBPP, "```python\ndef add(a, b):\n    return a - b\n```")
    assert not v["correct"] and "AssertionError" in v["error"]
    assert score("mbpp", MBPP, "def add(a, b): return a +")["error"]


def test_humaneval_body_or_full_function():
    assert score("humaneval", HE, HE["canonical_solution"])["correct"]
    full = "```python\ndef inc(x):\n    return math.floor(x) + 1\n```"  # uses the prompt's import
    assert score("humaneval", HE, full)["correct"]
    assert not score("humaneval", HE, "    return x\n")["correct"]


def test_timeout_and_isolation(tmp_path):
    v = score("mbpp", MBPP, "def add(a, b):\n    while True: pass", timeout=1.0)
    assert not v["correct"] and v["error"] == "timeout"
    v = score("mbpp", MBPP, f"open({str(tmp_path / 'x')!r}, 'w').write('1')\n"
                            "def add(a, b): return a + b")
    assert v["correct"]  # the program runs, but in its own temporary directory
    v = score("mbpp", MBPP, "import os\nassert os.getcwd() != " + repr(os.getcwd()) +
              "\ndef add(a, b): return a + b")
    assert v["correct"]


def test_cli_aggregates_per_config(tmp_path):
    task = tmp_path / "gsm8k.jsonl"
    recs = [dict(GSM, doc_id=i) for i in range(1319)]  # load_task checks the split size
    task.write_text("".join(json.dumps(r) + "\n" for r in recs))
    run = tmp_path / "run"
    run.mkdir()
    rows = [{"task": "gsm8k", "config": "a", "doc_id": i, "completion": c, "stopped": True}
            for i, c in enumerate(["18", "18", "17"])]
    rows += [{"task": "gsm8k", "config": "b", "doc_id": i, "completion": "18", "stopped": False}
             for i in range(2)]
    rows.append(rows[0])  # a resumed shard repeats a record: scored once
    (run / "t12.shard0of1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "summary.json"
    subprocess.run([sys.executable, str(ROOT / "scripts/score_t12.py"), str(run),
                    "--task-jsonl", str(task), "--out", str(out),
                    "--per-record", str(tmp_path / "v.jsonl")], check=True, capture_output=True)
    s = json.loads(out.read_text())
    by = {r["config"]: r for r in s["rows"]}
    assert (by["a"]["n"], by["a"]["n_correct"]) == (3, 2)
    assert (by["b"]["n"], by["b"]["n_correct"], by["b"]["n_unstopped"]) == (2, 2, 2)
    assert by["a"]["ci95"][0] < 2 / 3 < by["a"]["ci95"][1]
    assert any(w.startswith("gsm8k a:") for w in s["warnings"])  # a has doc 2, b does not
    assert "git_commit" in s["provenance"]
    assert len((tmp_path / "v.jsonl").read_text().splitlines()) == 5


TASK_DIR = os.environ.get("C3_TASK_DIR")


@pytest.mark.skipif(not TASK_DIR, reason="set C3_TASK_DIR to the save_task JSONL directory")
@pytest.mark.parametrize("task", ["gsm8k", "mbpp", "humaneval"])
def test_reference_solutions_score_100(task):
    from concurrent.futures import ThreadPoolExecutor

    from c3.data.tasks import load_task

    recs = load_task(task, Path(TASK_DIR) / f"{task}.jsonl")
    with ThreadPoolExecutor(8) as pool:
        verdicts = list(pool.map(lambda r: score(task, r, reference_completion(task, r)), recs))
    failed = [(r["doc_id"], v["error"]) for r, v in zip(recs, verdicts) if not v["correct"]]
    assert not failed
