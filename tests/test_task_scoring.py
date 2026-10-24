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
    assert any(" gsm8k a:" in w for w in s["warnings"])  # a has doc 2, b does not
    assert "git_commit" in s["provenance"]
    assert s["inputs"][0]["sha256"]
    assert len((tmp_path / "v.jsonl").read_text().splitlines()) == 5


def _shard(d, rows, meta=None):
    d.mkdir(parents=True, exist_ok=True)
    (d / "t12.shard0of1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if meta is not None:
        (d / "t12.shard0of1.meta.json").write_text(json.dumps(meta))


def _cli(tmp_path, *dirs):
    task = tmp_path / "gsm8k.jsonl"
    task.write_text("".join(json.dumps(dict(GSM, doc_id=i)) + "\n" for i in range(1319)))
    out = tmp_path / "summary.json"
    subprocess.run([sys.executable, str(ROOT / "scripts/score_t12.py"), *map(str, dirs),
                    "--task-jsonl", str(task), "--out", str(out)], check=True, capture_output=True)
    return json.loads(out.read_text())


def test_cli_driver_error_records(tmp_path):
    rows = [{"task": "gsm8k", "config": "a", "doc_id": 0, "completion": "18", "error": None},
            {"task": "gsm8k", "config": "a", "doc_id": 1, "completion": None,
             "error": "RuntimeError: CUDA out of memory"}]
    _shard(tmp_path / "run", rows)
    (r,) = _cli(tmp_path, tmp_path / "run")["rows"]
    assert (r["n"], r["n_correct"], r["n_driver_error"]) == (2, 1, 1)


def test_cli_runs_never_merge(tmp_path):
    rec = {"task": "gsm8k", "config": "topk_conf[k=2]", "doc_id": 0}
    meta = {"model": "m", "precision": "fp16", "task": "gsm8k", "half": "report"}
    _shard(tmp_path / "lm", [{**rec, "run_id": "lmeval1", "completion": "#### 18"}],
           {**meta, "prompt": {"format": "lmeval", "fewshot": 5}, "run_id": "lmeval1"})
    _shard(tmp_path / "chat", [{**rec, "run_id": "chat1", "completion": "#### 1"}],
           {**meta, "prompt": {"format": "chat", "fewshot": 0}, "run_id": "chat1"})
    rows = _cli(tmp_path, tmp_path / "lm", tmp_path / "chat")["rows"]
    by = {r["prompt_format"]: r for r in rows}
    assert (by["lmeval"]["n_correct"], by["chat"]["n_correct"]) == (1, 0)
    # without meta files, the records' run_id still keeps them apart
    for d in ("lm", "chat"):
        (tmp_path / d / "t12.shard0of1.meta.json").unlink()
    assert len(_cli(tmp_path, tmp_path / "lm", tmp_path / "chat")["rows"]) == 2


def test_cli_shards_of_one_run_pool(tmp_path):
    meta = {"model": "m", "precision": "fp16", "task": "gsm8k", "half": "report",
            "prompt": {"format": "chat"}}
    for i in range(2):
        d = tmp_path / "run"
        d.mkdir(exist_ok=True)
        (d / f"t12.shard{i}of2.jsonl").write_text(json.dumps(
            {"task": "gsm8k", "config": "a", "doc_id": i, "completion": "18",
             "run_id": f"r{i}"}) + "\n")
        (d / f"t12.shard{i}of2.meta.json").write_text(json.dumps(
            {**meta, "shard": i, "num_shards": 2, "run_id": f"r{i}"}))
    (r,) = _cli(tmp_path, tmp_path / "run")["rows"]
    assert (r["n"], r["n_correct"]) == (2, 2)


@pytest.mark.parametrize("tail", [
    "exit()", "import sys; sys.exit(0)", "import os; os._exit(0)",
    "import unittest\nclass T(unittest.TestCase):\n    def test(self):\n        pass\n"
    "if __name__ == '__main__':\n    unittest.main()"])
def test_early_exit_after_wrong_code_fails(tail):
    v = score("mbpp", MBPP, f"def add(a, b):\n    return a - b\n{tail}\n")
    assert not v["correct"]


def test_main_block_reading_input_is_inert():
    c = ("def add(a, b):\n    return a + b\n"
         "if __name__ == '__main__':\n    a, b = map(int, input().split())\n")
    assert score("mbpp", MBPP, c)["correct"]


def test_stdout_flood_and_grandchild():
    flood = "def add(a, b):\n    return a + b\nwhile True:\n    print('x' * 100000)\n"
    assert score("mbpp", MBPP, flood, timeout=2.0)["error"] == "timeout"
    fork = ("import os, time\nif os.fork() == 0:\n    time.sleep(20)\n    os._exit(0)\n"
            "def add(a, b):\n    return a + b\n")
    assert score("mbpp", MBPP, fork, timeout=5.0)["correct"]


def test_unfenced_code_with_prose():
    code = "def add(a, b):\n    return a + b\n"
    assert score("mbpp", MBPP, "Here is the function:\n" + code)["correct"]
    assert score("mbpp", MBPP, code + "\nThis function returns the sum of a and b.")["correct"]
    assert score("humaneval", HE, "    return x + 1\n\nThis adds one to x.")["correct"]
    v = score("mbpp", MBPP, "def add(a, b) return a + b")
    assert v["error"].startswith("SyntaxError")


@pytest.mark.parametrize("text,ours,strict,flexible", [
    ("#### 18.0", True, False, False),
    ("So the total is 18.", True, False, True),
    ("#### 18\n\n#### 20", False, True, False),
    ("#### 18", True, True, True),
    ("It is $1,8.", True, False, True),  # both drop commas before comparing
])
def test_gsm8k_lmeval_parity(text, ours, strict, flexible):
    v = score("gsm8k", GSM, text)
    assert (v["correct"], v["lmeval_strict"], v["lmeval_flexible"]) == (ours, strict, flexible)


def test_gsm8k_edge_cases():
    assert gsm8k_extract("\\boxed{18 \\text{ dollars}} per week for 7 days") == "18 \\text{ dollars}"
    assert parse_number(gsm8k_extract("It took 10-18 min")) == 18
    assert parse_number(gsm8k_extract("It fell to -4 degrees")) == -4


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
