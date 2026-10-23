"""slurm/prereg_analysis.sbatch end to end, on the laptop, reading no trace.

The real job script runs against stand-ins: `ssh` runs its command locally and `rsync`
maps `tester@testlogin:` paths to a directory standing in for /share1 (the shims of
tests/test_stage_fineweb.py); `module` is a no-op; the env interpreter is this python,
with activate_env.sh's `import torch, numpy` check answered for it; a stub
`transformers` on PYTHONPATH stands in for the HF cache; and PY is
tests/prereg_stub_py.sh, which logs each call and writes its results.json.  The
repository is a scratch git repo holding the real prereg/analysis_commands.sh, slurm/,
c3/ and artifacts, tagged prereg-v1, so the gates are the production ones.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "slurm" / "prereg_analysis.sbatch"
STUB = ROOT / "tests" / "prereg_stub_py.sh"
RSYNC = shutil.which("rsync")
pytestmark = pytest.mark.skipif(not (RSYNC and shutil.which("git")), reason="needs rsync and git")

RSYNC_SHIM = r'''#!/bin/bash
echo "rsync $*" >> "$SHIM_LOG"
args=(); skip=0
for a in "$@"; do
    if [ "$skip" = 1 ]; then skip=0; continue; fi
    case "$a" in
        -e) skip=1; continue ;;
        *@testlogin:*) a="${a#*@testlogin:}" ;;
    esac
    args+=("$a")
done
exec REAL_RSYNC "${args[@]}"
'''

SSH_SHIM = r'''#!/bin/bash
echo "ssh $*" >> "$SHIM_LOG"
while [ "$1" = -o ]; do shift 2; done
[ "$1" = tester@testlogin ] || exit 255
shift
exec bash -c "$*"
'''

ENV_PY = '''#!/bin/bash
[ "$*" = "-c import torch, numpy" ] && exit 0
exec REAL_PY "$@"
'''

FAKE_TRANSFORMERS = '''import os


class AutoTokenizer:
    @staticmethod
    def from_pretrained(path, **kw):
        if path in os.environ.get("FAKE_HF_MISSING", "").split(","):
            raise OSError(f"{path} is not in the HF cache")
        return object()
'''

ARMS_8B = ("c1-llada-base", "c1-dream-base", "c1-llada-instruct-gsm8k",
           "c1-llada-instruct-mbpp-v2", "c1-dream-instruct-gsm8k", "c1-dream-instruct-mbpp")


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                   check=True, capture_output=True)


def _run_dir(share: Path, run: str, dataset: str, parts: int = 2) -> Path:
    d = share / run
    d.mkdir(parents=True)
    files = []
    for i in range(parts):
        name = f"{run}.shard0of1.p{i:04d}.parquet"
        data = f"{run} part {i}\n".encode() * 50
        (d / name).write_bytes(data)
        files.append({"file": name, "sha256": hashlib.sha256(data).hexdigest(), "rows": 50})
    (d / f"{run}.dataset.json").write_text(json.dumps(
        {"format": "c3-trace-dataset-v1", "model": "local-110m", "dataset": dataset, "files": files}))
    return d


@pytest.fixture
def job(tmp_path):
    """run(**env) -> (returncode, output, shim calls, stub calls) of the runner."""
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"doc_id": 0, "text": "stand-in"}\n')
    repo = tmp_path / "repo"
    for d in ("c3", "prereg", "slurm"):
        shutil.copytree(ROOT / d, repo / d, ignore=shutil.ignore_patterns("__pycache__"))
    (repo / "artifacts" / "t2_pilot").mkdir(parents=True)
    for p in ROOT.glob("artifacts/*.json"):
        shutil.copy(p, repo / "artifacts" / p.name)
    for p in ROOT.glob("artifacts/t2_pilot/*.json"):
        shutil.copy(p, repo / "artifacts" / "t2_pilot" / p.name)
    meta = repo / "artifacts" / "split_manifest.meta.json"
    m = json.loads(meta.read_text())
    m["corpus_sha256"] = hashlib.sha256(corpus.read_bytes()).hexdigest()
    meta.write_text(json.dumps(m))
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "frozen")
    _git(repo, "tag", "prereg-v1")

    share = tmp_path / "share1" / "traces"
    _run_dir(share, "main-110m-v3", "fineweb")
    ts = _run_dir(share, "timesplit-110m-v3", "timesplit")
    (ts / "timesplit-110m-v3.parquet").write_bytes(b"merged copy, not in the manifest")
    (share / "c1ts-llada-base").mkdir()                    # named by the commands, never merged
    (share / "t2floor-110m-v3-fp32").mkdir()               # not named by the commands
    _run_dir(share, "some-other-run", "fineweb")           # merged, but not named either

    shims = tmp_path / "shims"
    shims.mkdir()
    for name, src in (("rsync", RSYNC_SHIM.replace("REAL_RSYNC", RSYNC)), ("ssh", SSH_SHIM)):
        (shims / name).write_text(src)
        (shims / name).chmod(0o755)
    scratch = tmp_path / "scratch"
    (scratch / "c3-env" / "bin").mkdir(parents=True)
    env_py = scratch / "c3-env" / "bin" / "python"
    env_py.write_text(ENV_PY.replace("REAL_PY", sys.executable))
    env_py.chmod(0o755)
    extra = tmp_path / "py-extra"
    (extra / "transformers").mkdir(parents=True)
    (extra / "transformers" / "__init__.py").write_text(FAKE_TRANSFORMERS)
    log, stub_log = tmp_path / "shim.log", tmp_path / "stub.log"
    base = {"PATH": f"{shims}:{os.environ['PATH']}", "HOME": str(tmp_path), "USER": "tester",
            "LANG": "C.UTF-8", "BASH_FUNC_module%%": "() {  return 0\n}",
            "C3_REPO": str(repo), "C3_SCRATCH": str(scratch), "C3_PERSIST": str(tmp_path / "persist"),
            "C3_LOGIN_HOST": "testlogin", "TRACES_SRC": str(share), "PYTHONPATH": str(extra),
            "CORPUS": str(corpus), "PY": str(STUB), "STUB_REAL_PY": sys.executable,
            "STUB_LOG": str(stub_log), "SHIM_LOG": str(log), "C3_REPORT_HALF_OK": "1"}

    def run(**extra_env):
        log.write_text("")
        stub_log.write_text("")
        env = {k: v for k, v in {**base, **extra_env}.items() if v is not None}
        p = subprocess.run(["bash", str(RUNNER)], env=env, capture_output=True, text=True,
                           timeout=300, cwd=tmp_path)
        return (p.returncode, p.stdout + p.stderr, log.read_text().splitlines(),
                [line.split() for line in stub_log.read_text().splitlines()])

    run.repo, run.share, run.scratch = repo, share, scratch
    run.out = tmp_path / "persist" / "prereg-v1"
    run.corpus = corpus
    return run


def _arg(call, flag):
    return call[call.index(flag) + 1]


def test_the_runner_stages_this_accounts_runs_and_runs_only_their_arms(job):
    rc, out, calls, stub = job()
    assert rc == 0, out
    assert "[prereg] prereg-v1 = " in out
    assert "staging main-110m-v3 timesplit-110m-v3 from" in out
    assert "c1ts-llada-base has no c1ts-llada-base.dataset.json" in out
    assert not any("some-other-run" in c or "t2floor" in c for c in calls if c.startswith("rsync"))
    # Only the parts each manifest lists come over, and they are checked.
    stage = job.scratch / "prereg" / "traces"
    assert sorted(p.name for p in (stage / "timesplit-110m-v3").iterdir()) == [
        "timesplit-110m-v3.dataset.json", "timesplit-110m-v3.shard0of1.p0000.parquet",
        "timesplit-110m-v3.shard0of1.p0001.parquet"]
    assert "main-110m-v3: 2 parts, sha256 ok" in out

    got = sorted((c[0], Path(_arg(c, "--out")).relative_to(job.out).as_posix()) for c in stub)
    assert got == sorted([("scripts/analyze_traces.py", "c1/Mdlm"),
                          ("scripts/analyze_traces.py", "c1/Mdlm_knots5"),
                          ("scripts/analyze_traces.py", "c1/Mdlm_knots13"),
                          ("scripts/analyze_traces.py", "c1/MdlmTimeSplit"),
                          ("scripts/analyze_e3.py", "e3/Mdlm"),
                          ("scripts/run_controls.py", "t8/Mdlm")])
    views = {Path(_arg(c, "--traces")).parent.parent for c in stub}
    assert len(views) == 1 and views.pop().name.startswith("view-")
    for c in stub:
        run = "timesplit-110m-v3" if "MdlmTimeSplit" in _arg(c, "--out") else "main-110m-v3"
        assert Path(_arg(c, "--traces")).name == f"{run}.dataset.json"
        assert "--split" in c and _arg(c, "--split") == "half"
    e1 = [c for c in stub if c[0] == "scripts/analyze_traces.py"]
    assert {_arg(c, "--floor") for c in e1} == {"0.0"}
    assert {_arg(c, "--manifest") for c in e1} == {"artifacts/split_manifest.json",
                                                   "artifacts/split_manifest_timesplit.json"}
    t8 = next(c for c in stub if c[0] == "scripts/run_controls.py")
    assert _arg(t8, "--tokenizer") == "gpt2" and _arg(t8, "--docs") == str(job.corpus)
    assert _arg(t8, "--half") == "report"
    # The other account's arms are skipped and named; the summary waits for them.
    for run in ARMS_8B:
        assert f"[prereg] {run} not in TRACES; skipped here" in out
    assert "c1ts-dream-base not collected; skipped" in out
    assert ("Holm summary waits for: LladaBase DreamBase LladaInstructGsm LladaInstructMbpp "
            "DreamInstructGsm DreamInstructMbpp") in out
    assert (job.out / "c1" / "Mdlm" / "results.json").exists()
    assert "tokenizer gpt2 (T8 of main-110m-v3): loads offline" in out
    assert f"{job.corpus}: sha256 ok" in out
    assert "[prereg] done ->" in out
    assert not list((job.scratch / "prereg").glob("view-*"))       # the job's view is removed


def test_a_resubmitted_job_keeps_the_steps_an_earlier_job_finished(job):
    rc, out, calls, stub = job()
    assert rc == 0 and len(stub) == 6, out
    (job.out / "e3" / "Mdlm" / ".prereg-done").unlink()        # as if killed during E3
    rc, out, calls, stub = job(N_JOBS="7")                     # N_JOBS changes no number
    assert rc == 0, out
    assert [(c[0], _arg(c, "--out")) for c in stub] == [("scripts/analyze_e3.py",
                                                          str(job.out / "e3" / "Mdlm"))]
    assert out.count("finished by an earlier job at this commit") == 5
    # At another commit nothing is kept.
    (job.repo / "prereg" / "deviations.md").write_text("a later change\n")
    _git(job.repo, "commit", "-qam", "after the tag")
    rc, out, calls, stub = job(C3_PREREG_DRYRUN="1", OUT=str(job.out))
    assert rc == 0 and len(stub) == 6 and "earlier job" not in out, out


@pytest.mark.parametrize("case", ["no-ok", "past-tag", "dirty", "dryrun-python"])
def test_the_runner_refuses_before_staging_anything(job, case):
    env = {}
    if case == "no-ok":
        env["C3_REPORT_HALF_OK"] = None
        why = "C3_REPORT_HALF_OK=1 is not set"
    elif case == "past-tag":
        (job.repo / "prereg" / "deviations.md").write_text("a later change\n")
        _git(job.repo, "commit", "-qam", "after the tag")
        why = "is not prereg-v1"
    elif case == "dirty":
        (job.repo / "prereg" / "analysis_commands.sh").write_text("echo edited\n")
        why = "uncommitted changes"
    else:
        env.update(C3_PREREG_DRYRUN="1", PY=sys.executable)
        why = "stub PY only"
    rc, out, calls, stub = job(**env)
    assert rc != 0 and why in out, out
    assert calls == [] and stub == []


def test_a_dry_run_skips_the_tag_check_and_writes_apart(job):
    (job.repo / "prereg" / "deviations.md").write_text("a later change\n")
    _git(job.repo, "commit", "-qam", "after the tag")
    rc, out, calls, stub = job(C3_PREREG_DRYRUN="1")
    assert rc == 0, out
    assert "DRY RUN" in out and len(stub) == 6
    assert all("/prereg-dryrun/" in c[c.index("--out") + 1] for c in stub)
    assert not job.out.exists()


@pytest.mark.parametrize("case", ["corrupt-part", "no-tokenizer", "other-corpus"])
def test_the_runner_stops_before_the_commands_when_inputs_are_wrong(job, case):
    env = {}
    if case == "corrupt-part":
        part = job.share / "main-110m-v3" / "main-110m-v3.shard0of1.p0001.parquet"
        part.write_bytes(part.read_bytes()[::-1])           # same size, other bytes
        why = "main-110m-v3: a staged part does not match its manifest"
    elif case == "no-tokenizer":
        env["FAKE_HF_MISSING"] = "gpt2"
        why = "tokenizer gpt2@None for main-110m-v3 does not load offline"
    else:
        job.corpus.write_text('{"doc_id": 0, "text": "another corpus"}\n')
        why = "sha256 differs from the recorded"
    rc, out, calls, stub = job(**env)
    assert rc != 0 and why in out, out
    assert stub == []
