"""The job scripts reach /share1 through a login host that still accepts SSH.

`ada` stopped accepting SSH in October 2026; a script that still rsyncs through
it cannot fetch the environment, stage a model or persist a trace part.
"""
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SWEEP_SCRIPTS = ["slurm/activate_env.sh", "slurm/stage_model.sh",
                 "slurm/collect_traces.sbatch", "slurm/build_env.sbatch"]


@pytest.mark.parametrize("path", SWEEP_SCRIPTS)
def test_sweep_scripts_do_not_hard_code_the_old_login_node(path):
    src = (ROOT / path).read_text()
    assert not re.search(r"\$USER@ada[:\"]", src), f"{path} still goes through $USER@ada"


def _pick(tmp_path, accepting: str | None, env_extra: dict | None = None) -> str:
    """Source login_host.sh with a fake ssh that only `accepting` answers."""
    fake = tmp_path / "bin"
    fake.mkdir(exist_ok=True)
    (fake / "ssh").write_text(
        "#!/bin/bash\nfor a in \"$@\"; do case \"$a\" in *@" + (accepting or "NONE")
        + ") exit 0;; esac; done\nexit 255\n")
    (fake / "ssh").chmod(0o755)
    env = {"PATH": f"{fake}:{os.environ['PATH']}", "USER": "u", "HOME": str(tmp_path)}
    env.update(env_extra or {})
    r = subprocess.run(["bash", "-c", f"source {ROOT / 'slurm/login_host.sh'}; echo \"$C3_LOGIN_HOST\""],
                       env=env, capture_output=True, text=True, check=True)
    return r.stdout.strip().splitlines()[-1]


def test_login_host_takes_the_first_candidate_that_answers(tmp_path):
    assert _pick(tmp_path, "ada-gw2") == "ada-gw2"
    assert _pick(tmp_path, "ada-gw1") == "ada-gw1"


def test_login_host_respects_an_explicit_choice(tmp_path):
    assert _pick(tmp_path, "ada-gw1", {"C3_LOGIN_HOST": "ada-gw2"}) == "ada-gw2"


def test_login_host_warns_and_falls_back_when_nothing_answers(tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "ssh").write_text("#!/bin/bash\nexit 255\n")
    (fake / "ssh").chmod(0o755)
    env = {"PATH": f"{fake}:{os.environ['PATH']}", "USER": "u", "HOME": str(tmp_path)}
    r = subprocess.run(["bash", "-c", f"source {ROOT / 'slurm/login_host.sh'}; echo \"$C3_LOGIN_HOST\""],
                       env=env, capture_output=True, text=True, check=True)
    assert "WARNING" in r.stdout and r.stdout.strip().splitlines()[-1] == "ada-gw1"
