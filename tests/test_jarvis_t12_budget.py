"""scripts/jarvis_t12.sh start_budget: the global deadline, with the clock of provisioning in it.

The function is cut out of the script (the script runs a command when sourced) and run in a
scratch directory.
"""
from shell_helpers import ROOT, needs_bash, posix, run_bash, script_ok

pytestmark = needs_bash

LOAD = ('die() { echo "FATAL: $*" >&2; exit 1; }\n'
        'eval "$(sed -n "/^start_budget() {/,/^}/p" "%s/scripts/jarvis_t12.sh")"\n' % posix(ROOT))


def budget(tmp_path, body, **env):
    script_ok("jarvis_t12.sh")
    return run_bash(LOAD + body, env, cwd=tmp_path, timeout=60)


def left(r):
    return int(r.stdout.split("BUDGET_LEFT_S=")[1].split()[0])


def test_a_new_budget_is_the_hours_and_provisioning_time_is_taken_off(tmp_path):
    r = budget(tmp_path, 'start_budget calib 2.5 >/dev/null; echo BUDGET_LEFT_S=$BUDGET_LEFT_S')
    assert abs(left(r) - 2.5 * 3600) <= 5, r.stderr
    # the laptop spent 10 min creating the instance: it uploads what is left of the budget
    (tmp_path / "x").mkdir()
    r = budget(tmp_path / "x", 'start_budget calib 2.5 >/dev/null; echo BUDGET_LEFT_S=$BUDGET_LEFT_S',
               T12_BUDGET_REMAINING_S=str(2 * 3600 + 1200))
    assert abs(left(r) - (2 * 3600 + 1200)) <= 5, r.stderr


def test_an_existing_budget_is_kept_not_restarted_and_a_spent_one_refuses(tmp_path):
    body = ('mkdir -p runs/t12; echo $(( $(date +%s) + 1000 )) > runs/t12/.deadline-calib\n'
            'start_budget calib 2.5 >/dev/null; echo BUDGET_LEFT_S=$BUDGET_LEFT_S\n')
    assert 990 <= left(budget(tmp_path, body)) <= 1000              # not reset to 2.5 h
    spent = ('mkdir -p runs/t12; echo $(( $(date +%s) - 5 )) > runs/t12/.deadline-calib\n'
             'start_budget calib 2.5\n')
    (tmp_path / "s").mkdir()
    r = budget(tmp_path / "s", spent)
    assert r.returncode == 1 and "NEW_BUDGET=1" in r.stderr
    (tmp_path / "n").mkdir()
    r = budget(tmp_path / "n", spent + 'echo BUDGET_LEFT_S=$BUDGET_LEFT_S\n', NEW_BUDGET="1")
    assert r.returncode == 0 and abs(left(r) - 2.5 * 3600) <= 5


def test_the_laptops_remaining_budget_caps_an_existing_deadline(tmp_path):
    body = ('mkdir -p runs/t12; echo $(( $(date +%s) + 20000 )) > runs/t12/.deadline-report\n'
            'start_budget report 9.5 >/dev/null; echo BUDGET_LEFT_S=$BUDGET_LEFT_S\n')
    assert abs(left(budget(tmp_path, body, T12_BUDGET_REMAINING_S="3600")) - 3600) <= 5


def test_the_absolute_deadline_epoch_caps_the_budget(tmp_path):
    import time
    r = budget(tmp_path, 'start_budget report 9 >/dev/null; echo BUDGET_LEFT_S=$BUDGET_LEFT_S',
               T12_DEADLINE_EPOCH=str(int(time.time()) + 5000))
    assert abs(left(r) - 5000) <= 5, r.stderr
