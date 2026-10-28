"""scripts/c3_paper_outputs.py on a synthetic T12 fixture.

The report-half results do not exist yet: these tests build records, verdicts, grids and
calibrator metadata in the formats ``scripts/run_t12.py``, ``scripts/score_t12.py``,
``scripts/pick_knobs.py`` and ``scripts/fit_c3a_calibrator.py`` write, with numbers chosen so
every expected value can be computed by hand.  A schema and arithmetic check, not results.
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import c3_paper_outputs as P  # noqa: E402

TASKS = ("gsm8k", "mbpp")
TASK_WORD = {"gsm8k": "Gsm", "mbpp": "Mbpp"}
DEC_POLICY = {"TopK": "topk_conf", "Threshold": "threshold", "Margin": "margin",
              "Entropy": "entropy_bounded", "Klass": "klass", "PcSampler": "pc_sampler",
              "CThreeA": "c3a"}
SPEED_WORD = {1: "SpeedOne", 2: "SpeedTwo", 3: "SpeedThree"}
N_DOCS = 30
# forward passes per problem (before the +-1 wobble that keeps the sum fixed): tokens per
# step of the configuration is 256 / NFE0.
NFE0 = {
    "TopK": (128, 64, 32), "Margin": (128, 64, 32), "PcSampler": (128, 64, 32),
    "CThreeA": (128, 64, 32),
    # slot 1 is the nearest to C3a's speed-2 (4.063), not slot 2 (3.657)
    "Threshold": (63, 70, 30),
    # never within 10% of C3a at any slot
    "Entropy": (100, 80, 20),
    # exactly C3a's speed at slot 3, but flagged unmatched at the target
    "Klass": (130, 64, 32),
}
ALPHA = {"gsm8k": (0.022, 0.271, 0.638), "mbpp": (0.039, 0.322, 0.735)}
KNOBS = {
    "TopK": lambda s, t: {"k": (2, 4, 8)[s - 1]},
    "Margin": lambda s, t: {"k": (2, 4, 8)[s - 1]},
    "PcSampler": lambda s, t: {"k": (2, 4, 8)[s - 1], "lam": 0.25, "alpha": 10.0},
    "Threshold": lambda s, t: {"tau": (0.99, 0.86, 0.49)[s - 1]},
    "Entropy": lambda s, t: {"gamma": (0.017, 0.5, 6.1)[s - 1]},
    "Klass": lambda s, t: {"tau": (0.8, 0.29, 0.2)[s - 1], "eps_kl": 0.01, "history": 2,
                           "fallback": 1},
    "CThreeA": lambda s, t: {"alpha": ALPHA[t][s - 1]},
}


def config_key(cfg):
    knobs = ",".join(f"{k}={cfg[k]}" for k in sorted(cfg) if k != "policy")
    return f"{cfg['policy']}[{knobs}]"


def cfg_of(dec, slot, task):
    return {"policy": DEC_POLICY[dec], **KNOBS[dec](slot, task)}


def nfe_of(dec, slot, i):
    return NFE0[dec][slot - 1] + (i % 3 - 1)          # +-1: the sum over 30 docs is exact


def correct_of(task, dec, slot, i):
    if dec == "CThreeA":
        return i % 5 != 0 if slot != 3 else i % 3 != 0
    if dec in ("TopK", "Margin"):
        return i % 5 != 0 and i % 7 != 0
    return i % 5 != 0 and i % 11 != 0


def outside_of(task, i):
    return i % 2 == 0 if task == "gsm8k" else i % 3 == 0


def failed(task, dec, slot, i):
    return task == "mbpp" and dec == "TopK" and slot == 3 and i == 4


def make_grid(task):
    configs, picks, speed_of = [], [], {}
    for dec in DEC_POLICY:
        for s in (1, 2, 3):
            cfg = cfg_of(dec, s, task)
            configs.append(cfg)
            speed_of[config_key(cfg)] = s
            if dec in ("Threshold", "Entropy", "Klass", "CThreeA"):
                picks.append({"policy": cfg["policy"], "speed": s, "target_tokens_per_step": 2.0 * 2 ** (s - 1),
                              "config_key": config_key(cfg),
                              "knobs": {k: v for k, v in cfg.items() if k != "policy"},
                              "unmatched_at_target": dec == "Klass" and s == 3,
                              "achieved_tokens_per_step": None, "bracket": [1.0, 9.0]})
    return {"task": task, "status": "calibrated", "targets": {"speed1": 2.0, "speed2": 4.0, "speed3": 8.0},
            "configs": configs, "speed_of_config": speed_of, "picks": picks}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def make_fixture(root: Path, with_records=True, with_verdicts=True, with_grids=True,
                 with_cal=True, with_sanity=True):
    """Write every input under ``root``; returns the paths."""
    paths = {"records": root / "runs", "verdicts": root / "verdicts.jsonl", "grids": root / "grids",
             "cal": root / "cal", "sanity": root / "sanity.json", "out": root / "out"}
    verdicts = []
    for task in TASKS:
        shard_rows = {0: [], 1: []}
        for i in range(N_DOCS):
            for dec in DEC_POLICY:
                for s in (1, 2, 3):
                    cfg = cfg_of(dec, s, task)
                    key = config_key(cfg)
                    rec = {"config": key, **cfg, "task": task, "doc_id": i, "run_id": f"run{i % 2}",
                           "precision": "fp16", "git_commit": "abc1234"}
                    if failed(task, dec, s, i):
                        rec.update({"completion": None, "error": "OutOfMemoryError: out of memory"})
                    else:
                        nfe = nfe_of(dec, s, i)
                        rec.update({"completion": "x", "stopped": True, "n_forward": nfe,
                                    "n_committed": 256, "tokens_per_step": 256 / nfe,
                                    "seconds": 1.0 + 0.01 * NFE0[dec][s - 1], "error": None})
                    if dec == "CThreeA":
                        rec["c3a_steps"] = [{"step": st, "n_commit": 3, "n_eligible": 5,
                                             "median_ell": 0.1, "mask_rate": 0.5, "log_n_masked": 3.0,
                                             "outside": ["mask_rate"] if (st == 0 and outside_of(task, i)) else []}
                                            for st in range(10)]
                    shard_rows[i % 2].append(rec)
                    ok = (not failed(task, dec, s, i)) and correct_of(task, dec, s, i)
                    verdicts.append({"run": "r", "run_id": f"run{i % 2}", "prompt_format": "chat",
                                     "task": task, "config": key, "doc_id": i, "stopped": True,
                                     "correct": bool(ok), "extracted": None, "driver_error": failed(task, dec, s, i),
                                     "error": None})
        if with_records:
            d = paths["records"] / f"sweep-{task}"
            d.mkdir(parents=True, exist_ok=True)
            for sh, rows in shard_rows.items():
                write_jsonl(d / f"t12.shard{sh}of2.jsonl", rows)
                (d / f"t12.shard{sh}of2.meta.json").write_text(json.dumps(
                    {"task": task, "half": "report", "gen_length": 256, "run_id": f"run{sh}",
                     "prompt": {"format": "chat", "fewshot": 0}}))
        if with_grids:
            paths["grids"].mkdir(exist_ok=True)
            (paths["grids"] / f"t12_grid_{task}.json").write_text(json.dumps(make_grid(task)))
        if with_cal:
            paths["cal"].mkdir(exist_ok=True)
            (paths["cal"] / f"llada-8b-instruct_fp16_c3a-{task}.json").write_text(json.dumps({
                "n_fit_docs": 660 if task == "gsm8k" else 255, "monotone_share": 1.0 if task == "gsm8k" else 0.9876,
                "dev_check": {"ece": {"raw": {"value": 0.0607, "ci": [0.05, 0.07]},
                                      "calibrated": {"value": 0.0059, "ci": [0.003, 0.01]}}},
                "provenance": {"git_commit": "0b5c8b8"}}))
    if with_verdicts:
        write_jsonl(paths["verdicts"], verdicts)
    # the Ada sanity run, score_t12.py output (n=200)
    if with_sanity:
        paths["sanity"].write_text(json.dumps({
            "created": "2026-10-25T18:55:24+00:00", "inputs": [], "warnings": [],
            "provenance": {"git_commit": "0b5c8b8", "git_dirty": False},
            "rows": [{"task": "gsm8k", "config": "threshold[tau=0.9]", "prompt_format": "lmeval", "half": "fit",
                      "n": 200, "n_correct": 169, "accuracy": 0.845, "ci95": [0.7884, 0.8886]},
                     {"task": "gsm8k", "config": "topk_conf[k=1]", "prompt_format": "lmeval", "half": "fit",
                      "n": 200, "n_correct": 167, "accuracy": 0.835, "ci95": [0.7773, 0.8800]}]}))
    return paths


def run_main(paths, *extra, records=True, verdicts=True, grids=True, cal=True, sanity=True):
    argv = ["--out-dir", str(paths["out"])]
    if records:
        argv += ["--records", str(paths["records"])]
    if verdicts:
        argv += ["--verdicts", str(paths["verdicts"])]
    nope = str(paths["out"].parent / "does-not-exist")   # an omitted flag would use the repo defaults
    argv += ["--grids", str(paths["grids"]) if grids else nope]
    argv += ["--calibrators", str(paths["cal"]) if cal else nope]
    argv += ["--sanity", str(paths["sanity"]) if sanity else nope]
    return P.main(argv + list(extra))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("c3paper")
    paths = make_fixture(root)
    assert run_main(paths) == 0
    out = paths["out"]
    return {"paths": paths, "out": out,
            "summary": json.loads((out / "t12" / "llada-8b-instruct_fp16_t12-summary.json").read_text()),
            "macros": parse_macros((out / "c3" / "macros_c3.tex").read_text())}


NEWCOMMAND = re.compile(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}$")


def parse_macros(text):
    out = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("%"):
            continue
        m = NEWCOMMAND.match(line)
        assert m, f"not a \\newcommand line: {line!r}"
        assert m.group(1) not in out, f"{m.group(1)} defined twice"
        out[m.group(1)] = m.group(2)
    return out


def val(body):
    """The number in a macro body (the sign may be \\ensuremath{-})."""
    body = body.replace("\\ensuremath{-}", "-")
    return float(body)


# ------------------------------------------------------------------ pure functions
def test_aggregate_speed_is_not_the_mean_of_ratios():
    nfe = [10, 20, 40]                       # ratios 25.6, 12.8, 6.4
    assert P.aggregate_speed(nfe) == pytest.approx(256 * 3 / 70)
    assert P.aggregate_speed(nfe) != pytest.approx(np.mean([256 / n for n in nfe]))
    assert P.aggregate_speed(nfe, gen_length=128) == pytest.approx(128 * 3 / 70)
    assert math.isnan(P.aggregate_speed([]))


def test_aggregate_speed_in_summary_matches_hand_calculation(built):
    gsm = built["summary"]["tasks"]["gsm8k"]["decoders"]["TopK"]["speeds"]
    # NFE per doc = 64 + (i % 3 - 1) over 30 docs: the sum is 30 * 64
    assert gsm["2"]["tps"] == pytest.approx(256 * 30 / (30 * 64))
    assert gsm["2"]["nfe"] == pytest.approx(64.0)
    assert gsm["2"]["seconds"] == pytest.approx(1.0 + 0.64)
    # the mean of per-problem ratios is larger (Jensen) and is recorded separately
    assert gsm["2"]["mean_of_ratios"] > gsm["2"]["tps"]
    thr = built["summary"]["tasks"]["gsm8k"]["decoders"]["Threshold"]["speeds"]
    assert thr["1"]["tps"] == pytest.approx(256 / 63)


def test_failed_records_are_counted_and_leave_the_speed_alone(built):
    sp = built["summary"]["tasks"]["mbpp"]["decoders"]["TopK"]["speeds"]["3"]
    assert sp["failures"] == 1 and sp["n"] == N_DOCS and sp["n_ok"] == N_DOCS - 1
    nfes = [nfe_of("TopK", 3, i) for i in range(N_DOCS) if i != 4]
    assert sp["tps"] == pytest.approx(256 * len(nfes) / sum(nfes))
    # a failed problem scores wrong: its verdict is 0
    assert sp["n_correct"] == sum(correct_of("mbpp", "TopK", 3, i) for i in range(N_DOCS) if i != 4)
    assert built["summary"]["tasks"]["gsm8k"]["decoders"]["TopK"]["speeds"]["3"]["failures"] == 0


SETTINGS = {1: {"tps": 4.063, "unmatched": False}, 2: {"tps": 3.657, "unmatched": False},
            3: {"tps": 8.533, "unmatched": False}}


def test_match_picks_nearest_setting_not_the_same_slot():
    m = P.match_baseline(2, 4.0, SETTINGS)
    assert m["slot"] == 1 and m["tps"] == pytest.approx(4.063)
    assert P.match_baseline(3, 8.0, SETTINGS)["slot"] == 3


def test_match_requires_within_ten_percent():
    assert P.match_baseline(1, 2.0, SETTINGS) is None            # nearest is 3.657
    just_in = {1: {"tps": 4.4, "unmatched": False}}               # exactly +10%
    just_out = {1: {"tps": 4.41, "unmatched": False}}
    assert P.match_baseline(2, 4.0, just_in)["slot"] == 1
    assert P.match_baseline(2, 4.0, just_out) is None
    assert P.match_baseline(2, 4.0, {1: {"tps": 3.6, "unmatched": False}})["slot"] == 1   # -10%
    assert P.match_baseline(2, 4.0, {}) is None
    assert P.match_baseline(2, float("nan"), SETTINGS) is None


def test_match_is_dashed_where_the_grid_says_unmatched_at_target():
    klass = {1: {"tps": 2.0, "unmatched": False}, 2: {"tps": 4.0, "unmatched": False},
             3: {"tps": 8.0, "unmatched": True}}              # numerically exactly C3a's speed
    assert P.match_baseline(2, 4.0, klass)["slot"] == 2
    assert P.match_baseline(3, 8.0, klass) is None


def test_bootstrap_is_deterministic_and_pins_the_algorithm():
    a = {i: int(i % 4 != 0) for i in range(40)}
    b = {i: int(i % 3 != 0) for i in range(40)}
    r1 = P.paired_bootstrap(a, b)
    assert r1 == P.paired_bootstrap(a, b)
    assert P.SEED == 20261030 and P.N_BOOT == 2000
    # the specified algorithm: sorted shared doc_ids, 2,000 resamples of problems, percentile CI
    ids = sorted(a)
    d = np.array([a[i] - b[i] for i in ids], dtype=float)
    idx = np.random.default_rng(20261030).integers(0, len(ids), size=(2000, len(ids)))
    means = d[idx].mean(axis=1)
    assert r1["diff"] == pytest.approx(d.mean())
    assert r1["lo"] == pytest.approx(np.percentile(means, 2.5))
    assert r1["hi"] == pytest.approx(np.percentile(means, 97.5))
    assert r1["n"] == 40
    assert P.paired_bootstrap(a, b, seed=1) != r1


def test_bootstrap_is_paired_on_shared_doc_ids_and_ignores_order():
    a = {i: int(i % 4 != 0) for i in range(40)}
    b = {i: int(i % 3 != 0) for i in range(40)}
    full = P.paired_bootstrap(a, b)
    # docs only one side has are dropped; dict insertion order does not matter
    a_extra = {**{i: a[i] for i in reversed(range(40))}, 100: 1, 101: 0}
    b_extra = {**b, 200: 1}
    assert P.paired_bootstrap(a_extra, b_extra) == full
    # paired: if a == b on every doc the difference is exactly 0 with a zero-width interval,
    # which an unpaired bootstrap of two copies of the same sample would not give
    same = P.paired_bootstrap(a, dict(a))
    assert (same["diff"], same["lo"], same["hi"]) == (0.0, 0.0, 0.0)
    assert P.paired_bootstrap({1: 1}, {2: 0})["n"] == 0
    assert math.isnan(P.paired_bootstrap({1: 1}, {2: 0})["diff"])


def test_wilson_matches_score_t12():
    lo, hi = P.wilson(167, 200)
    assert (lo, hi) == pytest.approx((0.7773, 0.8800), abs=1e-4)


# ------------------------------------------------------------------ the fixture run
def test_matched_differences_in_summary(built):
    t = built["summary"]["tasks"]["gsm8k"]
    m = t["matched"]["2"]
    # C3a at speed 2 is correct for i % 5 != 0 (24/30); top-k for neither i % 5 == 0 nor
    # i % 7 == 0 (20/30); the C3a-better docs are 7, 14, 21, 28
    assert m["TopK"]["matched"] is True and m["TopK"]["slot"] == 2
    assert m["TopK"]["diff"] == pytest.approx(4 / 30)
    assert m["TopK"]["lo"] <= 4 / 30 <= m["TopK"]["hi"] and m["TopK"]["n_paired"] == 30
    assert m["Margin"]["diff"] == pytest.approx(4 / 30)
    # threshold: slot 1 (4.063) is nearer to 4.0 than its own slot 2 (3.657)
    assert m["Threshold"]["slot"] == 1
    # entropy is never within 10%
    for s in "123":
        assert t["matched"][s]["Entropy"]["matched"] is False
    assert "CThreeA" not in m


def test_klass_unmatched_at_target_gives_dash_even_at_equal_speed(built):
    t = built["summary"]["tasks"]["gsm8k"]
    assert t["matched"]["3"]["Klass"]["matched"] is False
    assert t["matched"]["3"]["Klass"]["reason"] == "unmatched_at_target"
    assert t["matched"]["2"]["Klass"]["matched"] is True
    assert built["macros"]["CThreeTTwelveGsmMatchedDiffKlassSpeedThree"] == "---"
    assert built["macros"]["CThreeTTwelveGsmMatchedDiffKlassSpeedThreeLo"] == "---"
    # KLASS's own speed-3 setting is still reported
    assert "CThreeTTwelveGsmKlassSpeedThreeAcc" in built["macros"]


def test_accuracy_wilson_and_n(built):
    s = built["summary"]["tasks"]["gsm8k"]["decoders"]["CThreeA"]["speeds"]["2"]
    assert s["n_correct"] == 24 and s["acc"] == pytest.approx(0.8)
    assert s["ci95"] == pytest.approx(list(P.wilson(24, 30)))
    assert built["macros"]["CThreeTTwelveGsmN"] == "30"
    assert built["macros"]["CThreeTTwelveGsmCThreeASpeedTwoAcc"] == "0.800"
    assert built["macros"]["CThreeTTwelveGsmCThreeASpeedTwoAccLo"] == f"{P.wilson(24, 30)[0]:.3f}"
    assert built["macros"]["CThreeTTwelveGsmCThreeASpeedTwoTPS"] == "4.00"


def test_outside_share_from_c3a_step_logs(built):
    # gsm8k: docs 0,2,..28 (15 of 30) have one outside step of 10; mbpp: docs i % 3 == 0 (10)
    assert built["macros"]["CThreeTTwelveGsmOutsideShare"] == "0.050"
    assert built["macros"]["CThreeTTwelveMbppOutsideShare"] == "0.033"
    assert built["summary"]["tasks"]["gsm8k"]["outside_share"]["n_steps"] == 3 * 300


def test_alpha_macros_and_primary_alias(built):
    mac = built["macros"]
    assert mac["CThreeTTwelveGsmCThreeASpeedOneAlpha"] == "0.022"
    assert mac["CThreeTTwelveMbppCThreeASpeedThreeAlpha"] == "0.735"
    assert mac["CThreeTTwelveGsmAlphaPrimary"] == mac["CThreeTTwelveGsmCThreeASpeedTwoAlpha"] == "0.271"
    assert mac["CThreeTTwelveMbppAlphaPrimary"] == "0.322"


def test_t11_macros(built):
    mac = built["macros"]
    assert mac["CThreeTElevenGsmFitDocs"] == "660"
    assert mac["CThreeTElevenMbppFitDocs"] == "255"
    assert mac["CThreeTElevenMbppMonotoneShare"] == "0.988"
    assert mac["CThreeTElevenGsmDevECERaw"] == "0.061"
    assert mac["CThreeTElevenGsmDevECECal"] == "0.006"


def test_sanity_macros(built):
    mac = built["macros"]
    assert mac["CThreeTTwelveSanityN"] == "200"
    assert mac["CThreeTTwelveSanityTopKAcc"] == "0.835"
    assert mac["CThreeTTwelveSanityThresholdAcc"] == "0.845"
    assert mac["CThreeTTwelveSanityTopKAccLo"] == "0.777"
    assert mac["CThreeTTwelveSanityThresholdAccHi"] == "0.889"


NUMBER = re.compile(r"^(\\ensuremath\{-\})?\d+(\.\d+)?$")


def test_macros_are_wellformed_latex_with_three_decimals(built):
    for name, body in built["macros"].items():
        assert re.fullmatch(r"[A-Za-z]+", name)
        assert name.startswith(("CThreeTEleven", "CThreeTTwelve"))
        assert body == "---" or NUMBER.match(body), (name, body)
        assert body.count("{") == body.count("}")
        three_dp = re.search(r"(Acc|AccLo|AccHi|MatchedDiff\w*|Share|Alpha|AlphaPrimary|DevECE\w+)$", name)
        if three_dp and body != "---":
            assert re.search(r"\.\d{3}$", body), (name, body)
        if name.endswith("N") or name.endswith("FitDocs"):
            assert re.fullmatch(r"\d+", body), (name, body)
    # no placeholders, no nan, and the name set carries no digit
    assert not any("nan" in b.lower() or "inf" in b.lower() for b in built["macros"].values())


def test_negative_difference_is_a_math_minus(built):
    # C3a at speed 3 is correct for i % 3 != 0 (20/30), top-k at speed 3 for 20/30 too, but a
    # different set: the sign of any difference must be \ensuremath{-} when negative
    neg = [b for b in built["macros"].values() if b.startswith("\\ensuremath{-}")]
    assert neg, "the fixture has at least one negative difference or interval end"
    for b in neg:
        assert NUMBER.match(b)


def test_macro_value_of_matched_difference(built):
    mac = built["macros"]
    assert mac["CThreeTTwelveGsmMatchedDiffTopKSpeedTwo"] == "0.133"
    lo, hi = val(mac["CThreeTTwelveGsmMatchedDiffTopKSpeedTwoLo"]), val(mac["CThreeTTwelveGsmMatchedDiffTopKSpeedTwoHi"])
    assert lo <= 0.133 <= hi
    assert mac["CThreeTTwelveGsmMatchedDiffEntropySpeedTwo"] == "---"
    assert not any(k.startswith("CThreeTTwelveGsmMatchedDiffCThreeA") for k in mac)


# ------------------------------------------------------------------ files
def test_summary_carries_provenance_inputs_and_the_pairing_rule(built):
    s = built["summary"]
    assert "git_commit" in s["provenance"]
    assert s["pairing"]["seed"] == 20261030 and s["pairing"]["n_boot"] == 2000
    assert s["pairing"]["tolerance"] == 0.10
    assert "shared doc_ids" in s["pairing"]["rule"]
    assert s["primary_speed"] == 2
    kinds = {k.split(":")[0] for k in s["inputs"]}
    assert {"records", "verdicts", "grid", "calibrator", "sanity"} <= kinds
    for v in s["inputs"].values():
        assert re.fullmatch(r"[0-9a-f]{64}", v["sha256"])
    assert s["sanity"]["published"] == {"topk_conf[k=1]": 0.793, "threshold[tau=0.9]": 0.792}


def test_table_is_a_bare_booktabs_tabular_one_row_per_decoder_and_task(built):
    tex = (built["out"] / "t12" / "t12_matched_speed.tex").read_text()
    assert "\\begin{tabular}" in tex and "\\end{tabular}" in tex
    assert "\\toprule" in tex and "\\bottomrule" in tex and "\\midrule" in tex
    assert "\\begin{table" not in tex and "\\caption" not in tex
    assert tex.count("\\begingroup") == tex.count("\\endgroup") == 1
    assert "GSM8K" in tex and "MBPP" in tex
    for name in ("Top-$k$", "Threshold", "Margin", "Entropy", "KLASS", "PC-Sampler", "C3a"):
        assert tex.count(name) >= 2, name                         # once per task
    # entropy is unmatched at speed 2: its row is dashes
    row = next(l for l in tex.splitlines() if l.startswith("Entropy"))
    assert row.count("---") == 3
    # C3a's own row is the reference: no difference
    assert not any(c in tex for c in ("nan", "None"))
    # ~1 column: five narrow columns
    assert re.search(r"\\begin\{tabular\}\{l[rcl]+\}", tex)


def test_sanity_cost_table_for_the_slides_and_makefile(built):
    tex = (built["out"] / "t12" / "t12_sanity_cost.tex").read_text()
    assert "\\begin{tabular}" in tex and "0.793" in tex and "0.792" in tex
    assert "0.835" in tex and "0.845" in tex


def test_sanity_cost_table_rows_and_columns(built):
    tex = (built["out"] / "t12" / "t12_sanity_cost.tex").read_text()
    assert "\\begin{tabular}" in tex and "\\begin{table" not in tex and "\\caption" not in tex
    assert tex.count("\\begingroup") == tex.count("\\endgroup") == 1
    assert "$n=200$" in tex and "published $0.793$" in tex and "published $0.792$" in tex
    assert "[$0.777$, $0.880$]" in tex                            # the Wilson interval of top-k
    # MBPP top-k: speed-3 has one failed problem -> fail column 1
    row = [l for l in tex.splitlines() if l.startswith("Top-$k$ &")][1]
    assert row.rstrip(" \\").endswith("& 1")
    assert "tok/st" in tex and "NFE" in tex


def test_unstopped_and_timeouts_count_as_failures(tmp_path):
    paths = make_fixture(tmp_path)
    key = config_key(cfg_of("TopK", 2, "gsm8k"))
    # one record that never hit a stop string, one verdict that timed out
    f = paths["records"] / "sweep-gsm8k" / "t12.shard0of2.jsonl"
    rows = [json.loads(l) for l in f.read_text().splitlines()]
    for r in rows:
        if r["config"] == key and r["doc_id"] == 2:
            r["stopped"] = False
    write_jsonl(f, rows)
    vs = [json.loads(l) for l in paths["verdicts"].read_text().splitlines()]
    for v in vs:
        if v["config"] == key and v["doc_id"] == 3:
            v["error"] = "timeout"
    write_jsonl(paths["verdicts"], vs)
    assert run_main(paths) == 0
    s = json.loads((paths["out"] / "t12" / "llada-8b-instruct_fp16_t12-summary.json").read_text())
    sp = s["tasks"]["gsm8k"]["decoders"]["TopK"]["speeds"]["2"]
    assert (sp["failures"], sp["n_unstopped"], sp["n_timeout"], sp["n_failed_any"]) == (0, 1, 1, 2)


def test_zip_tables_markdown_and_csv(built):
    z = built["out"] / "t12" / "zip"
    for stem in ("t12_sanity_cost", "t12_speed1_speed3"):
        assert (z / f"{stem}.md").read_text().startswith("#")
        assert (z / f"{stem}.csv").exists()
    md = (z / "t12_sanity_cost.md").read_text()
    assert "0.793" in md and "0.835" in md and "0.792" in md and "0.845" in md
    rows = (z / "t12_sanity_cost.csv").read_text().splitlines()
    head = rows[0].split(",")
    assert {"task", "decoder", "speed", "tokens_per_step", "nfe", "seconds_per_problem", "failures"} <= set(head)
    assert len([r for r in rows[1:] if r.startswith("mbpp,TopK,3,")]) == 1
    sp = (z / "t12_speed1_speed3.csv").read_text().splitlines()
    sh = sp[0].split(",")
    assert {"task", "decoder", "speed", "acc", "acc_lo", "acc_hi", "matched_diff", "diff_lo", "diff_hi"} <= set(sh)
    assert all(r.split(",")[sh.index("speed")] in ("1", "3") for r in sp[1:])


def test_figure_is_written(built):
    pytest.importorskip("matplotlib")
    pdf = built["out"] / "t12" / "figures" / "t12_matched_speed.pdf"
    assert pdf.exists() and pdf.read_bytes().startswith(b"%PDF")


def test_rerun_is_byte_stable(tmp_path):
    paths = make_fixture(tmp_path)
    assert run_main(paths) == 0
    a = (paths["out"] / "c3" / "macros_c3.tex").read_bytes()
    t = (paths["out"] / "t12" / "t12_matched_speed.tex").read_bytes()
    assert run_main(paths) == 0
    assert (paths["out"] / "c3" / "macros_c3.tex").read_bytes() == a
    assert (paths["out"] / "t12" / "t12_matched_speed.tex").read_bytes() == t


def test_records_are_read_from_a_glob_and_report_half_only(tmp_path):
    paths = make_fixture(tmp_path)
    # a fit-half calibration directory next to the sweeps must not leak into the report numbers
    cal = paths["records"] / "calibrate-gsm8k"
    cal.mkdir()
    bogus = [{"config": "topk_conf[k=2]", "policy": "topk_conf", "k": 2, "task": "gsm8k", "doc_id": 999,
              "completion": "x", "n_forward": 1, "n_committed": 256, "seconds": 1.0, "error": None}]
    write_jsonl(cal / "t12.shard0of1.jsonl", bogus)
    (cal / "t12.shard0of1.meta.json").write_text(json.dumps({"task": "gsm8k", "half": "fit", "gen_length": 256}))
    argv = ["--out-dir", str(paths["out"]), "--records", str(paths["records"] / "*" / "t12.shard*.jsonl"),
            "--verdicts", str(paths["verdicts"]), "--grids", str(paths["grids"]),
            "--calibrators", str(paths["cal"]), "--sanity", str(paths["sanity"])]
    assert P.main(argv) == 0
    mac = parse_macros((paths["out"] / "c3" / "macros_c3.tex").read_text())
    assert mac["CThreeTTwelveGsmN"] == "30"
    assert mac["CThreeTTwelveGsmTopKSpeedOneTPS"] == "2.00"


def test_duplicate_records_from_a_resumed_shard_count_once(tmp_path):
    paths = make_fixture(tmp_path)
    f = paths["records"] / "sweep-gsm8k" / "t12.shard0of2.jsonl"
    f.write_text(f.read_text() + f.read_text().splitlines()[0] + "\n")
    assert run_main(paths) == 0
    s = json.loads((paths["out"] / "t12" / "llada-8b-instruct_fp16_t12-summary.json").read_text())
    assert s["tasks"]["gsm8k"]["n"] == 30


def test_score_rows_are_the_accuracy_when_there_are_no_verdicts(tmp_path):
    paths = make_fixture(tmp_path, with_verdicts=False)
    key = config_key(cfg_of("CThreeA", 2, "gsm8k"))
    rows = [{"task": "gsm8k", "config": key, "half": "report", "prompt_format": "chat",
             "n": 30, "n_correct": 24, "accuracy": 0.8, "ci95": [0.627, 0.905]}]
    sc = tmp_path / "scores.json"
    sc.write_text(json.dumps({"rows": rows}))
    assert run_main(paths, "--scores", str(sc), verdicts=False) == 0
    mac = parse_macros((paths["out"] / "c3" / "macros_c3.tex").read_text())
    assert mac["CThreeTTwelveGsmCThreeASpeedTwoAcc"] == "0.800"
    assert mac["CThreeTTwelveGsmCThreeASpeedTwoAccLo"] == "0.627"
    assert "CThreeTTwelveGsmTopKSpeedTwoAcc" not in mac           # no score row, no verdict
    assert "CThreeTTwelveGsmMatchedDiffTopKSpeedTwo" not in mac   # a difference needs the verdicts


# ------------------------------------------------------------------ missing inputs
@pytest.mark.parametrize("missing", ["records", "verdicts", "grids", "cal", "sanity"])
def test_a_missing_input_never_crashes(tmp_path, missing):
    paths = make_fixture(tmp_path)
    flags = {"records": True, "verdicts": True, "grids": True, "cal": True, "sanity": True}
    flags[missing] = False
    assert run_main(paths, **flags) == 0
    out = paths["out"]
    assert (out / "t12" / "t12_matched_speed.tex").exists()
    assert (out / "c3" / "macros_c3.tex").exists()
    s = json.loads((out / "t12" / "llada-8b-instruct_fp16_t12-summary.json").read_text())
    assert missing in "".join(s["missing_inputs"])


def test_no_inputs_at_all_gives_dashes(tmp_path):
    out = tmp_path / "out"
    assert P.main(["--out-dir", str(out), "--records", str(tmp_path / "nope"),
                   "--verdicts", str(tmp_path / "nope.jsonl"), "--grids", str(tmp_path / "nogrid"),
                   "--calibrators", str(tmp_path / "nocal"), "--sanity", str(tmp_path / "nosanity.json")]) == 0
    tex = (out / "t12" / "t12_matched_speed.tex").read_text()
    assert "\\begin{tabular}" in tex and "---" in tex
    mac = parse_macros((out / "c3" / "macros_c3.tex").read_text())
    assert mac == {}                       # pending: the paper prints its bold ?
    assert (out / "t12" / "llada-8b-instruct_fp16_t12-summary.json").exists()
    assert (out / "t12" / "zip" / "t12_sanity_cost.md").exists()


def test_missing_records_leave_t11_and_sanity_macros(tmp_path):
    paths = make_fixture(tmp_path)
    assert run_main(paths, records=False) == 0
    mac = parse_macros((paths["out"] / "c3" / "macros_c3.tex").read_text())
    assert "CThreeTElevenGsmFitDocs" in mac and "CThreeTTwelveSanityN" in mac
    assert "CThreeTTwelveGsmCThreeASpeedTwoAlpha" in mac            # the grid's design constant
    assert "CThreeTTwelveGsmN" not in mac and "CThreeTTwelveGsmTopKSpeedTwoAcc" not in mac


def test_unreadable_json_is_a_dash_not_a_crash(tmp_path):
    paths = make_fixture(tmp_path)
    (paths["cal"] / "llada-8b-instruct_fp16_c3a-gsm8k.json").write_text("{not json")
    (paths["sanity"]).write_text("")
    assert run_main(paths) == 0
    mac = parse_macros((paths["out"] / "c3" / "macros_c3.tex").read_text())
    assert "CThreeTElevenGsmFitDocs" not in mac and "CThreeTElevenMbppFitDocs" in mac
    assert "CThreeTTwelveSanityN" not in mac


# ------------------------------------------------------------------ the paper
PAPER = ROOT / "paper"
USED = re.compile(r"\\Cres(CI)?\{([A-Za-z]+)\}")
BARE = re.compile(r"\\(CThreeT(?:Eleven|Twelve)[A-Za-z]+)(?![A-Za-z<{\[])")


def paper_macro_uses():
    names = set()
    for f in ("sections/06_c3.tex", "sections/appendix_c3.tex", "main.tex"):
        text = (PAPER / f).read_text(encoding="utf-8")
        for ci, name in USED.findall(text):
            names |= {name} | ({name + "Lo", name + "Hi"} if ci else set())
        names |= set(BARE.findall(text))
    return names


def paper_file_uses():
    files = set()
    for f in ("sections/06_c3.tex", "sections/appendix_c3.tex", "main.tex", "Makefile"):
        text = (PAPER / f).read_text(encoding="utf-8")
        files |= set(re.findall(r"\.\./artifacts/((?:c3|t12)/[A-Za-z0-9_/\.]+)", text))
    # the viva slides (ojas): \CThreeTable{name} -> artifacts/t12/name.tex, \CThreeFigure{name} -> figures/name.pdf
    slides = (ROOT / "documents" / "viva-slides" / "slides.tex").read_text(encoding="utf-8")
    files |= {f"t12/{n}.tex" for n in re.findall(r"\\CThreeTable\{([A-Za-z0-9_]+)\}", slides)}
    files |= {f"t12/figures/{n}.pdf" for n in re.findall(r"\\CThreeFigure\{([A-Za-z0-9_]+)\}", slides)}
    out = set()
    for f in files:
        f = f.rstrip(".")
        out.add(f if Path(f).suffix else f + ".pdf")
    return out


def test_the_paper_uses_macros_the_script_defines(built):
    used = paper_macro_uses()
    assert len(used) >= 20
    missing = sorted(used - set(built["macros"]))
    assert not missing, f"the paper uses macros the script does not define: {missing}"


def test_the_paper_inputs_files_the_script_writes(built):
    files = paper_file_uses()
    assert {"c3/macros_c3.tex", "t12/t12_matched_speed.tex", "t12/figures/t12_matched_speed.pdf",
            "t12/t12_sanity_cost.tex"} <= files
    for f in files:
        assert (built["out"] / f).exists(), f"the paper reads artifacts/{f}; the script did not write it"


def test_no_macro_is_defined_that_the_naming_scheme_does_not_allow(built):
    dec = "|".join(DEC_POLICY)
    task = "Gsm|Mbpp"
    speed = "SpeedOne|SpeedTwo|SpeedThree"
    allowed = [
        rf"CThreeTEleven({task})(FitDocs|MonotoneShare|DevECERaw|DevECECal)",
        rf"CThreeTTwelve({task})N",
        rf"CThreeTTwelve({task})({dec})({speed})(Acc|AccLo|AccHi|TPS|NFE|Sec)",
        rf"CThreeTTwelve({task})MatchedDiff({dec})({speed})(Lo|Hi)?",
        rf"CThreeTTwelve({task})OutsideShare",
        r"CThreeTTwelveSanity(TopK|Threshold)Acc(Lo|Hi)?", r"CThreeTTwelveSanityN",
        rf"CThreeTTwelve({task})CThreeA({speed})Alpha", rf"CThreeTTwelve({task})AlphaPrimary",
    ]
    for name in built["macros"]:
        assert any(re.fullmatch(p, name) for p in allowed), name


def test_cost_table_never_uses_scientific_notation(built):
    tex = (built["out"] / "t12" / "t12_sanity_cost.tex").read_text()
    assert "\times" not in tex and "$128.0$" in tex
