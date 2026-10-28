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


JUNK_DOC = 1000                                      # a fit-half doc_id in the fixture manifests


def meta_of(task, shard, nshards=2, n_docs=N_DOCS, **over):
    """The meta file ``run_t12.py`` writes for one shard (the fields score_t12.RUN_FIELDS and
    the completeness checks read)."""
    m = {"task": task, "half": "report", "gen_length": 256, "run_id": f"run{shard}", "precision": "fp16",
         "model": "llada-8b-instruct", "model_revision": "rev1", "task_jsonl_sha256": "t" * 8,
         "manifest_sha256": "m" * 8, "configs_sha256": "c" * 8, "calibrator_sha256": "k" * 8,
         "c3a_clamp": False, "pc_sampler_variant": "full", "decoding": "greedy", "block_length": 32,
         "prompt": {"format": "chat", "fewshot": 0}, "limit": 0, "shard": shard, "num_shards": nshards,
         "label": "jl-report", "provenance": {"host": "jarvis-1", "gpu": "RTX PRO 6000", "git_commit": "abc1234"},
         "doc_ids": [i for i in range(n_docs) if i % nshards == shard]}
    m.update(over)
    return m


def make_fixture(root: Path, with_records=True, with_verdicts=True, with_grids=True,
                 with_cal=True, with_sanity=True, with_manifests=True, nshards=2, n_docs=N_DOCS):
    """Write every input under ``root``; returns the paths."""
    paths = {"records": root / "runs", "verdicts": root / "verdicts.jsonl", "grids": root / "grids",
             "cal": root / "cal", "sanity": root / "sanity.json", "out": root / "out",
             "manifests": root / "manifests"}
    verdicts = []
    for task in TASKS:
        shard_rows = {sh: [] for sh in range(nshards)}
        for i in range(n_docs):
            for dec in DEC_POLICY:
                for s in (1, 2, 3):
                    cfg = cfg_of(dec, s, task)
                    key = config_key(cfg)
                    rec = {"config": key, **cfg, "task": task, "doc_id": i, "run_id": f"run{i % nshards}",
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
                    shard_rows[i % nshards].append(rec)
                    ok = (not failed(task, dec, s, i)) and correct_of(task, dec, s, i)
                    verdicts.append({"run": P.run_key(meta_of(task, i % nshards, nshards, n_docs)), "run_id": f"run{i % nshards}", "prompt_format": "chat",
                                     "task": task, "config": key, "doc_id": i, "stopped": True,
                                     "correct": bool(ok), "extracted": None, "driver_error": failed(task, dec, s, i),
                                     "error": None})
        if with_records:
            d = paths["records"] / f"sweep-{task}"
            d.mkdir(parents=True, exist_ok=True)
            for sh, rows in shard_rows.items():
                write_jsonl(d / f"t12.shard{sh}of{nshards}.jsonl", rows)
                (d / f"t12.shard{sh}of{nshards}.meta.json").write_text(json.dumps(meta_of(task, sh, nshards, n_docs)))
        if with_manifests:
            paths["manifests"].mkdir(exist_ok=True)
            man = {str(i): 1 for i in range(n_docs)} | {str(JUNK_DOC + i): 0 for i in range(10)}
            (paths["manifests"] / f"split_manifest_{task}.json").write_text(json.dumps(man))
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


def run_main(paths, *extra, records=True, verdicts=True, grids=True, cal=True, sanity=True, manifests=True):
    argv = ["--out-dir", str(paths["out"])]
    if records:
        argv += ["--records", str(paths["records"])]
    if verdicts:
        argv += ["--verdicts", str(paths["verdicts"])]
    nope = str(paths["out"].parent / "does-not-exist")   # an omitted flag would use the repo defaults
    argv += ["--grids", str(paths["grids"]) if grids else nope]
    argv += ["--calibrators", str(paths["cal"]) if cal else nope]
    argv += ["--sanity", str(paths["sanity"]) if sanity else nope]
    argv += ["--manifest-dir", str(paths["manifests"]) if manifests else nope]
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


def test_bootstrap_known_numbers():
    """Pinned values (numpy default_rng, seed 20261030, 2,000 resamples), not a re-implementation."""
    assert P.SEED == 20261030 and P.N_BOOT == 2000
    # d = [1, 0, -1, 1, 0, 0]: mean 1/6; resampled means are multiples of 1/6
    a = {0: 1, 1: 1, 2: 0, 3: 1, 4: 1, 5: 0}
    b = {0: 0, 1: 1, 2: 1, 3: 0, 4: 1, 5: 0}
    r = P.paired_bootstrap(a, b)
    assert r["n"] == 6
    assert (r["diff"], r["lo"], r["hi"]) == pytest.approx((1 / 6, -1 / 3, 2 / 3), abs=1e-12)
    # every problem favours a: a zero-width interval at 1
    r = P.paired_bootstrap({0: 1, 1: 1, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 0, 3: 0})
    assert (r["diff"], r["lo"], r["hi"], r["n"]) == (1.0, 1.0, 1.0, 4)
    a = {i: int(i % 4 != 0) for i in range(40)}
    b = {i: int(i % 3 != 0) for i in range(40)}
    r = P.paired_bootstrap(a, b)
    assert (r["diff"], r["lo"], r["hi"]) == pytest.approx((0.1, -0.075625, 0.3), abs=1e-12)
    assert r == P.paired_bootstrap(a, b)                        # deterministic
    assert P.paired_bootstrap(a, b, seed=1)["lo"] == pytest.approx(-0.1, abs=1e-12)


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
    assert built["macros"]["CThreeTTwelveGsmMatchedDiffKlassSpeedThree"] == DASH_MACRO
    assert built["macros"]["CThreeTTwelveGsmMatchedDiffKlassSpeedThreeLo"] == DASH_MACRO
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


DASH_MACRO = r"\textnormal{---}"        # an em dash in text, never a minus inside the ensuremath of CresCI
NUMBER = re.compile(r"^(\\ensuremath\{-\})?\d+(\.\d+)?$")


def test_macros_are_wellformed_latex_with_three_decimals(built):
    for name, body in built["macros"].items():
        assert re.fullmatch(r"[A-Za-z]+", name)
        assert name.startswith(("CThreeTEleven", "CThreeTTwelve"))
        assert body == DASH_MACRO or NUMBER.match(body), (name, body)
        assert body.count("{") == body.count("}")
        three_dp = re.search(r"(Acc|AccLo|AccHi|MatchedDiff\w*|Share|Alpha|AlphaPrimary|DevECE\w+)$", name)
        if three_dp and body != DASH_MACRO:
            assert re.search(r"\.\d{3,5}$", body), (name, body)
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
    assert mac["CThreeTTwelveGsmMatchedDiffEntropySpeedTwo"] == DASH_MACRO
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


def body_rows(tex):
    """The data rows of the tabular: the lines ending in a row break between the first and the last
    rule (midrules and cmidrules end no row)."""
    lines = tex.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("\\toprule"))
    j = next(k for k, l in enumerate(lines) if l.startswith("\\bottomrule"))
    return [l for l in lines[i + 1:j] if l.rstrip().endswith("\\\\")]


def test_table_is_a_bare_booktabs_tabular_one_row_per_decoder_tasks_side_by_side(built):
    tex = (built["out"] / "t12" / "t12_matched_speed.tex").read_text()
    assert "\\begin{tabular}" in tex and "\\end{tabular}" in tex
    assert "\\toprule" in tex and "\\bottomrule" in tex and "\\midrule" in tex and "\\cmidrule" in tex
    assert "\\begin{table" not in tex and "\\caption" not in tex
    assert tex.count("\\begingroup") == tex.count("\\endgroup") == 1
    assert "GSM8K" in tex and "MBPP" in tex
    for name in ("Top-$k$", "Threshold", "Margin", "Entropy", "KLASS", "PC-Sampler", "C3a"):
        assert sum(l.startswith(name + " &") for l in tex.splitlines()) == 1, name   # one row, both tasks
    # entropy is unmatched at speed 2: both of its task groups are dashes
    row = next(l for l in tex.splitlines() if l.startswith("Entropy"))
    assert row.count("---") == 6                                   # (tok/step, acc, diff) x 2 tasks
    # C3a's own row is the reference: no difference, in either task
    assert next(l for l in tex.splitlines() if l.startswith("C3a &")).count("ref.") == 2
    assert not any(c in tex for c in ("nan", "None"))
    # seven columns: the decoder, then tok/step, acc, diff per task
    assert re.search(r"\\begin\{tabular\}\{lrll(@\{\\hspace\{[^}]*\}\})?rll\}", tex)


def test_table_is_compact_at_most_nine_body_rows(built):
    """Sec. 6 has ~0.15 page: 2 header lines + one row per decoder, the tasks side by side."""
    tex = (built["out"] / "t12" / "t12_matched_speed.tex").read_text()
    rows = body_rows(tex)
    assert len(rows) <= 9
    assert len(rows) == 2 + len(P.DECODERS)
    assert rows[0].count("\\multicolumn{3}") == 2                  # the task groups
    assert rows[1].startswith("Decoder")
    assert all(r.count("&") == 6 for r in rows[1:])               # 7 columns
    assert "\\footnotesize" in tex and "\\tabcolsep" in tex
    assert "$n$" not in tex                                       # no n column


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
    assert row.rstrip(" \\").endswith("& $1$ & $0$ & $0$")             # errors, timeouts, unstopped
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


def test_records_are_read_from_a_glob(tmp_path):
    paths = make_fixture(tmp_path)
    argv = ["--out-dir", str(paths["out"]), "--records", str(paths["records"] / "sweep-*" / "t12.shard*.jsonl"),
            "--verdicts", str(paths["verdicts"]), "--grids", str(paths["grids"]),
            "--calibrators", str(paths["cal"]), "--sanity", str(paths["sanity"]),
            "--manifest-dir", str(paths["manifests"])]
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
             "run": P.run_key(meta_of("gsm8k", 0)), "n": 30, "n_correct": 24, "accuracy": 0.8, "ci95": [0.627, 0.905]}]
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


# ------------------------------------------------------------------ plumbing: completeness, runs, halves
def edit_rows(path, fn):
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    write_jsonl(path, fn(rows))


def cap_at(paths, k, tasks=("gsm8k", "mbpp")):
    """Every configuration stopped after doc k-1 (problems are the outer loop: a time cap)."""
    for t in tasks:
        for f in (paths["records"] / f"sweep-{t}").glob("t12.shard*.jsonl"):
            edit_rows(f, lambda rows: [r for r in rows if r["doc_id"] < k])


def summary_of(paths):
    return json.loads((paths["out"] / "t12" / "llada-8b-instruct_fp16_t12-summary.json").read_text())


def test_missing_shard_is_refused_and_writes_nothing(tmp_path):
    paths = make_fixture(tmp_path)
    d = paths["records"] / "sweep-gsm8k"
    (d / "t12.shard1of2.jsonl").unlink()
    (d / "t12.shard1of2.meta.json").unlink()
    with pytest.raises(SystemExit) as e:
        run_main(paths)
    assert "shard" in str(e.value) and "gsm8k" in str(e.value)
    assert not (paths["out"] / "c3" / "macros_c3.tex").exists()


def test_a_cap_at_doc_k_is_refused_with_and_without_the_manifest(tmp_path):
    paths = make_fixture(tmp_path)
    cap_at(paths, 20)
    with pytest.raises(SystemExit) as e:
        run_main(paths)
    assert "20/30" in str(e.value)
    with pytest.raises(SystemExit):                       # the metas' doc_ids alone give the 30
        run_main(paths, manifests=False)


def test_a_config_cut_short_inside_the_last_problem_is_refused(tmp_path):
    paths = make_fixture(tmp_path)
    f = paths["records"] / "sweep-gsm8k" / "t12.shard0of2.jsonl"
    edit_rows(f, lambda rows: [r for r in rows if not (r["doc_id"] == 28 and r["config"].startswith("c3a"))])
    with pytest.raises(SystemExit) as e:
        run_main(paths)
    assert "29/30" in str(e.value)


def test_allow_partial_with_a_missing_shard_reports_n_of_the_complete_problems(tmp_path):
    paths = make_fixture(tmp_path)
    d = paths["records"] / "sweep-gsm8k"
    (d / "t12.shard1of2.jsonl").unlink()
    (d / "t12.shard1of2.meta.json").unlink()
    assert run_main(paths, "--allow-partial") == 0
    s = summary_of(paths)
    g = s["tasks"]["gsm8k"]
    assert (g["n"], g["n_complete"], g["n_planned"], g["partial"]) == (15, 15, 30, True)
    assert (g["n_dropped_problems"], g["n_dropped_records"]) == (0, 0)     # the shard is absent, not cut
    assert g["decoders"]["TopK"]["speeds"]["2"]["n_planned"] == 30
    assert s["tasks"]["mbpp"]["partial"] is False
    tex = (paths["out"] / "t12" / "t12_matched_speed.tex").read_text()
    assert "$N=15$" in tex and "$N=30$" in tex and "partial" not in tex
    mtex = (paths["out"] / "c3" / "macros_c3.tex").read_text()
    assert "% PARTIAL: gsm8k N=15 of 30 planned" in mtex
    assert "\\newcommand{\\CThreeTTwelveGsmN}{15}" in mtex
    assert "gsm8k N=15" in (paths["out"] / "t12" / "zip" / "t12_sanity_cost.md").read_text()


def test_complete_run_has_no_partial_marker(built):
    assert "partial" not in (built["out"] / "t12" / "t12_matched_speed.tex").read_text()
    assert "PARTIAL" not in (built["out"] / "c3" / "macros_c3.tex").read_text()
    g = built["summary"]["tasks"]["gsm8k"]
    assert (g["n"], g["n_complete"], g["n_planned"], g["partial"]) == (30, 30, 30, False)
    assert (g["n_dropped_problems"], g["n_dropped_records"]) == (0, 0)
    assert g["prefix_check"] == {"ok": True, "violations": []}
    assert "kept problems" in g["rule"] and built["summary"]["completeness_rule"] == g["rule"]


def test_two_runs_for_one_task_are_refused(tmp_path):
    paths = make_fixture(tmp_path)
    # an Ada copy of the same sweep next to the Jarvis run: different model revision / machine
    src = paths["records"] / "sweep-gsm8k"
    dst = paths["records"] / "ada-copy-gsm8k"
    dst.mkdir()
    for f in src.glob("t12.shard*"):
        (dst / f.name).write_text(f.read_text())
    for sh in (0, 1):
        m = json.loads((dst / f"t12.shard{sh}of2.meta.json").read_text())
        m["model_revision"] = "other"
        (dst / f"t12.shard{sh}of2.meta.json").write_text(json.dumps(m))
    with pytest.raises(SystemExit) as e:
        run_main(paths)
    assert "more than one run" in str(e.value) and "gsm8k" in str(e.value)


def test_verdicts_of_another_run_are_ignored_and_non_verdict_lines_skipped(tmp_path):
    paths = make_fixture(tmp_path)
    key = config_key(cfg_of("CThreeA", 2, "gsm8k"))
    vs = [json.loads(l) for l in paths["verdicts"].read_text().splitlines()]
    other = [dict(v, run="deadbeef0000", correct=False) for v in vs if v["config"] == key and v["task"] == "gsm8k"]
    rec_line = {"config": key, "task": "gsm8k", "doc_id": 0, "completion": "x", "n_forward": 1}   # no `correct`
    write_jsonl(paths["verdicts"], other + vs[:1] + [rec_line] + vs[1:])
    assert run_main(paths) == 0
    s = summary_of(paths)["tasks"]["gsm8k"]["decoders"]["CThreeA"]["speeds"]["2"]
    assert s["n_correct"] == 24 and s["n_scored"] == 30


def test_a_fit_half_doc_id_is_refused(tmp_path):
    paths = make_fixture(tmp_path)
    f = paths["records"] / "sweep-gsm8k" / "t12.shard0of2.jsonl"
    edit_rows(f, lambda rows: rows + [dict(rows[0], doc_id=JUNK_DOC)])
    with pytest.raises(SystemExit) as e:
        run_main(paths)
    assert "fit" in str(e.value)


def test_a_shard_without_a_meta_file_is_refused(tmp_path):
    paths = make_fixture(tmp_path)
    (paths["records"] / "sweep-gsm8k" / "t12.shard0of2.meta.json").unlink()
    with pytest.raises(SystemExit) as e:
        run_main(paths)
    assert "meta" in str(e.value)


@pytest.mark.parametrize("over,word", [({"precision": "bf16"}, "fp16"),
                                       ({"prompt": {"format": "lmeval", "fewshot": 5}}, "chat")])
def test_meta_must_say_fp16_and_the_chat_prompt(tmp_path, over, word):
    paths = make_fixture(tmp_path)
    for sh in (0, 1):
        mp = paths["records"] / "sweep-gsm8k" / f"t12.shard{sh}of2.meta.json"
        m = json.loads(mp.read_text())
        m.update(over)
        mp.write_text(json.dumps(m))
    with pytest.raises(SystemExit) as e:
        run_main(paths)
    assert word in str(e.value)


def test_duplicate_with_a_different_completion_warns_but_identical_does_not(tmp_path):
    paths = make_fixture(tmp_path)
    f = paths["records"] / "sweep-gsm8k" / "t12.shard0of2.jsonl"
    rows = [json.loads(l) for l in f.read_text().splitlines()]
    write_jsonl(f, rows + [rows[0]])                       # identical resumed line
    assert run_main(paths) == 0
    assert not any("different completion" in w for w in summary_of(paths)["warnings"])
    write_jsonl(f, rows + [dict(rows[0], completion="y")])
    assert run_main(paths) == 0
    assert any("different completion" in w for w in summary_of(paths)["warnings"])


# ------------------------------------------------------------------ driver errors, columns, rule text
def inject_driver_error(paths, dec="TopK", slot=2, doc=5, task="gsm8k"):
    key = config_key(cfg_of(dec, slot, task))
    f = paths["records"] / f"sweep-{task}" / f"t12.shard{doc % 2}of2.jsonl"

    def fn(rows):
        for r in rows:
            if r["config"] == key and r["doc_id"] == doc:
                for k in ("n_forward", "n_committed", "tokens_per_step", "seconds", "stopped"):
                    r.pop(k, None)
                r.update({"completion": None, "error": "OutOfMemoryError: out of memory"})
        return rows
    edit_rows(f, fn)
    vs = [json.loads(l) for l in paths["verdicts"].read_text().splitlines()]
    for v in vs:
        if v["task"] == task and v["config"] == key and v["doc_id"] == doc:
            v.update({"correct": False, "driver_error": True})
    write_jsonl(paths["verdicts"], vs)
    return key


def test_driver_errors_warn_are_scored_wrong_and_leave_the_speed_over_the_rest(tmp_path):
    paths = make_fixture(tmp_path)
    inject_driver_error(paths)
    assert run_main(paths) == 0
    s = summary_of(paths)
    sp = s["tasks"]["gsm8k"]["decoders"]["TopK"]["speeds"]["2"]
    assert (sp["n"], sp["n_ok"], sp["failures"]) == (30, 29, 1)
    nfes = [nfe_of("TopK", 2, i) for i in range(N_DOCS) if i != 5]
    assert sp["tps"] == pytest.approx(256 * 29 / sum(nfes))               # 256 N_ok / sum NFE_ok, no imputing
    assert sp["n_scored"] == 30 and sp["n_correct"] == sum(correct_of("gsm8k", "TopK", 2, i) for i in range(N_DOCS) if i != 5)
    assert any("driver error" in w and "TopK" in w or "driver error" in w and "topk_conf" in w for w in s["warnings"])
    tex = (paths["out"] / "t12" / "t12_matched_speed.tex").read_text()
    # Sec. 6 has no line to spare: the matched table carries the note as a LaTeX comment only; the
    # footnote is in the sanity/cost table, n_ok in the zip tables, the counts in the summary JSON
    assert "% driver errors scored wrong; speed is over the problems that decoded: GSM8K Top-$k$: 29/30 decoded" in tex
    assert "\\scriptsize" not in tex.replace("{\\scriptsize[", "") and len(body_rows(tex)) == 2 + len(P.DECODERS)
    cost = (paths["out"] / "t12" / "t12_sanity_cost.tex").read_text()
    assert "speed is over the problems that decoded" in cost


def test_no_driver_errors_no_footnote(built):
    assert "driver errors scored wrong" not in (built["out"] / "t12" / "t12_matched_speed.tex").read_text()


def test_n_ok_is_in_the_zip_tables(built):
    z = built["out"] / "t12" / "zip"
    head = (z / "t12_sanity_cost.csv").read_text().splitlines()[0].split(",")
    assert "n_ok" in head
    row = next(r for r in (z / "t12_sanity_cost.csv").read_text().splitlines() if r.startswith("mbpp,TopK,3,"))
    assert row.split(",")[head.index("n_ok")] == "29"
    assert "n_ok" in (z / "t12_speed1_speed3.csv").read_text().splitlines()[0].split(",")


def test_matched_table_has_no_n_column_but_names_a_short_pairing(built):
    tex = (built["out"] / "t12" / "t12_matched_speed.tex").read_text()
    assert "$n$" not in tex and "$N=30$" in tex                       # N is in the task header
    assert "paired on fewer" not in tex                               # every difference is paired on all N
    assert built["summary"]["tasks"]["gsm8k"]["matched"]["2"]["TopK"]["n_paired"] == 30
    S = json.loads(json.dumps(built["summary"]))
    S["tasks"]["gsm8k"]["matched"]["2"]["TopK"]["n_paired"] = 28
    tex = P.matched_table(S)
    assert "% difference paired on fewer than N problems: GSM8K Top-$k$: 28" in tex
    assert len(body_rows(tex)) == 2 + len(P.DECODERS)                 # still no footnote row


def test_fail_columns_are_split(built):
    tex = (built["out"] / "t12" / "t12_sanity_cost.tex").read_text()
    assert "err" in tex and "t/o" in tex and "uns" in tex
    assert tex.count("{13}") >= 1 and "{11}" not in tex


def test_tie_rule_is_in_the_pairing_rule(built):
    assert "tie" in built["summary"]["pairing"]["rule"] and "lower" in built["summary"]["pairing"]["rule"]
    equal = {1: {"tps": 3.6, "unmatched": False}, 2: {"tps": 4.4, "unmatched": False}}
    assert P.match_baseline(2, 4.0, equal)["slot"] == 1               # equidistant: the lower slot


# ------------------------------------------------------------------ a second complete run of a task
def copy_run(paths, dst_name, num_shards=None, **meta_over):
    """A complete copy of the gsm8k sweep in a sibling directory (the reviewer's 'Ada copy next to
    the Jarvis run'), optionally with another num_shards / meta fields.  The pinned inputs are the same, so
    score_t12's RUN_FIELDS hash is the same: only the other facts can tell the two apart."""
    src = paths["records"] / "sweep-gsm8k"
    dst = paths["records"] / dst_name
    dst.mkdir()
    for sh in (0, 1):
        n = num_shards or 2
        (dst / f"t12.shard{sh}of{n}.jsonl").write_text((src / f"t12.shard{sh}of2.jsonl").read_text())
        m = json.loads((src / f"t12.shard{sh}of2.meta.json").read_text())
        m.update({"num_shards": n, **meta_over})
        (dst / f"t12.shard{sh}of{n}.meta.json").write_text(json.dumps(m))
    return dst


def test_the_runs_of_two_copies_share_the_run_key():
    # the premise of the reviewer's case: nothing in RUN_FIELDS tells the copies apart
    assert P.run_key(meta_of("gsm8k", 0)) == P.run_key(meta_of("gsm8k", 0, num_shards=12, label="ada"))


def test_a_complete_copy_with_other_num_shards_is_refused(tmp_path):
    paths = make_fixture(tmp_path)
    copy_run(paths, "ada-copy-gsm8k", num_shards=12)
    with pytest.raises(SystemExit) as e:
        run_main(paths)
    assert "more than one run" in str(e.value) and "num_shards" in str(e.value)
    assert not (paths["out"] / "c3" / "macros_c3.tex").exists()


def test_a_complete_copy_with_the_same_num_shards_in_another_directory_is_refused(tmp_path):
    paths = make_fixture(tmp_path)
    copy_run(paths, "ada-copy-gsm8k")
    with pytest.raises(SystemExit) as e:
        run_main(paths)
    assert "more than one run" in str(e.value) and "director" in str(e.value)


@pytest.mark.parametrize("over,word", [({"label": "ada"}, "label"),
                                       ({"provenance": {"host": "gnode006", "gpu": "RTX 2080 Ti"}}, "host")])
def test_shards_of_one_task_must_agree_on_label_and_machine(tmp_path, over, word):
    paths = make_fixture(tmp_path)
    mp = paths["records"] / "sweep-gsm8k" / "t12.shard1of2.meta.json"
    m = json.loads(mp.read_text())
    m.update(over)
    mp.write_text(json.dumps(m))
    with pytest.raises(SystemExit) as e:
        run_main(paths)
    assert word in str(e.value)


def test_the_same_problem_in_two_shard_files_is_refused(tmp_path):
    paths = make_fixture(tmp_path)
    d = paths["records"] / "sweep-gsm8k"
    rows0 = [json.loads(l) for l in (d / "t12.shard0of2.jsonl").read_text().splitlines()]
    f1 = d / "t12.shard1of2.jsonl"
    edit_rows(f1, lambda rows: rows + [rows0[0]])            # doc 0 is shard 0's
    with pytest.raises(SystemExit) as e:
        run_main(paths)
    assert "two shard files" in str(e.value)


def test_an_identical_duplicate_inside_one_shard_file_still_only_warns(tmp_path):
    paths = make_fixture(tmp_path)
    f = paths["records"] / "sweep-gsm8k" / "t12.shard0of2.jsonl"
    f.write_text(f.read_text() + f.read_text().splitlines()[0] + "\n")
    assert run_main(paths) == 0
    assert any("repeated" in w for w in summary_of(paths)["warnings"])


# ------------------------------------------------------------------ PR #62 review: accuracy and verdict plumbing
C3A_KEY = config_key(cfg_of("CThreeA", 2, "gsm8k"))


def drop_verdicts(paths, key, keep=lambda v: False):
    vs = [json.loads(l) for l in paths["verdicts"].read_text().splitlines()]
    write_jsonl(paths["verdicts"], [v for v in vs if not (v["config"] == key and v["task"] == "gsm8k") or keep(v)])


def test_failures_alone_never_become_the_accuracy(tmp_path):
    """C3a with driver errors and no verdict at all used to score 0.0 on its two failures, and a
    paired difference of -1.0."""
    paths = make_fixture(tmp_path)
    inject_driver_error(paths, "CThreeA", 2, 5)
    inject_driver_error(paths, "CThreeA", 2, 6)
    drop_verdicts(paths, C3A_KEY, keep=lambda v: v["driver_error"])      # only the two failed docs have verdicts
    drop_verdicts(paths, C3A_KEY)                                        # ... and now none at all
    assert run_main(paths) == 0
    s = summary_of(paths)
    sp = s["tasks"]["gsm8k"]["decoders"]["CThreeA"]["speeds"]["2"]
    assert sp.get("acc") is None and "n_scored" not in sp
    mac = parse_macros((paths["out"] / "c3" / "macros_c3.tex").read_text())
    assert "CThreeTTwelveGsmCThreeASpeedTwoAcc" not in mac
    assert "CThreeTTwelveGsmMatchedDiffTopKSpeedTwo" not in mac           # a difference needs both accuracies
    assert not any(v.endswith("1.000") for k, v in mac.items() if "MatchedDiff" in k)
    assert s["tasks"]["gsm8k"]["matched"]["2"]["TopK"]["diff"] is None
    assert any("accuracy left blank" in w and "gsm8k" in w for w in s["warnings"])


def test_failures_are_imputed_to_zero_next_to_real_verdicts_only(tmp_path):
    paths = make_fixture(tmp_path)
    inject_driver_error(paths, "CThreeA", 2, 5)                          # its verdict says wrong; keep it
    drop_verdicts(paths, C3A_KEY, keep=lambda v: v["doc_id"] != 5)       # the failed doc has no verdict line
    assert run_main(paths) == 0
    sp = summary_of(paths)["tasks"]["gsm8k"]["decoders"]["CThreeA"]["speeds"]["2"]
    assert (sp["n_scored"], sp["n_correct"]) == (30, 24 - (1 if correct_of("gsm8k", "CThreeA", 2, 5) else 0))


def test_the_score_row_is_used_when_a_config_has_no_verdicts(tmp_path):
    paths = make_fixture(tmp_path)
    inject_driver_error(paths, "CThreeA", 2, 5)
    inject_driver_error(paths, "CThreeA", 2, 6)
    drop_verdicts(paths, C3A_KEY)
    sc = tmp_path / "scores.json"
    sc.write_text(json.dumps({"rows": [{"task": "gsm8k", "config": C3A_KEY, "half": "report", "prompt_format": "chat",
                                        "run": P.run_key(meta_of("gsm8k", 0)), "n": 30, "n_correct": 22,
                                        "accuracy": 22 / 30, "ci95": [0.55, 0.85]}]}))
    assert run_main(paths, "--scores", str(sc)) == 0
    s = summary_of(paths)
    sp = s["tasks"]["gsm8k"]["decoders"]["CThreeA"]["speeds"]["2"]
    assert (sp["n_correct"], sp["n_scored"]) == (22, 30)
    assert s["tasks"]["gsm8k"]["matched"]["2"]["TopK"]["diff"] is None   # no verdicts: no paired difference


def test_a_score_row_with_fewer_problems_than_records_is_blank(tmp_path):
    paths = make_fixture(tmp_path)
    drop_verdicts(paths, C3A_KEY)
    sc = tmp_path / "scores.json"
    sc.write_text(json.dumps({"rows": [{"task": "gsm8k", "config": C3A_KEY, "half": "report", "prompt_format": "chat",
                                        "run": P.run_key(meta_of("gsm8k", 0)), "n": 20, "n_correct": 15,
                                        "accuracy": 0.75, "ci95": [0.5, 0.9]}]}))
    assert run_main(paths, "--scores", str(sc)) == 0
    assert summary_of(paths)["tasks"]["gsm8k"]["decoders"]["CThreeA"]["speeds"]["2"].get("acc") is None


def test_a_config_with_some_verdicts_missing_has_no_accuracy(tmp_path):
    paths = make_fixture(tmp_path)
    drop_verdicts(paths, C3A_KEY, keep=lambda v: v["doc_id"] < 22)       # 8 records lack a verdict
    assert run_main(paths) == 0
    s = summary_of(paths)
    assert s["tasks"]["gsm8k"]["decoders"]["CThreeA"]["speeds"]["2"].get("acc") is None
    assert any("8 of 30" in w for w in s["warnings"])


def test_a_stale_run_less_verdict_file_in_the_directory_does_not_win(tmp_path):
    paths = make_fixture(tmp_path)
    good = [json.loads(l) for l in paths["verdicts"].read_text().splitlines()]
    vdir = tmp_path / "verdicts"
    vdir.mkdir()
    stale = [{k: v for k, v in dict(g, correct=False).items() if k != "run"} for g in good]
    write_jsonl(vdir / "a_stale.jsonl", stale)                       # read first, no run key
    write_jsonl(vdir / "b_current.jsonl", good)
    paths["verdicts"] = vdir
    assert run_main(paths) == 0
    s = summary_of(paths)
    sp = s["tasks"]["gsm8k"]["decoders"]["CThreeA"]["speeds"]["2"]
    assert sp["n_correct"] == 24
    assert any("without a run" in w for w in s["warnings"])


def test_verdicts_are_keyed_by_run(tmp_path):
    paths = make_fixture(tmp_path)
    vs = [json.loads(l) for l in paths["verdicts"].read_text().splitlines()]
    write_jsonl(paths["verdicts"], vs)
    vd = P.load_verdicts([paths["verdicts"]], [], {}, {P.run_key(meta_of("gsm8k", 0))})
    assert (P.run_key(meta_of("gsm8k", 0)), "gsm8k", C3A_KEY) in vd


def test_macros_round_like_the_table():
    # a CI that excludes 0 but prints 0.000 at 3 decimals: both raise the decimals together
    m = P.Macros()
    P.est_macros(m, "X", 0.0004, 0.0002, 0.0007)
    bodies = {line.split("{")[1][1:-1]: line.split("}{")[1][:-1] for line in m.lines}
    assert bodies == {"X": "0.0004", "XLo": "0.0002", "XHi": "0.0007"}
    cell = P.est_ci({"value": 0.0004, "ci": [0.0002, 0.0007]})
    assert "0.0004" in cell and "0.0002" in cell and "0.0007" in cell
    m = P.Macros()
    P.est_macros(m, "Y", -0.0204, -0.04, 0.0004)
    assert [line.split("}{")[1][:-1] for line in m.lines] == ["\\ensuremath{-}0.020", "\\ensuremath{-}0.040", "0.000"]


def test_the_difference_header_matches_the_caption(built):
    tex = (built["out"] / "t12" / "t12_matched_speed.tex").read_text()
    assert "C3a $-$ decoder [95\\% CI]" in tex and "\\Delta" not in tex


def test_a_fit_half_shard_under_records_is_refused(tmp_path):
    paths = make_fixture(tmp_path)
    cal = paths["records"] / "calibrate-gsm8k"
    cal.mkdir()
    write_jsonl(cal / "t12.shard0of1.jsonl", [{"config": "topk_conf[k=2]", "task": "gsm8k", "doc_id": 1000}])
    (cal / "t12.shard0of1.meta.json").write_text(json.dumps(meta_of("gsm8k", 0, half="fit", num_shards=1)))
    with pytest.raises(SystemExit) as e:
        run_main(paths)
    assert "fit" in str(e.value)


# ------------------------------------------------------------------ the completeness rule (deviation 17)
# 3 shards of 4 problems (shard s holds docs s, s+3, s+6, s+9); the last problem of each shard (9, 10,
# 11) is incomplete: doc 9 lacks C3a, doc 10 has top-k only, doc 11 lacks one configuration.
R_DOCS, R_SHARDS = 12, 3
PARTIAL_DOCS = (9, 10, 11)
KEPT_N = 9


def _keep(doc, r):
    if doc == 9:
        return not r["config"].startswith("c3a")
    if doc == 10:
        return r["config"].startswith("topk_conf")
    return r["config"] != config_key(cfg_of("Klass", 3, r["task"]))


def partial_fixture(root):
    paths = make_fixture(root, nshards=R_SHARDS, n_docs=R_DOCS)
    for task in TASKS:
        for f in (paths["records"] / f"sweep-{task}").glob("t12.shard*.jsonl"):
            edit_rows(f, lambda rows: [r for r in rows if r["doc_id"] not in PARTIAL_DOCS or _keep(r["doc_id"], r)])
    return paths


def n_records_cut():
    return sum(1 for doc in PARTIAL_DOCS for dec in DEC_POLICY for s in (1, 2, 3)
               if _keep(doc, {"config": config_key(cfg_of(dec, s, "gsm8k")), "task": "gsm8k"}))


def test_partial_problems_are_refused_without_the_flag_and_nothing_is_written(tmp_path):
    paths = partial_fixture(tmp_path)
    with pytest.raises(SystemExit) as e:
        run_main(paths)
    assert "--allow-partial" in str(e.value) and "fewer than the 12" in str(e.value)
    assert not (paths["out"] / "c3" / "macros_c3.tex").exists()


def test_n_is_the_number_of_complete_problems_everywhere(tmp_path):
    paths = partial_fixture(tmp_path)
    assert run_main(paths, "--allow-partial") == 0
    s = summary_of(paths)
    for task in TASKS:
        t = s["tasks"][task]
        assert (t["n"], t["n_complete"], t["n_planned"]) == (KEPT_N, KEPT_N, R_DOCS)
        assert t["n_dropped_problems"] == 3
        assert t["n_dropped_records"] == n_records_cut() == 18 + 3 + 20
        assert "kept problems" in t["rule"] and "every configuration" in t["rule"]
        assert t["prefix_check"] == {"ok": True, "violations": []}
        assert {sp["n"] for d in t["decoders"].values() for sp in d["speeds"].values()} == {KEPT_N}
        assert {sp["n_scored"] for d in t["decoders"].values() for sp in d["speeds"].values()
                if "n_scored" in sp} == {KEPT_N}
        assert {e["n_paired"] for per in t["matched"].values() for e in per.values() if e["matched"]} == {KEPT_N}
    mac = parse_macros((paths["out"] / "c3" / "macros_c3.tex").read_text())
    assert mac["CThreeTTwelveGsmN"] == mac["CThreeTTwelveMbppN"] == str(KEPT_N)
    tex = (paths["out"] / "t12" / "t12_matched_speed.tex").read_text()
    assert tex.count(f"$N={KEPT_N}$") == 2 and "/12" not in tex and "partial" not in tex
    assert f"$N={KEPT_N}$" in (paths["out"] / "t12" / "t12_sanity_cost.tex").read_text()
    assert "% PARTIAL: gsm8k N=9 of 12 planned" in (paths["out"] / "c3" / "macros_c3.tex").read_text()
    assert any("kept 9 of 12 planned problems" in w for w in s["warnings"])


def _artifacts(paths):
    out = paths["out"]
    files = {"macros": out / "c3" / "macros_c3.tex", "table": out / "t12" / "t12_matched_speed.tex",
             "cost": out / "t12" / "t12_sanity_cost.tex"}
    files |= {f"zip:{f.name}": f for f in sorted((out / "t12" / "zip").glob("*.csv"))}
    return {k: f.read_text() for k, f in files.items()}


def test_dropped_records_change_no_number(tmp_path):
    """The same run without the three incomplete problems at all gives identical accuracy, speed,
    matching, bootstrap, macros, tables and CSVs."""
    a = partial_fixture(tmp_path / "partial")
    b = make_fixture(tmp_path / "clean", nshards=R_SHARDS, n_docs=KEPT_N)
    assert run_main(a, "--allow-partial") == 0
    assert run_main(b) == 0
    sa, sb = summary_of(a), summary_of(b)
    for task in TASKS:
        ta, tb = sa["tasks"][task], sb["tasks"][task]
        for t in (ta, tb):                                             # only the planned N differs (12 vs 9)
            for d in t["decoders"].values():
                for sp in d["speeds"].values():
                    sp.pop("n_planned")
        assert ta["decoders"] == tb["decoders"]                        # acc, ci, tps, nfe, seconds, counts
        assert ta["matched"] == tb["matched"]                          # diff, lo, hi (the bootstrap), n_paired
        assert ta["outside_share"] == tb["outside_share"]
        assert ta["alpha"] == tb["alpha"]
        assert tb["n_dropped_problems"] == 0 and ta["n_dropped_problems"] == 3
        assert {m["n_paired"] for per in ta["matched"].values() for m in per.values() if m["matched"]} == {KEPT_N}
        assert any(m["lo"] is not None for per in ta["matched"].values() for m in per.values() if m["matched"])
    art_a, art_b = _artifacts(a), _artifacts(b)
    macros_a = parse_macros(art_a.pop("macros"))
    macros_b = parse_macros(art_b.pop("macros"))
    assert macros_a == macros_b
    assert art_a == art_b


def test_an_error_record_counts_as_finished(tmp_path):
    """A driver error is a finished record (scored wrong): its problem is kept, no flag needed."""
    paths = make_fixture(tmp_path, nshards=R_SHARDS, n_docs=R_DOCS)
    key = config_key(cfg_of("CThreeA", 2, "gsm8k"))

    def to_error(rows):
        for r in rows:
            if r["doc_id"] == R_DOCS - 1 and r["config"] == key:
                r.update({"completion": None, "error": "OutOfMemoryError: out of memory", "n_forward": None,
                          "tokens_per_step": None, "seconds": None})
                r.pop("c3a_steps", None)
        return rows

    for f in (paths["records"] / "sweep-gsm8k").glob("t12.shard*.jsonl"):
        edit_rows(f, to_error)
    assert run_main(paths) == 0                                        # complete: no --allow-partial
    g = summary_of(paths)["tasks"]["gsm8k"]
    assert (g["n"], g["n_complete"], g["n_dropped_problems"], g["n_dropped_records"]) == (R_DOCS, R_DOCS, 0, 0)
    sp = g["decoders"]["CThreeA"]["speeds"]["2"]
    assert (sp["n"], sp["failures"], sp["n_ok"]) == (R_DOCS, 1, R_DOCS - 1)


def test_a_missing_record_is_not_an_error_record(tmp_path):
    paths = make_fixture(tmp_path, nshards=R_SHARDS, n_docs=R_DOCS)
    key = config_key(cfg_of("CThreeA", 2, "gsm8k"))
    for f in (paths["records"] / "sweep-gsm8k").glob("t12.shard*.jsonl"):
        edit_rows(f, lambda rows: [r for r in rows if not (r["doc_id"] == R_DOCS - 1 and r["config"] == key)])
    assert run_main(paths, "--allow-partial") == 0
    g = summary_of(paths)["tasks"]["gsm8k"]
    assert (g["n"], g["n_dropped_problems"], g["n_dropped_records"]) == (R_DOCS - 1, 1, 20)
    assert summary_of(paths)["tasks"]["mbpp"]["n"] == R_DOCS


def test_a_gap_that_is_not_a_prefix_is_a_loud_warning_not_a_failure(tmp_path, capsys):
    paths = make_fixture(tmp_path, nshards=R_SHARDS, n_docs=R_DOCS)
    key = config_key(cfg_of("CThreeA", 2, "gsm8k"))
    f = paths["records"] / "sweep-gsm8k" / "t12.shard0of3.jsonl"              # shard 0: docs 0, 3, 6, 9
    edit_rows(f, lambda rows: [r for r in rows if not (r["doc_id"] == 3 and r["config"] == key)])
    assert run_main(paths, "--allow-partial") == 0                     # warns, does not fail
    s = summary_of(paths)
    g = s["tasks"]["gsm8k"]
    assert g["n"] == R_DOCS - 1 and g["n_dropped_problems"] == 1
    pc = g["prefix_check"]
    assert pc["ok"] is False and len(pc["violations"]) == 1
    v = pc["violations"][0]
    assert v["file"].endswith("t12.shard0of3.jsonl") and v["first_dropped"] == 3 and v["kept_after_gap"] == [6, 9]
    assert any("NOT A PREFIX" in w for w in s["warnings"])
    assert "NOT A PREFIX" in capsys.readouterr().err
    assert "% WARNING: gsm8k: kept problems are NOT a prefix" in (paths["out"] / "c3" / "macros_c3.tex").read_text()
    assert s["tasks"]["mbpp"]["prefix_check"]["ok"] is True


def test_a_grid_configuration_with_no_records_is_refused_or_gives_n_zero(tmp_path):
    paths = make_fixture(tmp_path, nshards=R_SHARDS, n_docs=R_DOCS)
    key = config_key(cfg_of("Klass", 3, "gsm8k"))
    for f in (paths["records"] / "sweep-gsm8k").glob("t12.shard*.jsonl"):
        edit_rows(f, lambda rows: [r for r in rows if r["config"] != key])
    with pytest.raises(SystemExit) as e:
        run_main(paths)
    assert "no records" in str(e.value)
    assert run_main(paths, "--allow-partial") == 0
    g = summary_of(paths)["tasks"]["gsm8k"]
    assert (g["n"], g["n_complete"], g["n_planned"], g["n_dropped_problems"]) == (0, 0, R_DOCS, R_DOCS)
    assert "$N=0$" in (paths["out"] / "t12" / "t12_matched_speed.tex").read_text()


def test_score_rows_are_not_used_for_a_task_that_lost_problems(tmp_path):
    """A score_t12 aggregate covers the dropped problems too, so with drops the accuracy is blank
    (never an aggregate), even when its n happens to equal the kept number."""
    paths = partial_fixture(tmp_path)
    drop_verdicts(paths, C3A_KEY)
    sc = tmp_path / "scores.json"
    run = P.run_key(meta_of("gsm8k", 0))
    sc.write_text(json.dumps({"rows": [{"task": "gsm8k", "config": C3A_KEY, "half": "report", "prompt_format": "chat",
                                        "run": run, "n": KEPT_N, "n_correct": 7, "accuracy": 7 / KEPT_N,
                                        "ci95": [0.4, 0.9]}]}))
    assert run_main(paths, "--scores", str(sc), "--allow-partial") == 0
    assert summary_of(paths)["tasks"]["gsm8k"]["decoders"]["CThreeA"]["speeds"]["2"].get("acc") is None
