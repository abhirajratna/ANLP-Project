#!/usr/bin/env python3
"""Write prereg/frozen_config.json: every analysis choice fixed before the eval half is read.

    python scripts/freeze_prereg.py --out prereg/frozen_config.json

Values are read from the code that applies them (c3 constants, the G1/G2 gates,
the trace design, the declaration rule), not retyped, so the frozen file cannot
drift from what runs.  Run it on a clean tree right before tagging: the git
commit it records is the one the pre-registration points at, and a dirty tree
is recorded as such.  Items the plan leaves open are listed under ``open``
rather than decided here.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import inspect
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]


def c3_constants() -> dict:
    import c3
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(c3).items()
            if k.isupper() and not k.startswith("_")}


def g2_thresholds() -> dict:
    """check_g2.py's argparse defaults, read from its source."""
    src = (ROOT / "scripts" / "check_g2.py").read_text()

    def default(flag):
        m = re.search(rf'"--{flag}"[^)]*default=([^,)]+)', src)
        if not m:
            raise ValueError(f"check_g2.py has no default for --{flag}")
        return json.loads(m.group(1).replace("'", '"'))

    return {"metric": default("metric"), "tail_fraction": default("tail"),
            "max_relative_change": default("max-rel-change"),
            "rule": "the metric is flat over the last tail_fraction of training",
            "source": "scripts/check_g2.py"}


def _sha(path: str) -> str | None:
    from c3.provenance import sha256_file
    p = ROOT / path
    return sha256_file(p) if p.exists() else None


def primary_arms() -> list[str]:
    """The E1/E2 Holm-family arms, read from prereg/analysis_commands.sh so they cannot drift."""
    src = (ROOT / "prereg" / "analysis_commands.sh").read_text()
    m = re.search(r"^PRIMARY_ARMS=\(([^)]*)\)", src, re.M)
    if not m:
        raise ValueError("prereg/analysis_commands.sh defines no PRIMARY_ARMS")
    return m.group(1).split()


def floors() -> dict:
    out = {}
    for fam, path in (("llada", "artifacts/t2_pilot/floor_llada_base_fp16.json"),
                      ("dream", "artifacts/t2_pilot/floor_dream-7b-base_fp16.json")):
        p = ROOT / path
        out[fam] = ({"floor_delta": json.loads(p.read_text())["plan_v2"]["floor_delta"],
                     "source": path, "sha256": _sha(path)} if p.exists() else
                    {"floor_delta": "PENDING: slurm/submit_c1.sh floor dream-7b-base, then "
                                    "scripts/measure_floor.py, before any Dream arm is analysed",
                     "source": path})
    out["rule"] = ("one floor per model family, measured on the Base checkpoint (20 reserve "
                   "documents x 5 repeats, t2-pilot design); the Instruct arms use their family's")
    return out


def e3_config() -> dict:
    from c3.endpoints import e3

    sig = inspect.signature(e3.e3_compare).parameters
    return {"model": "a ~ alpha_rate_bin + f(ell) + type + f(ell) x type, f a restricted cubic "
                     "spline (knots from the fit half), pool weights within each arm",
            "rate_bin_centres": e3.RATE_CENTRES.tolist(),
            "rate_bin_edges": e3.RATE_EDGES.tolist(),
            "references": {"primary": "uniform_matched (same |M| as each decoder state)",
                           "secondary": "uniform grid, reweighted to the decoder arm's rate-bin mix"},
            "endpoint": "type effect E_dec[pi_ref(l) - pi_dec(l)] (> 0: decoder states more "
                        "overconfident at equal confidence)",
            "joint_test": "sup-t test that the difference curve is zero (c3.endpoints.e3.sup_t)",
            "difference_curve": {"grid": "decoder-ell quantiles 2.5%..97.5%",
                                 "grid_size": sig["grid_size"].default,
                                 "bands": "pointwise percentile and simultaneous sup-t"},
            "n_ell_bins": 100, "reliability_bins": 15,
            "by_rate": "exploratory", "implementation": "scripts/analyze_e3.py"}


def t8_config() -> dict:
    from c3 import controls

    return {"rows": "E1 Delta(32) - Delta(1) and E2 field(32) - field(1), without and with each "
                    "control, and their paired difference, on one resample matrix",
            "position": {"covariates": list(controls.POSITION_COVARIATES),
                         "pc_sampler": "exp(-0.25 i) (position-only: no unigram table)",
                         "with": "Delta - E_m[pi(l) - pi(l, x)]; field with x in both calibrators",
                         "strata": {"distance_lower_edges": list(controls.DIST_STRATA),
                                    "position": "terciles of position / doc_len"}},
            "token_type": {"classes": list(controls.TOKEN_CLASSES), "of": "the slot's true token",
                           "stop_words": len(controls.STOP_WORDS),
                           "rows": "covariates, per-class split, class-standardised contrast"},
            "temperature": {"primary": "logodds: c = sigma(ell / T_bin), fitted by the NLL of "
                                       "correctness (exact from the trace)",
                            "sensitivity": "softmax: logits / T rebuilt from the top-20 and an "
                                           "even tail, fitted by true-token NLL",
                            "bins": controls.TEMPERATURE_EDGES.tolist(),
                            "pools": "re-ranked on the tempered confidence"},
            "precision": {"documents": "artifacts/t8_precision_docs.json",
                          "documents_sha256": _sha("artifacts/t8_precision_docs.json"),
                          "comparison": "LLaDA-8B-Base fp16 vs 4-bit NF4, same documents; E1, "
                                        "E2, beta_z (paired resamples) and the paired logit shift"},
            "implementation": "scripts/run_controls.py, c3/controls.py"}


def frozen() -> dict:
    from c3 import stats
    from c3.calibrators import ContextLogistic
    from c3.frame import FIELD_FEATURES
    from c3.gates import FALLBACKS, G1_CORPUS_DOCS, G1_MAX_DOCS
    from c3.metrics import ece_equal_mass
    from c3.provenance import collect as provenance
    from c3.traces.collect import PRESETS, SCHEMA_VERSION, CollectConfig

    consts = c3_constants()
    ece_bins = inspect.signature(ece_equal_mass).parameters["n_bins"].default
    return {
        "constants": consts,
        "knots": {"primary": consts["DEFAULT_KNOTS"], "sensitivity": [5, 13]},
        "declaration_rule": {
            "text": "an effect is declared only if its 95% document-bootstrap percentile "
                    "interval excludes 0 AND |effect| > 2 x the non-determinism floor",
            "implementation": "c3.stats.declare",
            "source": inspect.getsource(stats.declare)},
        "bootstrap": {"unit": "document", "resamples": consts["N_BOOT"],
                      "interval": "percentile, 95%"},
        "split": {"manifest": "artifacts/split_manifest.json", "fit_half": 0, "report_half": 1,
                  "rule": "everything fitted (calibrators, splines, temperatures, C3) uses the "
                          "fit half; every reported number comes from the report half",
                  "manifests": {p: _sha(p) for p in (
                      "artifacts/split_manifest.json", "artifacts/split_manifest_gsm8k.json",
                      "artifacts/split_manifest_mbpp.json",
                      "artifacts/split_manifest_timesplit.json")},
                  "tasks": "scripts/make_task_split.py: the same document_split and seed",
                  "time_split": "scripts/make_split_manifest.py --any-ids on timesplit_docs.jsonl "
                                "(same seed; doc_ids 200000-200499)"},
        "floors": floors(),
        "gates": {
            "G1": {"max_docs_full_plan": G1_MAX_DOCS, "corpus_docs": G1_CORPUS_DOCS,
                   "outcomes": FALLBACKS, "n_formula": "7.85 sigma_d^2 / delta_min^2",
                   "delta_min": consts["DELTA_MIN"],
                   "taken_on": "max N over mask arms and the two primary contrasts",
                   "implementation": "c3.gates"},
            "G2": g2_thresholds()},
        "primary_endpoints": {
            "E1": "Delta(32) - Delta(1), Delta(m) = E_m[conf - agree] over pool winners, "
                  "document-first",
            "E2": "field(32) - field(1), field(m) = E_m[pi_hat(l) - pi_hat(l, s)]"},
        "holm_families": {
            "E1_contrast": "Delta(32) - Delta(1) across models x datasets",
            "E2_field_contrast": "field(32) - field(1) across models x datasets",
            "Eq3_rank_profile": "gamma_2..gamma_4 within each arm",
            "primary_arms": primary_arms(),
            "everything_else": "exploratory, labelled as such",
            "implementation": "scripts/summarize_c1.py (c3.stats.holm on bootstrap p-values)"},
        "e3": e3_config(),
        "t8": t8_config(),
        "analysis_commands": {"path": "prereg/analysis_commands.sh",
                              "sha256": _sha("prereg/analysis_commands.sh")},
        "deviations": {"path": "prereg/deviations.md", "sha256": _sha("prereg/deviations.md")},
        "calibrators": {
            "own": "isotonic in ell, pool-weighted",
            "context": {"features": list(FIELD_FEATURES),
                        "interactions_default": ContextLogistic().interactions,
                        "ridge": ContextLogistic().ridge},
            "ece": {"bins": ece_bins, "binning": "equal mass", "also": "smooth ECE"}},
        "trace_design": {"schema_version": SCHEMA_VERSION,
                         "main": CollectConfig().to_dict(),
                         "t2_pilot": {"model": PRESETS["t2-pilot"]["model"],
                                      **CollectConfig(**PRESETS["t2-pilot"]["config"]).to_dict()}},
        "contamination": {"threshold": consts["CONTAM_THRESHOLD"],
                          "rule": "flag a document whose greedy 50-token continuation of its "
                                  "50-token prefix matches the true text for a run of at least "
                                  "the threshold; report every FineWeb endpoint with and "
                                  "without flagged documents",
                          "screens": {m: _sha(f"artifacts/contamination_{m}.json")
                                      for m in ("llada-8b-base", "dream-7b-base")},
                          "deviation": "the threshold was fixed in code after the 29 Sep "
                                       "screens were seen (prereg/deviations.md)"},
        "open": [
            "CONFIRM the E1/E2 Holm-family arms: " + ", ".join(primary_arms()) + " (decoder "
            "k = 2, 8 and uniform_matched exploratory)",
            "CONFIRM the interacted context calibrator (code default) over plan v2 T6(2)'s "
            "additive form; the additive fit is the sensitivity check",
            "CONFIRM the GSM8K/MBPP split: the FineWeb rule applied to the task test sets",
            "CONFIRM log-odds temperature as the T8 primary (softmax with the even-tail "
            "approximation is a sensitivity row; prereg/deviations.md)",
            "CONFIRM the time-split set (artifacts/timesplit_manifest.json on branch "
            "t3-110m-probe, sha256 3374de4d...)"],
        "provenance": provenance(),
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = frozen()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(cfg, indent=2))
    prov = cfg["provenance"]
    print(f"[prereg] -> {a.out} (commit {prov['git_commit'][:12]}"
          f"{', DIRTY TREE: regenerate on a clean commit before tagging' if prov['git_dirty'] else ''})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
