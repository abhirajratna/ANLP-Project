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
                          "fit half; every reported number comes from the report half"},
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
            "everything_else": "exploratory, labelled as such"},
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
        "contamination": {"threshold": "PENDING: to be fixed before the screen runs "
                                       "(plan suggests 32 of 50 tokens)",
                          "report": "every endpoint with and without flagged documents"},
        "open": [
            "which mask arms are members of the E1/E2 Holm families (uniform, and which of "
            "the decoder k = 2, 4, 8 arms)",
            "additive (plan v2 T6(2)) or interacted (code default) context calibrator",
            "the fit/report split for GSM8K/MBPP documents (the manifest covers FineWeb)",
            "the contamination threshold",
            "trace storage: top-20 fp32 probabilities exceed /share1 "
            "(artifacts/trace_size_estimate.json)"],
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
