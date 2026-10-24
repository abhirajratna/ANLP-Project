# Source of notebooks/c3_dorigo_methods.ipynb (percent format).  Edit this file, then
# rebuild:  python notebooks/build_notebook.py notebooks/c3_dorigo_methods.py

# %% [markdown]
# # C3 × Dorigo (2026): winner's-curse methods on stored traces
#
# ## Phase 1 verdict: **PARTIAL now, GO expected on the v2 traces** (written 24 Oct 2026, before Phase 2)
#
# **Basis.** The repo, without the full traces. The full traces are on Ada (`/share1/ojas.k/traces/main-110m-v3/`, `/share1/abhiraj.ratna/traces/`) and not yet on Drive. Cells 1.1–1.6 redo every check below on the files you mount. If cell 1.6 prints a different verdict, cell 1.6 wins, and this cell must be edited to match.
#
# | check | what was found | source |
# |---|---|---|
# | format | parquet: one file, or a folder of `part-*.parquet` (+ `*.meta.json`) | `c3/traces/collect.py`, `scripts/merge_traces.py` |
# | schema v2 columns | `model, variant, revision, precision, dataset, git_commit, seed, mask_kind, policy, k, target_rate, mask_rate, step, n_masked, doc_id, state_id, state_idx, doc_len, span_start, span_len, position, top1, conf, ell, p_true, entropy, margin, topk_ids, topk_p (top-20), delta_ell, agree, committed, commit_rank` | `c3/traces/collect.py` `TRACE_COLUMNS` |
# | fit/report flag | **not a column**. It comes from `artifacts/split_manifest.json` (`doc_id → 0 fit / 1 report`). That file equals `document_split(2000, seed=20260917)`; checked 24 Oct, as is the time-split one (`500`, offset 200000) | `c3/data/fineweb.py` |
# | pilot sample on hand | `artifacts/pilot/llada-8b-4bit-n20.parquet`: 21,710 rows, 20 docs, 160 states, kinds `uniform` and `decoder`, **old schema**: no `margin`, `topk_*`, `p_true`, `entropy`, `delta_ell`, `commit_rank`, and no `uniform_matched` | read 24 Oct |
# | size | 110M main: 119.6 M rows, 1,979 docs (≈60k rows/doc, ≈12× the 1e5-per-20-docs guess) ≈ 20 GB with top-20 lists. Per kind ≈ 57 M decoder, ≈ 57 M matched, ≈ 4.6 M uniform | `artifacts/traces_110m/*.json`, `artifacts/trace_size_estimate.json` |
# | memory plan | load one dataset × mask kind at a time, 10 columns, downcast (≈41 B/row) → ≈2.4 GB for the largest kind, plus ≈1.5 GB for the sort. Top-20 lists are read in batches only if `margin` is missing. Fits 12 GB | this notebook, cell 1.5 |
# | \|M\| ≥ 32 | pilot: 160/160 states. 110M by design: uniform rate 0.1 gives 26 masked (excluded), so ≈89% of uniform states qualify. Decoder/matched: the last steps fall below 32, so ≈87% qualify. **Estimate; counted exactly in 1.4** | trace config |
# | ties | the pilot's decoder step-0 state (all 256 positions masked) gives every position the same ℓ, so random tie-breaking is needed and is handled exactly | read 24 Oct |
#
# **Modules.** On the pilot sample: **A, B, C, D, E** can run; **F cannot** (no `margin`/`topk_p`). On the pilot, D runs only for small m: with 10 report documents spread over 10 rate bins, shuffled pools exist up to m = 4 (uniform) and m = 8 (decoder), never at m = 32; checked by running the notebook on the pilot, 24 Oct. On v2 traces, every module, A–F, can run.
#
# **Runtime estimate (CPU; the T4 GPU is not used).** Module A ≈ 1–2 min; self-tests ≈ 1 min. Per dataset × kind: load ≈ 1–3 min from local disk (Drive reads can be several times slower), pools ≈ 1–3 min, 10k bootstrap < 1 min. 110M, three kinds: **≈ 15–30 min**; each 8B model about the same. Cell 1.5 prints a measured projection. If it exceeds 20 min, the run falls back to 2,000 resamples and asks first (see `CONFIRM_FULL_BOOT`).
#
# **Methods transferred from Dorigo (arXiv:2609.03581), not his formulas.** These are: the maximum is biased; the exact local + remote split; the shuffle test; the paired (control-variate) contrast. His closed forms (D/(2Q), the trials factor) are for Gaussian matched filters. They appear only in the Module A toy, where they apply, and nothing here claims they hold for language models.

# %% [markdown]
# ## 0.1 Config

# %%
USE_SYNTHETIC = True          # True: run everything on Module B synthetic traces (no data needed)

# One entry per dataset: a parquet file or a folder of parquet parts, how its documents split
# into fit/report, and (optionally) which `model` value to keep when a file holds several.
# The released traces live in the private HF dataset ANLP-project/c3-traces (slurm/hf_upload.sh):
# main-110m-v3/ (1,979 calibration docs, split = document_split(2000)), timesplit-110m-v3/
# (doc_ids 200000-200499, split = document_split(500) offset 200000); its README lists the rest.
# "hf" entries are downloaded in 1.1 with the token in Colab secrets (HF_TOKEN); the org must have
# given you read access.  The rules below equal artifacts/split_manifest*.json (checked 24 Oct).
HF_REPO = "ANLP-project/c3-traces"
HF_LOCAL = "/content/c3_traces" if __import__("os").path.isdir("/content") else "c3_traces"
DATASETS_REAL = {
    "local-110m-v3": {"hf": "main-110m-v3",
                      "split": {"rule": {"n_docs": 2000, "seed": 20260917, "doc_offset": 0}}},
    "local-110m-v3-timesplit": {"hf": "timesplit-110m-v3",
                                "split": {"rule": {"n_docs": 500, "seed": 20260917, "doc_offset": 200000}}},
}
DATASETS = {**DATASETS_REAL,
    # Other sources: a local / Drive folder or file instead of "hf", e.g.
    # "local-110m-v3": {"path": "/content/drive/MyDrive/c3_traces/main-110m-v3",
    #                   "split": {"manifest": "/content/drive/MyDrive/c3_traces/split_manifest.json"}},
    # "llada-8b-base": {"path": "/content/drive/MyDrive/c3_traces/llada-8b-base",
    #                   "split": {"manifest": "/content/drive/MyDrive/c3_traces/split_manifest.json"},
    #                   "model": None},
    # "pilot-llada-8b-4bit": {"path": "artifacts/pilot/llada-8b-4bit-n20.parquet",
    #                         "split": {"rule": {"n_docs": 2000, "seed": 20260917, "doc_offset": 0}}},
}
MOUNT_DRIVE = True            # Colab only; ignored elsewhere
MASK_KINDS = ["uniform", "decoder", "uniform_matched"]   # "matched" is stored as uniform_matched
M_GRID = [1, 2, 4, 8, 16, 32]
POOLS_PER_STATE = 20
N_BOOT = 10_000
DRY_RUN_BOOT = 2_000          # used instead of N_BOOT when the projected runtime exceeds 20 min
CONFIRM_FULL_BOOT = False     # set True (after reading the warning) to run N_BOOT anyway
RUNTIME_LIMIT_MIN = 20
SEED = 20260917
MIN_STATE_SIZE = max(M_GRID)  # states with |M| below this are excluded from every m (logged)
RATE_BINS = 10                # mask-rate deciles for matching shuffled pools
N_QUINTILES = 5               # Module F
ECE_BINS = 15
TOY_N = 50_000                # Module A toys per Q
SELFTEST_BOOT = 2_000         # bootstrap resamples inside the Module B self-tests
OUT_DIR = "/content/c3_dorigo_out" if __import__("os").path.isdir("/content") else "c3_dorigo_out"
NOTEBOOK_VERSION = "2026-10-24.1"

# %% [markdown]
# ## 0.2 Setup: imports, versions, seeds, exclusion log

# %%
import os, sys, json, time, math, zlib, platform, warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import scipy
from scipy.special import gammaln, expit
import sklearn
from sklearn.isotonic import IsotonicRegression
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

os.makedirs(OUT_DIR, exist_ok=True)
FIG_DIR = os.path.join(OUT_DIR, "figures"); os.makedirs(FIG_DIR, exist_ok=True)
VERSIONS = {"python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pa.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "matplotlib": matplotlib.__version__, "notebook": NOTEBOOK_VERSION}
print(json.dumps(VERSIONS, indent=1))
pd.set_option("display.width", 160, "display.max_columns", 40)


def rng_for(*keys) -> np.random.Generator:
    """A generator fixed by SEED and the given keys, so every stream is reproducible on its own.

    >>> rng_for("a", 1).random() == rng_for("a", 1).random()
    """
    return np.random.default_rng([SEED] + [zlib.crc32(str(k).encode()) for k in keys])


EXCLUSIONS = []


def log_exclusion(dataset, kind, stage, unit, count, reason):
    """Record (and print) an exclusion; nothing is dropped without a line here."""
    count = int(count)
    EXCLUSIONS.append({"dataset": dataset, "kind": kind, "stage": stage, "unit": unit,
                       "count": count, "reason": reason})
    if count:
        print(f"[excluded] {dataset} / {kind} / {stage}: {count} {unit} ({reason})")


assert rng_for("x", 1).random() == rng_for("x", 1).random() != rng_for("x", 2).random()

# %% [markdown]
# ## 0.3 Library
# All functions used later are defined here. Each has a docstring, and the self-tests in Module B
# exercise each one on synthetic data.

# %%
# ---------------------------------------------------------------- schema / field map
CANON = {  # canonical name -> candidate column names, in order of preference
    "doc": ["doc_id", "document_id"], "state": ["state_id"], "model": ["model"],
    "kind": ["mask_kind"], "rate": ["mask_rate", "target_rate"], "n_masked": ["n_masked"],
    "doc_len": ["doc_len"], "step": ["step"], "position": ["position"], "top1": ["top1"],
    "conf": ["conf", "confidence"], "ell": ["ell", "log_odds", "logodds"],
    "agree": ["agree", "correct"], "p_true": ["p_true"], "entropy": ["entropy"],
    "margin": ["margin"], "topk_ids": ["topk_ids"], "topk_p": ["topk_p"],
    "delta_ell": ["delta_ell"], "commit_rank": ["commit_rank"], "half": ["half", "split"],
}
FIELD_LABEL = {
    "doc": "document id", "state": "state id", "model": "model", "kind": "mask kind",
    "rate": "mask rate", "n_masked": "number masked", "doc_len": "document length",
    "step": "step", "position": "position", "top1": "top-1 token id", "conf": "confidence",
    "ell": "log-odds", "agree": "correctness", "p_true": "p(true token)", "entropy": "entropy",
    "margin": "margin (top1 - top2)", "topk_ids": "top-20 ids", "topk_p": "top-20 probs",
    "delta_ell": "delta-ell since previous step", "commit_rank": "commit rank",
    "half": "fit/report flag",
}
USED_BY = {"doc": "C D E F ICC", "state": "C D E F ICC", "kind": "C D E F", "rate": "D",
           "conf": "C D E F", "ell": "C D E F ICC", "agree": "C D E F ICC", "margin": "F",
           "topk_p": "F (if no margin)", "half": "all (or split manifest)",
           "position": "duplicate check", "n_masked": "consistency check / rate",
           "doc_len": "rate (if no mask_rate)"}
FORMAT_OF = {".parquet": "parquet", ".pq": "parquet", ".arrow": "arrow", ".feather": "arrow",
             ".ipc": "arrow", ".jsonl": "jsonl", ".json": "json", ".npz": "npz"}


def trace_files(path) -> dict:
    """Files under a trace path, grouped by format (only parquet is read by the pipeline)."""
    p = Path(path)
    files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else []
    out = defaultdict(list)
    for f in files:
        out[FORMAT_OF.get(f.suffix.lower(), "other")].append(f)
    return dict(out)


def field_map(columns) -> dict:
    """Canonical field -> actual column name, or None when no candidate is present."""
    cols = set(columns)
    return {k: next((c for c in cands if c in cols), None) for k, cands in CANON.items()}


def module_feasibility(fmap: dict, has_split: bool) -> dict:
    """Which modules can run with these columns, and what is missing for the others."""
    def need(*fields):
        miss = [f for f in fields if not fmap.get(f)]
        return miss
    base = need("doc", "state", "kind", "agree") + ([] if (fmap.get("ell") or fmap.get("conf"))
                                                    else ["ell or conf"])
    split = [] if (has_split or fmap.get("half")) else ["fit/report split (column or manifest)"]
    rate = [] if fmap.get("rate") or (fmap.get("n_masked") and fmap.get("doc_len")) \
        else ["mask_rate (or n_masked + doc_len)"]
    marg = [] if fmap.get("margin") or fmap.get("topk_p") else ["margin (or topk_p)"]
    req = {"A": [], "B": [], "C": base + split, "D": base + split + rate, "E": base + split,
           "F": base + split + rate + marg, "ICC": base + split}
    return {m: (not miss, miss) for m, miss in req.items()}


# ---------------------------------------------------------------- split
def document_split_rule(n_docs: int, seed: int = 20260917, frac: float = 0.5):
    """c3.data.fineweb.document_split, copied: (fit indices, report indices)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_docs)
    k = int(round(frac * n_docs))
    return np.sort(perm[:k]), np.sort(perm[k:])


def make_split(spec: dict):
    """A function doc_ids -> half (0 fit, 1 report, -1 unknown) from a manifest or the rule."""
    if "manifest" in spec:
        with open(spec["manifest"]) as fh:
            man = {int(k): int(v) for k, v in json.load(fh).items()}
    else:
        r = spec["rule"]
        fit, rep = document_split_rule(r["n_docs"], r.get("seed", 20260917))
        off = r.get("doc_offset", 0)
        man = {**{int(i) + off: 0 for i in fit}, **{int(i) + off: 1 for i in rep}}
    keys = np.array(sorted(man), dtype=np.int64)
    vals = np.array([man[k] for k in keys], dtype=np.int8)

    def half_of(doc_ids):
        doc_ids = np.asarray(doc_ids, dtype=np.int64)
        i = np.clip(np.searchsorted(keys, doc_ids), 0, len(keys) - 1)
        return np.where(keys[i] == doc_ids, vals[i], -1).astype(np.int8)
    half_of.n = len(man)
    return half_of


# ---------------------------------------------------------------- Module B generator
SCENARIOS = {
    "calibrated": "perfectly calibrated, independent positions",
    "flat_miscal": "overconfident by a constant 0.08 in every context",
    "context": "miscalibration grows with a context variable that also raises confidence",
    "shared_err": "per-state random effect: confidence moves with it, correctness does not",
    "shared_indep": "per-state random effect on correctness only, independent of confidence",
}


def make_synthetic(scenario: str, n_docs=200, n_states=5, n_pos=100, seed=0,
                   round_ell: float | None = None) -> pd.DataFrame:
    """Synthetic trace rows (schema-v2 names) for one Module B scenario, with a known answer.

    calibrated   a ~ Bern(sigmoid(l)), l iid               -> Delta(m) = 0, gap = 0, ICC = 0
    flat_miscal  a ~ Bern(max(c - 0.08, 0))                -> Delta(m) = 0.08 for every m
    context      l = 0.5 + 1.2 z + e, logit p = l - 0.8 z  -> Delta(m) grows with m, gap = 0
    shared_err   l = 0.5 + u_s + e, logit p = l - u_s      -> gap != 0, ICC > 0
    shared_indep l = 0.5 + e, logit p = l + u_s            -> ICC > 0 but gap = 0 (the shuffle
                 test only sees dependence that reaches the winner's selection)
    ``round_ell`` rounds l (creates ties) for the tie-break tests.
    """
    rng = rng_for("synthetic", scenario, seed)
    S = n_docs * n_states
    doc = np.repeat(np.arange(n_docs), n_states * n_pos)
    state_idx = np.tile(np.repeat(np.arange(n_states), n_pos), n_docs)
    state_id = doc * 1000 + state_idx
    e = rng.normal(size=S * n_pos)
    u = np.repeat(rng.normal(size=S), n_pos)
    if scenario == "calibrated":
        ell = 0.5 + 1.5 * e
        logit_p = ell
    elif scenario == "flat_miscal":
        ell = 1.0 + 1.2 * e
        logit_p = None
    elif scenario == "context":
        z = rng.normal(size=S * n_pos)
        ell = 0.5 + 1.2 * z + e
        logit_p = ell - 0.8 * z
    elif scenario == "shared_err":
        ell = 0.5 + u + 1.2 * e
        logit_p = ell - u
    elif scenario == "shared_indep":
        ell = 0.5 + 1.5 * e
        logit_p = ell + u
    else:
        raise KeyError(scenario)
    if round_ell:
        ell = np.round(ell / round_ell) * round_ell
    conf = expit(ell)
    p = np.clip(conf - 0.08, 0, 1) if scenario == "flat_miscal" else expit(logit_p)
    agree = (rng.random(p.size) < p).astype(np.int64)
    rate = np.repeat(rng.uniform(0.1, 0.9, size=S), n_pos)
    second = (1 - conf) * rng.uniform(0.2, 1.0, size=conf.size)
    return pd.DataFrame({
        "model": f"synthetic-{scenario}", "mask_kind": "uniform", "mask_rate": rate,
        "n_masked": n_pos, "doc_len": int(n_pos / 0.5), "doc_id": doc, "state_id": state_id,
        "position": np.tile(np.arange(n_pos), S), "conf": conf, "ell": ell,
        "agree": agree, "margin": (conf - second).astype(np.float32)})


# ---------------------------------------------------------------- loading
@dataclass
class Traces:
    """One dataset x mask kind, rows sorted by (doc, state, ell descending).

    Per row: ell, conf, agree, margin, row_state, blk_start/blk_len (the run of rows with the
    same state and ell: a tie block).  Per state s: st_start, st_n, st_doc, st_rate, st_mean_ell.
    """
    dataset: str
    kind: str
    ell: np.ndarray
    conf: np.ndarray
    agree: np.ndarray
    margin: np.ndarray
    row_state: np.ndarray
    blk_start: np.ndarray
    blk_len: np.ndarray
    st_start: np.ndarray
    st_n: np.ndarray
    st_doc: np.ndarray
    st_rate: np.ndarray
    st_mean_ell: np.ndarray

    @property
    def n_states(self):
        return self.st_n.size


def _margin_from_topk(col) -> np.ndarray:
    """top1 - top2 from a (fixed-size) list column of top-k probabilities."""
    lens = pc.list_value_length(col).to_numpy(zero_copy_only=False)
    k = int(lens.max()) if lens.size else 0
    if k < 2 or (lens != k).any():
        return np.full(len(col), np.nan, np.float32)
    flat = pc.list_flatten(col).to_numpy(zero_copy_only=False).astype(np.float32)
    arr = -np.sort(-flat.reshape(-1, k), axis=1)
    return (arr[:, 0] - arr[:, 1]).astype(np.float32)


def read_kind(files, fmap: dict, kind: str, model: str | None = None,
              need_margin: bool = True, batch_rows: int = 2_000_000) -> dict:
    """The rows of one mask kind as pruned, downcast numpy columns (batched; no list columns
    kept).  Keys: doc, state, ell, conf, agree, margin, rate, n_masked, position (when present)."""
    want = {"doc": np.int64, "state": np.int64, "ell": np.float64, "conf": np.float32,
            "agree": np.int8, "margin": np.float32, "rate": np.float32,
            "n_masked": np.int32, "doc_len": np.int32, "position": np.int32}
    cols = {k: fmap[k] for k in want if fmap.get(k)}
    use_topk = need_margin and not fmap.get("margin") and fmap.get("topk_p")
    read_cols = list(cols.values()) + ([fmap["topk_p"]] if use_topk else [])
    dset = pads.dataset([str(f) for f in files], format="parquet")
    filt = pc.field(fmap["kind"]) == kind
    if model is not None and fmap.get("model"):
        filt = filt & (pc.field(fmap["model"]) == model)
    parts = defaultdict(list)
    for b in dset.to_batches(columns=read_cols, filter=filt, batch_size=batch_rows):
        for k, c in cols.items():
            parts[k].append(b.column(c).to_numpy(zero_copy_only=False).astype(want[k]))
        if use_topk:
            parts["margin"].append(_margin_from_topk(b.column(fmap["topk_p"])))
    out = {}
    for k in list(parts):  # one column at a time, so the batches and the result never all coexist
        v = parts.pop(k)
        out[k] = np.concatenate(v) if v else np.array([], want.get(k, np.float32))
        del v
    n = len(next(iter(out.values()))) if out else 0
    if "ell" not in out and "conf" in out:
        c = np.clip(out["conf"].astype(np.float64), 1e-6, 1 - 1e-6)
        out["ell"] = np.log(c) - np.log1p(-c)
        print(f"[derived] {kind}: ell = logit(conf) (no log-odds column)")
    if "conf" not in out and "ell" in out:
        out["conf"] = expit(out["ell"]).astype(np.float32)
        print(f"[derived] {kind}: conf = sigmoid(ell) (no confidence column)")
    if "rate" not in out and "n_masked" in out and "doc_len" in out:
        out["rate"] = (out["n_masked"] / np.maximum(out["doc_len"], 1)).astype(np.float32)
        print(f"[derived] {kind}: mask rate = n_masked / doc_len")
    out.setdefault("margin", np.full(n, np.nan, np.float32))
    out.setdefault("rate", np.full(n, np.nan, np.float32))
    return out


def build_traces(cols: dict, dataset: str, kind: str) -> Traces:
    """Validate, deduplicate and sort the columns of one dataset x kind into a Traces.

    Logged, never silent: duplicate (state, position) rows, rows with non-finite ell/conf or
    agree outside {0, 1}, states spanning several documents, states whose row count differs
    from n_masked (counted only), states whose mask rate is not constant (first value used).
    """
    n0 = len(cols["doc"])
    ok = np.isfinite(cols["ell"]) & np.isfinite(cols["conf"]) & np.isin(cols["agree"], (0, 1))
    log_exclusion(dataset, kind, "load", "rows", (~ok).sum(), "non-finite ell/conf or agree not 0/1")
    if "position" in cols:
        order = np.lexsort((cols["position"], cols["state"], cols["doc"]))
        d, s, p = cols["doc"][order], cols["state"][order], cols["position"][order]
        dup_sorted = np.zeros(n0, bool)
        dup_sorted[1:] = (d[1:] == d[:-1]) & (s[1:] == s[:-1]) & (p[1:] == p[:-1])
        dup = np.zeros(n0, bool); dup[order] = dup_sorted
        log_exclusion(dataset, kind, "load", "rows", (dup & ok).sum(),
                      "duplicate (doc, state, position); first kept")
        ok &= ~dup
        del order, d, s, p, dup_sorted, dup
    keep = {}
    for k in list(cols):  # pops from the caller's dict: one column copied at a time
        keep[k] = cols.pop(k)[ok]
    keep.pop("position", None)
    order = np.lexsort((-keep["ell"], keep["state"], keep["doc"]))
    for k in list(keep):
        keep[k] = keep[k][order]
    del order
    n = len(keep["doc"])
    new_state = np.ones(n, bool)
    new_state[1:] = (keep["doc"][1:] != keep["doc"][:-1]) | (keep["state"][1:] != keep["state"][:-1])
    st_start = np.flatnonzero(new_state)
    st_n = np.diff(np.append(st_start, n))
    row_state = np.cumsum(new_state) - 1
    # same state_id under two doc_ids is two states here; count it
    sid = keep["state"][st_start]
    n_multi = len(sid) - np.unique(sid).size
    log_exclusion(dataset, kind, "check", "state ids", n_multi,
                  "state_id appears under several doc_ids (kept apart as separate states)")
    if "n_masked" in keep:
        mism = (keep["n_masked"][st_start] != st_n).sum()
        if mism:
            print(f"[check] {dataset} / {kind}: {mism} states with row count != n_masked "
                  "(kept; counted here)")
        EXCLUSIONS.append({"dataset": dataset, "kind": kind, "stage": "check",
                           "unit": "states (count only)", "count": int(mism),
                           "reason": "row count != n_masked"})
    rate = keep["rate"]
    st_rate = rate[st_start]
    if np.isfinite(rate).any():
        nonconst = (np.abs(rate - st_rate[row_state]) > 1e-6).any()
        if nonconst:
            print(f"[check] {dataset} / {kind}: mask rate varies within a state; first value used")
    new_blk = new_state.copy()
    new_blk[1:] |= keep["ell"][1:] != keep["ell"][:-1]
    b_start = np.flatnonzero(new_blk)
    b_len = np.diff(np.append(b_start, n))
    blk_id = np.cumsum(new_blk) - 1
    st_mean_ell = np.bincount(row_state, weights=keep["ell"]) / st_n
    print(f"[load] {dataset} / {kind}: {n:,} rows ({n0 - n:,} excluded), {len(st_start):,} states, "
          f"{np.unique(keep['doc']).size:,} docs, tied rows {(b_len[blk_id] > 1).mean():.1%}")
    return Traces(dataset, kind, keep["ell"], keep["conf"], keep["agree"], keep["margin"],
                  row_state.astype(np.int64), b_start[blk_id].astype(np.int64),
                  b_len[blk_id].astype(np.int64), st_start.astype(np.int64),
                  st_n.astype(np.int64), keep["doc"][st_start], st_rate, st_mean_ell)


def traces_from_frame(df: pd.DataFrame, dataset: str, kind: str = "uniform") -> Traces:
    """build_traces on an in-memory DataFrame with schema-v2 column names (tests, Module B)."""
    sub = df[df["mask_kind"] == kind]
    fm = field_map(sub.columns)
    want = {"doc": np.int64, "state": np.int64, "ell": np.float64, "conf": np.float32,
            "agree": np.int8, "margin": np.float32, "rate": np.float32,
            "n_masked": np.int32, "position": np.int32}
    cols = {k: sub[fm[k]].to_numpy().astype(t) for k, t in want.items() if fm.get(k)}
    cols.setdefault("margin", np.full(len(sub), np.nan, np.float32))
    cols.setdefault("rate", np.full(len(sub), np.nan, np.float32))
    return build_traces(cols, dataset, kind)


# ---------------------------------------------------------------- pools
def lgcomb(a, b):
    """log C(a, b) for arrays with 0 <= b <= a."""
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1)


def sample_winner_slots(n, m: int, rng) -> np.ndarray:
    """Slot (0 = highest ell) of the winner of a uniformly random m-subset of n sorted slots.

    Exact: P(slot >= t) = C(n - t, m) / C(n, m), inverted by vectorised binary search.
    """
    n = np.asarray(n, np.int64)
    logu = np.log(rng.random(n.shape))
    lo = np.zeros_like(n)
    hi = n - m
    if (hi < 0).any():
        raise ValueError("a state has fewer than m positions")
    lg_nm = lgcomb(n, m)
    while True:
        act = lo < hi
        if not act.any():
            return lo
        mid = (lo + hi + 1) // 2
        ok = (lgcomb(n - mid, m) - lg_nm) >= logu
        lo = np.where(act & ok, mid, lo)
        hi = np.where(act & ~ok, mid - 1, hi)


def actual_pools(tr: Traces, states: np.ndarray, m: int, pools: int, rng):
    """`pools` random m-subsets of each state: (state, winner row, random-member row) per pool.

    Ties in ell are broken uniformly at random (exact: the winner is a uniform member of the
    tie block holding the winning slot).  The random member is a uniform member of the same
    pool (the m = 1 arm of the paired contrast); for m = 1 it is the winner itself.
    """
    s = np.repeat(np.asarray(states, np.int64), pools)
    n, start = tr.st_n[s], tr.st_start[s]
    j = sample_winner_slots(n, m, rng)
    r = start + j
    w = tr.blk_start[r] + (rng.random(r.size) * tr.blk_len[r]).astype(np.int64)
    same = rng.random(r.size) < 1.0 / m
    below = np.maximum(n - 1 - j, 1)
    rs = start + np.minimum(j + 1 + (rng.random(r.size) * below).astype(np.int64), n - 1)
    in_blk = tr.blk_start[rs] == tr.blk_start[r]
    k = (rng.random(r.size) * np.maximum(tr.blk_len[r] - 1, 1)).astype(np.int64)
    idx_blk = tr.blk_start[r] + k
    idx_blk += idx_blk >= w
    idx_other = tr.blk_start[rs] + (rng.random(r.size) * tr.blk_len[rs]).astype(np.int64)
    u = np.where(same, w, np.where(in_blk, idx_blk, idx_other))
    return s, w, u


def rate_bins_of(tr: Traces, states: np.ndarray, n_bins: int):
    """Mask-rate quantile bin of every state (edges from `states`); None if no rate stored."""
    rates = tr.st_rate[states]
    if not np.isfinite(rates).any():
        return None, 0
    edges = np.unique(np.quantile(rates[np.isfinite(rates)], np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return np.zeros(tr.n_states, np.int64), 1
    b = np.clip(np.searchsorted(edges, tr.st_rate, side="right") - 1, 0, len(edges) - 2)
    return b.astype(np.int64), len(edges) - 1


def shuffled_pools(tr: Traces, states: np.ndarray, m: int, pools: int, rbin, rng,
                   chunk_entries: int = 8_000_000, max_iter: int = 1000):
    """Pools of m positions from m different states of m different documents, within the same
    mask kind and mask-rate bin; `pools` x (states in the bin) pools per bin.

    Each member: a uniform state of the bin (re-drawn while its document repeats in the pool),
    then a uniform position of that state.  Winner = largest ell, ties at random.
    Returns (winner state, winner row, n_pools_dropped); bins with fewer than m documents are
    logged and skipped.
    """
    ws_all, w_all, dropped = [], [], 0
    states = np.asarray(states, np.int64)
    for b in np.unique(rbin[states]):
        st_b = states[rbin[states] == b]
        docs_b = tr.st_doc[st_b]
        n_docs = np.unique(docs_b).size
        n_pools = pools * st_b.size
        if n_docs < m:
            log_exclusion(tr.dataset, tr.kind, f"shuffle m={m}", "pools", n_pools,
                          f"rate bin {b}: {n_docs} documents < m")
            dropped += n_pools
            continue
        per = max(1, chunk_entries // m)
        for c0 in range(0, n_pools, per):
            c = min(per, n_pools - c0)
            pick = rng.integers(0, st_b.size, size=(c, m))
            for _ in range(max_iter):
                d = docs_b[pick]
                order = np.argsort(d, axis=1, kind="stable")
                ds_ = np.take_along_axis(d, order, 1)
                dup_sorted = np.zeros(d.shape, bool)
                dup_sorted[:, 1:] = ds_[:, 1:] == ds_[:, :-1]
                if not dup_sorted.any():
                    break
                dup = np.zeros(d.shape, bool)
                np.put_along_axis(dup, order, dup_sorted, 1)
                pick[dup] = rng.integers(0, st_b.size, size=int(dup.sum()))
            else:
                bad = dup.any(1)
                log_exclusion(tr.dataset, tr.kind, f"shuffle m={m}", "pools", bad.sum(),
                              f"rate bin {b}: no distinct-document draw in {max_iter} tries")
                dropped += int(bad.sum())
                pick = pick[~bad]
            s = st_b[pick]
            rows = tr.st_start[s] + (rng.random(s.shape) * tr.st_n[s]).astype(np.int64)
            e = tr.ell[rows]
            key = np.where(e == e.max(1, keepdims=True), rng.random(e.shape), -1.0)
            col = key.argmax(1)
            ix = np.arange(len(col))
            w_all.append(rows[ix, col]); ws_all.append(s[ix, col])
    if not w_all:
        return np.array([], np.int64), np.array([], np.int64), dropped
    return np.concatenate(ws_all), np.concatenate(w_all), dropped


def exact_winner_mean(tr: Traces, states: np.ndarray, m: int, values: np.ndarray) -> np.ndarray:
    """Per state, the exact mean of `values` at the winner over *all* m-subsets (ties shared
    equally within a tie block): sum_j C(n-1-j, m-1)/C(n, m) v_j.  Checks the sampler."""
    smask = np.zeros(tr.n_states, bool); smask[states] = True
    rows = np.flatnonzero(smask[tr.row_state])
    rs = tr.row_state[rows]
    n = tr.st_n[rs]; j = rows - tr.st_start[rs]
    okj = (n - 1 - j) >= (m - 1)
    lw = np.full(rows.size, -np.inf)
    lw[okj] = lgcomb(n[okj] - 1 - j[okj], m - 1) - lgcomb(n[okj], m)
    w = np.exp(lw)
    bstart = tr.blk_start[rows]
    seg = np.flatnonzero(np.r_[True, bstart[1:] != bstart[:-1]])
    blen = np.diff(np.append(seg, rows.size))
    wb = np.add.reduceat(w, seg) / blen
    w_row = np.repeat(wb, blen)
    val = np.bincount(rs, weights=w_row * values[rows], minlength=tr.n_states)
    return val[states]


# ---------------------------------------------------------------- calibrator, ICC, ECE
def fit_calibrator(ell: np.ndarray, agree: np.ndarray):
    """Isotonic pi_hat(ell), increasing, fit on (ell rounded to 1e-4, mean agree) with counts."""
    key = np.round(ell, 4)
    u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    ybar = np.bincount(inv, weights=agree.astype(np.float64)) / cnt
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    iso.fit(u, ybar, sample_weight=cnt)
    return lambda x: iso.predict(np.round(np.asarray(x, np.float64), 4))


def icc_from_stats(st: np.ndarray) -> np.ndarray:
    """One-way ANOVA ICC for unequal group sizes from summed sufficient statistics.

    st[..., :] = (N, sum r, sum_s (S_s^2 / n_s), sum r^2, k groups, sum_s n_s^2).
    """
    N, S, C, Q, k, n2 = (st[..., i] for i in range(6))
    msb = (C - S ** 2 / N) / (k - 1)
    msw = (Q - C) / (N - k)
    n0 = (N - n2 / N) / (k - 1)
    return (msb - msw) / (msb + (n0 - 1) * msw)


def ece_equal_mass(conf: np.ndarray, agree: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-mass confidence bins (row level)."""
    order = np.argsort(conf, kind="stable")
    bins = np.array_split(order, n_bins)
    return float(sum(abs(conf[b].astype(np.float64).sum() - agree[b].sum()) for b in bins)
                 / conf.size)


# ---------------------------------------------------------------- bootstrap
def doc_weights(D: int, n_boot: int, rng) -> np.ndarray:
    """Multinomial document weights (n_boot x D): one bootstrap = resample D documents."""
    return rng.multinomial(D, np.full(D, 1.0 / D), size=n_boot).astype(np.float64)


class DocAccumulator:
    """Per-document sums and counts of named quantities; estimates average within document first,
    then across documents, with every quantity bootstrapped on the same document weights."""

    def __init__(self, D: int):
        self.D, self.sum, self.cnt = D, {}, {}

    def add(self, name, doc_idx, values):
        doc_idx = np.asarray(doc_idx, np.int64)
        self.sum[name] = np.bincount(doc_idx, weights=np.asarray(values, np.float64),
                                     minlength=self.D)
        self.cnt[name] = np.bincount(doc_idx, minlength=self.D).astype(np.float64)

    def estimate(self, W: np.ndarray):
        """(point, boot): point = mean over documents with data of their mean; boot = n_boot x K."""
        names = list(self.sum)
        S = np.column_stack([self.sum[k] for k in names])
        Cn = np.column_stack([self.cnt[k] for k in names])
        H = (Cn > 0).astype(np.float64)
        M = np.divide(S, Cn, out=np.zeros_like(S), where=Cn > 0)
        point = (M * H).sum(0) / np.maximum(H.sum(0), 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = (W @ (M * H)) / (W @ H)
        return dict(zip(names, point)), {k: boot[:, i] for i, k in enumerate(names)}


def ci(boot: np.ndarray, point: float) -> dict:
    """95% percentile interval and bootstrap SE (NaN replicates, if any, are counted)."""
    b = boot[np.isfinite(boot)]
    lo, hi = (np.percentile(b, [2.5, 97.5]) if b.size else (np.nan, np.nan))
    return {"est": float(point), "lo": float(lo), "hi": float(hi),
            "se": float(b.std(ddof=1)) if b.size > 1 else np.nan,
            "n_nan_boot": int(boot.size - b.size)}


# ---------------------------------------------------------------- the analysis of one dataset x kind
def analyze(tr: Traces, st_half: np.ndarray, *, m_grid=M_GRID, pools=POOLS_PER_STATE,
            n_boot=N_BOOT, rate_bins=RATE_BINS, min_state=MIN_STATE_SIZE,
            n_quint=N_QUINTILES, key="main") -> dict:
    """Modules C, D, E, F and the ICC for one dataset x mask kind.

    Report-half states only (|M| >= min_state for every m); the isotonic calibrator for the ICC
    is fit on fit-half states.  Returns tables `main` (per m), `paired` (per m >= 2) and
    `quint` (per m x margin quintile), plus `info`.
    """
    t0 = time.time()
    ds_, kd = tr.dataset, tr.kind
    for h, name in ((0, "fit"), (1, "report")):
        log_exclusion(ds_, kd, "analysis", f"{name}-half states",
                      ((st_half == h) & (tr.st_n < min_state)).sum(), f"|M| < {min_state}")
    rep = np.flatnonzero((st_half == 1) & (tr.st_n >= min_state))
    fit = np.flatnonzero((st_half == 0) & (tr.st_n >= min_state))
    if rep.size == 0 or fit.size == 0:
        raise ValueError(f"{ds_}/{kd}: no report or fit states with |M| >= {min_state}")
    S = tr.n_states
    smask_rep = np.zeros(S, bool); smask_rep[rep] = True
    smask_fit = np.zeros(S, bool); smask_fit[fit] = True
    row_rep = smask_rep[tr.row_state]; row_fit = smask_fit[tr.row_state]

    rep_docs, inv = np.unique(tr.st_doc[rep], return_inverse=True)
    D = rep_docs.size
    didx = np.full(S, -1, np.int64); didx[rep] = inv
    W = doc_weights(D, n_boot, rng_for(key, ds_, kd, "boot"))

    # ICC of a - pi_hat(ell), calibrator from the fit half
    cal = fit_calibrator(tr.ell[row_fit], tr.agree[row_fit])
    rs = tr.row_state[row_rep]
    resid = tr.agree[row_rep] - cal(tr.ell[row_rep])
    n_s = np.bincount(rs, minlength=S)[rep].astype(np.float64)
    S_s = np.bincount(rs, weights=resid, minlength=S)[rep]
    Q_s = np.bincount(rs, weights=resid ** 2, minlength=S)[rep]
    dstat = np.column_stack([np.bincount(inv, weights=v, minlength=D) for v in
                             (n_s, S_s, S_s ** 2 / n_s, Q_s, np.ones_like(n_s), n_s ** 2)])
    icc = ci(icc_from_stats(W @ dstat), float(icc_from_stats(dstat.sum(0))))
    ece = ece_equal_mass(tr.conf[row_rep], tr.agree[row_rep], ECE_BINS)

    rbin, n_rbins = rate_bins_of(tr, np.concatenate([rep, fit]), rate_bins)
    has_margin = bool(np.isfinite(tr.margin[row_rep]).all())
    acc = DocAccumulator(D)
    counts, exact = {}, {}
    gap_row = tr.conf.astype(np.float64) - tr.agree
    for m in m_grid:
        s, w, u = actual_pools(tr, rep, m, pools, rng_for(key, ds_, kd, "actual", m))
        di = didx[s]
        g = gap_row[w]
        acc.add(f"c_w|{m}", di, tr.conf[w]); acc.add(f"a_w|{m}", di, tr.agree[w])
        acc.add(f"delta|{m}", di, g); acc.add(f"infl|{m}", di, tr.ell[w] - tr.st_mean_ell[s])
        if m >= 2:
            acc.add(f"pdiff|{m}", di, g - gap_row[u])
        ex = exact_winner_mean(tr, rep, m, gap_row)
        exact[m] = float(pd.Series(ex).groupby(didx[rep]).mean().mean())
        counts[m] = {"n_pools": int(s.size), "n_shuf_pools": 0, "n_shuf_docs": 0,
                     "n_shuf_dropped": 0}
        if has_margin:
            edges = np.quantile(tr.margin[w], np.linspace(0, 1, n_quint + 1)[1:-1])
            qb = np.searchsorted(edges, tr.margin[w], side="right")
            for q in range(n_quint):
                acc.add(f"dq{q}|{m}", di[qb == q], g[qb == q])
        if m >= 2 and rbin is not None:
            ss, sw, dropped = shuffled_pools(tr, rep, m, pools, rbin,
                                             rng_for(key, ds_, kd, "shuffle", m))
            counts[m].update(n_shuf_pools=int(sw.size), n_shuf_dropped=int(dropped),
                             n_shuf_docs=int(np.unique(didx[ss]).size))
            if sw.size:
                acc.add(f"shuf|{m}", didx[ss], gap_row[sw])
                if has_margin:
                    qs = np.searchsorted(edges, tr.margin[sw], side="right")
                    for q in range(n_quint):
                        acc.add(f"sq{q}|{m}", didx[ss][qs == q], gap_row[sw][qs == q])
    point, boot = acc.estimate(W)

    def get(name):
        return ci(boot[name], point[name]) if name in point else None

    def diff(a, b):
        if a not in point or b not in point:
            return None
        return ci(boot[a] - boot[b], point[a] - point[b])

    main, paired, quint = [], [], []
    for m in m_grid:
        row = {"dataset": ds_, "kind": kd, "m": m, "n_states": int(rep.size), "n_docs": int(D),
               **counts[m], "delta_exact": exact[m], "ece": ece, "n_rate_bins": n_rbins}
        for nm, k in (("E_c_w", f"c_w|{m}"), ("E_a_w", f"a_w|{m}"), ("delta", f"delta|{m}"),
                      ("ell_infl", f"infl|{m}"), ("delta_shuf", f"shuf|{m}")):
            r = get(k)
            for f in ("est", "lo", "hi", "se"):
                row[f"{nm}_{f}"] = r[f] if r else np.nan
        gap = diff(f"delta|{m}", f"shuf|{m}") if m >= 2 else None
        for f in ("est", "lo", "hi", "se"):
            row[f"gap_{f}"] = gap[f] if gap else np.nan
        for f in ("est", "lo", "hi"):
            row[f"icc_{f}"] = icc[f]
            v = icc[f] if f == "est" else icc["hi" if f == "lo" else "lo"]
            row[f"m_eff_{f}"] = m / (1 + (m - 1) * v)
        main.append(row)
        if m >= 2:
            pr, up = get(f"pdiff|{m}"), diff(f"delta|{m}", f"delta|{m_grid[0]}")
            se_ind = math.hypot(get(f"delta|{m}")["se"], get(f"delta|{m_grid[0]}")["se"])
            paired.append({"dataset": ds_, "kind": kd, "m": m,
                           "paired_est": pr["est"], "paired_lo": pr["lo"], "paired_hi": pr["hi"],
                           "paired_se": pr["se"], "unpaired_est": up["est"],
                           "unpaired_lo": up["lo"], "unpaired_hi": up["hi"],
                           "unpaired_se_joint": up["se"], "unpaired_se_indep": se_ind,
                           "var_reduction_vs_joint": 1 - pr["se"] ** 2 / up["se"] ** 2,
                           "var_reduction_vs_indep": 1 - pr["se"] ** 2 / se_ind ** 2})
        if has_margin:
            for q in range(n_quint):
                a, sh = get(f"dq{q}|{m}"), get(f"sq{q}|{m}")
                gq = diff(f"dq{q}|{m}", f"sq{q}|{m}")
                quint.append({"dataset": ds_, "kind": kd, "m": m, "quintile": q + 1,
                              "n_pools": int(acc.cnt[f"dq{q}|{m}"].sum()),
                              **{f"delta_{f}": a[f] if a else np.nan for f in ("est", "lo", "hi")},
                              **{f"delta_shuf_{f}": sh[f] if sh else np.nan for f in ("est", "lo", "hi")},
                              **{f"gap_{f}": gq[f] if gq else np.nan for f in ("est", "lo", "hi")}})
    info = {"dataset": ds_, "kind": kd, "n_rep_states": int(rep.size), "n_fit_states": int(fit.size),
            "n_docs": int(D), "n_boot": int(n_boot), "icc": icc, "ece": ece, "has_margin": has_margin,
            "n_rate_bins": n_rbins, "seconds": round(time.time() - t0, 1)}
    return {"main": pd.DataFrame(main), "paired": pd.DataFrame(paired),
            "quint": pd.DataFrame(quint), "info": info}


# ---------------------------------------------------------------- Module A toy
def toy_scores(Q: float, n_toys: int, L: float, rng, sigma=1.0, dx=0.12, step=0.01,
               chunk=5000, local_half=2.0):
    """Gaussian matched filter: data = Q s(0) + N(0, I) on a grid over [-L, L] (101 points for
    L = 6); per toy the score at m0 = 0, the local max M_S (|m| <= 2 sigma) and the remote max M_R.
    Templates are unit-norm on the grid, so the score at m0 is Q + N(0, 1)."""
    x = np.linspace(-L, L, int(round(2 * L / dx)) + 1)
    mg = np.round(np.arange(-L, L + step / 2, step), 6)
    T = np.exp(-0.5 * ((x[None, :] - mg[:, None]) / sigma) ** 2)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    i0 = int(np.argmin(np.abs(mg)))
    loc = np.abs(mg) <= local_half * sigma
    s0 = T[i0]
    q0, ms, mr = [], [], []
    for c0 in range(0, n_toys, chunk):
        c = min(chunk, n_toys - c0)
        data = (Q * s0[None, :] + rng.standard_normal((c, x.size))).astype(np.float32)
        sc = data @ T.T.astype(np.float32)
        q0.append(sc[:, i0]); ms.append(sc[:, loc].max(1)); mr.append(sc[:, ~loc].max(1))
    return np.concatenate(q0).astype(np.float64), np.concatenate(ms).astype(np.float64), \
        np.concatenate(mr).astype(np.float64), x.size


def mean_se(v):
    """(mean, standard error) of a 1-D sample."""
    v = np.asarray(v, np.float64)
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(v.size))


print("library loaded")

# %% [markdown]
# # Phase 1: feasibility
# ## 1.1 Locate the traces
# Put the trace folders (or single parquet files) and the split manifest (`artifacts/split_manifest.json`
# from the repo) on Drive, then set `DATASETS` in 0.1. Uploading works too: `from google.colab import files; files.upload()`
# writes the files to `/content`. Reads from Drive are slow, so copying a folder to `/content` first
# (`!cp -r /content/drive/MyDrive/c3_traces /content/`) makes loading much faster.
#
# With `USE_SYNTHETIC = True`, the five Module B scenarios are written as parquet files and used in place of real traces.

# %%
if USE_SYNTHETIC:
    SYN_DIR = Path(OUT_DIR) / "synthetic_traces"
    SYN_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS = {}
    for scen in SCENARIOS:
        f = SYN_DIR / f"{scen}.parquet"
        make_synthetic(scen).to_parquet(f, index=False)
        DATASETS[f"synthetic-{scen}"] = {"path": str(f), "split": {"rule": {"n_docs": 200}}}
    MASK_KINDS_RUN = ["uniform"]
    print(f"synthetic traces written to {SYN_DIR}")
else:
    MASK_KINDS_RUN = list(MASK_KINDS)
    if MOUNT_DRIVE and any("path" in s and str(s["path"]).startswith("/content/drive") for s in DATASETS.values()):
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except ImportError:
            print("not on Colab: no Drive mount")
    hf_dirs = [s["hf"] for s in DATASETS.values() if "hf" in s]
    if hf_dirs:
        from huggingface_hub import snapshot_download
        token = os.environ.get("HF_TOKEN")
        if token is None:
            try:
                from google.colab import userdata
                token = userdata.get("HF_TOKEN")
            except Exception as exc:  # not on Colab, or the secret is missing / not shared
                print(f"no HF_TOKEN in the environment or Colab secrets ({type(exc).__name__})")
        t_dl = time.time()
        try:
            snapshot_download(HF_REPO, repo_type="dataset", token=token, local_dir=HF_LOCAL,
                              allow_patterns=[f"{d}/*" for d in hf_dirs] + ["README.md"])
        except Exception as exc:
            raise RuntimeError(
                f"could not download {HF_REPO} ({type(exc).__name__}). It is private: add a read token as "
                "the Colab secret HF_TOKEN (key icon, left bar; enable notebook access) for an HF account "
                "the ANLP-project org has given access, or point DATASETS at a local/Drive copy.") from None
        print(f"downloaded {hf_dirs} from {HF_REPO} to {HF_LOCAL} in {time.time() - t_dl:.0f} s")
        for s in DATASETS.values():
            if "hf" in s:
                s["path"] = os.path.join(HF_LOCAL, s["hf"])
assert DATASETS, "set DATASETS in 0.1 (or USE_SYNTHETIC = True)"

FILES = {}
for name, spec in DATASETS.items():
    if not spec.get("path"):
        print(f"{name}: no path (download failed or not configured)")
        FILES[name] = []
        continue
    by = trace_files(spec["path"])
    FILES[name] = by.get("parquet", [])
    sizes = {fmt: (len(fs), sum(f.stat().st_size for f in fs) / 1e9) for fmt, fs in by.items()}
    print(f"{name}: {spec['path']}")
    for fmt, (k, gb) in sizes.items():
        print(f"   {fmt:8s} {k:5d} files  {gb:8.3f} GB")
    if not by:
        print("   NOT FOUND")
    elif not FILES[name]:
        print("   no parquet files: only parquet is supported; convert first")

# %% [markdown]
# ## 1.2 Schema, row counts, 5-row preview

# %%
SCHEMA, NROWS, FMAP = {}, {}, {}
for name, files in FILES.items():
    if not files:
        continue
    schemas = {}
    n = 0
    for f in files:
        pf = pq.ParquetFile(f)
        n += pf.metadata.num_rows
        schemas.setdefault(str(pf.schema_arrow), []).append(f.name)
    NROWS[name] = n
    first = pq.ParquetFile(files[0])
    SCHEMA[name] = first.schema_arrow
    FMAP[name] = field_map(first.schema_arrow.names)
    print(f"== {name}: {n:,} rows in {len(files)} files; {len(schemas)} distinct schema(s)")
    if len(schemas) > 1:
        print("   WARNING: files differ in schema:", {k[:60]: v[:3] for k, v in schemas.items()})
    print("   columns:", ", ".join(f"{fl.name}:{fl.type}" for fl in first.schema_arrow))
    prev = pa.Table.from_batches([next(first.iter_batches(batch_size=5))]).slice(0, 5).to_pandas()
    for c in prev.columns:
        if prev[c].dtype == object and prev[c].map(lambda v: hasattr(v, "__len__") and not isinstance(v, str)).any():
            prev[c] = prev[c].map(lambda v: f"[{len(v)} values]")
    display(prev)

# %% [markdown]
# ## 1.3 Field map and module feasibility

# %%
SPLITS, FEASIBLE = {}, {}
for name, fm in FMAP.items():
    spec = DATASETS[name]["split"]
    try:
        SPLITS[name] = make_split(spec)
        has_split = True
    except (OSError, KeyError, ValueError) as exc:
        print(f"{name}: split not available: {exc}")
        has_split = False
    tab = pd.DataFrame([{"field": FIELD_LABEL[k], "canonical": k, "column": fm[k] or "missing",
                         "used by": USED_BY.get(k, "-")} for k in CANON])
    print(f"== {name}  (split: {'manifest' if 'manifest' in spec else 'rule'}"
          f"{', ' + str(SPLITS[name].n) + ' docs' if has_split else ', MISSING'})")
    print(tab.to_string(index=False))
    FEASIBLE[name] = module_feasibility(fm, has_split)
    print("modules:", {k: ("yes" if ok else f"NO, needs {', '.join(miss)}")
                       for k, (ok, miss) in FEASIBLE[name].items()})

# %% [markdown]
# ## 1.4 Mask kinds, state sizes, and how many states allow m = 32
# This cell reads only `doc_id`, `state_id` and `mask_kind` (plus `ell` on a sample, to count ties).

# %%
STATE_TABLE = {}
for name, files in FILES.items():
    fm = FMAP.get(name)
    if not fm or not fm["kind"]:
        continue
    cols = [fm["doc"], fm["state"], fm["kind"]]
    dset = pads.dataset([str(f) for f in files], format="parquet")
    filt = (pc.field(fm["model"]) == DATASETS[name]["model"]) if DATASETS[name].get("model") and fm["model"] else None
    parts = []
    for b in dset.to_batches(columns=cols, filter=filt, batch_size=4_000_000):
        df = b.to_pandas()
        parts.append(df.groupby([fm["kind"], fm["doc"], fm["state"]], observed=True).size())
    st = pd.concat(parts).groupby(level=[0, 1, 2]).sum().rename("n").reset_index()
    st.columns = ["kind", "doc", "state", "n"]
    st["half"] = SPLITS[name](st["doc"].to_numpy()) if name in SPLITS else -1
    STATE_TABLE[name] = st
    summ = st.groupby("kind").agg(rows=("n", "sum"), docs=("doc", "nunique"), states=("n", "size"),
                                  M_min=("n", "min"), M_median=("n", "median"), M_max=("n", "max"),
                                  ge_32=("n", lambda v: int((v >= MIN_STATE_SIZE).sum())))
    summ["share_ge_32"] = (summ["ge_32"] / summ["states"]).round(3)
    halves = st.assign(ok=st["n"] >= MIN_STATE_SIZE).groupby(["kind", "half"]).agg(
        docs=("doc", "nunique"), states_ok=("ok", "sum"))
    print(f"== {name}")
    print(summ.to_string())
    print("by half (-1 = document not in the split):")
    print(halves.to_string())
    missing_kinds = [k for k in MASK_KINDS_RUN if k not in set(st["kind"])]
    if missing_kinds:
        print("   mask kinds not present:", missing_kinds)
    n_nohalf = st.loc[st["half"] < 0, "doc"].nunique()
    if n_nohalf:
        print(f"   {n_nohalf} documents are not in the split and will be excluded")
    sample = pq.ParquetFile(files[0]).read_row_group(0, columns=[fm["state"], fm["kind"], fm["ell"] or fm["conf"]]).to_pandas()
    sample.columns = ["state", "kind", "ell"]
    tied = sample.duplicated(["kind", "state", "ell"], keep=False)
    full_tie = sample.groupby(["kind", "state"])["ell"].nunique() == 1
    print(f"   ties in the first row group: {tied.mean():.1%} of rows; "
          f"{int(full_tie.sum())} of {full_tie.size} states fully tied")

# %% [markdown]
# ## 1.5 Memory and runtime estimate
# Each dataset × kind is loaded on its own with about 10 downcast columns (≈41 B/row, plus about 2× that during the sort).
# Runtime is measured on a synthetic benchmark of the same pool and bootstrap code, then scaled by the real state counts.

# %%
try:
    import psutil
    AVAIL_GB = psutil.virtual_memory().available / 1e9
except ImportError:
    AVAIL_GB = float("nan")
BYTES_PER_ROW, PEAK_FACTOR = 41, 2.0

bench_tr = traces_from_frame(make_synthetic("calibrated", n_docs=400, n_states=10, n_pos=100), "bench")
bench_states = np.arange(bench_tr.n_states)
t = time.time()
for m in M_GRID:
    actual_pools(bench_tr, bench_states, m, POOLS_PER_STATE, rng_for("bench", m))
t_act = (time.time() - t) / (bench_tr.n_states * POOLS_PER_STATE * len(M_GRID))
rb, _ = rate_bins_of(bench_tr, bench_states, RATE_BINS)
t = time.time()
ent = 0
for m in M_GRID[1:]:
    shuffled_pools(bench_tr, bench_states, m, POOLS_PER_STATE, rb, rng_for("bench-s", m))
    ent += bench_tr.n_states * POOLS_PER_STATE * m
t_shuf = (time.time() - t) / ent
t = time.time()
Wb = doc_weights(1000, 2000, rng_for("bench-w"))
_ = Wb @ np.random.default_rng(0).random((1000, 150))
t_boot = (time.time() - t) / (2000 * 1000)
EXCLUSIONS[:] = [e for e in EXCLUSIONS if e["dataset"] != "bench"]

PROJ = []
for name, st in STATE_TABLE.items():
    for kind in MASK_KINDS_RUN:
        sk = st[(st["kind"] == kind)]
        if sk.empty:
            continue
        rows = int(sk["n"].sum())
        S_rep = int(((sk["half"] == 1) & (sk["n"] >= MIN_STATE_SIZE)).sum())
        D = int(sk.loc[sk["half"] == 1, "doc"].nunique())
        sec_pools = S_rep * POOLS_PER_STATE * (len(M_GRID) * t_act / 1 + sum(M_GRID[1:]) * t_shuf)
        sec_boot = N_BOOT * D * t_boot * 1.5
        sec_load = rows / 5e6
        PROJ.append({"dataset": name, "kind": kind, "rows": rows,
                     "mem_GB": rows * BYTES_PER_ROW * PEAK_FACTOR / 1e9, "report_states": S_rep,
                     "report_docs": D, "load_min": sec_load / 60, "pools_min": sec_pools / 60,
                     "boot_min": sec_boot / 60})
PROJ = pd.DataFrame(PROJ)
if not PROJ.empty:
    PROJ["total_min"] = PROJ[["load_min", "pools_min", "boot_min"]].sum(1)
    print(PROJ.round(2).to_string(index=False))
TOY_MIN = 2.0
PROJECTED_MIN = float(PROJ["total_min"].sum() if not PROJ.empty else 0) + TOY_MIN + 1.0
print(f"available RAM {AVAIL_GB:.1f} GB; largest dataset x kind needs ≈ "
      f"{PROJ['mem_GB'].max() if not PROJ.empty else 0:.1f} GB")
print(f"projected total runtime ≈ {PROJECTED_MIN:.1f} min at N_BOOT = {N_BOOT:,} "
      "(load assumes local disk at ~5M rows/s; Drive can be several times slower)")

# %% [markdown]
# ## 1.6 Verdict (computed)
# This cell is the gate for Phase 2. It must agree with the verdict in the first cell. If it doesn't, edit the first cell.

# %%
problems, partial = [], []
for name in DATASETS:
    if not FILES.get(name):
        problems.append(f"{name}: no parquet files")
        continue
    if name not in FEASIBLE:
        problems.append(f"{name}: schema not read")
        continue
    no = [k for k, (ok, _) in FEASIBLE[name].items() if not ok]
    if "C" in no:
        problems.append(f"{name}: core fields missing ({FEASIBLE[name]['C'][1]})")
    elif no:
        partial.append(f"{name}: cannot run {', '.join(no)}")
    st = STATE_TABLE.get(name)
    if st is not None:
        for kind in MASK_KINDS_RUN:
            sk = st[(st["kind"] == kind) & (st["half"] == 1)]
            if sk.empty:
                partial.append(f"{name}: no '{kind}' states in the report half")
            elif (sk["n"] >= MIN_STATE_SIZE).sum() == 0:
                partial.append(f"{name}/{kind}: no report state with |M| >= {MIN_STATE_SIZE}")
if not PROJ.empty and not (PROJ["mem_GB"] < 0.8 * AVAIL_GB).all() and np.isfinite(AVAIL_GB):
    partial.append("some dataset x kind may not fit in RAM: "
                   + ", ".join(f"{r.dataset}/{r.kind}" for r in PROJ.itertuples()
                               if r.mem_GB >= 0.8 * AVAIL_GB))
PHASE1_VERDICT = "NO-GO" if problems and len(problems) == len(DATASETS) else \
    ("PARTIAL" if problems or partial else "GO")
print(f"PHASE 1 VERDICT: {PHASE1_VERDICT}" + ("  [synthetic data]" if USE_SYNTHETIC else ""))
for p in problems + partial:
    print("  -", p)
print(f"estimated total runtime ≈ {PROJECTED_MIN:.0f} min")

# %% [markdown]
# # Phase 2
# ## Runtime guard
# If the projection exceeds the limit, the run uses `DRY_RUN_BOOT` resamples and prints a warning. Set `CONFIRM_FULL_BOOT = True` and re-run to get the full count.

# %%
assert PHASE1_VERDICT != "NO-GO", "Phase 1 says NO-GO: fix the inputs before Phase 2"
N_BOOT_RUN = N_BOOT
if PROJECTED_MIN > RUNTIME_LIMIT_MIN and not CONFIRM_FULL_BOOT:
    N_BOOT_RUN = DRY_RUN_BOOT
    print(f"WARNING: projected {PROJECTED_MIN:.0f} min > {RUNTIME_LIMIT_MIN} min. This is a DRY RUN "
          f"with {N_BOOT_RUN:,} bootstrap resamples. Check the timings, then set CONFIRM_FULL_BOOT = True "
          f"and re-run for {N_BOOT:,}.")
print(f"bootstrap resamples for this run: {N_BOOT_RUN:,}")

# %% [markdown]
# ## Module A: toy reproduction (synthetic matched filter; checks the methods, not the LMs)
# * data = Q·s(0) + N(0, I) on 101 points over [−6, 6] (for L = 10 the grid is extended at the same spacing: 167 points over [−10, 10]). The template is a Gaussian with σ = 1, unit-norm on the grid.
# * Local fit: Q̂ = max of the matched filter over |m| ≤ 2σ. Bias is estimated two ways: raw, E[Q̂] − Q, and paired, E[Q̂ − score(m0)]. The paired version is Dorigo's control variate; it works because score(m0) has mean exactly Q.
# * Global search over [−L, L]: B_global = B_local + B_remote with B_remote = E[(M_R − M_S)₊]. This is an identity, because max(M_S, M_R) = M_S + (M_R − M_S)₊.
# * Shuffle test at Q = 1: B_actual pairs a toy's M_S with its own M_R. B_shuf pairs it with another toy's M_R. B_null takes M_R from a signal-free toy.
# * The same noise draws are reused across Q (common random numbers). Nothing is tuned; disagreements are printed as FLAG.

# %%
t0 = time.time()
Q_LOCAL = [2, 3, 5, 8, 12, 20]
Q_DECOMP = [1, 2, 3, 5, 8, 12, 20]
toy_rows, toy_dec, toy_shuf = [], [], []
for L in (6, 10):
    for Q in sorted(set(Q_LOCAL) | set(Q_DECOMP) | {0}):
        # common random numbers across Q; the signal-free null (Q = 0) gets its own noise
        q0, ms, mr, npts = toy_scores(Q, TOY_N, L, rng_for("toy-null" if Q == 0 else "toy", L))
        if L == 6 and Q in Q_LOCAL:
            b_raw, se_raw = mean_se(ms - Q)
            b_cv, se_cv = mean_se(ms - q0)
            toy_rows.append({"Q": Q, "Qbias_raw": Q * b_raw, "Qbias_raw_se": Q * se_raw,
                             "Qbias_cv": Q * b_cv, "Qbias_cv_se": Q * se_cv, "grid_points": npts})
        if Q in Q_DECOMP:
            g = np.maximum(ms, mr)
            Bg, Bl, Br = mean_se(g - Q), mean_se(ms - Q), mean_se(np.maximum(mr - ms, 0))
            toy_dec.append({"L": L, "Q": Q, "B_global": Bg[0], "B_global_se": Bg[1],
                            "B_local": Bl[0], "B_local_se": Bl[1], "B_remote": Br[0],
                            "B_remote_se": Br[1], "P_remote_wins": float((mr > ms).mean()),
                            "identity_error": abs(Bg[0] - Bl[0] - Br[0]), "grid_points": npts})
        if Q == 1:
            keep_1 = (ms, mr)
        if Q == 0:
            mr_null = mr
    ms1, mr1 = keep_1
    # pair toy i with toy i + k (k != 0): every toy gets another toy's M_R
    perm = (np.arange(TOY_N) + int(rng_for("toy-perm", L).integers(1, TOY_N))) % TOY_N
    for nm, mrx in (("actual", mr1), ("shuf", mr1[perm]), ("null", mr_null)):
        b, se = mean_se(np.maximum(ms1, mrx) - 1.0)
        toy_shuf.append({"L": L, "pairing": nm, "B": b, "se": se})
TOY_LOCAL, TOY_DEC, TOY_SHUF = pd.DataFrame(toy_rows), pd.DataFrame(toy_dec), pd.DataFrame(toy_shuf)
print(TOY_LOCAL.round(4).to_string(index=False))
print(TOY_DEC.round(4).to_string(index=False))
print(TOY_SHUF.round(4).to_string(index=False))
for df, nm in ((TOY_LOCAL, "toy_local"), (TOY_DEC, "toy_decomposition"), (TOY_SHUF, "toy_shuffle")):
    df.to_csv(os.path.join(OUT_DIR, f"{nm}.csv"), index=False)

TOY_CHECKS = {}
last = TOY_LOCAL.iloc[-1]
TOY_CHECKS["Q*bias -> 0.5 at Q=20 (paired estimator within 3 SE)"] = abs(last.Qbias_cv - 0.5) < 3 * last.Qbias_cv_se + 0.02
for L in (6, 10):
    d = TOY_DEC[TOY_DEC.L == L].set_index("Q")
    TOY_CHECKS[f"L={L}: B_global > B_local at Q=1"] = d.loc[1, "B_global"] > d.loc[1, "B_local"] + 2 * d.loc[1, "B_remote_se"]
    TOY_CHECKS[f"L={L}: remote term < 0.01 by Q=5"] = d.loc[5, "B_remote"] < 0.01
    TOY_CHECKS[f"L={L}: decomposition identity exact"] = d["identity_error"].max() < 1e-9
    s = TOY_SHUF[TOY_SHUF.L == L].set_index("pairing")
    TOY_CHECKS[f"L={L}: B_actual < B_shuf at Q=1"] = s.loc["actual", "B"] < s.loc["shuf", "B"]
for k, v in TOY_CHECKS.items():
    print(("ok   " if v else "FLAG ") + k)
TOY_CHECKS_ALL = all(TOY_CHECKS.values())
for L in (6, 10):
    s = TOY_SHUF[TOY_SHUF.L == L].set_index("pairing")
    rel = 1 - s.loc["actual", "B"] / s.loc["shuf", "B"]
    print(f"note L={L}: B_actual is {rel:.0%} below B_shuf (the brief expected 'slightly below'). M_S and M_R are "
          "maxima over adjacent windows of the same noise, so they are positively correlated near |m| = 2σ, "
          "which lowers E[max]. Reported as found, not tuned.")
print(f"Module A: {time.time() - t0:.0f} s")

# %%
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].errorbar(TOY_LOCAL.Q, TOY_LOCAL.Qbias_raw, 1.96 * TOY_LOCAL.Qbias_raw_se, fmt="o-", capsize=3, label="raw  Q(E[Q̂]−Q)")
ax[0].errorbar(TOY_LOCAL.Q * 1.03, TOY_LOCAL.Qbias_cv, 1.96 * TOY_LOCAL.Qbias_cv_se, fmt="s-", capsize=3, label="paired  Q·E[Q̂−score(m₀)]")
ax[0].axhline(0.5, ls="--", c="k", lw=1, label="D/2 = 0.5 (Gaussian theory, D = 1)")
ax[0].set(xscale="log", xlabel="signal strength Q", ylabel="Q × local bias", title="Local fit bias")
for a in ax[:2]:
    a.set_xticks(Q_DECOMP, [str(q) for q in Q_DECOMP]); a.minorticks_off()
ax[0].legend(fontsize=8)
for L, mk in ((6, "o"), (10, "s")):
    d = TOY_DEC[TOY_DEC.L == L]
    ax[1].errorbar(d.Q, d.B_global, 1.96 * d.B_global_se, fmt=mk + "-", capsize=3, label=f"B_global, L={L}")
    ax[1].errorbar(d.Q, d.B_remote, 1.96 * d.B_remote_se, fmt=mk + ":", capsize=3, label=f"B_remote, L={L}")
d6 = TOY_DEC[TOY_DEC.L == 6]
ax[1].errorbar(d6.Q, d6.B_local, 1.96 * d6.B_local_se, fmt="k^-", capsize=3, label="B_local")
ax[1].set(xscale="log", xlabel="signal strength Q", ylabel="bias of the maximum", title="Global = local + remote")
ax[1].legend(fontsize=8)
for i, L in enumerate((6, 10)):
    s = TOY_SHUF[TOY_SHUF.L == L]
    ax[2].errorbar(np.arange(3) + 0.2 * i, s.B, 1.96 * s.se, fmt="o", capsize=4, label=f"L={L}")
ax[2].set_xticks(range(3), ["actual", "shuffled", "null"])
ax[2].set(ylabel="B at Q = 1", title="Shuffle test (Q = 1)")
ax[2].legend(fontsize=8)
for a in ax:
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "moduleA_toy.png"), dpi=300)
plt.show()

# %% [markdown]
# ## Module B: synthetic C3 traces and self-tests
# There are five scenarios, each with a known answer: about 200 documents × 5 states × 100 masked positions, with a 50/50 document split. The checks were fixed before the first run, at m = 32, the size the paper reports. Expected-null checks use 20 independent replicates rather than one draw: one 95% interval misses 0 in 1 of 20 draws by design. The single-draw results are still printed. If one fails, the failure is printed and stays a failure; neither the seed nor the threshold is changed to make it pass.

# %%
t0 = time.time()
SELFTEST = {}


def check(name, cond):
    """Record one self-test result."""
    SELFTEST[name] = bool(cond)
    print(("ok   " if cond else "FAIL ") + name)


# --- unit tests on small cases and Module B data
rng = rng_for("unit")
n, m = 10, 3
draws = sample_winner_slots(np.full(200_000, n), m, rng)
pmf = np.exp(lgcomb(n - 1 - np.arange(n - m + 1), m - 1) - lgcomb(n, m))
check("winner-slot sampler matches C(n-1-j, m-1)/C(n, m)",
      np.abs(np.bincount(draws, minlength=n - m + 1) / draws.size - pmf).max() < 0.005)
brute = np.array([rng.choice(n, m, replace=False).min() for _ in range(20_000)])
check("winner-slot sampler matches brute-force subsets",
      np.abs(np.bincount(brute, minlength=n - m + 1) / brute.size - pmf).max() < 0.015)

tie_df = make_synthetic("calibrated", n_docs=40, n_states=5, n_pos=40, round_ell=0.5)
tie_tr = traces_from_frame(tie_df, "ties")
st_all = np.arange(tie_tr.n_states)
s_, w_, u_ = actual_pools(tie_tr, st_all, 8, 4000, rng_for("tie"))
gap_t = tie_tr.conf.astype(np.float64) - tie_tr.agree
mc = pd.Series(gap_t[w_]).groupby(s_).mean().to_numpy()
ex = exact_winner_mean(tie_tr, st_all, 8, gap_t)
check("tie-break: Monte Carlo winner mean = exact (ties shared) within 4 SE",
      np.abs(mc - ex).max() < 4 * 0.5 / math.sqrt(4000) + 1e-9 and (tie_tr.blk_len > 1).mean() > 0.5)
cnt_u = np.bincount(u_ - tie_tr.st_start[s_], minlength=40).astype(float) / u_.size
check("random pool member is uniform over the state", np.abs(cnt_u - 1 / 40).max() < 0.004)
check("random member differs from winner unless m = 1", (u_ != w_).mean() > 0.8)

rb_t, nb = rate_bins_of(tie_tr, st_all, 5)
ss_, sw_, _ = shuffled_pools(tie_tr, st_all, 8, 5, rb_t, rng_for("shuf-unit"))
check("shuffled winners come from the stratum's states", np.isin(ss_, st_all).all() and nb == 5)
pick_docs = tie_tr.st_doc  # distinct-document property, checked on a direct draw
s_chk = st_all[rb_t[st_all] == 0]
d0 = tie_tr.st_doc[s_chk]
rng_c = rng_for("distinct")
pk = rng_c.integers(0, s_chk.size, size=(500, 6))
for _ in range(1000):
    d = d0[pk]; o = np.argsort(d, 1, kind="stable"); dsd = np.take_along_axis(d, o, 1)
    dd = np.zeros(d.shape, bool); dd[:, 1:] = dsd[:, 1:] == dsd[:, :-1]
    if not dd.any():
        break
    du = np.zeros(d.shape, bool); np.put_along_axis(du, o, dd, 1); pk[du] = rng_c.integers(0, s_chk.size, size=int(du.sum()))
check("re-draw loop yields distinct documents per pool",
      all(np.unique(r).size == r.size for r in d0[pk]))

Wt = doc_weights(50, 300, rng_for("w"))
check("bootstrap weights resample D documents", (Wt.sum(1) == 50).all())
acc_t = DocAccumulator(3); acc_t.add("x", [0, 0, 1, 2], [1.0, 3.0, 5.0, 8.0])
p_t, b_t = acc_t.estimate(np.ones((4, 3)))
check("document-first averaging: mean of doc means", abs(p_t["x"] - (2 + 5 + 8) / 3) < 1e-12 and np.allclose(b_t["x"], p_t["x"]))
r_i = rng_for("icc")
k_g, n_g, rho = 2000, 20, 0.1
y = np.sqrt(rho) * np.repeat(r_i.normal(size=k_g), n_g) + np.sqrt(1 - rho) * r_i.normal(size=k_g * n_g)
g_ = np.repeat(np.arange(k_g), n_g)
Sg = np.bincount(g_, y); Qg = np.bincount(g_, y ** 2)
stt = np.array([k_g * n_g, Sg.sum(), (Sg ** 2 / n_g).sum(), Qg.sum(), k_g, k_g * n_g ** 2])
check("ANOVA ICC recovers a known ICC of 0.10", abs(icc_from_stats(stt) - rho) < 0.015)
cal_t = fit_calibrator(tie_tr.ell, tie_tr.agree)
xs = np.linspace(-5, 5, 101)
check("isotonic calibrator is non-decreasing and in [0, 1]", (np.diff(cal_t(xs)) >= -1e-12).all() and (cal_t(xs) >= 0).all() and (cal_t(xs) <= 1).all())
fit_i, rep_i = document_split_rule(200)
check("document split is 50/50 and disjoint", len(fit_i) == len(rep_i) == 100 and not set(fit_i) & set(rep_i))
check("ECE of perfectly calibrated data is small", ece_equal_mass(*(lambda c: (c, (rng.random(c.size) < c).astype(np.int8)))(rng.random(200_000))) < 0.01)
check("field map finds schema-v2 names", field_map(make_synthetic("calibrated", 2, 1, 4).columns)["ell"] == "ell")

# --- the five scenarios through the full analysis
SYN = {}
for scen in SCENARIOS:
    df = make_synthetic(scen)
    tr = traces_from_frame(df, f"synthetic-{scen}")
    half = make_split({"rule": {"n_docs": 200}})(tr.st_doc)
    SYN[scen] = analyze(tr, half, n_boot=SELFTEST_BOOT, key="selftest")


def row(scen, m):
    """The `main` row of a scenario at pool size m."""
    t = SYN[scen]["main"]
    return t[t.m == m].iloc[0]


def prow(scen, m):
    """The `paired` row of a scenario at pool size m."""
    t = SYN[scen]["paired"]
    return t[t.m == m].iloc[0]


M = max(M_GRID)
covers = lambda lo, hi: lo <= 0 <= hi
for scen in SCENARIOS:
    r = row(scen, M)
    check(f"{scen}: Monte Carlo Delta({M}) = exact within 4 SE",
          abs(r.delta_est - r.delta_exact) < 4 * 0.5 / math.sqrt(r.n_pools))
# Single draws (seed fixed in 0.1).  A 95% CI around a true null misses 0 in 1 of 20 draws by
# design, so these are reported, not asserted; the replicate tests below are the asserted ones.
SINGLE_DRAW = {
    f"(1) calibrated: shuffle gap CI at m={M} covers 0": covers(row("calibrated", M).gap_lo, row("calibrated", M).gap_hi),
    f"(3) context: shuffle gap CI at m={M} covers 0": covers(row("context", M).gap_lo, row("context", M).gap_hi),
    f"(4b) independent state effect: shuffle gap CI at m={M} covers 0": covers(row("shared_indep", M).gap_lo, row("shared_indep", M).gap_hi),
}
for k, v in SINGLE_DRAW.items():
    print(("info ok   " if v else "info MISS ") + k + ("" if v else "  (single draw; expected in ~5% of draws)"))
check(f"(1) calibrated: paired Delta({M}) - Delta(1) CI covers 0", covers(prow("calibrated", M).paired_lo, prow("calibrated", M).paired_hi))
check("(1) calibrated: ICC CI covers 0", covers(row("calibrated", M).icc_lo, row("calibrated", M).icc_hi))
check(f"(2) flat: Delta(1) and Delta({M}) CIs exclude 0", row("flat_miscal", 1).delta_lo > 0 and row("flat_miscal", M).delta_lo > 0)
check(f"(2) flat: paired Delta({M}) - Delta(1) CI covers 0", covers(prow("flat_miscal", M).paired_lo, prow("flat_miscal", M).paired_hi))
check(f"(3) context: paired Delta({M}) - Delta(1) CI above 0", prow("context", M).paired_lo > 0)
check(f"(4) shared errors: shuffle gap CI at m={M} excludes 0", not covers(row("shared_err", M).gap_lo, row("shared_err", M).gap_hi))
check("(4) shared errors: ICC CI above 0", row("shared_err", M).icc_lo > 0)
check("(4b) independent state effect: ICC CI above 0", row("shared_indep", M).icc_lo > 0)

# Replicates: R independent synthetic data sets per scenario (fresh seeds), shuffle gap at m = M.
# Null scenarios: no bias (|mean gap| < 3 SD / sqrt(R)), CI coverage >= 16/20 (a correct 95% CI
# fails this with probability 1.6%), bootstrap SE within 30% of the replicate SD.
# Shared errors: mean gap != 0 (|mean| > 3 SD / sqrt(R)).
R_REP = 20
REPS = []
for scen in ("calibrated", "context", "shared_indep", "shared_err"):
    for rep_i in range(R_REP):
        tr_r = traces_from_frame(make_synthetic(scen, seed=1000 + rep_i), f"synthetic-rep-{scen}")
        res_r = analyze(tr_r, make_split({"rule": {"n_docs": 200}})(tr_r.st_doc), m_grid=[1, M],
                        n_boot=1000, key=f"rep{rep_i}")["main"]
        x = res_r[res_r.m == M].iloc[0]
        REPS.append({"scenario": scen, "rep": rep_i, "gap": x.gap_est, "se": x.gap_se,
                     "covers": covers(x.gap_lo, x.gap_hi)})
REPS = pd.DataFrame(REPS)
rep_summ = REPS.groupby("scenario").agg(mean_gap=("gap", "mean"), sd_gap=("gap", "std"),
                                         mean_se=("se", "mean"), coverage=("covers", "mean"))
print(rep_summ.round(4).to_string())
for scen in ("calibrated", "context", "shared_indep"):
    r_ = rep_summ.loc[scen]
    check(f"replicates, {scen}: shuffle gap unbiased (|mean| < 3 SD/sqrt(R))", abs(r_.mean_gap) < 3 * r_.sd_gap / math.sqrt(R_REP))
    check(f"replicates, {scen}: CI coverage >= 16/{R_REP}", r_.coverage >= 16 / R_REP)
    check(f"replicates, {scen}: bootstrap SE within 30% of replicate SD", 0.7 < r_.mean_se / r_.sd_gap < 1.3)
r_ = rep_summ.loc["shared_err"]
check("replicates, shared_err: shuffle gap != 0 (|mean| > 3 SD/sqrt(R))", abs(r_.mean_gap) > 3 * r_.sd_gap / math.sqrt(R_REP))
# pairing removes document-level noise: a clear gain where states share an effect, none to lose elsewhere
check(f"paired SE < unpaired SE at m={M} with state effects", all(prow(s, M).paired_se < 0.9 * prow(s, M).unpaired_se_indep for s in ("shared_err", "shared_indep")))
check(f"paired SE <= 1.1 x unpaired SE at m={M} in every scenario", all(prow(s, M).paired_se <= 1.1 * prow(s, M).unpaired_se_indep for s in SCENARIOS))
print(f"self-tests: {sum(SELFTEST.values())}/{len(SELFTEST)} passed in {time.time() - t0:.0f} s")
SELFTEST_ALL = all(SELFTEST.values())
EXCLUSIONS[:] = [e for e in EXCLUSIONS if not e["dataset"].startswith(("synthetic-", "ties", "bench"))]
assert SELFTEST_ALL, "a self-test failed: fix the code before trusting Modules C-F"
pd.DataFrame([{"test": k, "passed": v} for k, v in SELFTEST.items()]).to_csv(os.path.join(OUT_DIR, "selftests.csv"), index=False)

# %%
syn_tab = pd.concat([SYN[s]["main"].assign(scenario=s) for s in SCENARIOS])
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for scen in SCENARIOS:
    t = syn_tab[syn_tab.scenario == scen]
    ax[0].errorbar(np.log2(t.m), t.delta_est, [t.delta_est - t.delta_lo, t.delta_hi - t.delta_est], fmt="o-", capsize=3, label=scen)
    t2 = t[t.m >= 2]
    ax[1].errorbar(np.log2(t2.m), t2.gap_est, [t2.gap_est - t2.gap_lo, t2.gap_hi - t2.gap_est], fmt="o-", capsize=3, label=scen)
ax[0].set(xlabel="log2 pool size m", ylabel="Δ(m) = mean(c − a) at the winner", title="Module B: Δ(m) by scenario")
ax[1].set(xlabel="log2 pool size m", ylabel="Δ_actual − Δ_shuffled", title="Module B: shuffle gap by scenario")
for a in ax:
    a.axhline(0, c="k", lw=0.8); a.grid(alpha=0.3); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "moduleB_scenarios.png"), dpi=300); plt.show()

# %% [markdown]
# ## Modules C–F on the configured datasets
# The datasets are processed one at a time, for each dataset × mask kind: load (pruned and downcast), check, sort, then pools and the bootstrap. Every reported number comes from report-half documents. Calibrators are fit on the fit half.
# * C: E[c_w], E[a_w], Δ(m), and ℓ inflation (E[ℓ_w] − the state's mean ℓ, the expected ℓ of a random position).
# * D: shuffle gap and ICC.
# * E: paired contrast.
# * F: margin quintiles. Quintile edges come from the report-half actual winners' margins at each m (a covariate only, no outcome); the shuffled winners use the same edges.

# %%
RESULTS = {}
for name, spec in DATASETS.items():
    if not FILES.get(name) or name not in FEASIBLE or not FEASIBLE[name]["C"][0]:
        print(f"skip {name}: not feasible (Phase 1)")
        continue
    fm = FMAP[name]
    need_margin = FEASIBLE[name]["F"][0]
    for kind in MASK_KINDS_RUN:
        t0 = time.time()
        cols = read_kind(FILES[name], fm, kind, model=spec.get("model"), need_margin=need_margin)
        if len(cols.get("doc", [])) == 0:
            log_exclusion(name, kind, "load", "mask kind", 1, "no rows of this kind")
            continue
        tr = build_traces(cols, name, kind)
        del cols
        st_half = SPLITS[name](tr.st_doc)
        log_exclusion(name, kind, "split", "states", (st_half < 0).sum(),
                      "document not in the fit/report split")
        res = analyze(tr, st_half, n_boot=N_BOOT_RUN, key="main")
        res["info"]["load_seconds"] = round(time.time() - t0 - res["info"]["seconds"], 1)
        RESULTS[(name, kind)] = res
        i = res["info"]
        print(f"   {name}/{kind}: {i['n_rep_states']:,} report states, {i['n_docs']:,} docs, "
              f"ICC {i['icc']['est']:.4f}, ECE {i['ece']:.4f}, {i['seconds']} s analysis")
        del tr

MAIN = pd.concat([r["main"] for r in RESULTS.values()], ignore_index=True)
PAIRED = pd.concat([r["paired"] for r in RESULTS.values()], ignore_index=True)
QUINT = pd.concat([r["quint"] for r in RESULTS.values()], ignore_index=True) if any(len(r["quint"]) for r in RESULTS.values()) else pd.DataFrame()
MAIN.to_csv(os.path.join(OUT_DIR, "summary_main.csv"), index=False)
PAIRED.to_csv(os.path.join(OUT_DIR, "summary_paired.csv"), index=False)
QUINT.to_csv(os.path.join(OUT_DIR, "summary_quintiles.csv"), index=False)
pd.DataFrame(EXCLUSIONS).to_csv(os.path.join(OUT_DIR, "exclusions.csv"), index=False)
show = ["dataset", "kind", "m", "n_states", "n_docs", "delta_est", "delta_lo", "delta_hi",
        "delta_exact", "delta_shuf_est", "gap_est", "gap_lo", "gap_hi", "icc_est", "m_eff_est", "ece"]
print(MAIN[show].round(4).to_string(index=False))

# %% [markdown]
# ### Summary table per dataset × mask kind (m = 1 and m = 32)

# %%
def fmt(e, lo, hi):
    """'est [lo, hi]' with 4 decimals."""
    return "n/a" if not np.isfinite(e) else f"{e:+.4f} [{lo:+.4f}, {hi:+.4f}]"


SUMMARY = []
for (name, kind), r in RESULTS.items():
    t = r["main"].set_index("m"); p = r["paired"].set_index("m")
    SUMMARY.append({"dataset": name, "kind": kind, "report docs": r["info"]["n_docs"],
                    "report states": r["info"]["n_rep_states"], "ECE": round(r["info"]["ece"], 4),
                    "Δ(1)": fmt(*t.loc[1, ["delta_est", "delta_lo", "delta_hi"]]),
                    f"Δ({M})": fmt(*t.loc[M, ["delta_est", "delta_lo", "delta_hi"]]),
                    f"Δ({M})−Δ(1) paired": fmt(*p.loc[M, ["paired_est", "paired_lo", "paired_hi"]]),
                    f"var. reduction": f"{p.loc[M, 'var_reduction_vs_indep']:.0%}",
                    f"shuffle gap m={M}": fmt(*t.loc[M, ["gap_est", "gap_lo", "gap_hi"]]),
                    "ICC": fmt(*t.loc[M, ["icc_est", "icc_lo", "icc_hi"]]),
                    f"m_eff({M})": f"{t.loc[M, 'm_eff_est']:.1f}"})
SUMMARY = pd.DataFrame(SUMMARY)
SUMMARY.to_csv(os.path.join(OUT_DIR, "summary_table.csv"), index=False)
display(SUMMARY)

# %% [markdown]
# ### Figures (300 dpi PNG in `OUT_DIR/figures`)

# %%
def errbar(ax, x, t, col, **kw):
    """Error bars from est/lo/hi columns."""
    e, lo, hi = t[f"{col}_est"], t[f"{col}_lo"], t[f"{col}_hi"]
    return ax.errorbar(x, e, [e - lo, hi - e], capsize=3, **kw)


for name in dict.fromkeys(k[0] for k in RESULTS):
    kinds = [k for (n_, k) in RESULTS if n_ == name]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for i, kind in enumerate(kinds):
        t = RESULTS[(name, kind)]["main"]; x = np.log2(t.m) + 0.04 * i
        c = f"C{i}"
        errbar(ax[0, 0], x, t, "E_c_w", fmt="o-", c=c, label=kind)
        errbar(ax[0, 1], x, t, "E_a_w", fmt="o-", c=c, label=kind)
        errbar(ax[1, 0], x, t, "delta", fmt="o-", c=c, label=f"{kind} Δ(m)")
        ax[1, 0].axhline(t.ece.iloc[0], c=c, ls="--", lw=1, label=f"{kind} ECE (all positions)")
        errbar(ax[1, 1], x, t, "ell_infl", fmt="o-", c=c, label=kind)
    ax[0, 0].set(ylabel="E[c_winner]", title="raw winner confidence")
    ax[0, 1].set(ylabel="E[a_winner]", title="winner accuracy")
    ax[1, 0].set(ylabel="Δ(m) = E[c − a] at winner", title="calibrated gap (+ = overconfident)")
    ax[1, 1].set(ylabel="E[ℓ_w] − E[ℓ random position]", title="log-odds inflation of the winner")
    for a in ax.flat:
        a.set_xlabel("log2 pool size m"); a.grid(alpha=0.3); a.legend(fontsize=7)
    ax[1, 0].axhline(0, c="k", lw=0.8)
    fig.suptitle(f"Module C: {name} (report half, 95% document-bootstrap CIs)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, f"moduleC_{name}.png"), dpi=300); plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for i, kind in enumerate(kinds):
        t = RESULTS[(name, kind)]["main"]; t = t[t.m >= 2]; x = np.log2(t.m) + 0.04 * i
        errbar(ax[0], x, t, "delta", fmt="o-", c=f"C{i}", label=f"{kind} actual")
        errbar(ax[0], x, t, "delta_shuf", fmt="s:", c=f"C{i}", label=f"{kind} shuffled")
        ax[0].axhline(t.ece.iloc[0], c=f"C{i}", ls="--", lw=1, label=f"{kind} ECE")
        errbar(ax[1], x, t, "gap", fmt="o-", c=f"C{i}", label=f"{kind} (ICC {t.icc_est.iloc[0]:.3f})")
    ax[0].set(xlabel="log2 pool size m", ylabel="Δ(m)", title="actual vs shuffled pools")
    ax[1].set(xlabel="log2 pool size m", ylabel="Δ_actual − Δ_shuffled", title="shuffle gap (within-state dependence)")
    for a in ax:
        a.axhline(0, c="k", lw=0.8); a.grid(alpha=0.3); a.legend(fontsize=7)
    fig.suptitle(f"Module D: {name}"); fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"moduleD_{name}.png"), dpi=300); plt.show()

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, kind in enumerate(kinds):
        p = RESULTS[(name, kind)]["paired"]; x = np.log2(p.m) + 0.05 * i
        ax.plot(x, p.paired_se, "o-", c=f"C{i}", label=f"{kind} paired")
        ax.plot(x, p.unpaired_se_indep, "s:", c=f"C{i}", label=f"{kind} unpaired")
    ax.set(xlabel="log2 pool size m", ylabel="bootstrap SE of Δ(m) − Δ(1)", title=f"Module E: {name}")
    ax.grid(alpha=0.3); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"moduleE_{name}.png"), dpi=300); plt.show()

    qk = [k for k in kinds if len(RESULTS[(name, k)]["quint"])]
    if qk:
        fig, ax = plt.subplots(1, len(qk), figsize=(5.5 * len(qk), 4), squeeze=False)
        for j, kind in enumerate(qk):
            q = RESULTS[(name, kind)]["quint"]
            for i, m in enumerate([mm for mm in M_GRID if mm >= 2]):
                t = q[q.m == m]; x = t.quintile + 0.05 * i
                errbar(ax[0, j], x, t, "gap", fmt="o-", c=f"C{i}", label=f"m={m}")
            ax[0, j].axhline(0, c="k", lw=0.8)
            ax[0, j].set(xlabel="winner margin quintile (1 = weakest)", ylabel="shuffle gap",
                         title=f"Module F: {name}/{kind}")
            ax[0, j].grid(alpha=0.3); ax[0, j].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, f"moduleF_{name}.png"), dpi=300); plt.show()

# %% [markdown]
# ## Provenance and exclusions

# %%
RUN_META = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "versions": VERSIONS,
            "config": {k: globals()[k] for k in ("USE_SYNTHETIC", "MASK_KINDS", "M_GRID", "POOLS_PER_STATE",
                                                 "N_BOOT", "SEED", "MIN_STATE_SIZE", "RATE_BINS",
                                                 "N_QUINTILES", "ECE_BINS", "TOY_N")},
            "n_boot_run": N_BOOT_RUN, "phase1_verdict": PHASE1_VERDICT,
            "datasets": {n: {"path": s.get("path"), "hf": s.get("hf"), "split": s["split"], "model": s.get("model"),
                             "files": [{"name": f.name, "bytes": f.stat().st_size,
                                        "rows": pq.ParquetFile(f).metadata.num_rows} for f in FILES.get(n, [])]}
                         for n, s in DATASETS.items()},
            "selftests_passed": SELFTEST_ALL, "toy_checks_passed": TOY_CHECKS_ALL,
            "info": {f"{k[0]}/{k[1]}": v["info"] for k, v in RESULTS.items()},
            "git_commit": None, "note": "notebook run outside the repo; record the repo commit of this notebook by hand"}
with open(os.path.join(OUT_DIR, "run_meta.json"), "w") as fh:
    json.dump(RUN_META, fh, indent=1, default=str)
ex = pd.DataFrame(EXCLUSIONS)
print(ex[ex["count"] > 0].to_string(index=False) if len(ex) else "no exclusions")
print("outputs in", OUT_DIR, ":", sorted(os.listdir(OUT_DIR)))

# %% [markdown]
# ## Results in plain language
# The next cell writes this summary from the numbers above; nothing in it is typed by hand. "CI" means the 95% document-bootstrap percentile interval.

# %%
side = lambda lo, hi: "excludes 0" if (lo > 0 or hi < 0) else "includes 0"
lines = []
if USE_SYNTHETIC:
    lines.append("**SYNTHETIC RUN: these are Module B test scenarios, not results about any model.**\n")
lines.append(f"Phase 1 verdict: **{PHASE1_VERDICT}**. Self-tests: {sum(SELFTEST.values())}/{len(SELFTEST)} passed; "
             f"toy checks: {sum(TOY_CHECKS.values())}/{len(TOY_CHECKS)} as expected. Bootstrap resamples: {N_BOOT_RUN:,}"
             + (" (dry run)." if N_BOOT_RUN < N_BOOT else "."))
tl = TOY_LOCAL.iloc[-1]
d1 = TOY_DEC[(TOY_DEC.L == 6) & (TOY_DEC.Q == 1)].iloc[0]
lines.append(f"\n**Toy (Module A).** In the Gaussian matched filter, Q × local bias reaches {tl.Qbias_cv:.3f} ± {1.96 * tl.Qbias_cv_se:.3f} at Q = {int(tl.Q)} "
             f"(theory 0.5). At Q = 1 the global search adds a remote term of {d1.B_remote:.3f} to the local bias of {d1.B_local:.3f}, and "
             f"the remote side wins {d1.P_remote_wins:.0%} of the time (L = 6). This checks the code; it says nothing about language models.")
for (name, kind), r in RESULTS.items():
    t = r["main"].set_index("m"); p = r["paired"].set_index("m"); i = r["info"]
    a1, aM = t.loc[1], t.loc[M]; pm = p.loc[M]
    lines.append(f"\n**{name} / {kind}** ({i['n_docs']:,} report documents, {i['n_rep_states']:,} states, ECE {i['ece']:.3f}).")
    lines.append(f"- The winner's raw confidence rises from {a1.E_c_w_est:.3f} (m = 1) to {aM.E_c_w_est:.3f} (m = {M}), "
                 f"and its log-odds sit {aM.ell_infl_est:.2f} above the state's average position. That is the expected inflation of a maximum.")
    lines.append(f"- The calibrated gap Δ goes from {a1.delta_est:+.4f} [{a1.delta_lo:+.4f}, {a1.delta_hi:+.4f}] at m = 1 to "
                 f"{aM.delta_est:+.4f} [{aM.delta_lo:+.4f}, {aM.delta_hi:+.4f}] at m = {M}. The paired change Δ({M}) − Δ(1) = "
                 f"{pm.paired_est:+.4f} [{pm.paired_lo:+.4f}, {pm.paired_hi:+.4f}] (CI {side(pm.paired_lo, pm.paired_hi)}); "
                 + (f"pairing cut its variance by {pm.var_reduction_vs_indep:.0%}." if pm.var_reduction_vs_indep > 0 else f"pairing did not reduce its variance here ({pm.var_reduction_vs_indep:+.0%})."))
    if np.isfinite(aM.gap_est):
        lines.append(f"- Shuffle test at m = {M}: actual pools {aM.delta_est:+.4f}, shuffled pools {aM.delta_shuf_est:+.4f}, gap "
                     f"{aM.gap_est:+.4f} [{aM.gap_lo:+.4f}, {aM.gap_hi:+.4f}] (CI {side(aM.gap_lo, aM.gap_hi)}). "
                     + ("A gap that excludes 0 means a state's positions are not exchangeable with positions from other states: "
                        "errors, or confidence shifts, are shared within a state. It does not by itself say which."
                        if not covers(aM.gap_lo, aM.gap_hi) else
                        "No sign that a winner behaves differently next to its own state's competitors than next to strangers."))
    else:
        lines.append(f"- Shuffle test at m = {M}: not available (see exclusions).")
    lines.append(f"- Residual ICC {i['icc']['est']:.4f} [{i['icc']['lo']:.4f}, {i['icc']['hi']:.4f}], so a pool of {M} behaves like "
                 f"{aM.m_eff_est:.1f} independent positions (Kish).")
    q = r["quint"]
    if len(q):
        qm = q[q.m == M]
        lines.append(f"- Margin quintiles at m = {M}: Δ runs " + ", ".join(f"{v:+.3f}" for v in qm.delta_est)
                     + " from the weakest to the strongest winners; shuffle gaps " + ", ".join(f"{v:+.3f}" for v in qm.gap_est) + ".")
lines.append("\nCaveats: the CIs are conditional on the pool draws (20 per state); pool-sampling noise is small next to the "
             "document bootstrap but is not included. A nonzero shuffle gap can come from shared errors or from state-level "
             "shifts in confidence (Module B scenario 4); a zero gap does not rule out shared errors that are independent of "
             "confidence (scenario 4b, which the ICC does detect).")
SUMMARY_MD = "\n".join(lines)
with open(os.path.join(OUT_DIR, "summary.md"), "w") as fh:
    fh.write(SUMMARY_MD)
display(Markdown(SUMMARY_MD))
