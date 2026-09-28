# C3: Committed Confidence Calibration

Selection bias in confidence-based token commitment in diffusion language models.
ANLP Monsoon 2026, team **DuckDuckDuckGoose**.

Specification: [`documents/DuckDuckDuckGoose-Proposal.pdf`](documents/DuckDuckDuckGoose-Proposal.pdf).
Schedule: [`documents/project-timeline.md`](documents/project-timeline.md).
Cluster facts and their consequences: [`slurm/CLUSTER.md`](slurm/CLUSTER.md).

## Project links

- **Weights & Biases:** https://wandb.ai/ANLP-project/c3-dllm-calibration
  (110M training run: `train110m`)
- **HuggingFace:** not used yet. The 110M checkpoint and the released traces are
  still on Ada only — see [`artifacts/README.md`](artifacts/README.md) for where.
  The proposal commits to releasing the traces, so this needs an upload before
  the final submission.

**Standing requirement, unverified:** the guidelines require this repository to be
public *or* to grant the TA mentors access. Someone should confirm which, once.

Large artefacts (trained model, shards, corpus) live on Ada and are inventoried
in [`artifacts/README.md`](artifacts/README.md). Everything small enough to
version — the gate decisions, the exclusion manifest, the pilot traces — is in
`artifacts/`.

## What is here

The measurement instrument for C1 and C2, the Stage 0 pilot and its written gate,
and the 110M training stack — everything the mid submission needs to demonstrate
as progress. The write-up itself is not here and must be written by hand: course
guidelines allow AI for implementation and code but **not** for generating reports.

```
c3/
  order_stats.py    z_{r,m}, the expected normal order statistic in Eq. 2
  splines.py        restricted cubic spline for the flexible control f(ell)
  design.py         post-hoc pool draws and ranks with random tie-breaking
  logit.py          logistic IRLS with high-cardinality state fixed effects
  frame.py          pool-row and position-level tables; fixed document halves
  regression.py     Eqs. 2-3 with and without state FE, cross-fitting, bootstrap
  stats.py          document-first means, shared doc-bootstrap, Holm, declare rule
  metrics.py        equal-mass ECE, smooth ECE, reliability bins, isotonic (PAV)
  calibrators.py    pi_hat(l) and the context-aware pi_hat(l, s), cross-fitted
  endpoints/        E1 Delta(m) (sign conf - agree) and E2 field-term decomposition
  sim/              position-level traces with known truth (test fixture)
  provenance.py     git commit, versions, GPU, SLURM job for every output
  calibration.py    proposal-era commitment gap (old sign; kept for the records)
  dispersion.py     dispersion of position-specific noise (proposal-era gate input)
  power.py          plan v2 sample size (7.85 sigma_d^2 / delta_min^2); old gate
  simulate.py       proposal-era synthetic DGPs with known truth
  assemble.py       stored traces -> analysis tables
  traces/           masks, model adapters, the collection harness
  data/             FineWeb-Edu sampling and the exclusion manifest
  training/         110M MDLM: model, loss, resumable loop, compute budget
scripts/            command-line entry points
slurm/              Ada job scripts and cluster documentation
tests/              pytest suite
```

## Design decisions worth knowing

**Traces are stored at position level; pools are drawn at analysis time.** One row
per (masked state, masked position) rather than per pool membership. That is ~4M
rows for the full corpus instead of hundreds of millions, and the pool design
becomes a free parameter — sizes, counts and seeds can change without re-running a
forward pass. It is also what makes the proposal's promise to release the traces
cheap to keep.

**Cross-fitting is how "fit on half 1, report from half 2" is implemented.** The
confidence-accuracy curve `f(ell)` is estimated on the fit half and carried into
the eval half as a fixed offset, so the deliberately flexible control cannot be
bent on the same rows that produce the reported coefficient.

**Ranks are tie-broken at random.** Reported confidence saturates in low precision,
so several positions can share the clipped maximum. Letting array order decide the
winner would manufacture a positional artefact in exactly the coefficient under
audit. Clipping is counted and reported rather than hidden.

**The bootstrap resamples documents by reweighting, not by rebuilding the design.**
Same estimating equation, far cheaper, and it is what makes 10,000 resamples
affordable at full scale.

## Running it

```bash
# environment (local, analysis only; add the "models" extra for torch)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev,analysis]"

.venv/bin/python -m pytest tests/ -q      # torch-dependent tests skip without torch

# plan v2 endpoints on a trace (or on a synthetic trace with known truth)
.venv/bin/python scripts/analyze_traces.py \
    --traces artifacts/traces/stage0_llada8b_4bit_n20.parquet \
    --split kfold-doc --eq-boot 2000 --n-jobs 8 --out artifacts/v2_pilot --macro-prefix Pilot
.venv/bin/python scripts/analyze_traces.py --traces sim:context --split half --out /tmp/sim
# (E1/E2 bootstraps take seconds; the Eq. 2/3 bootstrap refits a logistic model per
#  resample, so --eq-boot and --n-jobs control most of the runtime.)

# the week 3-4 requirement: validate the estimator on known truth
.venv/bin/python scripts/validate_estimator.py --out artifacts/estimator_validation.json

# compute budget for the 110M arm, on this cluster's actual GPUs
.venv/bin/python scripts/train_110m.py --budget-only
```

On Ada (see [`slurm/CLUSTER.md`](slurm/CLUSTER.md) first):

```bash
bash slurm/submit_t3.sh                     # T3 end to end, env build included (see the runbook)
sbatch slurm/build_env.sbatch               # the env alone, if you only need trace collection
sbatch --export=ALL,C3_MODEL=llada-8b,C3_LIMIT=20 slurm/collect_traces.sbatch
```

Proposal-era Stage 0 (these scripts pin the proposal-era pool design so the
committed `artifacts/stage0_gate_*.json` and `estimator_validation.json` still
reproduce; plan v2 analyses go through `scripts/analyze_traces.py`):

```bash
python scripts/collect_traces.py --model llada-8b --limit 20 --repeat 5 \
    --docs runs/corpus/calibration_docs.jsonl --out runs/traces/pilot.parquet
python scripts/measure_floor.py --traces runs/traces/pilot.rep*.parquet
python scripts/run_stage0.py --traces runs/traces/pilot.rep0.parquet --floor <from above>
```

`run_stage0.py` writes the gate decision to `artifacts/stage0_gate.json`. Commit it:
the proposal pre-registers the gate, so the decision has to be recorded before the
full study rather than reconstructed after it.

## Stage 0 pilot: first results on LLaDA-8B

20 documents, 4-bit, RTX 2080 Ti, 160 masked states. Pilot-scale and fitted pooled
rather than cross-fitted, so these are feasibility estimates, not results. Full
output in `artifacts/stage0_gate_*.json`.

> **Sign convention.** This section and `artifacts/stage0_gate_*.json` use agree − conf, so a *negative* gap means overconfidence. Every new output use conf − agree (positive = overconfident). The pilot's checkpoint variant (Base or Instruct) was not recorded and is being verified.

| | uniform masks | decoder-induced masks |
|---|---|---|
| `beta_z` | +0.061 `[-0.024, +0.142]` | +0.149 `[-0.070, +0.351]` |
| noise dispersion | **0.000** (9/10 strata show no attenuation) | **0.428** (3 strata attenuated) |
| docs for 80% power | 75 | 82 |
| **pre-registered gate** | **RESTRICT** | **PROCEED** |
| commitment gap slope in log2(m) | +0.002 (flat) | **-0.008** (widening) |

**The two mask distributions give opposite gate decisions, and that is the
finding.** Under uniform masks -- the ones matching training -- LLaDA's confidence
log-odds track correctness essentially one-for-one (attenuation slope 1.12-1.24,
i.e. above 1), so there is no detectable position-specific approximation error for
selection to act on, and Sec. 5.1 says the question is then empty. Under
decoder-induced masks, which is the regime inference actually runs in, dispersion
is 0.428 and the commitment gap widens with field size overall, though not
monotonically (-0.013 at m=2, -0.018 at m=4, -0.012 at m=8, -0.049 at m=32; the
m=32 interval almost excluding zero).
That is the winner's-curse signature, appearing only where the decoder operates.

Read with care. Every interval includes zero at n=20. `beta_z` is **positive** in
both arms while a winner's curse predicts negative, and +0.149 sits near the
spurious band the synthetic validation maps out below. The honest summary is that
the decoder arm is where the effect could live and is worth the full sample; the
uniform arm looks empty.

## Two findings from the validation, before any model was touched

Both come from `scripts/validate_estimator.py` and both change how the audit
should be run. They are the kind of thing the synthetic check exists to catch.

**1. The spline's flexibility is not innocuous.** Under a null DGP with *no* rank
effect but heteroskedastic approximation error, a 5-knot control for `f(ell)`
leaves residual curvature that projects onto `z_{r,m}` and produces a spurious
`beta_z` of roughly +0.12 — the same order as any effect the study hopes to find.
It shrinks toward zero as knots increase, but at high dispersion it overshoots to
*negative* values, which would read as a winner's curse. The knot count must be
fixed in advance, reported with a sensitivity panel, and the effect bar must clear
the spurious magnitude at the dispersion Stage 0 actually measures.

**2. Dispersion is only estimable when it tracks something observable.** Sec. 5.1
makes dispersion the precondition for the whole audit, and the gate turns on it.
The estimator recovers it cleanly when the noise scale varies with position, and
returns ~0.1 whether the true dispersion is 0 or 1.2 when the scale varies
idiosyncratically with no observable correlate. So the gate's dispersion criterion
is a statement about *visible* heterogeneity and is a lower bound. Worth raising
with the TA mentor: a "restrict" decision may reflect an unidentified nuisance
rather than a genuinely homoskedastic model.

## Status

Done: estimator and its validation; trace harness; pool design; Stage 0 gate
arithmetic; dispersion estimation; 110M model and resumable trainer; Ada job
scripts; test suite. **On the cluster:** the 2,000-document calibration corpus is
sampled and its exclusion manifest verified (hashes check against content), and
the Stage 0 pilot has run on LLaDA-8B in 4-bit with both gates recorded.

Not done: the 110M training has not started (decide the token budget first --
`scripts/train_110m.py --budget-only`); Dream-7B has not been run; the full 2,000
document C1 sweep has not been launched; the mid write-up is unwritten by design.
