# Artifacts

Small, reproducibility-critical outputs, tracked deliberately. Anything large
lives on Ada — see the inventory at the bottom.

| file | what it is | produced by |
|---|---|---|
| `estimator_validation.json` | Eq. 2 estimator checked against known truth: null DGP, planted effects, and the knots × dispersion grid | `scripts/validate_estimator.py` |
| `nondeterminism_floor.json` | trace-level non-determinism floor (measured: **exactly zero**, repeats bitwise identical) | `scripts/measure_floor.py` |
| `stage0_gate_uniform.json` | Stage 0 written gate, uniform masks → **RESTRICT** | `scripts/run_stage0.py` |
| `stage0_gate_decoder.json` | Stage 0 written gate, decoder-induced masks → **PROCEED** | `scripts/run_stage0.py` |
| `exclusion_manifest.json` | the 2,000 content hashes removed from the 110M training stream | `scripts/prepare_corpus.py` |
| `train_summary.json` | provenance of the trained 110M model (76,293 steps, 10.0B tokens) | `scripts/train_110m.py` |
| `pilot/*.parquet` | position-level traces, LLaDA-8B 4-bit, 20 documents (schema v1; a feasibility run, not the T2 pilot) | `scripts/collect_traces.py` |
| `task_manifest.json` | GSM8K / MBPP / HumanEval test splits: hub revision, record count and sha256 of the JSONL | `scripts/prepare_tasks.py` |
| `t0/` | T0: the four-scenario table (`table.json`, `table.tex`), Figure 1 (`figure1.pdf/png`), the true field term against m at five noise levels (`field_growth.png`), and `macros.tex` (prefix `TZero`); 200 simulated documents, 10,000 resamples, fit half -> report half | `scripts/t0_scenarios.py` |
| `trace_size_estimate.json` | rows, forward passes and parquet bytes per document for schema v2, and the disk the plan needs per variant (top-20, top-10, top-20 with fp16 probabilities) | `scripts/estimate_trace_size.py` |
| `exclusion_v2.json` | v2 held-out record: FineWeb id + text hash + first-256-token hash for the 2,000 main and 500 reserve documents (job 2722135); the v2 110M shards matched all 2,500 | `slurm/prepare_exclusion.sbatch` |
| `eval_110m_train110m.json` | proposal-era 110M (`/share1/ojas.k/train110m`): perplexity bound 38.4 [37.7, 39.2] on the 2,000 held-out docs vs GPT-2 small 28.5, plus 4 samples; no G2 (its history.jsonl was not kept) | `scripts/eval_110m.py` (job 2722239) |
| `contamination_llada-8b-base.json` | T4 screen, LLaDA-8B-Base fp16: 2 / 1,979 flagged at a run of >= 32 of 50 tokens (threshold from plan v2 T4, **not yet pre-registered**); median run 1, max 39 | `scripts/contamination_screen.py` (job 2722128) |
| `t2_pilot/` | T2 pilot, LLaDA-8B-Base fp16 over 2 GPUs, reserve docs 100000-100020 (20 with >= 256 tokens): `floor_llada_base_fp16.json` (5 repeats bitwise identical, floor 0), `gate_g1.json` (N = 20 -> full plan), `analysis/` (kfold-doc, prefix `TTwoPilot`). Traces: `/share1/ojas.k/traces/t2pilot-llada-base-fp16/` | `slurm/collect_traces.sbatch` (job 2722140), `scripts/measure_floor.py`, `scripts/gate_g1.py`, `scripts/analyze_traces.py` |

The gate files are committed on purpose: the proposal pre-registers the Stage 0
decision, so it has to be recorded before the full study rather than
reconstructed after it.

## What is on Ada, and why it is not here

`ojas.k@ada`. `/share1` is **not mounted on compute nodes** but is reachable from
them over ssh — see `slurm/CLUSTER.md`.

| artefact | location | size |
|---|---|---|
| calibration corpus (2,000 docs) | `/home2/ojas.k/c3-runs/corpus/calibration_docs.jsonl` | 11 MB |
| tokenised training shards (9.94B tokens, 50 shards) | `/share1/ojas.k/shards/` | 19 GB |
| **110M model** — `final.pt`, `checkpoint.pt`, summary | `/share1/ojas.k/train110m/` | 1.9 GB |
| prebuilt Python environment (tarball) | `/share1/ojas.k/c3-env.tar.gz` | 3.1 GB |
| job logs | `/home2/ojas.k/c3/logs/` | small |

The corpus is regenerable from a fixed seed (`CorpusSpec(seed=20260917)`), but
only while the upstream FineWeb-Edu stream order is stable — treat
`exclusion_manifest.json` here as the authoritative record of what was held out.
Regenerated off-cluster on 2026-09-28 with `scripts/prepare_corpus.py --out-dir
runs/corpus`, its manifest was byte-identical to the committed one (same 2,000 ids
and content hashes), so the stream order had not changed.

## Provenance

| run | SLURM job | outcome |
|---|---|---|
| corpus sampling | 2700106 | 2,000 docs from 200,000 streamed |
| Stage 0 pilot (5 repeats) | 2700123 | floor = 0, bitwise identical |
| Stage 0 pilot (fixed decoder masks) | 2700160 | traces in `pilot/` |
| shard build | 2700429 | 9.94B tokens, 2,157 held-out docs excluded |
| 110M training | 2700435 | 19 h, reached step 76,000 then crashed in NCCL teardown |
| 110M training (resumed) | 2704812 | completed 76,293 steps / 10.0B tokens |

**Weights & Biases:** https://wandb.ai/ANLP-project/c3-dllm-calibration
