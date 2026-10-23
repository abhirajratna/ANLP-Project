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
| `contamination_llada-8b-base.json` | T4 screen, LLaDA-8B-Base fp16: 2 / 1,979 flagged at a run of >= 32 of 50 tokens (`c3.CONTAM_THRESHOLD`; fixed in code after this screen ran, see `prereg/deviations.md`); median run 1, max 39 | `scripts/contamination_screen.py` (job 2722128) |
| `contamination_dream-7b-base.json` | T4 screen, Dream-v0-7B-Base fp16: 1 / 1,976 flagged at >= 32 (same threshold and caveat); median run 1, max 43 | `scripts/contamination_screen.py` (job 2722129) |
| `t2_pilot/` | T2 pilot, LLaDA-8B-Base fp16 over 2 GPUs, reserve docs 100000-100020 (20 with >= 256 tokens): `floor_llada_base_fp16.json` (5 repeats bitwise identical, floor 0), `floor_dream-7b-base_fp16.json` (same design, Dream-7B-Base, job 17555, 12 Oct: bitwise identical, floor 0; traces `/share1/abhiraj.ratna/traces/t2floor-dream-7b-base-fp16/`), `gate_g1.json` (N = 20 -> full plan), `analysis/` (kfold-doc, prefix `TTwoPilot`). Traces: `/share1/ojas.k/traces/t2pilot-llada-base-fp16/` | `slurm/collect_traces.sbatch` (job 2722140), `scripts/measure_floor.py`, `scripts/gate_g1.py`, `scripts/analyze_traces.py` |
| `split_manifest.json` (+ `.meta.json`) | the frozen fit/report split of the 2,000 FineWeb documents: `{doc_id: 0 fit, 1 report}`, 1,000 / 1,000, seed 20260917; the 500 reserve documents are the separate pilot file | `scripts/make_split_manifest.py` |
| `split_manifest_gsm8k.json`, `split_manifest_mbpp.json` | the same rule on the task test sets (660 / 659 and 250 / 250) | `scripts/make_task_split.py` |
| `t8_precision_docs.json` | the 200 T8 precision documents (100 fit, 100 report; fixed seed) and the subset file's sha256 | `scripts/make_precision_subset.py` |
| `c1/<run>.json` | per-run validation and summary of the C1 sweep (documents, rows, passes, wall time, GPU-hours): run diagnostics, **not results** | `scripts/validate_run.py` via `slurm/merge_validate.sbatch` |
| `gate_g2_train110m-v3.json` | **Gate G2 PASS** for the plan-v2 110M (`train110m-v3`): EMA validation NELBO changes 0.37% over the last 10% of steps (threshold 0.5%, 18 points); final 3.612 nats/token, bound 37.0 | `scripts/check_g2.py` in `slurm/verify_110m.sh` (job 17434) |
| `eval_110m_train110m-v3.json` | the same model on the 2,000 held-out docs (EMA weights): perplexity bound 37.25 [36.54, 37.99] vs GPT-2 small 28.49 [27.98, 29.02]; 4 samples | `scripts/eval_110m.py` (job 17434) |
| `history_train110m-v3.jsonl`, `train_summary_train110m-v3.json` | its training curve and provenance: 169,055 steps, 9.995B tokens, config 53f70713a312b3b2; fresh start on 4 GPUs x batch 16 x accum 4 (the v2 global batch), document-aligned shards, no packing | `scripts/train_110m.py` (job 17431, 36.6 h) |
| `figures/train110m-v3_curve.{pdf,png}` | the v3 validation curve (EMA and raw) and G2's tail window with its fitted line: G2 passes at 0.37%, but the EMA NELBO is still falling ~0.013 nats over the last 1B tokens | `scripts/plot_training.py` |
| `traces_110m/` | C1 traces of the plan-v2 110M (`train110m-v3`, fp32, batch 8, full plan v2 design: 9 rates x 2 uniform masks, top-k decoder states for k = 2, 4, 8, matched uniform states): run metas and validation. `main-110m-v3`: 1,979 calibration docs (the other 21 are shorter than 256 GPT-2 tokens and were left out), 119,562,570 rows; `timesplit-110m-v3`: 500 docs, 30,208,673 rows. Both: decoder and matched rows exactly the closed form, uniform rows within |z| < 3.6, 60,416 rows per 256-token doc (estimate 60,390). No calibration number computed | `slurm/collect_traces.sbatch` (jobs 20752, 20753), `slurm/merge_traces.sbatch` (21271, 21260) |
| `timesplit_docs.jsonl`, `timesplit_manifest.json` | T4 step 2: 500 English Wikipedia articles created 2025-04-04 .. 2026-08-29 (after both 8B releases), current-revision text with revision ids; every doc reaches 256 tokens under GPT-2, LLaDA-8B and Dream-7B. doc_ids 200000-200499 (relabelled from 100000+i, which collided with the reserve set; `scripts/relabel_timesplit.py`, both hashes in the manifest) | `scripts/build_timesplit.py` (run locally 2026-10-12, 567 probes) |
| `exclusion_timesplit.json` | v2 exclusion record of the time-split set (`external`: not required to match the FineWeb stream) | `scripts/build_exclusion.py` |
| `shard_check_exclusion_timesplit.json` | the rebuilt v2 shards (built before the time-split set existed) checked against it: 0 of 9,669,362 train+val documents match by first-256-token hash | `scripts/check_shard_exclusion.py` in `slurm/push_shards.sbatch` (job 17430) |
| `probe/spec.json`, `probe/eval_instances.jsonl`, `probe/manifest.json` | T9 JSON probe: 200 SchemaStore schemas (commit 5e44c8e), 2,415 fields whose values are each one token in context under GPT-2, LLaDA-8B and Dream-7B; marginals, pairs and follower maps; the 1,000 evaluation instances (200 per rho); the sha256 of the 200k training and 2k validation texts, which are regenerated from the spec | `scripts/build_probe.py` |
| `probe/results/grid_<tag>.json`, `steps_<tag>.parquet` | T9 grid rho x k x rule: mean (A), (B), (A)+(B) per instance and per field with document-bootstrap 95% intervals, the crossover k, and the identity check against the direct KL; one row per decoding step in the parquet. `untuned-v1`, `ft110m-v1-dryrun`: the **v1** fallback model, a pipeline dry run (jobs 17340, 17349). Plan-v2 110M (`train110m-v3`): `untuned-v3` (job 20749), `ft110m-v3` (100M tokens, job 20751), `ft110m-v3-visible` (rho marker never masked, 300M tokens, job 20750). `rho_check_<tag>.json`: how far the rho marker moves q(f(leader)) at rho=1 (+0.014 and +0.251; the pre-set bar for the main result was 0.5, **not met by either**) | `scripts/run_probe.py` (`slurm/probe_grid.sbatch`, `slurm/finetune_probe.sbatch`) |
| `probe/figures/` | crossover figures: (A) and (B) per field against k per rule and rho, and the (A)-share map | `scripts/plot_probe.py` |
| `c2/t10_<model>_<precision>_<run>.json` (+ `.text.parquet`, `.probe.parquet`) | T10, C2 on real text: parallel-sequential disagreement (the proxy for (A)) and Delta / CE at the committed positions (proxies for (B)) on the first 300 fit-half FineWeb documents x 10 uniform states (rates 0.1-1.0), per k and per k x rate with document-bootstrap intervals; and the disagreement proxy on every state of the T9 confidence-rule grids, per (rho, k) cell, with its Spearman against the exact (A) per field. `local-110m_fp32_train110m-v3`: the 110M with probes `ft110m-v3-visible` and `ft110m-v3` (job 36091; all 11,165 probe states per model reproduce `steps_<tag>.parquet` bit for bit). `llada-8b-base_fp16_main`: LLaDA-8B-Base fp16 over 2 x 2080 Ti, real text only (job 36106). Parquets: one row per (state, rank) and per probe state | `scripts/run_t10.py` (`slurm/t10.sbatch`) |

The gate files are committed on purpose: the proposal pre-registers the Stage 0
decision, so it has to be recorded before the full study rather than
reconstructed after it.

## What is on Ada, and why it is not here

`ojas.k@ada`. `/share1` is **not mounted on compute nodes** but is reachable from
them over ssh — see `slurm/CLUSTER.md`.

| artefact | location | size |
|---|---|---|
| calibration corpus (2,000 docs) | `/home2/ojas.k/c3-runs/corpus/calibration_docs.jsonl` | 11 MB |
| v2 training shards (document-aligned, 9.94B tokens, 50 + 1 val shards, `SHA256SUMS`) | `/share1/ojas.k/v2/shards/` | 20 GB |
| **plan-v2 110M model** `train110m-v3` — `final.pt` (with EMA), `checkpoint.pt`, history, summary | `/share1/ojas.k/v2/train110m-v3/` | 3.3 GB |
| abandoned v2 run (cancelled 29 Sep at step 11,500; checkpoint at step 10,000) | `/share1/ojas.k/v2/train110m-v2/` | 2.1 GB |
| proposal-era 110M (v1, packed sequences; fallback only) — `final.pt`, `checkpoint.pt`, summary | `/share1/ojas.k/train110m/` | 1.9 GB |
| T9 fine-tuned 110M checkpoints | `/share1/ojas.k/probe/<run>/` | 1.1 GB each |
| prebuilt Python environment (tarball) | `/share1/ojas.k/c3-env.tar.gz` | 3.1 GB |
| **110M C1 traces** (parts; merge with `slurm/merge_traces.sbatch`, ~6 min, onto node SSD: merged next to the parts they exceed the quota) | `/share1/ojas.k/traces/main-110m-v3/`, `/share1/ojas.k/traces/timesplit-110m-v3/` (also merged: `timesplit-110m-v3.parquet`) | 15 GB, 7 GB |
| deleted 2026-10-12/14: the FineWeb-Edu copy (27 GB; jobs now stage it per node, `slurm/stage_fineweb.sh`) and the v1 packed shards (19 GB) | — | — |
| job logs | `/home2/ojas.k/c3/logs/` | small |

`abhiraj.ratna@ada-gw1` (the C1 sweep on the 8B models; `slurm/submit_c1.sh`):

| artefact | location | size |
|---|---|---|
| corpus copy (hashes checked) | `/home2/abhiraj.ratna/c3-runs/corpus/{calibration,reserve,precision}_docs.jsonl`, `tasks/*_test.jsonl` | 15 MB |
| C1 traces, one directory per run: shard parts, shard metas, and the zero-copy `<run>.dataset.json` that `load_traces` reads | `/share1/abhiraj.ratna/traces/<run>/` (`c1-llada-base`, `c1-llada-instruct-{gsm8k,mbpp}`, `c1-dream-base`, `c1-dream-instruct-{gsm8k,mbpp}`, `c1ts-*`, `c1t8-llada-base-4bit`) | ~22 GB per FineWeb arm, ~3 GB per Instruct model; ~64 GB in all |
| Python environment (tarball) | `/share1/abhiraj.ratna/c3-env.tar.gz` | 3.1 GB |
| submission log | `/home2/abhiraj.ratna/c3/logs/c1_jobs.txt` (a copy is committed as `artifacts/c1/c1_jobs.txt`; `logs/` is gitignored) | small |

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

**Weights & Biases:** https://wandb.ai/ANLP-project/c3-dllm-calibration. From 12 Oct the live runs were rejected (HTTP 403: the key on `ojas.k` had lost write access), so these were uploaded afterwards from `history.jsonl` (`scripts/wandb_backfill.py`, tag `backfilled`): `train110m-v3-oct12` (the plan-v2 110M; the id `train110m-v3` belongs to a teammate's 28 Sep run), `ft110m-v1-dryrun`, `ft110m-v3`, `ft110m-v3-visible`. The W&B run `train110m-v2` mixes a teammate's 28 Sep run and our 29 Sep run, which resumed into it under the same id.
