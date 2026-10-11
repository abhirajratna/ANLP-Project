# C1 sweep on the 8B models: status

Branch `c1-sweep-8b` (from `origin/main` at `dc65f07`, which contains `f816d87`).
Last updated Mon 12 Oct 2026, 02:30 IST.

## Blocker: no GPU job can run yet

`abhiraj.ratna` has **no SLURM association** on Ada. `sacctmgr show assoc where
user=abhiraj.ratna` returns nothing, the cluster enforces associations, and every
`sbatch`/`srun` fails with `Invalid account or account/partition combination specified`.
Of the team accounts only `ojas.k` has one (`research`, QOS `medium`). The Ada admins
have to add `abhiraj.ratna` to `research` with QOS `medium`. No trace has been collected
and no job has been submitted. `logs/c1_jobs.txt` does not exist yet.

The account itself is set up. The laptop key is installed on `ada-gw1`. `ada.iiit.ac.in`
no longer accepts SSH, which broke every job script, and that is now fixed. Ada has its
own key in `authorized_keys`, `/home2/abhiraj.ratna/c3` is at the branch head, and these
are in place with matching hashes:

- `calibration_docs.jsonl`: 2,000 lines, `af88…8b07`
- `reserve_docs.jsonl`: 500 lines, `1ae2…9628`
- `precision_docs.jsonl`: `a6c4…5590`
- the task JSONLs, which match `artifacts/task_manifest.json`

`/share1` has 0 of 100,000 MB in use, and `/home2` has 0 of 50 GB.

## Once the association exists (in order, on `ada-gw1`, in `~/c3`)

```bash
bash slurm/submit_c1.sh env                   # env tarball, ~30 min, 1 GPU
bash slurm/submit_c1.sh smoke                 # after env: 6 runs x 5 docs, full design
bash slurm/submit_c1.sh floor dream-7b-base   # Dream's T1 floor (LLaDA's is 0)
bash slurm/submit_c1.sh main                  # (a)-(d), priority order, 8 array tasks
bash slurm/submit_c1.sh precision             # (f), after (d) is running
bash slurm/submit_c1.sh merge <run>           # per finished run -> artifacts/c1/<run>.json
```

Smoke checks: passes/s in each log's `[collect] throughput` line, no `SKIPPED document`
lines (the fp16 guard), rows per document against `scripts/validate_run.py`'s closed
form, and `[model] … fp16 over ['0', '1']`. Instruct formatting was checked off-GPU with
the pinned tokenizers: the chat template is applied, the question is fully visible, and
the maskable span is exactly the reference answer plus `<|eot_id|>` / `<|im_end|>`.
Dream's remote code is bidirectional (`is_causal=False`, hard-coded), and its mask id
151666 matches the registry.

ETA at the measured 7.6 passes/s (Dream not yet measured):

| run | documents | slot-hours |
|---|---|---|
| (a) LLaDA-8B-Base, FineWeb | 1,979 | 33.7 |
| (b) LLaDA-8B-Instruct, GSM8K / MBPP | 1,319 / 500 | 9.0 / 2.5 |
| (c) Dream-7B-Base, FineWeb | 1,976 | ~33.7 |
| (d) Dream-7B-Instruct, GSM8K / MBPP | 1,319 / 500 | ~9.0 / 2.5 |

That is about 90 slot-hours, or ~46 h of wall time on two slots. Adding env and smoke
gives ~48 h from the moment jobs can run. To have (a)–(d) merged by Fri 16 Oct the
association is needed by about **Wed 14 Oct, 10:00**. If it comes later, the planned
fallback is `C3_DREAM_NO_MATCHED=1` for the Dream runs (242 instead of 466 passes per
document, ~16 slot-hours saved), recorded as a deviation. The main runs never move to
4-bit.

(e) is skipped for now: branch `t3-110m-probe` is not on GitHub yet, so there is no
`timesplit_docs.jsonl`. The laptop has a 29 Sep `runs/corpus/timesplit_docs.jsonl`
(`9902…f903`) from an earlier session; it is not the one the other agent is building, so
it was not used.

## Done on this branch

| commit | what |
|---|---|
| `566ecee` | jobs reach `/share1` via `ada-gw1/gw2` (`slurm/login_host.sh`); test |
| `28fa6e3` | **E3 (T7)**: `c3/endpoints/e3.py`, `scripts/analyze_e3.py`, `c3/sim/trajectories.py`; tests |
| `f8afcbf` | **T8 controls**: `c3/controls.py`, `scripts/run_controls.py`, one ablation table; tests |
| `f71f9f6` | zero-copy merge (`<run>.dataset.json`) and `scripts/validate_run.py`; tests |
| `11278fe` | an overflowing (inf/NaN) document is skipped and flagged instead of killing the shard; test |
| `6abb41b` | `artifacts/split_manifest.json` (1,000/1,000), `c3.CONTAM_THRESHOLD = 32`, T8 precision subset |
| `0f8abc7` | `slurm/submit_c1.sh`, `slurm/merge_validate.sbatch`, `--merge-as` |
| `f202f8a` | `prereg/analysis_commands.sh`, `prereg/deviations.md`, task splits, `summarize_c1.py` (Holm), extended `freeze_prereg.py` |
| `680761e` | CLUSTER.md and artifacts/README.md |

All 234 tests pass, including `test_prereg_script_parses`, which was failing on `main`.
E3 and T8 have been run only on synthetic traces and on a toy-model trace from the real
harness. No real trace exists yet.

Why the merge is zero-copy: the parts for (a)–(f) come to ~64 GB, and a merge that copies
every run would exceed `/share1`'s ~98 GB quota.

## Pre-registration: ready, not yet tagged

The split manifests, the threshold, `prereg/analysis_commands.sh` and
`prereg/deviations.md` are committed. `scripts/freeze_prereg.py` reads every choice from
the code. The tag `prereg-v1` is held back on purpose: the brief asks for E3/T8 to be
developed on the fit half of real traces first, and a code fix after the tag would be a
deviation. Plan: tag once LLaDA-Base's fit half has been run through E1/E2/E3/T8 (or at
the end of the chunk, whichever is first), then `freeze_prereg.py` on a clean tree,
commit, tag, push. The analysis script refuses to run without `C3_REPORT_HALF_OK=1`.

## Decisions that are yours

1. Get the SLURM association added (above).
2. Confirm before the OSF upload (listed under `open` in the frozen config):
   - Holm families on arms `uniform` + `decoder:topk_conf:k4` (k = 2, 8 and matched exploratory)
   - the interacted context calibrator
   - the task splits
   - softmax temperature for T8
3. Deviations to accept (`prereg/deviations.md`):
   - the threshold went into code after the screens were seen
   - fp16, not bf16
   - E3 for top-k decoders only
   - a floor per model family
4. Branch housekeeping: the main checkout is now on `c1-sweep-8b`. The uncommitted 29 Sep
   work that was on `Abhiraj` is in `git stash list` as `stash@{0}` ("29-Sep uncommitted
   work…"). `git stash branch <name> stash@{0}` restores it intact on its own branch.
   Parts of it (E3, controls, summarize_c1, the stop-word list, the softmax temperature)
   were reviewed and reused here.
