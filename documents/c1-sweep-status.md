# C1 sweep on the 8B models: status

Branch `c1-sweep-8b` (from `origin/main` at `dc65f07`, which contains `f816d87`).
Last updated Thu 15 Oct 2026, 01:25 IST.

## Update, 12 Oct 10:50: unblocked

The SLURM association now exists (`research`, QOS `medium`, checked 10:48). Submitted, all
logged in `artifacts/c1/c1_jobs.txt`:

- env build 17548 (running on gnode043);
- smoke runs 17549-17554 and the Dream floor 17555, all `afterok:17548`.

`main` follows once the smoke logs check out.

**13:00.** The env build took 55 min (PyPI ran at ~94 kB/s). Its tarball reached
`/share1` from the compute node, so the login-host fix works. The LLaDA smoke runs all
pass `validate_run.py` (`artifacts/c1/c1smoke-*.json`):

| smoke run | passes/s | passes | rows | check |
|---|---|---|---|---|
| llada-base, 5 docs | 7.58 | 466 / doc | 60,391 / doc | all ok |
| llada-instruct gsm8k | 8.15 | 682 | 29,539 | all ok |
| llada-instruct mbpp | 7.47 | 492 | 15,671 | all ok |

Fit-half spot checks:

- Base uniform accuracy falls from 0.83 to 0.14 as the mask rate rises, and confidence
  tracks it.
- At step 0 the decoder and matched states are bitwise identical.
- Every task row lies inside the answer span.

Parts take ~7.9 MB per document, so a FineWeb arm is ~16 GB, not 22. Submitted (a)
17716 and (b) 17745/17746. Dream smoke runs 17552/17553 are running.

**13:55.** All six smoke runs pass `validate_run.py`:

| smoke run | passes/s |
|---|---|
| dream-base | 8.38 |
| dream-instruct gsm8k | 9.54 |
| dream-instruct mbpp | 9.07 |

Dream-Base's fit-half accuracy profile matches LLaDA's (uniform 0.81 falling to 0.13,
confidence tracking it), so Dream's logit shift is right. Submitted:

- (c) 17766;
- (d) 17898/17899.

Run (a) shard 0 started at 13:44. ETA at the measured rates is ~84 slot-hours, i.e.
(a)-(d) done around **Wed 14 Oct, 10:00**, before Fri 16 Oct, so the full design stays
for Dream. To be recomputed 2 h into (a).

**14:35.** Dream floor (job 17555, 8.41 passes/s): the 5 repeats are bitwise identical
(same sha256), so the floor is 0, as for LLaDA
(`artifacts/t2_pilot/floor_dream-7b-base_fp16.json`). Both shards of (a) are running,
at 60 docs/hour each.

**17:30, ETA check (run late: the 15:45 check-ins did not fire while the laptop slept).**

Measured on (a): shard 0 at 60.0 docs/h (7.8 passes/s), shard 1 at 56.9 docs/h (7.4
passes/s), at 200/987 and 175/992 documents. The two slots then run:

| when | slot 1 | slot 2 |
|---|---|---|
| Tue ~06:30 / ~07:45 | (a) ends | (a) ends |
| next | (b) gsm8k, until ~15:30 Tue | (b) mbpp, until ~10:45 Tue |
| then | (c) shard 1 | (c) shard 0, ~15 h |
| last | (d) mbpp | (d) gsm8k |

So **(a)-(d) are done around Wed 14 Oct, 10:15**, merges by ~11:30. That is before Fri
16 Oct, so the full design stays for Dream (no matched-arm drop).

**(e) submitted.** `artifacts/timesplit_docs.jsonl` appeared on `t3-110m-probe` (500
documents, ids 200000-200499, sha256 `3374de4d…`, matching its manifest). It is copied to
Ada and hash-checked; jobs 18196 (LLaDA) and 18197 (Dream) queue after (d), ~9 h each, so
done ~Wed evening. Its split manifest is `artifacts/split_manifest_timesplit.json`
(250/250, `make_split_manifest.py --any-ids`, new because these ids are not file
positions). The laptop's 29 Sep `runs/corpus/timesplit_docs.jsonl` (`9902…`) was left
untouched; the t3 file is at `runs/corpus/t3/timesplit_docs.jsonl`.

(f) waits for a free slot. 8 jobs are queued, the brief's limit.

**Tue 13 Oct, 08:25: (a) merged and validated** (merge job 18925, 8.5 min,
`artifacts/c1/c1-llada-base.json`, all checks ok):

- **Size:** 1,979 documents, 119,562,567 rows, 922,214 passes (466 per document), 40
  parts, 15 GB at `/share1/abhiraj.ratna/traces/c1-llada-base/` (with
  `c1-llada-base.dataset.json`).
- **Cost:** 68.0 GPU-hours.
- **Rate:** 7.54 passes/s mean; shard 0 7.73 on gnode043, shard 1 7.36 on gnode046.
- **Checks:** uniform-count z = -0.01 +- 0.98, no overflowing document, one commit
  (`c072bc6`), one GPU model.

(b) GSM8K 17745 and MBPP 17746 started at 06:17 and 07:29.

**10:30. Two failures, both understood.**

- **LLaDA-Instruct MBPP (17746)** died with CUDA OOM after 450 documents. MBPP test
  482 is 3,986 tokens (a 3,730-token prompt of test asserts), and its float32 logits
  are ~1.9 GiB. Fix: task documents over 1,024 tokens are excluded and listed
  (`--max-doc-tokens 1024`, only test 482; deviations item 6).
  - MBPP is re-collected from scratch as `c1-llada-instruct-mbpp-v2` (18972).
  - Dream MBPP, queued without the cap as 17899, was cancelled before it started and
    resubmitted as 18973.
  - The partial first run is kept and not used.
  - Both run from `~/c3-r2` at `d7965e0`, so each run still has one commit.
- **Time-split jobs (18196, 18197)** are held so the MBPP reruns go before (e); release
  them once 18972/18973 start.
- **Dev run 2 (18970)** died on a full `/ssd_scratch` while pulling parts it did not
  need. Fixed, and rerun as 18974.

Revised ETA:

- (b) GSM8K ends ~15:30 Tue, then (c) shard 1 runs until ~07:00 Wed.
- (c) shard 0 started 10:04 Tue and ends ~01:35 Wed, then (d) GSM8K until ~09:15 Wed.
- Then the two MBPP runs, ending ~11:35 Wed.
- **(a)-(d) done ~Wed 14 Oct, 11:35**, still before Fri 16 Oct.

**Dev run 1 (18932, 150 fit-half LLaDA-Base documents; development only):**

- E1/E2 took 30 min at 7.4 GB.
- T8 took 12 min at 8.1 GB.
- E3 crashed on missing matplotlib. Now skipped with a note; the frozen commands
  would have crashed the same way.
- Softmax temperature fitted T = 0.86-0.95 everywhere and sharpened confidences, so
  log-odds temperature is now the primary row (deviations).
- Memory buys CPUs on u22 (`MaxMemPerCPU=3000`), so a CPU job gets at most ~40 GB
  while two GPU tasks run.

**15:25: (b) GSM8K merged and validated.** Merge job 19432, all checks ok
(`artifacts/c1/c1-llada-instruct-gsm8k.json`). 1,319 documents, 13,821,447 rows,
245,152 passes (186 per document), 17.6 GPU-hours, 7.74 passes/s. (c) shard 1 started
at 15:05.

**Wed 14 Oct, 10:50.** The check-ins did not fire between Tue ~16:00 and Wed 10:40
(laptop asleep). Meanwhile:

| run | job | finished | docs | passes/s |
|---|---|---|---|---|
| (c) shard 0 | 17766_0 | 01:17 | 985 | 8.59 |
| (c) shard 1 | 17766_1 | 07:00 | 991 | 8.50 |
| (d) GSM8K | 17898 | 08:56 | 1,319 | 9.79 |
| (b) MBPP-v2 | 18972 | 09:30 | 499 (test 482 excluded) | 7.42 |
| (d) MBPP | 18973 | running, 450/499 at 10:41 | | |

All exited 0. **One lost slot:** (e) was held until the MBPP reruns started, and that
release needed a check-in, so a 2-GPU slot idled from 09:30 to 10:42. (e) is now
released (18196 running, 18197 queued). Also submitted:

- (f) precision, 20470 (1 GPU, queued);
- merges for (c), (d) GSM8K and (b) MBPP-v2 (20471-20473, from `~/c3-r2`, whose merge
  code carries the excluded document).

**11:00: (a)-(d) all collected, merged and validated** (summaries in `artifacts/c1/`;
merges 18925, 19432, 20471-20473, 20486):

| run | docs | rows | passes | GPU-h | passes/s | checks |
|---|---|---|---|---|---|---|
| c1-llada-base | 1,979 | 119,562,567 | 922,214 | 68.0 | 7.54 | ok |
| c1-llada-instruct-gsm8k | 1,319 | 13,821,447 | 245,152 | 17.6 | 7.74 | ok |
| c1-llada-instruct-mbpp-v2 | 499 | 3,148,663 | 66,658 | 5.0 | 7.42 | ok (excl. 482) |
| c1-dream-base | 1,976 | 119,381,308 | 920,816 | 59.9 | 8.54 | ok |
| c1-dream-instruct-gsm8k | 1,319 | 13,399,445 | 241,162 | 13.7 | 9.79 | ok |
| c1-dream-instruct-mbpp | 499 | 2,523,455 | 60,680 | 3.8 | 8.87 | ok (excl. 482) |

Total for (a)-(d): 167.9 GPU-hours (2 GPUs per task). Every run is at one
commit: `c072bc6`, except the two MBPP runs at `d7965e0` (the length cap). Done on Wed
14 Oct, ahead of the Fri 16 Oct target. Still running: (e) 18196 (LLaDA time-split);
queued: (e) 18197 and (f) 20470.

**19:00: (e) Dream-Base time-split merged and validated** (18197, then merge 21346; all
checks ok): 500/500 documents, 30,208,673 rows, 233,000 passes, 15.0 GPU-hours, 8.63
passes/s.

- The first merge (21330) compared the run with the FineWeb corpus in its informational
  corpus count, because `*-base` matched before `c1ts-*`. Fixed in `0d697d5` with a test
  and re-validated.
- Updating `~/c3-r2` hit four untracked merge summaries. They were moved, not deleted, to
  `~/c3-r2-untracked-20261014-1853/` and are byte-identical to the committed copies.

**19:35: (e) LLaDA-Base time-split merged and validated** (18196, then merge 21370; all
checks ok): 500/500 documents, 30,208,673 rows, 233,000 passes, 16.6 GPU-hours, 7.82
passes/s. (e) is complete. (f) precision 20470 started at 18:22 (4-bit, one GPU,
41 docs/h) and is due ~23:00.

**23:30: every C1 run collected, merged and validated, (a)-(f).**

| run | precision | docs | rows | passes | GPU-h | passes/s | checks |
|---|---|---|---|---|---|---|---|
| c1-llada-base | fp16 | 1,979 | 119,562,567 | 922,214 | 68.0 | 7.54 | ok |
| c1-llada-instruct-gsm8k | fp16 | 1,319 | 13,821,447 | 245,152 | 17.6 | 7.74 | ok |
| c1-llada-instruct-mbpp-v2 | fp16 | 499 | 3,148,663 | 66,658 | 5.0 | 7.42 | ok (excl. 482) |
| c1-dream-base | fp16 | 1,976 | 119,381,308 | 920,816 | 59.9 | 8.54 | ok |
| c1-dream-instruct-gsm8k | fp16 | 1,319 | 13,399,445 | 241,162 | 13.7 | 9.79 | ok |
| c1-dream-instruct-mbpp | fp16 | 499 | 2,523,455 | 60,680 | 3.8 | 8.87 | ok (excl. 482) |
| c1ts-llada-base | fp16 | 500 | 30,208,673 | 233,000 | 16.6 | 7.82 | ok |
| c1ts-dream-base | fp16 | 500 | 30,208,673 | 233,000 | 15.0 | 8.63 | ok |
| c1t8-llada-base-4bit | 4bit | 198 | 11,962,128 | 92,268 | 4.8 | 5.36 | ok |

Total: 204.3 GPU-hours. Traces: `/share1/abhiraj.ratna/traces/<run>/`, each with
`<run>.dataset.json`. (f) ran at 5.36 passes/s in 4-bit on one GPU, against 7.5-7.8 for
fp16 over two.

**Full-scale sizing (dev job 21625, the whole LLaDA-Base fit half, 993 docs, 60M rows,
200 resamples; development only):**

| step | time | peak memory |
|---|---|---|
| subset | 3.5 min | 33 GB |
| E1/E2 without Eq. 2/3 | 21 min | 38 GB |
| E3 | 3 min | 21 GB |
| T8 | 80 min | 52 GB |

The report half is the same size, so the frozen commands fit a 64-96 GB CPU job, except
the Eq. 2/3 bootstrap. That needs the resample decision (A: 1,000, recommended).

Ada checkouts `~/c3`, `~/c3-r2` and `~/c3-dev` are at `e14f6d3` since 23:21 Wed. Moved
aside, byte-identical to the committed copies, deletable on request:

- `~/c3-untracked-20261014-2321`
- `~/c3-r2-untracked-20261014-2321`
- `~/c3-r2-untracked-20261014-1853`

The Ada checkout stays at `c072bc6` until every collection job has started, so all runs
carry one commit. The later commits are analysis-side only. (Before submission, `sbatch --test-only`
estimated a 17:14 start for every job; the env build actually started at once.)

## Blocker (resolved 12 Oct 10:48): no GPU job could run

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
