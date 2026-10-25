# Deviations from plan v2 (documents/C3_Detailed_Plan_v2.pdf)

Recorded before the tag `prereg-v1`. Entry 16 discloses report-half rows that an
exploratory notebook read before the tag. Each entry gives what the plan says, what
was done, and why. Later deviations are appended with a date; nothing above them is
edited.

## Fixed before the freeze (12 Oct 2026)

1. **Contamination threshold seen before it was fixed in code.** Plan v2 T4 says to fix the
   flag threshold before looking, and suggests 32 of 50 tokens. That value was adopted on
   25 Sep, but it went into the code (`c3.CONTAM_THRESHOLD = 32`) only on 12 Oct. The
   screens on both base models ran on 29 Sep with `--threshold 32` passed by hand
   (`artifacts/contamination_*.json`, `threshold_source: "--threshold (not yet
   pre-registered)"`), so their results (2/1,979 and 1/1,976 flagged) were seen first.
   The threshold is the plan's suggested value, not one chosen after looking.

2. **fp16, not bfloat16, for the main 8B runs.** Plan v2 runs the 8B models in bfloat16.
   Ada's RTX 2080 Ti (compute 7.5) has no bfloat16, so the main runs use fp16 split over
   two cards (`device_map=auto`, 9 GiB each; user decision, 29 Sep). Every forward pass
   is checked for inf/NaN logits, and a document that overflows is left out and listed
   (`doc_ids_failed`), so the omission is visible. The T8 precision control compares
   4-bit with fp16, not with bfloat16.

3. **E3 for top-k decoders only.** Plan v2 Sec. 3.3 and T7 ask for decoder-induced
   states from every T12 decoder (top-k, threshold, margin, entropy). This sweep
   collects teacher-forced confidence top-k trajectories for k = 2, 4, 8 (the collection
   defaults). E3 is pre-registered for those arms. The other decoders' trajectories are
   not collected here.

4. **Non-determinism floor per model family.** Plan v2 T1 measures the floor once
   (LLaDA-8B-Base: 0, `artifacts/t2_pilot/floor_llada_base_fp16.json`). The declaration
   rule needs a floor for every arm, so the Dream floor is measured the same way
   (`slurm/submit_c1.sh floor dream-7b-base`) before any Dream arm is analysed. Each
   Instruct checkpoint uses its Base family's floor.

5. **Split for GSM8K and MBPP.** The plan's 50/50 rule ("documents are split 50/50 with a
   fixed seed") is applied to the task test sets with the same function and seed as
   FineWeb (`scripts/make_task_split.py`; `artifacts/split_manifest_{gsm8k,mbpp}.json`).

6. **One MBPP document excluded for length.** Task documents longer than 1,024 tokens
   (prompt plus answer) are excluded and listed in each run's metadata
   (`collect_traces.py --max-doc-tokens 1024`). Only MBPP test 482 is affected: 3,986
   tokens, 3,730 of them a prompt of test asserts holding a long literal. Every other
   task document is at most 943 tokens. The whole sequence goes through each forward
   pass, and 482's float32 logits (~1.9 GiB at LLaDA's vocabulary) do not fit a 2080 Ti
   beside the weights. The first LLaDA-Instruct MBPP run (job 17746) crashed with CUDA
   OOM on it after 450 documents. That run is kept, partial, at
   `/share1/abhiraj.ratna/traces/c1-llada-instruct-mbpp/` and is not used. MBPP is
   re-collected from scratch at one commit as `c1-llada-instruct-mbpp-v2`. Dream-Instruct
   MBPP carries the cap from the start.

## Added 24-25 Oct 2026, before the tag

Items 7-14 are reserved for the T13 ablation entries (branch `t13-ablations`).

15. **Eq. 2/3 bootstrap at 1,000 resamples.** Plan v2 Sec. 5 asks for 10,000 document
    resamples for every interval. The Eq. 2/3 interval (beta_z, gamma_k) refits the
    regression per resample: ~3.7 s on 150 documents with 8 CPUs (measured). On a whole
    FineWeb run (~120M rows) it takes ~310-475 s with one worker, where memory allows
    three workers; that figure is extrapolated from the 9M- and 30M-row jobs 36215, 36251
    and 36422, not measured on a whole run. At 10,000 that is weeks per run. It runs at
    1,000 in the primary run and in the T8 precision comparison's beta_z (Eq. 2 on
    winners), and at 200 in the two knot runs. Counting fixed costs and E1/E2, a knot run
    takes ~8-12 h and a whole FineWeb arm ~2.2-3.3 days. The knot runs are sensitivity
    panels and their intervals are descriptive: at 200 resamples an endpoint's Monte Carlo
    error is ~0.19 SE and p resolves only to 1/200, so nothing is declared from them. The
    contamination-clean, time-split and additive-calibrator runs report E1/E2 only
    (`--skip-regression`). Every other interval, including the E1 and E2 contrasts in the
    Holm families, stays at 10,000.

16. **Report-half rows read before the tag (disclosure).** On 24 Oct an exploratory
    notebook (`notebooks/c3_dorigo_methods.py`, PR #18; not part of this pre-registration)
    computed report-half statistics on `main-110m-v3` and `timesplit-110m-v3` before
    `prereg-v1` existed. Only ayush saw the numbers; they were not committed or shared.
    `main-110m-v3` is the `Mdlm` arm, a member of both Holm families. Since 2b72e12 the
    notebook reads the fit half only by default. Most frozen choices were posted on 23 Oct,
    before that run. Four were settled after it, on 25 Oct, and none depends on data:
    (a) the knot runs at 200 resamples (item 15; proposed by ojas on 23 Oct, accepted by
    abhiraj on 25 Oct); (b) the T8 precision beta_z at 1,000 resamples (item 15); (c) the
    claim rule: a family member is claimed only if it is declared AND rejected by Holm,
    which with every floor 0 is the Holm rejection alone, so it cannot change a result;
    (d) each Holm family's size is fixed at 7 labels x 2 arms = 14, and a missing member
    counts as p = 1, which can only make the test more conservative.

## Choices the plan leaves open, fixed here (for sign-off)

- E1/E2 Holm families: arms `uniform` and `decoder:topk_conf:k4` across models x datasets.
  k = 2, k = 8 and `uniform_matched` are exploratory.
- Context calibrator: interacted (ell x field features), the code default. Plan v2 T6(2)
  writes the additive form, which is reported as a sensitivity check.
- E3 joint test of the type terms: a sup-t test that the difference curve is zero, with
  the document bootstrap. A Wald test on the spline coefficients over-rejected on
  simulated nulls.
- T8 temperature: one temperature per mask-rate bin on the top-1 log-odds,
  c = sigma(ell / T), fitted by the NLL of correctness on the fit half. This is the
  primary row: it is exact from the trace and acts on the confidence Delta(m) uses. The
  softmax temperature (logits / T, rebuilt from the stored top-20 probabilities plus an
  even tail, fitted by true-token NLL) is a sensitivity row. The order was changed after
  a development run on 150 fit-half LLaDA-Base documents (job 18932, 13 Oct; no
  report-half document). There the softmax fit gave T = 0.86-0.95 in every bin and arm,
  sharpening confidences the model already reports nearly calibrated. The suspected
  cause is that the even tail makes sharpening look cheap: it spreads the residual mass
  over ~126k tokens, so raising it to 1/T > 1 makes it vanish. This could not be checked
  without the full logits.

## After the freeze

(none yet)
