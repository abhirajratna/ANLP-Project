# Deviations from plan v2 (documents/C3_Detailed_Plan_v2.pdf)

Recorded before any report-half number was computed. Each entry gives what the plan
says, what was done, and why. Later deviations are appended with a date; nothing
above them is edited.

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

## Exploratory: set-level C3 on train110m-v4 (recorded 15 Oct 2026, before any number below was computed)

Exploratory work outside the pre-registered families (E1/E2 primaries are unchanged).
Everything is developed on the fit half only; the report half is read once, for the
final run, and only after the team's freeze tag or written sign-off (recorded at the
freeze in item 13).

7. **A second 110M checkpoint.** Plan v2 T3 trains one 110M model. `train110m-v4` is a
   second run of the same recipe (10B tokens, batch 16 x accum 4 x 4 GPUs, EMA 0.9999),
   trained on 4 GPUs with logits computed only at masked positions (the loss and
   gradients equal the full computation; `tests/test_training.py`). EMA perplexity bound
   37.22 [36.51, 37.96] against `train110m-v3`'s 37.25; G2 passes (0.35%). This work uses
   v4 so it is self-contained on one account; the team's v3 results are unaffected.
8. **A proper-score field test (B) as a development gate.** Plan v2 T6 tests the field
   term through winner gaps only. B compares held-out log-loss and Brier of correctness
   over all masked positions between a spline-only own-confidence logistic and the same
   model with field features (same family), paired per document with the 10,000-resample
   document bootstrap. It is exploratory, outside the E2 Holm family. Gate rule, fixed
   now: if the log-loss gain is declared (interval excludes 0 and exceeds twice the
   floor) on `uniform` or `decoder:topk_conf:k4` using the decode-time features
   (median_ell, mask_rate, log_n_masked), C3a and C3-set use field features; otherwise
   they use own confidence only.
9. **C3-set and C3-indep added to T11; C3b not run.** C3a cannot reorder positions within
   a step (all positions share the field features), so at matched speed it is a
   relabelled threshold rule. C3-set commits the largest confidence-ordered prefix whose
   calibrated probability that every token in it is correct is at least 1 - alpha, using
   set features (product of token confidences, weakest member, sum of entropies minus
   the largest, set size, adjacent-pair share, field features). C3-indep uses the
   independence product only, which isolates the learned set model. Within one step the
   expected number of correct tokens does not depend on their correlation, so a token-
   accuracy gain can only come from errors carried into later steps or from set features
   that inform single tokens; the per-step all-correct rate is reported alongside.
10. **T12 for the 110M is whole-span FineWeb infilling.** The 110M cannot do GSM8K/MBPP.
    Each 256-token document has positions 64-191 masked and is decoded greedily over the
    whole span (no blocks, no seeds; greedy decoding is deterministic). Accuracy is the
    share of the 128 tokens matching the original; speed is 128 / forward passes. Every
    policy (top-k, margin, PC-Sampler, threshold, entropy-bounded, KLASS, C3a, C3-indep,
    C3-set) gets a knob grid of equal density tuned on development documents. Primary
    statistic: at 2, 4 and 8 tokens per step, C3-set's interpolated accuracy minus the
    best of the six existing decoders, the maximum taken inside each bootstrap draw,
    Holm over the three speeds.
11. **The fit half is re-split for development.** Calibrators and knob grids are chosen
    on `hash_halves(fit_docs, 1)`: dev-fit (0) to dev-eval (1). The frozen calibrators are
    then refitted on the whole fit half before the single report-half run.
12. **Probe rules and exact joint correctness.** Plan v2 T9 compares random and
    confidence selection by the exact (A) and (B). The probe grid adds threshold,
    entropy-bounded, C3a, C3-indep and C3-set, fine-tuned from v4 with the team's
    settings (`ft110m-v4`). C3-set targets greedy joint correctness, so its calibration is
    checked against the exact J(S) = p*(x_S = argmax | x_obs), with (A) and (B) reported as
    diagnostics. Probe calibrators are fitted on 2,000 new instances (label
    `probe-c3calib`), disjoint from the training, validation and evaluation sets.
14. **Scattered-mask infilling replaces whole-span infilling as the primary T12 task
    (recorded 15 Oct 2026, after dev sweep 1 on dev-eval documents, before any report-half
    number and before any scattered-mask number).** Dev sweep 1 (item 10's whole-span task,
    498 dev-eval documents, 82 configurations) put every decoder at 3-6% token accuracy,
    with accuracy *rising* with speed (top-k k=1 2.8%, k=8 4.0%, one-shot 5.8%). The
    on-policy logs show why: the first commit is right 50% of the time and tokens next to
    the visible context 23-33%, but after a few steps the decoder continues its own text
    (committed-token accuracy 1-2% while its confidence rises to 0.67), so matching the
    original 128-token gap measures drift, not decoding quality. The task is changed to
    mask a fixed random 50% of positions 64-191 per document (`c3.infill.scatter_masks`,
    keyed by document id; 64 masked tokens, speed = 64 / forward passes); everything else
    in item 10 (greedy, no blocks, the six decoders, the primary statistic at 2, 4 and 8
    tokens per step, Holm over speeds) is unchanged. The whole-span sweep is kept and
    reported as a secondary development result. C3-indep and C3-set get alpha up to 0.997
    (`prereg/c3set_policies_dev2.json`) because in sweep 1 they never passed 2.6 tokens per
    step.

