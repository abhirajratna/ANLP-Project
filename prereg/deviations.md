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

### Added 26 Oct 2026, after the prereg-v1 tag (6b4c25e)

17. **T12 decoder sweep: protocol fixed before any report-half generation (26 Oct 2026).**
    T12 is not a pre-registered endpoint (`documents/final-plan.md:55-56`) and `prereg-v1`
    freezes no T12 number. Its choices are fixed here, before any T12 report-half problem is
    decoded, so none can depend on a report-half result. Plan v2 T12 runs eight decoders on
    GSM8K, MBPP and HumanEval in bfloat16; the final plan defers (not run here) C3b and
    HumanEval, leaving seven decoders x three speeds on GSM8K and MBPP.
    - Fixed settings: LLaDA-8B-Instruct, zero-shot chat prompt, gen 256, block 32, greedy,
      fp16 (as item 2; the C3a calibrator's traces are fp16), on the report halves of
      `artifacts/split_manifest_{gsm8k,mbpp}.json`.
    - Platform: Jarvis Labs, 2x RTX PRO 6000 (Blackwell). A machine is not mixed within a
      task. The fit-half sanity reference ran on Ada (2080 Ti, fp16, lm-eval 5-shot) and
      scored 83.5 / 84.5 (top-k k=1 / threshold 0.9) against the published 79.3 / 79.2.
    - Speed of a configuration = 256 N / sum(NFE) over its N problems (total generated
      positions over total forward passes), not the mean of per-problem ratios.
    - Knobs are calibrated per task, on fit-half problems only (GSM8K and MBPP each from
      their own run of `prereg/t12_cal.json`, 32 configurations (after the extension below); 32 and 24 problems in
      `scripts/jarvis_t12.sh calibrate`). `scripts/pick_knobs.py` refuses any report-half
      record. top-k, margin and PC-Sampler are exact at k = 2, 4, 8 tokens per step.
      Threshold, entropy-bounded, KLASS and C3a knobs are interpolated, in log(speed), to
      2, 4 and 8 tokens per step. They are chosen by speed alone, never by fit-half accuracy.
    - Matching: for each C3a setting, each baseline is taken at its setting whose tokens per
      step is nearest, within +-10% of C3a's, else '---'. The tokens per step matched is each
      configuration's aggregate speed (256 N / sum(NFE)) in the report-half run of that
      task. Per-configuration accuracy carries a 95% Wilson interval. The difference is the
      paired accuracy difference on the shared doc_ids, with a 95% percentile CI from 2,000
      bootstrap resamples of problems (seed 20261030). There is no Holm family and no
      declaration rule: T12 numbers are exploratory.
    - KLASS: if its fastest calibrated setting cannot reach 8 tokens per step within +-10%
      (history 2 caps it at 32/3 = 10.7, and reaching 8 needs nearly every position stable
      by step 3), that setting takes the speed-3 slot. `pick_knobs.py` records
      `"unmatched_at_target": true` and the tokens per step achieved, and the speed-3
      comparison reads '---'.
    - Main-text table (primary operating point, chosen by parth on 26 Oct 2026, before any
      report-half generation): §6's matched-speed table shows, per task, the speed-2 slot
      (target 4 tokens per step), at the C3a alpha that `scripts/pick_knobs.py` writes to
      `prereg/t12_grid_<task>.json`; speeds 1 and 3 appear in full in the appendix figure and
      the code archive's results table, and no slot is chosen after a report-half number is read.
    - Grid extension (fit half only). The first fit-half calibration (375b079, 25
      configurations, Jarvis instance 550443, 1,400 records) could not reach 7.2 tokens per
      step for MBPP threshold, C3a and entropy-bounded, or for GSM8K entropy-bounded, so
      `pick_knobs.py` refused. Using those fit-half speeds only, the grid was extended to 32
      configurations (threshold tau 0.2 and 0.1; entropy-bounded gamma 10, 30 and 100; C3a
      alpha 0.75 and 0.9; `prereg/t12_cal.json` at 4435c13, sha256 7d58d9a0e633...) and
      recalibrated (run 7d58d9a0, 1,792 records, 0 errors) before any report-half
      generation. KLASS remains unmatched at speed 3 (4.77 and 4.32 tokens per step), as the
      KLASS rule allows.
    - Files: the calibration grid `prereg/t12_cal.json` is at branch `ayush/t12-grids`,
      commit `4435c13041227cfd3d6f13a6d424300c6b2b3092`; `scripts/pick_knobs.py` is at
      branch `parth/t12-driver`, commit `daff750c251f8e49fb62c40f8fad12b80cb4ee04`. The final per-task grids
      (`prereg/t12_grid_gsm8k.json`, `prereg/t12_grid_mbpp.json`) are committed before the
      report-half run, and their hashes are recorded on the coordination bus.
    - No T12 result changes any choice frozen in prereg-v1 or prereg-v1.1.
    - Amendment (28 Oct 2026, before any H200 report-half generation): no RTX PRO 6000
      container was free in Jarvis IN1 for most of 28 Oct, and the first report attempt (RTX
      PRO, 1 GPU, commit ffff6cb) was paused mid-run by a launcher fault (fixed in
      `parth/t12-jarvis-cli` 202fb7f). The report half is therefore run afresh with both
      tasks wholly on 1x H200 (Jarvis IN2), with the same code (ffff6cb), the same per-task
      grids and fp16. The fit-half knobs stay as calibrated on RTX PRO 6000: tokens per step
      is a forward-pass count and does not depend on the GPU, and matching uses each
      configuration's report-half speed. No task mixes machines. The first attempt's partial
      RTX PRO records are kept as an unreported cross-check.
    - Execution detail (recorded 28 Oct after the H200 launch; no change to decoding, knobs,
      grids or precision): one worker per GPU left the H200 ~44% idle (per-step Python
      bound), so the run uses 3 worker processes per GPU (`T12_WORKERS_PER_GPU`, commit
      a9abd7b on `ayush/t12-grids` = ffff6cb plus that option), and the two tasks run on
      separate 1x H200 instances (GSM8K and MBPP, each task wholly on one instance, 3
      shards). The 1-worker GSM8K start (213 records) is set aside unreported, like the RTX
      PRO attempt.
    - KLASS implementation differences (found 29 Oct, AFTER the GSM8K report half was
      scored; no code, knob or reported number changes): a review against the reference
      implementation (shkim0116/KLASS, `llada_klass.py`) found (A) our KL stability history
      covers only the active 32-token block (`c3/decode/loop.py`), so each block starts with
      two forced one-token steps, where the reference keeps history over all positions and
      forces about one; this lowers KLASS's speed at a given threshold, so the fixed knob
      path needed a lower confidence threshold (0.29 and 0.2 on GSM8K, 0.23 and 0.2 on MBPP, at
      the two faster settings; the published range is about 0.6) to reach them; (B) the KL is computed as KL(previous ||
      current), the reverse of the reference; (C) blocks are 32 tokens, not the paper's 64
      (a setting shared by every decoder). At its published thresholds our KLASS runs at
      about 2.6 tokens per step, as in the paper. KLASS's two faster rows are reported as
      computed and marked as off its published operating point; no comparison in the text
      rests on them. Any corrected-KLASS run would be exploratory and never replace these
      rows.

18. **Step order: E1/E2 banked before Eq. 2/3 (scheduling only).** The Eq. 2/3 FE bootstrap
    did not fit the 96 h Ada job cap on the 120M-row arms (measured: LladaBase job 39851,
    eq2_fe 5-8 min per resample per worker; DreamBase job 39510, eq3_fe ~100 s per resample
    per worker, first of 7 mask kinds still running after 16 h). Each arm's primary
    `c1/<Label>` command is therefore first run with `--skip-regression` into
    `prereg-v1-e1e2/` (`slurm/prereg_analysis.sbatch`, `C3_E1E2_ONLY=1`, PR #47). The E1/E2
    Holm families come only from the 10,000-draw reweighting bootstrap
    (`scripts/analyze_traces.py:154-196` at `prereg-v1`; bootstrap counts drawn once at
    `:401`), so their numbers are identical to a full run; Eq. 2/3 follows separately. No
    frozen choice changes.

19. **Eq. 2/3 bootstrap warm start.** From commits cc9cc85 and 51657cb (branch
    `abhiraj/eq23-warm-start`) each Eq. 2/3 bootstrap resample's IRLS starts from the
    full-sample fit instead of zero; estimator, ridge, tolerance, max_iter and resample
    seeds unchanged. The penalised objective is strictly concave, so a converged draw does
    not depend on where it started; against an independent high-precision solver warm draws
    are exact to <=2e-9 and CI endpoints move by <=3e-10. Where the frozen cold start hit
    max_iter, its draws were not optimal (errors up to 9e-2 on synthetic hard cases);
    unconverged point fits and draws are now restarted once from their last iterate, and any
    still unconverged are counted and marked §. `--eq-cold-start` reproduces `prereg-v1`
    output exactly. On real data (fit-half documents only; 50 and 200 documents of LladaBase
    decoder:topk_conf:k2; full-sample fit plus 10 bootstrap resamples each, using each tag's
    own code), the prereg-v1 cold start converged in every case but slowly: median 49 IRLS
    iterations (max 58) and about 270-340 step-halvings per fit, 16 s and 63 s per fit.
    Under prereg-v1.1 every resample converged in 5 iterations with no step-halving, 29-32
    times faster, and coefficients agreed with the cold fits to within 1.3e-12; the
    full-sample fits were identical. On the observed data the change is therefore a speed
    change only. Non-convergence was seen in synthetic hard cases only. Diagnostic scripts
    and output: ANLP-project/c3-traces-8b, diagnostics/fediag-2026-10-26/. The library default stays cold, so other callers (`run_controls`,
    `run_stage0`, `c3/sim`) are unchanged. Convergence of point fits and draws is now
    recorded (`point_converged`, `point_restarted`, `n_nonconverged_draws`).
