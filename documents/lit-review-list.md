# Reading List — "Selection Bias in Confidence-Based Token Commitment in Diffusion LMs"

Reading list for the DuckDuckDuckGoose project, assuming **no prior knowledge of diffusion
language models (dLLMs)**. Papers are ordered in recommended reading sequence, grouped by
phase. Each entry says *why* it is needed for this project. arXiv IDs marked (P) come from
the proposal's own reference list; the rest were verified externally.

**Priority legend**

- ★★★ — must read before doing any project work (unblocks the core ideas)
- ★★  — read carefully before the phase of the project it supports
- ★   — skim for related work / baselines
- ○   — deferred / optional (background or future work the proposal explicitly does not do)

**Minimum viable path (10 papers to unblock everything):**
#1 survey → #2 DDPM → #3 D3PM → #4 MaskGIT → #5 MDLM → #12 Fast-dLLM → #15 PC-Sampler → #21 Tang et al. → #29 Efron → #30 Andrews et al.

---

## Phase 0 — Foundations: what a diffusion LLM is (★★★ unless noted)

The proposal assumes fluency with masked discrete diffusion, its decoding loop, and the
"confidence = predicted probability of the top-1 token" convention. Start here.

1. **A Survey on Diffusion Language Models** — Li et al., 2025. arXiv:2508.10875
   ○/★★★ *(optional entry point)*
   **Why:** The fastest way to map the whole dLLM landscape — paradigms (AR vs masked vs
   block diffusion), training (masking schedules, ELBO vs CE loss), and inference techniques
   (top-k, remasking, caching). Read §2–§4 for orientation, then go to the primary sources.

2. **Denoising Diffusion Probabilistic Models (DDPM)** — Ho, Jain, Abbeel, NeurIPS 2020. arXiv:2006.11239
   ★★★
   **Why:** The original continuous-diffusion paper. Even though we work in discrete
   token space, the core machinery the whole field inherits — forward corruption,
   reverse denoising, noise schedule *t*, training a network to denoise — comes from here.
   You cannot read D3PM or the proposal's "denoising step" / "masked state" language without it.

3. **Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM)** — Austin et al., NeurIPS 2021. arXiv:2107.03006
   ★★★
   **Why:** The foundational *discrete* diffusion paper (cited as "Austin et al., 2021a" in
   the proposal). Introduces categorical corruption/transition matrices, the absorbing
   ("masked") chain, and ELBO-based training — the exact generative framework our 110M model
   and the audited checkpoints are built on. Also defines the reverse-process "unmasking"
   step that our instrument holds fixed.

4. **MaskGIT: Masked Generative Image Transformer** — Chang et al., CVPR 2022. arXiv:2202.04200
   ★★★
   **Why:** The origin of the commit rule under audit. MaskGIT's two-stage decoding —
   commit the highest-confidence positions first, refine the rest — is exactly the
   "rank by predicted probability, keep the winners" statistic the proposal investigates.
   The proposal says this convention "originates with MaskGIT"; read it to see where
   "confidence" and the fixed-schedule top-k rule came from.

5. **Simple and Effective Masked Diffusion Language Models (MDLM)** — Sahoo et al., NeurIPS 2024. arXiv:2406.07524
   ★★★
   **Why:** The "MDLM recipe of Sahoo et al. (2024)" is explicitly what we will use to
   train our controlled 110M model on FineWeb-Edu with GPT-2 BPE. Read for: the simplified
   mask-only corruption, the cross-entropy training objective, the SUBS sampling schedule,
   and the 110M-class training setup (OpenWebText) we are imitating at smaller scale.

6. **Large Language Diffusion Models (LLaDA)** — Nie et al., NeurIPS 2025. arXiv:2502.09992
   ★★★
   **Why:** LLaDA-8B(-Instruct) is one of the two deployed checkpoints we run. You need its
   architecture (bidirectional Transformer, no causal mask, no KV cache), its sampling
   defaults, and its tokenizer (the proposal stresses each model is evaluated under its own
   tokenizer). Also documents the pretraining corpus issue behind the proposal's
   contamination section.

7. **Dream 7B: Diffusion Large Language Models** — Ye et al., 2025. arXiv:2508.15487 (P)
   ★★★
   **Why:** The second audited checkpoint. Same reasons as LLaDA: default decoder,
   tokenizer, and the undisclosed pretraining web corpus that limits which claims can rest
   on it (levels → 110M arm; gradients only → 8B arm).

---

## Phase 1 — The commit statistics under audit: decoders & samplers

C1's endpoint is a coefficient on *these* statistics. You must know exactly what each
decoder commits on and how the cutoff is chosen.

8. **Fast-dLLM: Training-free acceleration of diffusion LLM by enabling KV cache and parallel decoding** — Wu et al., 2025. arXiv:2505.22618 (P)
   ★★★
   **Why:** The de facto workhorse decoder (KV caching + global confidence threshold +
   parallel decoding) and one of the C3 baselines. Its "one global threshold" is the
   canonical example of "the commit statistic is tuned but never audited."

9. **Accelerated Sampling from Masked Diffusion Models via Entropy Bounded Unmasking (EB-Sampler)** — Ben-Hamu et al., NeurIPS 2025. arXiv:2505.24857
   ★★
   **Why:** An alternative commit statistic (entropy rather than top-1 probability) plus an
   *error decomposition into model error and joint-dependence error* that is structurally
   the same split as the proposal's Eq. 1 terms (A)/(B). Important both as a baseline
   (if C3 runs) and as prior art for why the two error terms should be separated.

10. **Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions** — J. Kim et al., ICML 2025. arXiv:2502.06768
    ★★
    **Why:** Introduces top-k probability vs. **top-k margin** as position-selection
    statistics, and shows plain top-1 probability gives *misleading* uncertainty estimates.
    Directly motivates the proposal's claim that the margin variant (J. Kim et al., 2025)
    is an alternative statistic in use — you need to know it to interpret what "confidence"
    means in Eq. 2–3 and why the statistic itself may be miscalibrated.

11. **KLASS: KL-Guided Fast Inference in Masked Diffusion Models** — S. H. Kim et al., NeurIPS 2025. arXiv:2511.05664
    ★★
    **Why:** The stability-based selector: token-level KL divergence between consecutive
    steps + confidence threshold. A C3 baseline, and the proposal's statement that
    "corrections rest on unmeasured premises" is aimed squarely at this family —
    understand its premise (stability ⇒ trustworthy) to see what C1 would add.

12. **Beyond Static Cutoffs: One-shot Dynamic Thresholding for Diffusion Language Models (OSDT)** — Shen & Ro, 2025. arXiv:2511.02077 (P)
    ★
    **Why:** Justifies reusing a threshold schedule calibrated on a *single* sequence by
    claiming confidence trajectories are nearly identical across inputs. That is precisely
    the assumption C1 tests: if confidence means different things at different field
    sizes, one-shot transfer is not safe.

13. **A Tale of Two Temperatures: Simple, Efficient, and Diverse Sampling from Diffusion Language Models** — Olausson et al., 2026. arXiv:2604.09921 (P)
    ★★
    **Why:** Named explicitly in the proposal: a per-step, per-field-size **temperature
    sweep is the obvious competing explanation** for any rank/field-size effect we report,
    and temperature can reorder positions because confidence depends on the whole logit
    vector. Read so the temperature ablation in the evaluation plan is well-designed.

---

## Phase 2 — Calibration & stability corrections (closest prior work)

14. **PC-Sampler: Position-Aware Calibration of Decoding Bias in Masked Diffusion Models** — Huang et al., 2025. arXiv:2508.13021 (P)
    ★★★
    **Why:** The closest existing work to what we are doing — it already "calibrates" the
    commit statistic, with a **position-aware weight** plus a trivial-token adjustment.
    Two reasons to read it carefully: (a) its bias is *positional and content-driven*,
    while our claimed bias is a property of the ranking operation itself (should appear
    even among positions of equal difficulty, and should scale with field size) — the
    separation of predictions is a key part of the paper we are writing; (b) its
    position-aware weight enters **our Eq. 2 as a competing covariate** — if adding it
    drives βz to zero, our effect is positional, not selection-induced. You must know
    exactly how their weight is computed.

15. **Stability-Weighted Decoding for Diffusion Language Models (SWD)** — Wu & Huang, 2026. arXiv:2604.17068 (P)
    ★★
    **Why:** Proves inter-step instability is a **strict lower bound on a token's mutual
    information with the remaining masked context** and uses it to reweight any selector.
    The proposal warns that "SWD's theorem constrains any reading of inter-step movement
    as estimation error, since the raw signal carries dependency information" — essential
    for interpreting C3, whose shrinkage rule is contingent on C1 and would corroborate
    SWD's account.

16. **LESS is More: Mutual-Stability Sampling for Diffusion Language Models** — Mohamed et al., 2026. arXiv:2606.16908 (P)
    ★
    **Why:** Adds *persistence* (stability over a window of steps) to the stability-based
    selection family. Part of the related-work landscape of §3; skim for the claim and
    mechanism.

17. **STDec: Spatio-Temporal Stability Guided Decoding for dLLMs** — Y. Chen et al., 2026. arXiv:2604.06330 (P)
    ★
    **Why:** Adds *spatial* structure (neighboring positions) to inter-step stability.
    Rounds out the stability family so the related-work section is complete; skim.

---

## Phase 3 — Dependency-aware decoding (the literature targeting term (A))

Eq. 1 splits parallel-decoding error into a **dependency term (A)** (joint vs. marginals)
and an **estimation term (B)** (marginals wrong at the states selection visits). This
whole body of work targets (A) and treats (B) as a constant — that's the gap the proposal
fills. Read to (i) know the competing explanations and (ii) borrow how they measure (A).

18. **DAPD: Dependency-Aware Parallel Decoding via Attention for Diffusion LLMs** — B. Kim et al., ICML 2026.
    ★★
    **Why:** Commits *independent sets recovered from self-attention*. The canonical
    "dependency explains everything" account; understanding how it detects dependency is
    needed to argue why (A) is *indifferent to rank* and so cannot produce a rank gradient.

19. **Dependency-Guided Parallel Decoding in Discrete Diffusion Language Models (DEMASK)** — Ringel, Ali, Romano, 2026. arXiv:2604.02560 (P)
    ★★
    **Why:** Learns pairwise conditional influence with a **total-variation bound**.
    A principled measurement of term (A) that we can cite and contrast with our
    disagreement-rate proxy on real corpora.

20. **ParallelBench: Understanding the Trade-offs of Parallel Decoding in Diffusion LLMs** — Kang et al., ICLR 2026.
    ★★
    **Why:** The benchmark paper for parallel decoding; proves **subadditivity of
    conditional total correlation** — the formal quantity behind the joint-vs-marginal
    gap. Needed both for the "two error sources measured as one" motivation and for
    situating C2's separately estimated terms.

21. **Enabling Approximate Joint Sampling in Diffusion LMs** — Bansal & Sanghavi, 2025. arXiv:2509.22738 (P)
    ★
    **Why:** Adds an auxiliary sampler for *approximate joint* (rather than product of
    marginals) sampling. The main theoretical treatment of term (A); skim for the setup.

22. **Parallel Sampling from Masked Diffusion Models via Conditional Independence Testing (PUNT)** — Azangulov et al., ICLR 2026.
    ★
    **Why:** Selects which positions to commit via **conditional independence testing**
    instead of a confidence score — an alternative selection rule that is itself immune
    to the winner's-curse argument (it never ranks on a noisy score). Good counterpoint
    for the discussion section.

23. **Accelerating Diffusion LLMs via Adaptive Parallel Decoding (APD)** — Israel, Van den Broeck, Grover, NeurIPS 2025.
    ★
    **Why:** Uses an **autoregressive verifier** to decide parallel commitments. Another
    architecture for fixing (A); skim for the verifier idea.

24. **dParallel: Learnable Parallel Decoding for dLLMs** — Z. Chen et al., ICLR 2026.
    ★
    **Why:** "Distils a checkpoint into reaching high certainty faster" — named in §3 as
    *learning the statistic away* (with Jazbec et al., 2026). Part of the strongest
    challenge to our premise; skim.

---

## Phase 4 — Learning the statistic away (the strongest challenge to C1's premise)

25. **Learning Unmasking Policies for Diffusion Language Models** — Jazbec et al., ICML 2026.
    ★★
    **Why:** Trains unmasking policies with RL. The proposal states this is "the strongest
    challenge to our premise": a learned policy can *absorb* a selection bias without
    exposing it, so it cannot say whether the bias exists, how large it is, or what it
    tracks. Read so you can articulate — and defend — the difference between *learning
    away* and *auditing* the commit statistic.

26. **MDPO: Overcoming the Training-Inference Divide of Masked Diffusion Language Models** — He et al., 2025. arXiv:2508.13148 (P)
    ★
    **Why:** In the proposal's reference list; targets the *training* side (model
    approximation error) by aligning training with inference-time masking. Relevant to
    the 110M-model arm (how we train the controlled instrument) and to interpreting
    position-specific noise scale ŝi dispersion.

27. **TRIMS: Trajectory-Ranked Instruction Masked Supervision for Diffusion Language Models** — L. Chen et al., 2026. arXiv:2604.00666 (P)
    ★
    **Why:** In the proposal's reference list; another training-side approach (trajectory
    ranking of masked supervision). Skim to know the current state of training-side
    remedies when we argue our contribution is on the inference/audit side.

---

## Phase 5 — Theory, evaluation, and the statistics of selection

The identification logic of C1 (winner's curse / sufficiency / fixed effects) and the
entanglement problem C2 resolves live here.

28. **Is Your Diffusion Sampler Actually Correct? A Sampler-Centric Evaluation of Discrete Diffusion Language Models** — Tang et al., ICML 2026.
    ★★★
    **Why:** The paper that identifies the **entanglement of sampler error and model
    approximation error** in current dLLM evaluations — the exact entanglement C2
    separates, "exactly and in nats on the probe, and as identified contrasts on real
    corpora." Read closely; our C2 is positioned against it and extends it.

29. **Confidence-Based Decoding is Provably Efficient for Diffusion Language Models** — Cai & Li, 2026. arXiv:2603.22248 (P)
    ★★
    **Why:** Bounds the iteration complexity of an **entropy-sum decoder** (unmask until
    cumulative entropy crosses a threshold), taking the predicted distributions as given.
    The theory anchor for "the decoder family we are auditing": read to understand what
    is provable when the marginals are assumed correct — and why auditing the marginals
    (our C1) is the missing ingredient.

30. **Re-evaluating Confidence Remasking in Masked Diffusion Language Models** — Frkovic et al., 2026. arXiv:2606.12232 (P)
    ★★
    **Why:** Finds a post-hoc **remasking** method on par with confidence unmasking in
    the standard short-block greedy setting. Important negative-leaning prior result:
    if rematching/remasking performs as well as confidence gating, the value of the
    confidence statistic is questionable — contextualizes both our motivation and the
    null case for C1.

31. **Tweedie's Formula and Selection Bias** — Efron, JASA 2011.
    ★★★
    **Why:** The statistical foundation of the core claim: *selecting the maximum of
    noisy estimates biases the selected value upward* (the winner's curse). The
    proposal's model ℓi = ℓ*ᵢ + εᵢ, retain-the-max, "the maximum over several noisy
    estimates will tend to belong to a candidate whose error ran in the favourable
    direction" is Efron's machinery applied to token commitment. Read the parts on
    post-selection inference — Eq. 2's z_{r,m} (expected order statistic) comes from
    this literature.

32. **Inference on Winners** — Andrews, Kitagawa, McCloskey, QJE 2024.
    ★★★
    **Why:** The proposal's second citation for the winner's curse: the modern
    econometric treatment of selecting and then inferring about winners (maxima of
    noisy signals). Gives the formal language for "selection rules favour candidates
    whose estimation error happened to run high" and the bias *growing with field
    size* — our primary endpoint βz.

33. **On Calibration of Modern Neural Networks** — Guo et al., ICML 2017. arXiv:1706.04599
    ★ (optional, not in the proposal's references)
    **Why:** Standard vocabulary for what "calibrated confidence" means (ECE, reliability
    diagrams) in the ML literature. Useful for interpreting C1's sufficiency framing
    ("confidence remains a sufficient statistic for correctness") and for reporting
    reliability curves alongside the regression.

34. **Confident Adaptive Language Modeling (CALM)** — Schuster et al., NeurIPS 2022. arXiv:2207.07061
    ○
    **Why:** Deferred, per the proposal: risk-controlled *stopping and acceptance*
    decisions for adaptive LMs — the closest prior work to risk-controlled thresholding
    for commitment. Read only if you want the background for the work explicitly out of
    scope (so you can write the "future work" paragraph).

35. **Conformal Risk Control** — Angelopoulos et al., ICLR 2024. arXiv:2208.02814
    ○
    **Why:** Deferred machinery. C3 could have been a user-specified tolerance with
    finite-sample validity; the proposal cuts it for compute reasons. Know what it is
    (one line: conformal prediction applied to risk) to state the deferral accurately.

36. **Learn Then Test: Calibrating Predictive Algorithms to Achieve Risk Control** — Angelopoulos et al., Annals of Applied Statistics 2025. arXiv:2310.18547
    ○
    **Why:** The high-probability companion to #35 (Waudby-Smith–Ramdas bounds). Same
    deferral rationale.

37. **Fast yet Safe: Early-Exiting with Risk Control** — Jazbec et al., NeurIPS 2024. arXiv:2405.20915
    ○
    **Why:** Companion to #34: risk control applied to early-exit (i.e., stop-when-
    confident) decisions, including for diffusion generation. Together #34–#37 define
    the boundary of what we are *not* doing: stopping/acceptance vs. commitment.

---

## Phase 6 — Datasets & benchmarks

Short abstracts only; you need to know what the ground truth actually is, since C1's
measurement rests on teacher-forced infilling against held-out text.

38. **The FineWeb Datasets** — Penedo et al., 2024. arXiv:2406.17557 (P)
    ★★
    **Why:** Source of our 2,000-document held-out calibration corpus (FineWeb-Edu 10BT
    subset, truncated to 256 tokens). Read for the dataset's composition/filtering so you
    can justify document-level splits and the 50/50 fit/eval split.

39. **Training Verifiers to Solve Math Word Problems (GSM8K)** — Cobbe et al., 2021. arXiv:2110.14168 (P)
    ★
    **Why:** 1,319 test problems; reference/canonical solutions supply ground-truth
    tokens for teacher-forced infilling on math. Read the task format so the
    tokenization/ground-truth alignment is done right.

40. **Program Synthesis with Large Language Models (MBPP)** — Austin et al., 2021b. arXiv:2108.07732 (P)
    ★
    **Why:** 500-problem test split; same role as GSM8K for code.

41. **Evaluating Large Language Models Trained on Code (HumanEval)** — M. Chen et al., 2021. arXiv:2107.03374 (P)
    ★
    **Why:** 164 problems, **end-task evaluation only** (too small for the fitting
    splits). Know exactly what pass@k means here since it appears only in the C3
    comparison ("task accuracy at matched tokens per step").

42. **SchemaStore** — https://github.com/SchemaStore/schemastore (not a paper)
    ★★
    **Why:** Source of the 200 JSON schemas for the **controlled synthetic probe**
    (1,000 instances, values sampled under a known generative process). Because p* is
    known in closed form on the probe, both terms of Eq. 1 are computable exactly in
    nats — the probe is the anchor for all of C2. Skim the schema formats you will
    actually draw from.

---

## Quick reference table

| # | Paper | Role in project | Priority |
|---|-------|-----------------|----------|
| 1 | DLM survey (Li et al. 2025) | Orientation map | ○/★★★ |
| 2 | DDPM (Ho et al. 2020) | Denoising intuition | ★★★ |
| 3 | D3PM (Austin et al. 2021a) | Discrete/masked diffusion framework | ★★★ |
| 4 | MaskGIT (Chang et al. 2022) | Origin of confidence-based commit rule | ★★★ |
| 5 | MDLM (Sahoo et al. 2024) | Recipe for our 110M model | ★★★ |
| 6 | LLaDA (Nie et al. 2025) | Audited checkpoint #1 | ★★★ |
| 7 | Dream 7B (Ye et al. 2025) | Audited checkpoint #2 | ★★★ |
| 8 | Fast-dLLM (Wu et al. 2025) | Baseline decoder; global threshold | ★★★ |
| 9 | EB-Sampler (Ben-Hamu et al. 2025) | Entropy statistic; (A)/(B)-like error split | ★★ |
| 10 | Margin ordering (J. Kim et al. 2025) | Alternative statistic (top-k margin) | ★★ |
| 11 | KLASS (S. H. Kim et al. 2025) | Stability selector; C3 baseline | ★★ |
| 12 | OSDT (Shen & Ro 2025) | One-shot threshold transfer claim | ★ |
| 13 | Two temperatures (Olausson et al. 2026) | Competing explanation for βz | ★★ |
| 14 | PC-Sampler (Huang et al. 2025) | Closest prior work; competing covariate in Eq. 2 | ★★★ |
| 15 | SWD (Wu & Huang 2026) | Inter-step stability theorem; C3 corroboration | ★★ |
| 16 | LESS (Mohamed et al. 2026) | Stability family (persistence) | ★ |
| 17 | STDec (Y. Chen et al. 2026) | Stability family (spatial) | ★ |
| 18 | DAPD (B. Kim et al. 2026) | Dependency term (A) via attention | ★★ |
| 19 | DEMASK (Ringel et al. 2026) | Pairwise influence; TV bound on (A) | ★★ |
| 20 | ParallelBench (Kang et al. 2026) | Parallel-decoding trade-offs; total correlation | ★★ |
| 21 | Bansal & Sanghavi 2025 | Approximate joint sampling | ★ |
| 22 | PUNT (Azangulov et al. 2026) | CI-testing selection (rank-free) | ★ |
| 23 | APD (Israel et al. 2025) | AR-verifier parallel decoding | ★ |
| 24 | dParallel (Z. Chen et al. 2026) | Learned parallel decoding | ★ |
| 25 | Learned unmasking (Jazbec et al. 2026) | Strongest challenge: absorbs bias | ★★ |
| 26 | MDPO (He et al. 2025) | Training-inference divide (training arm) | ★ |
| 27 | TRIMS (L. Chen et al. 2026) | Training-side remedy (related work) | ★ |
| 28 | Tang et al. 2026 | Sampler/model error entanglement (C2 target) | ★★★ |
| 29 | Cai & Li 2026 | Theory of confidence-based decoding | ★★ |
| 30 | Frkovic et al. 2026 | Remasking ≈ confidence unmasking (context) | ★★ |
| 31 | Efron 2011 | Selection bias / winner's curse | ★★★ |
| 32 | Andrews et al. 2024 | Inference on winners (field-size bias) | ★★★ |
| 33 | Guo et al. 2017 | Calibration vocabulary (optional) | ★ |
| 34 | CALM (Schuster et al. 2022) | Deferred: risk-controlled stopping | ○ |
| 35 | Conformal risk control (2024) | Deferred machinery | ○ |
| 36 | Learn then test (2025) | Deferred machinery | ○ |
| 37 | Jazbec et al. 2024 | Deferred: early-exit risk control | ○ |
| 38 | FineWeb (Penedo et al. 2024) | Calibration corpus | ★★ |
| 39 | GSM8K (Cobbe et al. 2021) | Ground-truth infilling (math) | ★ |
| 40 | MBPP (Austin et al. 2021b) | Ground-truth infilling (code) | ★ |
| 41 | HumanEval (M. Chen et al. 2021) | End-task evaluation only | ★ |
| 42 | SchemaStore (repo) | Synthetic probe (p* known → C2 anchor) | ★★ |

**Notes**

- Papers dated 2026 (e.g., #18–20, #22, #25, #27–30) are taken as-is from the
  proposal's reference list, which is the authoritative source for this project.
- #33 (Guo et al. 2017) is the only paper not in the proposal's references; it is
  optional background for the calibration vocabulary.
- Recommended weekly pace: Phase 0 in week 1 (before Stage 0 pilot), Phase 1–2 by
  week 2–3 (before C1 regressions are specified), Phase 3–4 by week 4–5 (before
  writing related work), Phase 5 in parallel throughout, Phase 6 when data prep starts.
