# Viva question bank: C3 (Committed Confidence Calibration)

Answers use only the paper (draft of 28 Oct, PR #58) plus the exploratory C3 check (parth, 28 Oct). "?" numbers are pending (T11/T12, controls, robustness); do not quote them. Numbers marked (dry run) come from the v1.1 dry run: re-check after the Thursday regen. Sign: Delta = confidence minus accuracy, positive = overconfident.

## A. Motivation and framing

**Q1. In one minute: what is the problem, and why is "selection" the issue?**
A masked diffusion LM commits the positions whose top token is most probable, so confidence chooses which tokens it then describes; on committed tokens confidence is a selected quantity, yet calibration is normally audited over random masks (Abstract, §1). The largest of many noisy estimates tends to be one whose error pushed it up (winner's curse, Andrews et al. 2024), and a commit out of many masked positions is such a maximum. We ask whether a committed token's confidence overstates accuracy more the more candidates it beat (§4).

**Q2. Calibration of diffusion LMs has been studied already. What is new?**
Prior work measures calibration over whole answers (Yadav et al.; Jiang and Park, §1, §7), or evaluates decoders on speed and accuracy, not the calibration of what they commit. None conditions on how many candidates a committed token beat (§7). We add the field-size dependence, an exact split of the gap, and a probe separating dependency from marginal error.

**Q3. Why three contributions (C1, C2, C3) rather than one?**
C1 asks whether the problem exists and how much is a winner's curse; C2 separates the other error of parallel commitment (dependency vs marginal, Eq. 1); C3 is a rule that uses calibrated confidence (§1). Be honest: C3 results are pending (§6, Table 3).

## B. Theory

**Q4. State Proposition 1 in plain words. Why does winning not bias a correct posterior?**
If, given the state, each position's (score, correctness) pair is independent of the other positions' scores, and the pool is drawn independently, then winning depends only on the other scores, which say nothing about this token's correctness. So Pr(correct | score, state, won) equals the context-aware calibration curve (§2, Appendix A.3, via weak union). Selection acts only through context.

**Q5. What exactly is the "field term" and why is it the winner's curse?**
Prop. 2: Delta(m) = own miscalibration E[sigma(l) − pi(l)] + field term E[pi(l) − pi(l,s)]. The field term is nonzero only if accuracy at fixed confidence depends on context, i.e. a nonzero grouping loss, and is exactly zero at m = 1 (§2). Own miscalibration can also make Delta grow with m (winners sit at high confidence), with no winner's curse, so only the field term is the testable curse.

**Q6. What assumptions can fail, and what happens then?**
Conditional independence given the state can fail if positions share errors; then Prop. 1 fails and (3) gains a third term E[pi(l,s) − a] (§2). Also "field of m" is modelled as a random pool, not the decoder's actual candidate set. The paper checks the third term and the within-state ICC (§4.1): the ICC is nonzero in every arm (up to 0.006 base/110M, 0.036 to 0.056 instruct). Weak spot: shared errors are present, so the exact split holds only approximately.

## C. C1 method and statistics

**Q7. How do the Holm families work, and what is "declared" versus "claimed"?**
E1 contrasts (Delta(32) − Delta(1)) and E2 contrasts (field(32) − field(1)) each form one Holm family over 14 cells (7 arms x uniform/decoder k=4) (§3.4, §4). Declared: 95% interval excludes 0 and the effect exceeds twice the non-determinism floor. Claimed: declared and Holm-rejected at 0.05. Result: for E1, 14 declared, 12 claimed; for E2, 14/14/14 (§4.1). Everything else is exploratory.
Weak spot: the floor is 0 (repeats were bitwise identical), so "declared" reduces to "interval excludes 0"; the safeguard adds nothing.

**Q8. Why those two calibrators, and what does it mean that the field term is "estimator-limited"?**
pi-hat(l) is isotonic in l; pi-hat(l,s) is logistic in a cubic spline of l plus state features (median l, mask rate, log|M|, step) and their products with l (§3.3). The field term lives between states, so it is only as context-aware as these features. On simulated no-field-term data the estimator read 0.001 to 0.002 at m=32 where the truth is zero (§3.5), so small base-model E2 values are not interpretable as a winner's curse.

**Q9. Bootstrap details, and what do the E2 intervals condition on?**
Document is the unit, 10,000 percentile resamples with one shared matrix per run, so contrasts use differenced draws (§3.4). Calibrators are fitted on a separate fit half and not refitted, so E2 intervals condition on the fit half and understate uncertainty (§3.4, §4.1). Weak spot: E2 intervals such as [0.001, 0.001] look tighter than the calibrator uncertainty justifies.

**Q10. How did you validate the pipeline, and what did simulation show about bias?**
Four simulated worlds (calibrated; uniformly miscalibrated; miscalibrated by context; shared errors) (§3.5). E1 was zero only in the first; beta_z without fixed effects was strongly negative only in the third. Findings: the field-term estimator reads high (0.001, 0.002 where truth is zero), fixed effects do not fully clean beta_z, and shared errors give a negative field term. The independent all-subsets estimator reproduces E1 (§4.5).

**Q11. Why teacher forcing, and what does it cost?**
Correctness against the true document is observed only if the document is known. States are those of an always-right decoder, not those after a free decoder's mistakes, and a valid rewording counts as an error (Limitations).
Weak spot: this is where it bites hardest for instruct models on free-form answers (GSM8K, MBPP).

## D. C1 results and their reading

**Q12. Headline result? Does the gap grow with field size?**
In decoder states (k=4) E1 is positive and claimed for every arm: base 0.006 (LLaDA) and 0.015 (Dream), 110M 0.009, instruct 0.024 to 0.186 (Table 2). In random masked states the sign varies: negative and claimed for both bases and LLaDA-I GSM8K (e.g. LLaDA-Base −0.003 [−0.004, −0.002]) (§4.1). We do not pool signs.

**Q13. Base-model effects are tiny (0.006). Why does it matter?**
Sizes differ widely: |E1| <= 0.015 for base and 110M, up to 0.26 for instruct on MBPP (§4.1). The native gap, the one that actually reaches users, is 0.013 to 0.018 for base/110M and 0.079 to 0.176 for instruct, 1.8 to 4.4 times each arm's random-mask ECE, and 0.15 to 0.27 at 128 to 255 masked positions for instruct (§4.3). The practical size sits in instruct models.
Weak spot: the native-gap description mixes field size with lost context and tests nothing (§4.3).

**Q14. Is the growth a winner's curse?**
Only partly. The field term E2 grows in 7 of 8 instruct cells (0.013 to 0.176; Dream-I MBPP uniform 0.176 while E1 is 0.257), read as the curse: at equal confidence winners of large fields come from contexts where tokens are right less often (§4.1). For base and 110M models |E2| < 0.005, comparable to estimator bias, so we do not read it as a curse; the decoder growth there is mostly own miscalibration (§4.5).
Sensitivity (dry run): under the additive calibrator (no products of l with the state features) the instruct field term is smaller in all eight cells (0 to 72% of its primary size) and keeps its sign with the interval excluding 0 in seven; in LLaDA-I MBPP uniform states it drops to 0 (§4.5, Context calibrator).
Weak spot: E2's size, not only its interval, depends on the calibrator's form. The abstract scopes the mechanism correctly ("its field term does too for the instruct models"); be ready to say that in base models the growth is own miscalibration.

**Q15. Why do uniform and decoder states differ, and in what direction?**
In uniform states E1 is negative for the bases, in decoder states it flips positive (§4.1). Matched uniform states (exploratory) keep the negative sign for the bases (LLaDA −0.002), so the change comes from which positions the decoder leaves masked; for LLaDA-I GSM8K matched uniform states are already positive (0.016 [0.010, 0.022]), so partly from how many. The E3 numbers are still "?" (§4.3).

**Q16. Instruct versus base?**
Instruct models show much larger gaps and field terms, and a clear within-state ICC (0.036 to 0.056 vs <= 0.006) (§4.1). Weak spot: model type and text type are confounded (FineWeb-Edu for base, GSM8K/MBPP for instruct; Limitations), so "instruction tuning causes this" is not identified.

**Q17. How does the native gap relate to ECE, and which matters?**
ECE is over random masks; the native gap is c − a over tokens the decoder commits, grouped by masks at commit time. For LLaDA-I GSM8K ECE is 0.033 and the native gap at 32 to 63 masked positions 0.079 [0.071, 0.086] (dry run). Across all seven arms the native gap exceeds the ECE (1.8 to 4.4x) (§4.3). An ECE audit therefore under-reports commit-time overconfidence.

**Q18. How robust is it? What about contamination and unseen text?**
On 500 time-split Wikipedia articles (post-release), the small negative uniform E1 of both bases is not reproduced (intervals hold 0), so the FineWeb result may reflect seen text, domain change or low power; the decoder-state E1 stays positive for all three models (§4.5). The 110M model, which saw neither set, shows larger E1 on the new domain (0.012 vs 0.002), so domain, not contamination, explains that. Contamination-filtered, knot, additive-calibrator and 4-bit results are pending.

## E. C2

**Q19. What does C2 show and why a synthetic probe?**
On a JSON probe with known joint (200 schemas, leader/follower pairs with coupling rho) both errors of Eq. 1 are computed exactly in nats (§5.1). Dependency (A) overtakes estimation (B) only for rho >= 0.5 and several tokens per step: at rho=1, k=8 (confidence) A is 0.338 and B 0.109, 76% of total (§5.2). Confidence ranking raises A, not B (rho=1, k=2: 0.124 vs 0.040 random).
Weak spot: one synthetic distribution and one 110M fine-tune (Limitations); even the sibling fine-tune only partially reads the marker.

**Q20. Why does B change with rho when rho should only move A? And is real text affected by dependency?**
The model reads the marker only in part, so a decoded follower stays broad (error in B), and committed in the same step the missing coupling counts as A; at rho=1, B falls from k=1 to 8 (0.109) while A rises (0 to 0.338) (§5.2). On real text the proxy (parallel-versus-sequential disagreement) is 0.0021 (110M) and 0.0022 (LLaDA) at k=8 (§5.3). Weak spot: the proxy tracks A only weakly (Spearman 0.49 and 0.28) and sees only learned dependence, so it does not show A is small.

## F. C3

**Q21. What is C3a, and what do you have to show for it?**
It commits the n positions with recalibrated confidence >= 1 − alpha (n >= 1) in the model's own order, no extra forward pass, alpha as the speed knob (§6). The calibrated-error guarantee for non-forced commits holds only if the calibrator is monotone in l, which the l x feature terms do not guarantee (fraction pending). T12 matched-speed results are pending. An exploratory, teacher-forced check (report-half LLaDA-8B-Instruct decoder states, one commit step replayed) found C3a's unforced commits have per-token error 0.23 to 0.51 x alpha at every alpha, while the raw threshold's is 0.64 to 1.50 x alpha and exceeds alpha at alpha = 0.01 on GSM8K; at matched tokens per state the two differ by at most 0.004 on GSM8K (no speed-accuracy gain).
Weak spot: the paper's third contribution presently has no positive result.

**Q22. Is the C3 comparison fair?**
Knobs are set by speed alone on fit-half problems, never by accuracy; every decoder is compared at its nearest setting within 10%; paired bootstrap on shared problems (§6). But C3a is fitted on teacher-forced states and applied to free generation, one model, greedy, one seed, two tasks, uncorrected multiplicity (Limitations, §6).

## G. Limitations and threats to validity

**Q23. Biggest threats to validity?**
Teacher forcing; model and text type confounded; 8B models in fp16 only (4-bit control pending); the field term is only as context-aware as three or four state features (§Limitations); contamination screen catches verbatim recall only; most robustness numbers are pending; levels beyond the 110M arm are not claimed, only change with field size.

**Q24. Many analyses are "exploratory". Is the confirmatory part just a handful of contrasts?**
Yes: only E1/E2 contrasts in two families of 14 are claims; decoder k=2, 8, beta_z, gamma_k, E3, native gap, controls and robustness runs carry no claims (§3.4). That is honest, but the explanations (position, token type) rest on controls whose Table 4 is empty.

## H. Why care

**Q25. Why should a practitioner care, if base-model gaps are about 0.01?**
Because commit-time overconfidence is up to 0.08 to 0.18 for instruct models (native gap, §4.3), and threshold decoders such as Fast-dLLM treat confidence as a probability of being right. It tells decoder designers where a calibrated cutoff has headroom, and C2 tells designers when parallel commits fail from dependency versus marginal error. The paper does not yet show a speed-accuracy gain.

---

## Top 5 weak spots

1. **C3 has no positive result (T12 pending; exploratory check showed no speed-accuracy gain over a raw threshold).** Defuse: "We report C3a as a calibration-preserving rule (unforced commits stay under the target risk) and an honest null on speed-accuracy at matched tokens per step; the contribution stands on C1 and C2."
2. **The winner's curse (field term) is shown only for instruct models; for base/110M the field term is of the order of estimator bias, and its size in instruct models halves under the additive calibrator.** Defuse: the abstract already limits the field term to the instruct models; say that base-model growth is own miscalibration, and that the instruct field term keeps its sign in 7 of 8 cells under the second calibrator while its size depends on the calibrator's form.
3. **Teacher forcing, and the instruct versus base confound (text type and model type change together).** Defuse: state that effects are measured in always-right-decoder states and note that only the 110M arm isolates domain.
4. **Statistical design: floor is 0 so "declared" adds nothing; E2 intervals condition on fixed calibrators; Holm families only cover the 14-cell contrasts; effects with shared errors (ICC > 0 everywhere) break exactness.** Defuse: say that the claims rest on intervals and Holm alone, and that E2 intervals are conditional on the fit-half calibrators.
5. **Real-text reading of C2 and the uniform-versus-decoder sign flip: the dependency proxy tracks A only weakly (Spearman 0.49 and 0.28), the probe is one synthetic distribution and one 110M model, and the time-split run does not reproduce the base uniform E1.** Defuse: say that real text supports a direction, not a split in nats, and that the decoder-state positive E1 is the result replicated on unseen text.
