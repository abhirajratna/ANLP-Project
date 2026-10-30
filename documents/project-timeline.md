# Project Timeline & To-Do List
## Selection Bias in Confidence-Based Token Commitment in Diffusion LMs (ANLP Monsoon 2026)

**Status as of Fri 4 Sep 2026**
- [x] Interim proposal (deadline 14 Aug 2026) — submitted
- [x] Final proposal — submitted (`documents/DuckDuckDuckGoose-Proposal.pdf`)
- [x] Literature review reading list (`documents/lit-review-list.md`)
- [ ] **Mid submission — due Wed 30 Sep 2026** ← 26 days
- [ ] **Final submission — due Sat 31 Oct 2026** (submit by Fri 30 Oct) ← 57 days

> ⚠️ Guidelines: schedule is **strict, no extensions**. Every team member must make
> every submission. AI is allowed for code and, since 30 Sep, for report writing too. ACL style
> (LaTeX template from the course) for all submissions.

---

## 1. Hard deadline requirements (from the guidelines)

| Deadline | What | Format | Filename |
|----------|------|--------|----------|
| Wed 30 Sep 2026 | Mid submission | 7–8 pages excl. refs + optional 1 appendix page. Formal problem statement, **progress**, **concrete timeline for the rest of the project**, comprehensive literature review (viability + motivation). Graded: write-up + code + viva (slides optional) | `<TeamNameWithNoSpaces>-Mid.zip` |
| Fri 30 Oct 2026 | Final submission | ≤ 8 pages excl. refs + optional 2 appendix pages. Full research-paper structure; **every claim backed by prior or empirical results**. Optional "challenges / limitations / future work" section **immediately before references**, containing *only that* (any result placed there is treated as a limitation). Code as ZIP inside the submission; viva presentation (preferably PDF) in the zip | `<TeamNameWithNoSpaces>-Final.zip` |

Standing repo requirements for **both** submissions:
- [ ] Repo **public, or TA mentors granted access** (do this once, before mid)
- [ ] README with **HuggingFace / WandB project links** if used (keep updated for both deadlines)

---

## 2. Master timeline (4 Sep → 31 Oct 2026)

Maps the proposal's 10-week plan (§8) onto the actual calendar, compressed so that
writing runs in parallel with experiments instead of only at the end.

### Phase A — to the mid submission

**W0 · Fri 4–Sun 6 Sep — Kickoff & scaffolding**
- [ ] Set up team git repo; make public or add TA mentor access (standing requirement)
- [ ] Pull ACL-style LaTeX template; create `mid/` and `final/` skeletons (sections pre-filled from the proposal)
- [ ] Accounts/infra: HF org, WandB project, GPU allocation; decide who runs what
- [ ] Assign roles: data prep · 110M training · trace harness/statistics · writing
- [ ] Start lit-review reading (foundation papers first, per `lit-review-list.md`)
- [ ] **TA mentor check-in #1** (confirm scope, compute, gate plan)

**W1 · Mon 7–Sun 13 Sep — Stage 0 pilot + data (proposal weeks 1–2)**
- [ ] FineWeb-Edu 10BT subset: sample 2,000 docs (fixed seed), truncate to 256 tokens, exclude from 110M training set
- [ ] 50/50 document-level fit/eval split, fixed seed
- [ ] Contamination screen pipeline: 50-token prefix prompt → verbatim continuation length → threshold flags (report every endpoint with and without flagged docs)
- [ ] **Start 110M MDLM training early** (proposal says weeks 3–4, but the C1 audit needs it by week 5; MDLM recipe, FineWeb-Edu 10BT minus calibration docs, GPT-2 BPE, from scratch)
- [ ] Stage 0 pilot: 20 docs on an existing 8B checkpoint (LLaDA-8B) — one forward pass at fixed masked states, pools m ∈ {2,4,8,16,32}; estimate (a) overall bias scale, (b) dispersion of position-specific noise
- [ ] Measure trace-level non-determinism floor: re-run 20 docs × 5 times (this becomes the "twice the floor" effect bar)
- [ ] Compute required sample size for 80% power

**W2 · Mon 14–Sun 20 Sep — Gate decision + instruments (proposal weeks 3–4)**
- [ ] **Stage 0 written gate** (record decision in repo):
      - proceed full if required size ≤ 1,000 docs **and** noise dispersion sufficiently large
      - else restrict C1 to the 110M model and shift compute to C2
- [ ] Trace schema + collection harness: per position `ℓᵢ, vᵢ, agreeᵢ` at fixed masked states; **uniform masks** (matching training) and **decoder-induced masks** (simulating the inference trajectory)
- [ ] Synthetic probe: 200 schemas from SchemaStore → 1,000 JSON instances, values under the specified generative process (p* known in closed form)
- [ ] Validate regression code (Eqs. 2–3: state fixed effects, `f(ℓᵢ)` flexible control, expected order statistic `z_{r,m}`, rank indicators `γ_k`) on a **synthetic toy with known bias** before touching real models
- [ ] Finish lit-review reading phases 1–3; start drafting the lit-review section for the mid (reuse `lit-review-list.md` rationale)
- [ ] **TA mentor check-in #2** (walk through gate result + mid plan)

**W3 · Mon 21–Sun 27 Sep — C1 pilot + mid writing**
- [ ] C1 pilot run on a small slice (e.g., 100 docs, 110M model once it's usable) — check βz and γₖ estimate sensibly, bootstrap pipeline works
- [ ] **Write the mid submission**: formal problem statement · comprehensive lit review (viability + motivation) · progress (pilot, gate, training status) · concrete timeline for the rest (update this doc and include it)
- [ ] Mid repo state: README with HF/WandB links, data-prep + pilot + training code committed and runnable
- [ ] Optional: mid viva slides
- [ ] First pass of `final/` paper skeleton: intro + methodology sections drafted (they mostly exist in the proposal)

**W4 · Mon 28 Sep–Sat 4 Oct — Submit mid, launch full C1**
- [ ] **MID SUBMISSION by Wed 30 Sep — all 4 team members submit** (`-Mid.zip`)
- [ ] Full C1 audit launches on all three models: 110M (bf16, cheap), LLaDA-8B (4-bit full sweep), Dream-7B (4-bit)
- [ ] Collect traces: fit half (1,000 docs) for Eqs. 2–3, eval half (1,000 docs) for reporting; both uniform and decoder-induced masks
- [ ] GSM8K (1,319) + MBPP (500) reference solutions formatted for teacher-forced infilling

### Phase B — to the final submission

**W5 · Mon 5–Sun 11 Oct — C1 (proposal week 5–7)**
- [ ] Finish C1 trace collection on all models
- [ ] Fit Eqs. 2–3 (fit on half 1, report on half 2); 95% intervals via document-clustered bootstrap, 10,000 resamples
- [ ] Check effects against the "twice the trace-level non-determinism floor" bar
- [ ] Co-primary: rank profile γₖ (the discriminator against dependency accounts)

**W6 · Mon 12–Sun 18 Oct — C1 controls + C2 (proposal weeks 7–8)**
- [ ] bfloat16 re-run: 200-doc subsample of LLaDA-8B with CPU offload; report βz under both precisions
- [ ] Contamination screen results folded in (levels claims rest on 110M arm; 8B arm carries gradient claims only)
- [ ] **C2 on the probe**: both terms of Eq. 1 computed exactly in nats; parallel-vs-sequential contrast calibrated against true (A)
- [ ] **C2 on real corpora**: CE(p*ᵢ, pθ,ᵢ) identified; contrasts in (B) across field sizes and mask distributions (levels not identified — report contrasts); disagreement rate as monotone proxy for (A), calibrated on the probe
- [ ] Report both terms **against tokens committed per step**
- [ ] Write results tables/figures as they land; final paper draft in steady progress

**W7 · Mon 19–Sun 25 Oct — C3 (contingent) + ablations (proposal week 9)**
- [ ] **If gate passed and C1 found an effect:** implement zero-cost shrinkage rule (C3) from the same regression; decoder suite at matched tokens/step — confidence top-k, Fast-dLLM thresholding, entropy-bounded (EB), margin, KLASS
- [ ] **If C1 null:** report the null (it explains why plain thresholding works), drop C3 + suite, extend C2 analysis
- [ ] Ablations — the two named competing explanations:
      - per-step, per-field-size **temperature** sweep (temperature can reorder positions)
      - **PC-Sampler position-aware weight** as competing covariate in Eq. 2 (if adding it drives βz to zero → effect is positional, not selection-induced)
- [ ] **TA mentor check-in #3** (draft review, viva plan)
- [ ] Final paper: all experiments frozen, numbers locked; second full draft

**W8 · Mon 26–Sat 31 Oct — Final submission (proposal week 10)**
- [ ] Final write-up complete: ≤ 8 pages + 2 appendix; every claim backed by results or prior work; "challenges, limitations, future work" section just before references (limitations only — no results)
- [ ] **Release traces** (the audit must be repeatable without repeating the compute)
- [ ] Code: cleaned, reproducible instructions in README (with HF/WandB links); package **code ZIP** for the submission
- [ ] Viva **presentation slides as PDF** (final viva requires a presentation, preferably in the submission)
- [ ] **FINAL SUBMISSION by Fri 30 Oct (deadline Sat 31 Oct) — all team members submit** (`-Final.zip`)

---

## 3. Submission checklists (run on each submission day)

### Mid (30 Sep)
- [ ] 7–8 pages excl. refs (+ ≤1 appendix page)
- [ ] Formal refined problem statement ✓ · progress ✓ · concrete timeline for rest of project ✓
- [ ] Comprehensive literature review (viability + motivation) ✓
- [ ] Repo public / TA mentor access granted
- [ ] README: HF/WandB links (if used)
- [ ] Code in repo runs (pilot, data prep, training scripts)
- [ ] ACL style, compiled, no errors
- [ ] **Every team member has made the submission**
- [ ] Viva slides prepared (optional but recommended)

### Final (30/31 Oct)
- [ ] ≤ 8 pages excl. refs (+ ≤2 appendix pages)
- [ ] Research-paper structure; all claims backed by prior or empirical results + sufficient analysis
- [ ] Limitations section immediately before references, containing only that content
- [ ] Code ZIP included in `-Final.zip`
- [ ] Viva presentation (PDF) included
- [ ] README: HF/WandB links; traces released
- [ ] ACL style, compiled
- [ ] **Every team member has made the submission**

---

## 4. Recurring / cross-cutting
- [ ] **TA mentor check-ins** — at minimum 3 (W0, W2, W7); more welcome; log decisions in the repo
- [ ] Weekly team sync: what ran, what failed, who owns what next
- [ ] Experiment log (WandB or a `logs/` dir): seed, model, precision, mask type, doc split — every run
- [ ] Keep `final/` LaTeX skeleton updated weekly; never leave writing to the last week
- [x] ~~Write-up discipline: human-written text only~~ (dropped 30 Sep: agents may write the report)

## 5. Top risks & buffers
1. **110M training is on the critical path.** Start in W1, not W3. If it isn't usable by W4–W5, run C1 on the two 8B models first and add the 110M arm as soon as it's ready (level-estimation claims rest on it, so this delay matters most for C1 levels, not gradients).
2. **No extensions exist.** W8 is writing-only by design; any experiment not frozen by Sun 25 Oct is descoped, not delayed.
3. **C3 is contingent by design** — the decoder suite only happens if C1 is positive. If it slips past W7, cut it and report C1+C2 as the deliverable (the proposal already pre-commits to this).
4. **4-bit vs bf16 discrepancy risk:** the bfloat16 re-run (W6) exists exactly because quantisation perturbs the audited logits — if βz flips sign across precisions, say so; it is a finding, not a failure.
5. **Null-result path:** if βz ≈ 0, the proposal already defines the paper's shape (null + C2 term separation + why plain thresholding works). Don't chase significance; report the pre-registered analysis.
