# Plan to the final submission (23–30 Oct 2026)

Written 23 Oct by ojas. The final is `DuckDuckDuckGoose-Final.zip`: the paper (≤ 8 pages + ≤ 2
appendix pages, ACL style), the code as a ZIP, and the viva slides as a PDF. It is due
Sat 31 Oct. We submit on Fri 30, all four of us. Ack or object on the bus by Sat 24, 10:00.

Agents may draft report text. The course dropped the human-written rule on 30 Sep;
`documents/project-timeline.md` and the guidelines PDF predate the change.

## Where we stand

Done:
- T0–T4.
- C1 traces for the six 8B arms, the 8B time-split runs and the 4-bit precision subset,
  merged and validated (PR #2, `documents/c1-sweep-status.md`).
- The 110M model (G2 pass), its C1 and time-split traces, and T9 with both fine-tunes (PR #3).
- Code: T8 controls (sized on the fit half), the decode loop with all six baseline policies,
  and the task loaders.

Not done:
- the `prereg-v1` tag and the OSF upload, so there is no report-half number yet;
- the 110M floor, and the 110M arm in `prereg/analysis_commands.sh`;
- T10;
- T11: the C3a policy and the calibrator fit;
- T12: the generation driver, the scoring, the runs;
- the paper, the slides, the trace release.

We are about ten days behind the mid report's schedule, hence the cuts at the end.

## Compute

| what | who | use it for |
|---|---|---|
| Ada (4 GPUs, 40 CPUs, 125 GB per user) | abhiraj, ojas, parth | everything that reads traces; the 8B analysis stays here (the traces are ~64 GB and Ada's uplink is ~0.1 MB/s) |
| Jarvis Labs, $100 credit | parth (T12) | LLaDA-8B-Instruct generation in bf16 on an A100; stop the instance whenever it is idle |
| Colab | anyone | 110M-scale jobs and figures; a free T4 cannot hold an 8B model in fp16 |

Ayush has no Ada account. His work is CPU code, writing, and whatever fits on Colab.

The 8B traces live under `/share1/abhiraj.ratna/traces/` and only Abhiraj can read them.
A read grant to the other two Ada accounts does two things: Parth can fit the C3a
calibrator, and the 8B analysis can run under three accounts' limits instead of one.

## Critical path

1. Fri: merge PRs #2 → #1 → #3.
2. Sat 12:00: the 110M arm and the open prereg items go in.
3. Sat evening: tag `prereg-v1`, upload to OSF.
4. Sat–Mon: the report-half analysis.
5. Wed 28, 12:00: results freeze.
6. Thu 29: full draft and slides.
7. Fri 30: submit.

T11 and T12 do not wait for the tag. The calibrator is fit on the fit half, and T12 is not
a pre-registered endpoint, so both run in parallel with the analysis.

## Who does what

### Abhiraj: C1 analysis, pre-registration (Ada)

1. **Fri 23:** give `ojas.k` and Parth's account read access to the 8B traces. The commands
   are `setfacl -m u:<user>:x /share1/abhiraj.ratna` and
   `setfacl -R -m u:<user>:rX /share1/abhiraj.ratna/traces`. If `/share1` has no ACL
   support, say so on the bus.
2. **Sat 24, 12:00:** settle the open items in `documents/c1-sweep-status.md`:
   - the Holm families;
   - the interacted context calibrator;
   - the task splits;
   - the T8 softmax temperature;
   - the deviations;
   - 1,000 resamples for the Eq. 2/3 bootstrap, recorded as a deviation (your option A).

   Post them to `all`. No objection by 18:00 means accepted.
3. **Sat 24, evening:** once the 110M lines are in, run `scripts/freeze_prereg.py` on a clean
   tree, tag `prereg-v1` on main, upload to OSF, and post the link.
4. **Sat 24 – Mon 26:** run `C3_REPORT_HALF_OK=1 bash prereg/analysis_commands.sh` on the 8B
   arms and commit the outputs (JSON, `macros.tex`, figures) by PR. After the first arm,
   post its wall-clock time.
5. **By Thu 29, 09:00:** write §2 Setup and theory, §3 Measurement instrument and §4 Results
   (E1–E3, the T8 table, the Holm summary).
6. **Repo owner tasks:**
   - make the repository public, or give the TA mentors access;
   - review PR #3's edits to `c3/traces/collect.py`, `c3/traces/models.py` and
     `c3/decode/loop.py`, which are in your area.

### Ojas: 110M arm, C2 (Ada)

1. **Sat 24, 12:00:** by PR, before the tag:
   - the 110M floor (`scripts/measure_floor.py`, fp32);
   - the zero-copy `dataset.json` for `main-110m-v3` and `timesplit-110m-v3`;
   - the 110M arms in `prereg/analysis_commands.sh`.

   The levels claims rest on this arm.
2. **Sun 25 – Mon 26:** run the 110M arms on the report half, after the tag.
3. **T9:** pick the main variant by the pre-set rule. That is `ft110m-v3-visible`, whose
   val NELBO with the marker visible is 0.1751 against 0.1892 (`val_rescored.json`).
   Then make the crossover figure.
4. **T10, by Tue 27:** measure parallel vs sequential disagreement on 300 documents × 10
   states, k ∈ {2, 4, 8}, on the 110M model (LLaDA-8B-Base too if time allows). Report its
   rank correlation with the true (A) on the probe.
5. **Release:**
   - start the 110M trace upload to HF by Sat; it takes 2–3 days at Ada's uplink;
   - the 8B traces (~64 GB) cannot leave Ada by the 30th at that rate, so release a subset
     or aggregates and say so in the README.
6. Commit the mid-eval slide sources, so Ayush can build the final deck from them.
7. **By Thu 29, 09:00:** write §5 (C2) and the 110M parts of §4.

### Parth: C3 (Ada, Jarvis)

1. **Fri 23:** say what the W&B runs `train110m-v4` and `ft110m-v4` are. They look like
   re-runs of the 110M arm. Label or delete them, so the README links point to one set.
2. **Sat 24, T11:** add C3a as a policy in `c3/decode/policies.py`: commit every masked
   position in the active block with ĉ ≥ 1 − α, and at least one. Fit its calibrator on the
   fit half of the LLaDA-Instruct GSM8K/MBPP traces and freeze it under `artifacts/c3/`.
   This needs Abhiraj's read grant, or Abhiraj runs the fit.
3. **Sat 24, T12 driver:** free-running LLaDA-8B-Instruct through `c3/decode/loop.py`
   (length 256, block 32), scored with Ayush's module. Log accuracy, tokens per step and
   wall-clock.
4. **Sun 25:** sanity check on ~200 GSM8K problems: top-k and the Fast-dLLM threshold
   against the published numbers at matched settings, within 1–2 points.
5. **Sun 25 – Wed 28, 12:00, the sweep:**
   - decoders: top-k, threshold, entropy-bounded, margin, KLASS, PC-Sampler and C3a, each
     at 3 speeds;
   - tasks: the report halves of GSM8K and MBPP, greedy, one seed, on Jarvis;
   - cost: about 30 A100-hours at batch size 1 by my estimate, which fits $100 at up to
     ~$3/h. Measure it on the sanity run and drop speeds before the credit runs out;
   - overflow: Ada (fp16, 2 GPUs per model), for whole tasks only, so that no comparison
     mixes precisions;
   - C3b and HumanEval only if both time and credit remain.
6. **By Thu 29, 09:00:** write §6 (C3) with the matched-speed table, and the related-work
   paragraphs on the decoders.

### Ayush: paper and integration (CPU, Colab)

1. **Fri 23 – Sat 24, T12 scoring (CPU only):**
   - GSM8K: final-answer extraction and exact match;
   - MBPP (and HumanEval): pass@1 by sandboxed execution with timeouts;
   - test: the reference solutions score 100%.

   Open a PR. This unblocks Parth's runs.
2. **Writing lead:**
   - set up the ACL skeleton with the plan v2 §8 outline;
   - write §1 Introduction, §7 Related work, and the limitations/future-work section (only
     that, immediately before the references);
   - fold in the other sections as they arrive;
   - hold the 8 + 2 page limit.
3. **Figures and tables:** generate them only from committed artifacts (`macros.tex` and
   the analysis outputs). Nothing is typed in by hand.
4. **Viva slides:** the PDF deck, built from the mid-eval deck.
5. **Checks:**
   - give the Propositions 1–2 proofs their second check;
   - verify every citation against arXiv or the venue (moved from Parth, who has T11 and T12).
6. **Integration:**
   - a README with the HF and W&B links and how to reproduce the results;
   - CI green;
   - on Thu 29, build the code ZIP from the final commit, then the Final ZIP.

## Timeline

| day | milestone |
|---|---|
| Fri 23 | PRs #2, #1, #3 merged; read grant on the 8B traces; scoring started; the v4 runs explained |
| Sat 24 | 110M arm (12:00); open prereg items settled (12:00, objections by 18:00); `prereg-v1` and OSF (evening); C3a and the T12 driver working; trace upload started |
| Sun 25 | report-half analysis running (8B, 110M); T12 sanity check, then the sweep |
| Mon 26 | 8B and 110M results committed; section drafting starts |
| Tue 27 | T10 done; two-thirds of the T12 sweep done; §1–§5 drafted |
| Wed 28 | **12:00 results freeze**: whatever is missing is cut or becomes a limitation; final figures |
| Thu 29 | full draft and slides; each of us reads the whole paper; fixes; ZIPs built |
| Fri 30 | all four submit (deadline Sat 31) |

## Cut to future work

- T13: other confidence statistics, the calibrator's data needs, 512/1,024-token outputs,
  block-wise vs whole-sequence decoding, bf16 vs 4-bit beyond the 200-document control,
  domain transfer, and a third model.
- C3b and HumanEval, unless Parth has both time and credit left.
- All of C3, if T12 is not done at the freeze. The paper then stands on C1 and C2, as plan v2
  and the timeline already allow.
- The full 8B trace release.

## Rules for the week

- Report-half numbers come only from `prereg/analysis_commands.sh` at the tag. Anything else
  is a deviation (`prereg/deviations.md`) or labelled exploratory.
- Every number in the paper comes from a committed artifact.
- One PR per task. Never push to main. Claim on the bus before shared edits (`CLAUDE.md`).
- Post a status line every day: `python3 tools/coord.py status "..."`.
