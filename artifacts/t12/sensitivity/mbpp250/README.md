# T12 MBPP completion run, N=250 (sensitivity check only)

**Not for the paper.** Deviation 17 / PR #74: the MBPP completion run to 250 report-half problems was split across two H200
instances (and is not the machine run of the paper's N=121), so it is a sensitivity check. Section 6 and `artifacts/c3/macros_c3.tex` stay at
MBPP N=121. Nothing here is `\input` by the paper: the macros carry the prefix `CThreeTTwelveSens`.

## Inputs
- Records: HF `ANLP-project/c3-traces-8b`, `t12/report-a9abd7b/mbpp-250-sensitivity/` (commit `dee81ff`); local `runs/t12/report-a9abd7b/mbpp-250-sensitivity/` (3 shards, 5,250 records, 250 problems x 21 configurations).
- Scores: Ada job 48981, `score_t12.py` at `a9abd7b`; `runs/t12/report-a9abd7b/mbpp-250-sensitivity-scored/` (report json sha256 `3d818a10...`, verdicts sha256 `61cc155c...`).
- GSM8K, grids, calibrators, sanity file and manifests are the paper's; the GSM8K macros here equal `macros_c3.tex` (N=659).
- Command: `python scripts/c3_paper_outputs.py --allow-mixed-machines --tag mbpp250 --records runs/t12/report-a9abd7b/gsm8k runs/t12/report-a9abd7b/mbpp-250-sensitivity --verdicts <gsm8k-scored>/*.verdicts.jsonl <mbpp-250-sensitivity-scored>/*.verdicts.jsonl --scores <both report.json> --grids prereg --calibrators artifacts/c3 --sanity artifacts/t12/llada-8b-instruct_fp16_gsm8k-sanity.json --out-dir artifacts`.
- MBPP: N=250, 0 problems dropped, no shard missing. The summary's `mixed_machines` is `{}` because a resumed shard keeps the meta written on its first host (H200 instance 552094); the second instance (H200 VM 555317, the last 129 problems) is recorded only in deviation 17 (#74) and the HF commit title, so the guard could not see it, rather than being satisfied by it.

## Checks that the two machines agree (ayush, bus 9608 and ef0b)
- Byte prefix: each shard's first 868 / 821 / 862 records are byte-identical to the N=121 run (HF `6c6113c`).
- Re-decode: report-half docs 0, 1, 3 x 21 configurations = 63 records re-decoded on the completion VM at `a9abd7b`; 63/63 identical token-for-token to `6c6113c` (ef0b corrects 9608's "0/1/5"). Evidence: HF `.../mbpp-250-sensitivity/redecode-check/` (commit `3ca65a2448`).

## MBPP speed slot 2, N=121 (paper, `macros_c3.tex`) against N=250 (this directory)
tok/step = aggregate tokens per step; pass@1 with the 95% Wilson interval; difference = C3a minus the decoder at its matched
setting (positive = C3a higher), paired bootstrap 95% CI; `---` = no setting within +-10% of C3a's tok/step.

| decoder | N=121 tok/step | N=121 pass@1 | N=121 C3a - dec | N=250 tok/step | N=250 pass@1 | N=250 C3a - dec | change |
|---|---|---|---|---|---|---|---|
| Top-k | 4.00 | 0.207 [0.144, 0.287] | --- | 4.00 | 0.216 [0.169, 0.271] | --- |  |
| Threshold | 3.73 | 0.355 [0.276, 0.444] | 0.000 [-0.033, 0.033] | 3.78 | 0.344 [0.288, 0.405] | 0.028 [0.000, 0.056] | sign 0 -> +; CI includes 0 -> includes 0 (lower end 0.000: 0 is on the boundary) |
| Margin | 4.00 | 0.174 [0.116, 0.251] | --- | 4.00 | 0.212 [0.166, 0.267] | --- |  |
| Entropy | 3.83 | 0.355 [0.276, 0.444] | 0.000 [-0.058, 0.058] | 3.85 | 0.352 [0.295, 0.413] | 0.020 [-0.028, 0.068] | sign 0 -> +; CI includes 0 -> includes 0 |
| KLASS (off its published operating point) | 3.87 | 0.066 [0.034, 0.125] | 0.289 [0.198, 0.380] | 3.80 | 0.056 [0.034, 0.092] | 0.316 [0.256, 0.376] | sign + -> +; CI excludes 0 -> excludes 0 |
| PC-Sampler | 4.00 | 0.157 [0.103, 0.232] | --- | 4.00 | 0.172 [0.130, 0.224] | --- |  |
| C3a | 3.61 | 0.355 [0.276, 0.444] | (reference) | 3.61 | 0.372 [0.314, 0.433] | (reference) |  |

Fixed-count decoders at N=250, slot 2, against C3a's 3.608 tok/step: Top-k 4.000 (+10.9%); Margin 4.000 (+10.9%); PC-Sampler 4.000 (+10.9%). None is within +-10%, so none is matched
at slot 2 (as at N=121); their slot 1 / 2 / 3 tok/step are Top-k 2.00, 4.00, 8.00, Margin 2.00, 4.00, 8.00, PC-Sampler 2.00, 4.00, 8.00.
