# T12 at speeds 1 and 3

Report problems: gsm8k N=659; mbpp N=121.

Per decoder: accuracy at its own speed-s setting (95% Wilson), and, when matched, the paired accuracy difference C3a minus the decoder at the setting nearest to C3a's slot s (within 10%; matched_slot / matched_tps), 95% percentile bootstrap CI (2,000 resamples of problems, seed 20261030). A blank is '---': no setting within 10%, or KLASS unmatched at the target.

† = off the decoder's published operating point (KLASS slots 2-3: confidence threshold below its published range (deviation 17, PR #66)); the CSV has it as a boolean column.

| task | decoder | speed | config | n | n_ok | tokens_per_step | acc | acc_lo | acc_hi | matched_slot | matched_tps | matched_diff | diff_lo | diff_hi | n_paired |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k | TopK | 1 | topk_conf[k=2] | 659 | 659 | 2.000 | 0.778 | 0.745 | 0.808 | 1 | 2.000 | 0.021 | -0.006 | 0.050 | 659 |
| gsm8k | Threshold | 1 | threshold[tau=0.99] | 659 | 659 | 2.262 | 0.801 | 0.769 | 0.830 | 1 | 2.262 | -0.002 | -0.005 | 0.000 | 659 |
| gsm8k | Margin | 1 | margin[k=2] | 659 | 659 | 2.000 | 0.766 | 0.733 | 0.797 | 1 | 2.000 | 0.033 | 0.005 | 0.062 | 659 |
| gsm8k | Entropy | 1 | entropy_bounded[gamma=0.017] | 659 | 659 | 2.125 | 0.783 | 0.750 | 0.813 | 1 | 2.125 | 0.017 | -0.011 | 0.046 | 659 |
| gsm8k | Klass | 1 | klass[eps_kl=0.0096,fallback=1,history=2,tau=0.803] | 659 | 659 | 2.025 | 0.791 | 0.758 | 0.820 | 1 | 2.025 | 0.009 | -0.002 | 0.021 | 659 |
| gsm8k | PcSampler | 1 | pc_sampler[alpha=10.0,k=2,lam=0.25] | 659 | 659 | 2.000 | 0.777 | 0.744 | 0.807 | 1 | 2.000 | 0.023 | -0.005 | 0.052 | 659 |
| gsm8k | CThreeA | 1 | c3a[alpha=0.022] | 659 | 659 | 2.075 | 0.800 | 0.767 | 0.828 | --- | --- | --- | --- | --- | --- |
| gsm8k | TopK | 3 | topk_conf[k=8] | 659 | 659 | 8.000 | 0.505 | 0.467 | 0.543 | 3 | 8.000 | 0.105 | 0.059 | 0.149 | 659 |
| gsm8k | Threshold | 3 | threshold[tau=0.494] | 659 | 659 | 8.272 | 0.674 | 0.637 | 0.708 | 3 | 8.272 | -0.064 | -0.106 | -0.027 | 659 |
| gsm8k | Margin | 3 | margin[k=8] | 659 | 659 | 8.000 | 0.558 | 0.520 | 0.596 | 3 | 8.000 | 0.052 | 0.008 | 0.096 | 659 |
| gsm8k | Entropy | 3 | entropy_bounded[gamma=6.1] | 659 | 659 | 8.025 | 0.648 | 0.611 | 0.683 | 3 | 8.025 | -0.038 | -0.083 | 0.005 | 659 |
| gsm8k | Klass † | 3 | klass[eps_kl=0.3,fallback=1,history=2,tau=0.2] | 659 | 659 | 4.900 | 0.291 | 0.258 | 0.327 | --- | --- | --- | --- | --- | --- |
| gsm8k | PcSampler | 3 | pc_sampler[alpha=10.0,k=8,lam=0.25] | 659 | 659 | 8.000 | 0.390 | 0.353 | 0.428 | 3 | 8.000 | 0.220 | 0.175 | 0.267 | 659 |
| gsm8k | CThreeA | 3 | c3a[alpha=0.638] | 659 | 659 | 8.561 | 0.610 | 0.572 | 0.647 | --- | --- | --- | --- | --- | --- |
| mbpp | TopK | 1 | topk_conf[k=2] | 121 | 121 | 2.000 | 0.281 | 0.209 | 0.367 | 1 | 2.000 | 0.083 | 0.017 | 0.149 | 121 |
| mbpp | Threshold | 1 | threshold[tau=0.986] | 121 | 121 | 1.943 | 0.364 | 0.283 | 0.452 | 1 | 1.943 | 0.000 | 0.000 | 0.000 | 121 |
| mbpp | Margin | 1 | margin[k=2] | 121 | 121 | 2.000 | 0.306 | 0.231 | 0.393 | 1 | 2.000 | 0.058 | -0.008 | 0.124 | 121 |
| mbpp | Entropy | 1 | entropy_bounded[gamma=0.028] | 121 | 121 | 1.941 | 0.355 | 0.276 | 0.444 | 1 | 1.941 | 0.008 | -0.050 | 0.066 | 121 |
| mbpp | Klass | 1 | klass[eps_kl=0.018,fallback=1,history=2,tau=0.701] | 121 | 121 | 1.886 | 0.331 | 0.253 | 0.418 | 1 | 1.886 | 0.033 | 0.008 | 0.066 | 121 |
| mbpp | PcSampler | 1 | pc_sampler[alpha=10.0,k=2,lam=0.25] | 121 | 121 | 2.000 | 0.289 | 0.216 | 0.376 | 1 | 2.000 | 0.074 | 0.008 | 0.149 | 121 |
| mbpp | CThreeA | 1 | c3a[alpha=0.039] | 121 | 121 | 1.948 | 0.364 | 0.283 | 0.452 | --- | --- | --- | --- | --- | --- |
| mbpp | TopK | 3 | topk_conf[k=8] | 121 | 121 | 8.000 | 0.033 | 0.013 | 0.082 | 3 | 8.000 | 0.058 | 0.000 | 0.116 | 121 |
| mbpp | Threshold | 3 | threshold[tau=0.339] | 121 | 121 | 6.789 | 0.091 | 0.052 | 0.155 | 3 | 6.789 | 0.000 | -0.058 | 0.066 | 121 |
| mbpp | Margin | 3 | margin[k=8] | 121 | 121 | 8.000 | 0.074 | 0.040 | 0.135 | 3 | 8.000 | 0.017 | -0.050 | 0.083 | 121 |
| mbpp | Entropy | 3 | entropy_bounded[gamma=11.0] | 121 | 121 | 7.736 | 0.066 | 0.034 | 0.125 | 3 | 7.736 | 0.025 | -0.041 | 0.091 | 121 |
| mbpp | Klass † | 3 | klass[eps_kl=0.3,fallback=1,history=2,tau=0.2] | 121 | 121 | 4.317 | 0.033 | 0.013 | 0.082 | --- | --- | --- | --- | --- | --- |
| mbpp | PcSampler | 3 | pc_sampler[alpha=10.0,k=8,lam=0.25] | 121 | 121 | 8.000 | 0.033 | 0.013 | 0.082 | 3 | 8.000 | 0.058 | 0.008 | 0.116 | 121 |
| mbpp | CThreeA | 3 | c3a[alpha=0.735] | 121 | 121 | 7.529 | 0.091 | 0.052 | 0.155 | --- | --- | --- | --- | --- | --- |
