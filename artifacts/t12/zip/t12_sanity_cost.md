# T12 sanity check and cost

Report problems: gsm8k N=659; mbpp N=121.

Sanity: the Ada fit-half GSM8K run (5-shot, Fast-dLLM settings, fp16, 2080 Ti) against the published LLaDA-8B-Instruct numbers (0.793 top-k k=1; 0.792 threshold tau=0.9); 95% Wilson intervals.

| task | decoder | config | n | acc | acc_lo | acc_hi | published | acc_minus_published |
|---|---|---|---|---|---|---|---|---|
| gsm8k | TopK | topk_conf[k=1] | 200 | 0.835 | 0.777 | 0.880 | 0.793 | 0.042 |
| gsm8k | Threshold | threshold[tau=0.9] | 200 | 0.845 | 0.788 | 0.889 | 0.792 | 0.053 |


Cost, report halves (zero-shot chat, 256 tokens, blocks of 32, greedy): tokens per step = 256 N_ok / sum NFE over the n_ok problems that decoded (driver errors are not imputed); NFE and seconds are per-problem means over them; failures = problems the driver could not decode (scored wrong); unstopped = no stop string within 256 tokens; timeouts = programs that timed out when scored; failed_any = problems with any of the three.

† = off the decoder's published operating point (KLASS slots 2-3: confidence threshold below its published range (deviation 17, PR #66)); the CSV has it as a boolean column.


| task | decoder | speed | config | n | n_ok | failures | unstopped | timeouts | failed_any | tokens_per_step | nfe | seconds_per_problem | outside_share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k | TopK | 1 | topk_conf[k=2] | 659 | 659 | 0 | 0 | 0 | 0 | 2.000 | 128.00 | 7.103 | --- |
| gsm8k | TopK | 2 | topk_conf[k=4] | 659 | 659 | 0 | 0 | 0 | 0 | 4.000 | 64.00 | 3.548 | --- |
| gsm8k | TopK | 3 | topk_conf[k=8] | 659 | 659 | 0 | 2 | 0 | 2 | 8.000 | 32.00 | 1.773 | --- |
| gsm8k | Threshold | 1 | threshold[tau=0.99] | 659 | 659 | 0 | 0 | 0 | 0 | 2.262 | 113.16 | 6.273 | --- |
| gsm8k | Threshold | 2 | threshold[tau=0.859] | 659 | 659 | 0 | 0 | 0 | 0 | 4.033 | 63.47 | 3.522 | --- |
| gsm8k | Threshold | 3 | threshold[tau=0.494] | 659 | 659 | 0 | 1 | 0 | 1 | 8.272 | 30.95 | 1.716 | --- |
| gsm8k | Margin | 1 | margin[k=2] | 659 | 659 | 0 | 0 | 0 | 0 | 2.000 | 128.00 | 7.104 | --- |
| gsm8k | Margin | 2 | margin[k=4] | 659 | 659 | 0 | 1 | 0 | 1 | 4.000 | 64.00 | 3.549 | --- |
| gsm8k | Margin | 3 | margin[k=8] | 659 | 659 | 0 | 5 | 0 | 5 | 8.000 | 32.00 | 1.774 | --- |
| gsm8k | Entropy | 1 | entropy_bounded[gamma=0.017] | 659 | 659 | 0 | 0 | 0 | 0 | 2.125 | 120.45 | 6.694 | --- |
| gsm8k | Entropy | 2 | entropy_bounded[gamma=0.5] | 659 | 659 | 0 | 0 | 0 | 0 | 4.276 | 59.87 | 3.326 | --- |
| gsm8k | Entropy | 3 | entropy_bounded[gamma=6.1] | 659 | 659 | 0 | 1 | 0 | 1 | 8.025 | 31.90 | 1.773 | --- |
| gsm8k | Klass | 1 | klass[eps_kl=0.0096,fallback=1,history=2,tau=0.803] | 659 | 659 | 0 | 0 | 0 | 0 | 2.025 | 126.44 | 10.758 | --- |
| gsm8k | Klass † | 2 | klass[eps_kl=0.17,fallback=1,history=2,tau=0.29] | 659 | 659 | 0 | 2 | 0 | 2 | 3.974 | 64.42 | 5.484 | --- |
| gsm8k | Klass † | 3 | klass[eps_kl=0.3,fallback=1,history=2,tau=0.2] | 659 | 659 | 0 | 7 | 0 | 7 | 4.900 | 52.24 | 4.423 | --- |
| gsm8k | PcSampler | 1 | pc_sampler[alpha=10.0,k=2,lam=0.25] | 659 | 659 | 0 | 0 | 0 | 0 | 2.000 | 128.00 | 7.094 | --- |
| gsm8k | PcSampler | 2 | pc_sampler[alpha=10.0,k=4,lam=0.25] | 659 | 659 | 0 | 0 | 0 | 0 | 4.000 | 64.00 | 3.552 | --- |
| gsm8k | PcSampler | 3 | pc_sampler[alpha=10.0,k=8,lam=0.25] | 659 | 659 | 0 | 3 | 0 | 3 | 8.000 | 32.00 | 1.771 | --- |
| gsm8k | CThreeA | 1 | c3a[alpha=0.022] | 659 | 659 | 0 | 0 | 0 | 0 | 2.075 | 123.37 | 9.120 | 0.016 |
| gsm8k | CThreeA | 2 | c3a[alpha=0.271] | 659 | 659 | 0 | 0 | 0 | 0 | 4.043 | 63.31 | 4.689 | 0.016 |
| gsm8k | CThreeA | 3 | c3a[alpha=0.638] | 659 | 659 | 0 | 0 | 0 | 0 | 8.561 | 29.90 | 2.207 | 0.020 |
| mbpp | TopK | 1 | topk_conf[k=2] | 121 | 121 | 0 | 0 | 1 | 1 | 2.000 | 128.00 | 18.422 | --- |
| mbpp | TopK | 2 | topk_conf[k=4] | 121 | 121 | 0 | 0 | 0 | 0 | 4.000 | 64.00 | 9.203 | --- |
| mbpp | TopK | 3 | topk_conf[k=8] | 121 | 121 | 0 | 0 | 0 | 0 | 8.000 | 32.00 | 4.622 | --- |
| mbpp | Threshold | 1 | threshold[tau=0.986] | 121 | 121 | 0 | 0 | 0 | 0 | 1.943 | 131.78 | 19.191 | --- |
| mbpp | Threshold | 2 | threshold[tau=0.746] | 121 | 121 | 0 | 0 | 0 | 0 | 3.728 | 68.66 | 10.041 | --- |
| mbpp | Threshold | 3 | threshold[tau=0.339] | 121 | 121 | 0 | 0 | 0 | 0 | 6.789 | 37.71 | 5.497 | --- |
| mbpp | Margin | 1 | margin[k=2] | 121 | 121 | 0 | 0 | 1 | 1 | 2.000 | 128.00 | 18.456 | --- |
| mbpp | Margin | 2 | margin[k=4] | 121 | 121 | 0 | 0 | 0 | 0 | 4.000 | 64.00 | 9.193 | --- |
| mbpp | Margin | 3 | margin[k=8] | 121 | 121 | 0 | 0 | 0 | 0 | 8.000 | 32.00 | 4.593 | --- |
| mbpp | Entropy | 1 | entropy_bounded[gamma=0.028] | 121 | 121 | 0 | 0 | 2 | 2 | 1.941 | 131.90 | 19.214 | --- |
| mbpp | Entropy | 2 | entropy_bounded[gamma=0.8] | 121 | 121 | 0 | 0 | 2 | 2 | 3.830 | 66.83 | 9.711 | --- |
| mbpp | Entropy | 3 | entropy_bounded[gamma=11.0] | 121 | 121 | 0 | 0 | 0 | 0 | 7.736 | 33.09 | 4.761 | --- |
| mbpp | Klass | 1 | klass[eps_kl=0.018,fallback=1,history=2,tau=0.701] | 121 | 121 | 0 | 0 | 0 | 0 | 1.886 | 135.73 | 26.544 | --- |
| mbpp | Klass † | 2 | klass[eps_kl=0.25,fallback=1,history=2,tau=0.229] | 121 | 121 | 0 | 0 | 0 | 0 | 3.868 | 66.18 | 12.810 | --- |
| mbpp | Klass † | 3 | klass[eps_kl=0.3,fallback=1,history=2,tau=0.2] | 121 | 121 | 0 | 0 | 0 | 0 | 4.317 | 59.31 | 11.421 | --- |
| mbpp | PcSampler | 1 | pc_sampler[alpha=10.0,k=2,lam=0.25] | 121 | 121 | 0 | 0 | 0 | 0 | 2.000 | 128.00 | 18.330 | --- |
| mbpp | PcSampler | 2 | pc_sampler[alpha=10.0,k=4,lam=0.25] | 121 | 121 | 0 | 0 | 0 | 0 | 4.000 | 64.00 | 9.191 | --- |
| mbpp | PcSampler | 3 | pc_sampler[alpha=10.0,k=8,lam=0.25] | 121 | 121 | 0 | 0 | 1 | 1 | 8.000 | 32.00 | 4.612 | --- |
| mbpp | CThreeA | 1 | c3a[alpha=0.039] | 121 | 121 | 0 | 0 | 0 | 0 | 1.948 | 131.41 | 22.681 | 0.031 |
| mbpp | CThreeA | 2 | c3a[alpha=0.322] | 121 | 121 | 0 | 0 | 0 | 0 | 3.606 | 71.00 | 12.518 | 0.037 |
| mbpp | CThreeA | 3 | c3a[alpha=0.735] | 121 | 121 | 0 | 0 | 0 | 0 | 7.529 | 34.00 | 5.964 | 0.038 |
