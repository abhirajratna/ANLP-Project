# T12 sanity check and cost

Report problems: gsm8k N=659; mbpp N=250.

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
| mbpp | TopK | 1 | topk_conf[k=2] | 250 | 250 | 0 | 0 | 1 | 1 | 2.000 | 128.00 | 13.004 | --- |
| mbpp | TopK | 2 | topk_conf[k=4] | 250 | 250 | 0 | 0 | 0 | 0 | 4.000 | 64.00 | 6.479 | --- |
| mbpp | TopK | 3 | topk_conf[k=8] | 250 | 250 | 0 | 0 | 0 | 0 | 8.000 | 32.00 | 3.239 | --- |
| mbpp | Threshold | 1 | threshold[tau=0.986] | 250 | 250 | 0 | 0 | 1 | 1 | 1.957 | 130.81 | 13.351 | --- |
| mbpp | Threshold | 2 | threshold[tau=0.746] | 250 | 250 | 0 | 0 | 0 | 0 | 3.776 | 67.80 | 6.942 | --- |
| mbpp | Threshold | 3 | threshold[tau=0.339] | 250 | 250 | 0 | 0 | 0 | 0 | 6.900 | 37.10 | 3.794 | --- |
| mbpp | Margin | 1 | margin[k=2] | 250 | 250 | 0 | 0 | 1 | 1 | 2.000 | 128.00 | 12.964 | --- |
| mbpp | Margin | 2 | margin[k=4] | 250 | 250 | 0 | 0 | 1 | 1 | 4.000 | 64.00 | 6.464 | --- |
| mbpp | Margin | 3 | margin[k=8] | 250 | 250 | 0 | 0 | 0 | 0 | 8.000 | 32.00 | 3.233 | --- |
| mbpp | Entropy | 1 | entropy_bounded[gamma=0.028] | 250 | 250 | 0 | 0 | 3 | 3 | 1.954 | 131.00 | 13.332 | --- |
| mbpp | Entropy | 2 | entropy_bounded[gamma=0.8] | 250 | 250 | 0 | 0 | 4 | 4 | 3.851 | 66.48 | 6.751 | --- |
| mbpp | Entropy | 3 | entropy_bounded[gamma=11.0] | 250 | 250 | 0 | 0 | 0 | 0 | 7.950 | 32.20 | 3.279 | --- |
| mbpp | Klass | 1 | klass[eps_kl=0.018,fallback=1,history=2,tau=0.701] | 250 | 250 | 0 | 0 | 1 | 1 | 1.900 | 134.73 | 19.000 | --- |
| mbpp | Klass † | 2 | klass[eps_kl=0.25,fallback=1,history=2,tau=0.229] | 250 | 250 | 0 | 0 | 0 | 0 | 3.804 | 67.30 | 9.351 | --- |
| mbpp | Klass † | 3 | klass[eps_kl=0.3,fallback=1,history=2,tau=0.2] | 250 | 250 | 0 | 0 | 0 | 0 | 4.328 | 59.16 | 8.176 | --- |
| mbpp | PcSampler | 1 | pc_sampler[alpha=10.0,k=2,lam=0.25] | 250 | 250 | 0 | 0 | 1 | 1 | 2.000 | 128.00 | 12.917 | --- |
| mbpp | PcSampler | 2 | pc_sampler[alpha=10.0,k=4,lam=0.25] | 250 | 250 | 0 | 0 | 0 | 0 | 4.000 | 64.00 | 6.430 | --- |
| mbpp | PcSampler | 3 | pc_sampler[alpha=10.0,k=8,lam=0.25] | 250 | 250 | 0 | 0 | 1 | 1 | 8.000 | 32.00 | 3.226 | --- |
| mbpp | CThreeA | 1 | c3a[alpha=0.039] | 250 | 250 | 0 | 0 | 1 | 1 | 1.962 | 130.50 | 16.025 | 0.034 |
| mbpp | CThreeA | 2 | c3a[alpha=0.322] | 250 | 250 | 0 | 0 | 1 | 1 | 3.608 | 70.96 | 8.833 | 0.044 |
| mbpp | CThreeA | 3 | c3a[alpha=0.735] | 250 | 250 | 0 | 0 | 1 | 1 | 7.662 | 33.41 | 4.164 | 0.034 |
