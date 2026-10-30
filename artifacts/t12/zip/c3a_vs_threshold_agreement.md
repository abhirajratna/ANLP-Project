# C3a vs threshold: per-problem agreement at matched speed slots
Per-problem correctness of the C3a and threshold configs in each speed slot, on the problems kept in the paper tables (MBPP N=121, PR #65; GSM8K N=659). The C3a and threshold configs are distinct (alpha vs tau; NFE differs per problem), so identical text is not guaranteed.

## MBPP

| slot | C3a config | threshold config | N | TT | TF | FT | FF | identical text | sum NFE (C3a / thr) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `c3a[alpha=0.039]` | `threshold[tau=0.986]` | 121 | 44 | 0 | 0 | 77 | 121 (100.0%) | 15901 / 15945 |
| 2 | `c3a[alpha=0.322]` | `threshold[tau=0.746]` | 121 | 41 | 2 | 2 | 76 | 47 (38.8%) | 8591 / 8308 |
| 3 | `c3a[alpha=0.735]` | `threshold[tau=0.339]` | 121 | 4 | 7 | 7 | 103 | 1 (0.8%) | 4114 / 4563 |

TT = both correct, TF = C3a correct and threshold wrong, FT = C3a wrong and threshold correct, FF = both wrong.

## GSM8K

| slot | C3a config | threshold config | N | TT | TF | FT | FF | identical text | sum NFE (C3a / thr) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `c3a[alpha=0.022]` | `threshold[tau=0.99]` | 659 | 527 | 0 | 1 | 131 | 649 (98.5%) | 81301 / 74570 |
| 2 | `c3a[alpha=0.271]` | `threshold[tau=0.859]` | 659 | 511 | 9 | 10 | 129 | 467 (70.9%) | 41724 / 41826 |
| 3 | `c3a[alpha=0.638]` | `threshold[tau=0.494]` | 659 | 340 | 62 | 104 | 153 | 19 (2.9%) | 19707 / 20394 |

TT = both correct, TF = C3a correct and threshold wrong, FT = C3a wrong and threshold correct, FF = both wrong.

