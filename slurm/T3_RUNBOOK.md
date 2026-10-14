# T3 on Ada: commands

Trains the v2 110M model on Ada from your laptop's copy of the repo; no GitHub
access is needed. The only download is the training data, from
<https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/main/sample/10BT>
(14 parquet files, 28.5 GB). One job fetches it once, checks every file's size
and sha256, and keeps the verified copy on `/share1`. Every later job reads that
copy, not the live Hub.

Laptop commands are PowerShell. Ada commands are bash on the login node unless
they start with `srun`.

## 1. One-time setup

Connect the IIIT VPN, then from the laptop:

```powershell
ssh parth.sharma@ada-gw1.iiit.ac.in      # or ada-gw2; ada.iiit.ac.in no longer accepts SSH
```

On Ada:

```bash
# Jobs copy files to and from /share1 over ssh, so this must work without a password.
[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
grep -qf ~/.ssh/id_ed25519.pub ~/.ssh/authorized_keys 2>/dev/null \
  || cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
ssh -o BatchMode=yes parth.sharma@ada-gw1 true && echo "ssh to ada-gw1 works"   # jobs reach /share1 this way

mkdir -p /home2/parth.sharma/c3/logs /home2/parth.sharma/c3-runs/corpus /share1/parth.sharma
df -h /home2/parth.sharma /share1/parth.sharma      # /share1 needs about 60 GB free
```

## 2. Copy the code from the laptop

Repeat this whenever the code changes on the laptop. The first block runs in a
PowerShell window on the laptop, **not** in the ssh session (a prompt reading
`[parth.sharma@ada-gw1 ~]$` means you are on Ada). `scp` asks for your Ada password.

```powershell
cd "C:\Users\parth\OneDrive\Documents\College Stuff\Sem5\ANLP\ANLP-Project"
$ADA = "parth.sharma@ada-gw1.iiit.ac.in"
tar -czf "$env:TEMP\c3.tgz" --no-fflags --exclude=__pycache__ --exclude=*.pyc .git .gitattributes pyproject.toml README.md c3 scripts slurm artifacts tests
scp "$env:TEMP\c3.tgz" "${ADA}:c3.tgz"
```

On Ada:

```bash
tar -xzf ~/c3.tgz -C /home2/parth.sharma/c3 && rm ~/c3.tgz
cd /home2/parth.sharma/c3 && sed -i 's/\r$//' slurm/*.sh slurm/*.sbatch && mkdir -p logs
```

## 3. Submit the pipeline

```bash
cd /home2/parth.sharma/c3
# Cancel T3 jobs left from an earlier submission (your other jobs are untouched).
squeue -u parth.sharma -h -o "%i %j" \
  | awk '$2 ~ /^c3-(buildenv|fineweb|corpus|exclusion|shards|smoke|train110m|verify)$/ {print $1}' \
  | xargs -r scancel

bash slurm/submit_t3.sh                  # 2 GPUs for training, about 1.6 days
# or, with nothing else of yours on GPUs: 4 GPUs, same batch, about 19 h
# C3_NGPU=4 C3_ACCUM=4 bash slurm/submit_t3.sh

squeue -u parth.sharma -o "%.10i %.14j %.9T %.25R"
```

This queues seven jobs; each starts when the jobs it needs have succeeded. The
probe checks a node's GPUs, then submits training pinned to that node, and
verify after it:

```text
env -> fineweb -> corpus -> exclusion -> shards -> probe -> train -> verify
  `--> smoke ------------------------------------------^
```

Check each log in `/home2/parth.sharma/c3/logs/` as its job finishes (`tail -30 logs/<log>`):

| job | log | time | done when the log shows |
| --- | --- | --- | --- |
| c3-buildenv | `buildenv-<id>.out` | ~30 min | `[env] shipped` |
| c3-fineweb | `fineweb-<id>.out` | ~1 h at 10 MB/s | `[fetch] complete: 14 files, 28.52 GB verified` |
| c3-corpus | `corpus-<id>.out` | minutes | `IDENTICAL to the committed held-out record` |
| c3-exclusion | `exclusion-<id>.out` | minutes | `main: 2000 documents ...; 2000 with ids` and `reserve: 500 documents` |
| c3-shards | `shards-<id>.out` | several hours | `'matched': 2500` in the exclusion summary, then `[stage] done` |
| c3-smoke | `smoke-<id>.out` | ~10 min | `resumed from ... at step N` with N > 0, `val NELBO` lines, `smoke test done` |
| c3-probe | `probe-<id>.out` | ~5 min | `passed; submitted train=... verify=...`. On a faulty node it prints `FAILED; resubmitted as job ...` and tries another node |
| c3-train110m | `train110m-<id>.out` | 1.6 days | `[train] done`; watch <https://wandb.ai/ANLP-project/c3-dllm-calibration> |
| c3-verify | `verify-<id>.out` | ~30 min | `[verify] wrote ...` and `G2 PASS` |

The smoke test's G2 verdict doesn't matter; its curve is too short.

## 4. While the data jobs run

Once `c3-buildenv` has finished, log in to W&B and run the tests. Do this before
`c3-train110m` starts, or the run logs offline and has to be uploaded later with
`wandb sync wandb/offline-run-*`.

```bash
srun -p u22 -A research --qos=medium -c 4 --mem=8G -t 00:30:00 --pty bash
type module >/dev/null 2>&1 || . /opt/Modules/*/init/bash   # srun shells lack `module`
cd /home2/parth.sharma/c3 && module load u22/python/3.12.4 && source slurm/activate_env.sh
"$(dirname "$C3_PY")/wandb" login        # your W&B user must be in the ANLP-project team
"$C3_PY" -m pytest tests/test_training.py tests/test_fineweb.py -q
exit
```

## 5. After `c3-verify`: bring the results to the laptop

```powershell
cd "C:\Users\parth\OneDrive\Documents\College Stuff\Sem5\ANLP\ANLP-Project"
$ADA = "parth.sharma@ada-gw1.iiit.ac.in"
scp "${ADA}:/home2/parth.sharma/c3/artifacts/exclusion_v2.json" artifacts/
scp "${ADA}:/home2/parth.sharma/c3/artifacts/*_train110m-v4.*" artifacts/
```

That brings `exclusion_v2.json`, `eval_110m_train110m-v4.json`,
`gate_g2_train110m-v4.json` and `history_train110m-v4.jsonl`. Commit them.

## 6. Publish to HuggingFace

```bash
srun -p u22 -A research --qos=medium -c 4 --mem=12000M -t 01:00:00 --pty bash
type module >/dev/null 2>&1 || . /opt/Modules/*/init/bash   # srun shells lack `module`
cd /home2/parth.sharma/c3 && module load u22/python/3.12.4 && source slurm/activate_env.sh
mkdir -p "$C3_SCRATCH/publish"
rsync -a -e ssh "parth.sharma@ada-gw1:/share1/parth.sharma/train110m-v4/final.pt" \
    "parth.sharma@ada-gw1:/share1/parth.sharma/train110m-v4/train_summary.json" "$C3_SCRATCH/publish/"
"$(dirname "$C3_PY")/huggingface-cli" login   # once; a token with write access to the org
"$C3_PY" scripts/upload_hf.py --ckpt "$C3_SCRATCH/publish/final.pt" \
    --repo ANLP-project/c3-mdlm-110m --eval artifacts/eval_110m_train110m-v4.json
exit
```

Add `--dry-run` to write the files without uploading, or `--public` to make the
repo public. Put the printed link in `README.md` under Project links.

## If a job fails

Every job after a failed one waits forever with `DependencyNeverSatisfied`.
Read the failed job's log, fix the cause, cancel the stuck jobs with the
`scancel` command from step 3, and resubmit from the failed step:

```bash
bash slurm/submit_t3.sh <step>     # env | fineweb | corpus | exclusion | shards | smoke | train | verify
```

| failure | what to do |
| --- | --- |
| `c3-buildenv` or `c3-fineweb` network errors | resubmit from that step; both resume where they stopped |
| `c3-corpus` prints `MISMATCH` | the data no longer reproduces the committed corpus. Ask Ojas to copy `/home2/ojas.k/c3-runs/corpus/calibration_docs.jsonl` to `/home2/parth.sharma/c3-runs/corpus/`, then `bash slurm/submit_t3.sh exclusion` |
| `c3-shards`: `never matched` | a held-out document's keys don't line up; the build refuses on purpose. Don't bypass it |
| `c3-train110m`: `belongs to a different run` | `C3_RUN_NAME=train110m-v5 bash slurm/submit_t3.sh train` |
| `c3-train110m`: CUDA out of memory | `C3_RUN_NAME=train110m-v5 C3_BATCH=8 C3_ACCUM=16 bash slurm/submit_t3.sh train` |

With a new run name, replace `train110m-v4` with it in steps 5 and 6.

## Optional: the autoregressive reference

After `c3-shards` has finished, with 2 GPUs free:

```bash
C3_OBJECTIVE=ar C3_RUN_NAME=ar110m-v2 bash slurm/submit_t3.sh train
```

## Where things end up

| path on Ada | contents |
| --- | --- |
| `/home2/parth.sharma/c3-runs/corpus/` | `calibration_docs.jsonl` (2,000 held-out docs), `reserve_docs.jsonl` (500 spares; T2 pilot documents), `exclusion_v2.json` |
| `/share1/parth.sharma/fineweb-10BT/` | the verified FineWeb copy, 28.5 GB |
| `/share1/parth.sharma/shards/` | tokenised training and validation shards, ~20 GB |
| `/share1/parth.sharma/train110m-v4/final.pt` | the trained model: traces use `--model local-110m --model-path ...`, and T9 uses `--init-from ...` |

Once training has finished, free about 28.5 GB with
`rm -rf /share1/parth.sharma/fineweb-10BT /share1/parth.sharma/shards.old`. Resubmitting
from `fineweb` fetches the data again if you need it.
