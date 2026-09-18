#!/bin/bash
#SBATCH --job-name=c3-buildenv
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
# u22 is HETEROGENEOUS: nodes advertising no feature are GTX 1080 Ti (compute
# 6.1, Pascal) and nodes advertising `2080ti` are RTX 2080 Ti (compute 7.5,
# Turing). bitsandbytes NF4 needs >= 7.5, fp16 tensor cores need >= 7.0, and the
# throughput budget assumes a 2080 Ti. Without this constraint the scheduler will
# happily hand you a 1080 Ti.
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode066,gnode069
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/buildenv-%j.out
#
# Builds the Python environment and ships it as ONE file.
#
# Three constraints force this shape.
#
#  1. The python/3.12.4 module resolves to /usr/local/apps, which exists on
#     compute nodes but not the login node, so the build must happen in a job.
#  2. /share1 allows ~3,000 inodes and the built env is 47,361 files, so it
#     travels as one tarball rather than unpacked.
#  3. **A venv is not relocatable.** Console scripts carry an absolute shebang,
#     so an env built at /scratch and extracted to /ssd_scratch has a `pip` that
#     silently installs into the *build* path while `python` reads the *extracted*
#     one. That failure is invisible: the install reports success and the package
#     never appears. The build path below is therefore identical to the path
#     activate_env.sh extracts to.
set -euo pipefail

BLOBS="${C3_BLOBS:-/share1/$USER}"
SCRATCH_ENV="${C3_SCRATCH:-/ssd_scratch/$USER}/c3-env"
TARBALL_LOCAL="${C3_SCRATCH:-/ssd_scratch/$USER}/c3-env.tar.gz"

if ! type module >/dev/null 2>&1; then
    for f in /opt/Modules/*/init/bash; do [ -r "$f" ] && . "$f" && break; done
fi
module load u22/python/3.12.4 u22/cuda/12.4

echo "[env] node=$(hostname) python=$(command -v python3) $(python3 -V 2>&1)"
python3 -c 'import sys; assert sys.version_info[:2]==(3,12), sys.version' \
  || { echo "[env] FATAL: python 3.12 not on PATH"; exit 1; }

rm -rf "$SCRATCH_ENV"
python3 -m venv "$SCRATCH_ENV"
# Always `python -m pip`, never the `pip` script -- see constraint 3 above.
PIP="$SCRATCH_ENV/bin/python -m pip"
$PIP install --no-cache-dir --upgrade pip wheel

# cu124 wheels carry sm_75 kernels, which is what a 2080 Ti needs.
$PIP install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch

# transformers is pinned to the 4.x line. LLaDA-8B and Dream-7B both ship
# trust_remote_code modeling files written against transformers 4.x; on 5.x the
# bitsandbytes 4-bit path calls get_keys_to_not_convert(), which reads
# `model.all_tied_weights_keys`, an attribute those files do not define:
#   AttributeError: 'LLaDAModelLM' object has no attribute 'all_tied_weights_keys'
# 4.46.3 is the version Dream's own config records, and LLaDA-8B loads in 4-bit
# under it (verified: 5.67 GB VRAM, correct infilling).
$PIP install --no-cache-dir \
    "transformers==4.46.3" "tokenizers>=0.20,<0.21" "huggingface_hub<1.0" \
    "datasets==3.1.0" accelerate bitsandbytes wandb
$PIP install --no-cache-dir numpy scipy pandas pyarrow statsmodels pytest

echo "[env] verifying"
"$SCRATCH_ENV/bin/python" - <<'PY'
import torch, transformers, datasets, bitsandbytes
print("torch", torch.__version__, "cuda build", torch.version.cuda)
print("sm_75 in arch list:", any("75" in a for a in torch.cuda.get_arch_list()))
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0),
          "capability:", torch.cuda.get_device_capability(0))
print("transformers", transformers.__version__, "datasets", datasets.__version__)
assert transformers.__version__.startswith("4."), \
    f"transformers must be 4.x for LLaDA/Dream remote code, got {transformers.__version__}"
# The 4-bit path is what actually breaks under 5.x, so import it explicitly.
from transformers import BitsAndBytesConfig
from transformers.quantizers.base import HfQuantizer
print("bitsandbytes", bitsandbytes.__version__, "- 4bit quantizer importable")
PY

echo "[env] files in env: $(find "$SCRATCH_ENV" | wc -l)  (this is why it ships as a tarball, not unpacked)"
# Ship to /share1: it has ~76 GB free against /home2's ~5 GB, and a tarball costs
# one of its 3,000 inodes. /share1 is not mounted here, so push it over ssh.
tar -czf "$TARBALL_LOCAL" -C "$(dirname "$SCRATCH_ENV")" "$(basename "$SCRATCH_ENV")"
echo "[env] tarball $(du -h "$TARBALL_LOCAL" | cut -f1); pushing to $BLOBS"
rsync -a --inplace --partial -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new" \
    "$TARBALL_LOCAL" "$USER@ada:$BLOBS/c3-env.tar.gz" \
    || { echo "[env] FATAL: could not push tarball to $BLOBS"; exit 1; }
echo "[env] shipped -> $BLOBS/c3-env.tar.gz (1 inode)"
