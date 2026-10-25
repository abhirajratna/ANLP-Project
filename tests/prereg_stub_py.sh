#!/bin/bash
# Stand-in for the interpreter in a dry run of slurm/prereg_analysis.sbatch (PY=<this>):
# it logs each call and creates the results.json that call would write under --out, and
# reads no trace.  `-c` snippets go to a real python: prereg/analysis_commands.sh reads
# the T1 floors (artifacts/t2_pilot/floor_*.json) that way.
#
#   STUB_LOG      where the calls go (default $OUT/stub_calls.log)
#   STUB_REAL_PY  the python for -c (default python3)
if [ "${1:-}" = -c ]; then
    exec "${STUB_REAL_PY:-python3}" "$@"
fi
log="${STUB_LOG:-${OUT:?}/stub_calls.log}"
mkdir -p "$(dirname "$log")"
echo "$*" >> "$log"
out=
prev=
for a in "$@"; do
    [ "$prev" = --out ] && out=$a
    prev=$a
done
if [ -n "$out" ]; then
    mkdir -p "$out" && echo '{"stub": true}' > "$out/results.json"
fi
