#!/usr/bin/env python3
"""Execute a notebook headless, with config overrides, and save it with its outputs.

    python notebooks/run_notebook.py notebooks/c3_dorigo_methods.ipynb --out run.ipynb \\
        --set USE_SYNTHETIC=False --set 'HALF_MODE="fit_only"'

Each ``--set NAME=EXPR`` is appended to the first code cell (the config cell) as
``NAME = EXPR``, so it overrides what the cell sets; EXPR is Python source.  The
executed notebook is written to ``--out`` even when a cell fails, and the exit
status is 1 then.  Needs nbclient, nbformat and ipykernel in the interpreter
that runs this script; the kernel is that same interpreter.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebook")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cwd", default=".", help="working directory of the kernel")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=EXPR")
    ap.add_argument("--timeout", type=int, default=6 * 3600, help="per cell, seconds")
    a = ap.parse_args()

    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    nb = nbformat.read(a.notebook, as_version=4)
    first = next(c for c in nb.cells if c.cell_type == "code")
    lines = []
    for kv in a.set:
        name, sep, expr = kv.partition("=")
        if not sep or not name.isidentifier():
            ap.error(f"--set wants NAME=EXPR, got {kv!r}")
        compile(expr, f"--set {name}", "eval")
        lines.append(f"{name} = {expr}")
    if lines:
        first.source += "\n\n# overrides from notebooks/run_notebook.py\n" + "\n".join(lines)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    client = NotebookClient(nb, timeout=a.timeout, kernel_name="python3",
                            resources={"metadata": {"path": a.cwd}})
    try:
        client.execute()
        ok = True
    except CellExecutionError as exc:
        print(str(exc)[-4000:], file=sys.stderr)
        ok = False
    finally:
        nbformat.write(nb, out)
    print(f"[run_notebook] {'OK' if ok else 'FAILED'} in {time.time() - t0:.0f} s -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
