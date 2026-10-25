#!/usr/bin/env python3
"""Build a .ipynb from a percent-format .py (``# %%`` code cells, ``# %% [markdown]`` cells).

    python notebooks/build_notebook.py notebooks/c3_dorigo_methods.py

Writes the notebook next to the source, with no outputs.  Lines before the first ``# %%``
(the header comment) are left out.
"""
import json
import sys
from pathlib import Path


def cells(text: str):
    kind, buf, out = None, [], []
    for line in text.splitlines():
        if line.startswith("# %%"):
            if kind is not None:
                out.append((kind, buf))
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
            continue
        if kind is not None:
            buf.append(line[2:] if kind == "markdown" and line.startswith("# ") else
                       "" if kind == "markdown" and line == "#" else line)
    if kind is not None:
        out.append((kind, buf))
    for i, (kind, lines) in enumerate(out):
        while lines and not lines[-1].strip():
            lines.pop()
        src = [ln + "\n" for ln in lines[:-1]] + lines[-1:]
        cell = {"cell_type": kind, "id": f"c{i:03d}", "metadata": {}, "source": src}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        yield cell


def main(src: str) -> None:
    p = Path(src)
    nb = {"cells": list(cells(p.read_text(encoding="utf-8"))),
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"},
                       "colab": {"provenance": []}},
          "nbformat": 4, "nbformat_minor": 5}
    out = p.with_suffix(".ipynb")
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main(sys.argv[1])
