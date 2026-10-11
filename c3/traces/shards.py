"""Joining the shards of one collection run into a single trace.

A full sweep is 2,000 documents x ~466 forward passes per document on an 8B
model, which does not fit one job, so ``collect_shard`` splits it by
``doc_id % N`` and writes parts.  The merge is where silent mistakes would
enter: a shard collected at another precision, a revision that moved, a
document written twice after a resume.  So it refuses rather than repairs --
every shard meta must agree on the run's identity, every expected document must
be present exactly once, and no (state_id, position) pair may repeat.

A merge can also be **zero-copy**: with an output named ``*.dataset.json`` the
same checks run, but instead of a second parquet copy it writes a manifest of
the validated parts (relative path, sha256, rows) with the merged meta, and
``c3.assemble.load_traces`` reads the parts through it.  A full-design 8B arm is
~22 GB; copying every run would need twice the space, which /share1's quota
does not have.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .collect import IDENTITY_KEYS, SCHEMA_VERSION

_SHARD_META = re.compile(r"\.shard(\d+)of(\d+)\.meta\.json$")

DATASET_SUFFIX = ".dataset.json"
DATASET_FORMAT = "c3-trace-dataset-v1"


def is_dataset(path) -> bool:
    return str(path).endswith(DATASET_SUFFIX)


def dataset_files(manifest_path, verify: bool = False) -> list[Path]:
    """The part files a dataset manifest lists, checked to exist (and, with
    ``verify``, to still have the sha256 recorded at merge time)."""
    from ..provenance import sha256_file

    mp = Path(manifest_path)
    man = json.loads(mp.read_text())
    if man.get("format") != DATASET_FORMAT:
        raise MergeError(f"{mp}: not a {DATASET_FORMAT} manifest")
    files = []
    for f in man["files"]:
        path = (mp.parent / f["file"]).resolve()
        if not path.exists():
            raise MergeError(f"{mp}: part {f['file']} is missing")
        if verify and sha256_file(path) != f["sha256"]:
            raise MergeError(f"{mp}: part {f['file']} changed since the merge")
        files.append(path)
    return files


class MergeError(ValueError):
    """The shards do not form one consistent run."""


def find_shard_metas(stem: str | Path) -> list[Path]:
    stem = Path(stem)
    return sorted(p for p in stem.parent.glob(stem.name + ".shard*of*.meta.json")
                  if _SHARD_META.search(p.name))


def check_shards(metas: list[dict], allow_mixed_commits: bool = False) -> dict:
    """Validate shard metas against each other; returns a summary or raises MergeError."""
    if not metas:
        raise MergeError("no shard metas found")
    nums = {m["num_shards"] for m in metas}
    if len(nums) != 1:
        raise MergeError(f"shards disagree on num_shards: {sorted(nums)}")
    n = nums.pop()
    got = sorted(m["shard"] for m in metas)
    if got != list(range(n)):
        raise MergeError(f"expected shards 0..{n-1}, found {got}")
    ref = metas[0]
    for m in metas[1:]:
        diff = [k for k in IDENTITY_KEYS if m.get(k) != ref.get(k)]
        if diff:
            raise MergeError(f"shard {m['shard']} differs from shard {ref['shard']} in {diff}")
    commits = sorted({c for m in metas for c in m.get("part_commits", [m.get("git_commit")])})
    if len(commits) > 1 and not allow_mixed_commits:
        raise MergeError(f"shards were collected at several commits {commits}; "
                         f"pass allow_mixed_commits if that is intended")
    for m in metas:
        if not m.get("complete"):
            missing = sorted(set(m["doc_ids_expected"]) - set(m.get("doc_ids_done", []))
                             - set(m.get("doc_ids_failed", [])))
            raise MergeError(f"shard {m['shard']} is incomplete: {len(missing)} documents "
                             f"missing (first: {missing[:5]}); resume it before merging")
    return {"num_shards": n, "commits": commits}


def merge(stem: str | Path, out: str | Path, allow_mixed_commits: bool = False) -> Path:
    """Merge every shard of ``stem`` into ``out`` (+ ``.meta.json``).

    With ``out`` named ``*.dataset.json`` the merge is zero-copy: the parts are
    validated exactly as for a copy, and ``out`` becomes a manifest of them.
    """
    import os

    import pyarrow.parquet as pq

    from ..provenance import collect as provenance, sha256_file

    stem, out = Path(stem), Path(out)
    copy = not is_dataset(out)
    files: list[dict] = []
    schema = None
    meta_paths = find_shard_metas(stem)
    metas = [json.loads(p.read_text()) for p in meta_paths]
    summary = check_shards(metas, allow_mixed_commits)

    seen_docs: dict[int, str] = {}
    failed = sorted(d for m in metas for d in m.get("doc_ids_failed", []))
    # Documents skipped for non-finite logits are not in any part; they are
    # carried into the merged meta, and scripts/validate_run.py fails on them.
    expected = sorted(set(d for m in metas for d in m["doc_ids_expected"]) - set(failed))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    writer = None
    n_rows = 0
    parts_used = []
    try:
        for m in sorted(metas, key=lambda m: m["shard"]):
            for part in m["parts"]:
                path = stem.parent / part["file"]
                table = pq.read_table(path)
                doc = table.column("doc_id").to_numpy()
                # A document split across parts, or written twice, is a resume bug.
                for d in np.unique(doc):
                    if int(d) in seen_docs:
                        raise MergeError(f"document {int(d)} appears in {seen_docs[int(d)]} "
                                         f"and {part['file']}")
                    seen_docs[int(d)] = part["file"]
                key = (table.column("state_id").to_numpy().astype(np.int64) << 20) \
                    | table.column("position").to_numpy().astype(np.int64)
                if np.unique(key).size != key.size:
                    raise MergeError(f"{part['file']}: repeated (state_id, position) rows")
                if table.num_rows != part["n_rows"]:
                    raise MergeError(f"{part['file']}: {table.num_rows} rows, meta says "
                                     f"{part['n_rows']}")
                if schema is None:
                    schema = table.schema
                elif table.schema != schema:
                    raise MergeError(f"{part['file']}: schema differs from the first part")
                if copy:
                    if writer is None:
                        writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
                    writer.write_table(table)
                else:
                    files.append({"file": os.path.relpath(path, out.parent),
                                  "sha256": sha256_file(path), "n_rows": int(table.num_rows)})
                n_rows += table.num_rows
                parts_used.append(part["file"])
                del table
        if schema is None:
            raise MergeError("the shards contain no parts")
        if writer is not None:
            writer.close()
            writer = None
        if sorted(seen_docs) != expected:
            missing = sorted(set(expected) - set(seen_docs))
            raise MergeError(f"{len(missing)} expected documents missing (first: {missing[:5]})")
        if copy:
            tmp.replace(out)
    finally:
        if writer is not None:
            writer.close()
        tmp.unlink(missing_ok=True)

    ref = metas[0]
    merged = {k: ref[k] for k in IDENTITY_KEYS}
    merged.update({
        "schema_version": ref.get("schema_version", SCHEMA_VERSION),
        "git_commit": summary["commits"][0] if len(summary["commits"]) == 1 else summary["commits"],
        "path": ref.get("path"), "device_map": ref.get("device_map"),
        "n_rows": n_rows, "n_docs": len(seen_docs), "doc_ids": sorted(seen_docs),
        "doc_ids_failed": failed,
        "nonfinite_docs": [x for m in metas for x in m.get("stats", {}).get("nonfinite_docs", [])],
        "num_shards": summary["num_shards"], "parts": parts_used,
        "n_saturated": int(sum(m.get("stats", {}).get("n_saturated", 0) for m in metas)),
        "forward_passes": int(sum(m.get("stats", {}).get("forward_passes", 0) for m in metas)),
        "forward_passes_failed_docs": int(sum(m.get("stats", {}).get("forward_passes_failed_docs", 0)
                                              for m in metas)),
        "gpu_seconds": float(sum(m.get("stats", {}).get("seconds", 0.0) for m in metas)),
        "shard_provenance": [m.get("provenance") for m in metas],
        "shard_stats": [{"shard": m["shard"], **m.get("stats", {})} for m in metas],
        "provenance": provenance(),
    })
    if copy:
        merged["sha256"] = sha256_file(out)
        out.with_suffix(".meta.json").write_text(json.dumps(merged, indent=2))
    else:
        merged.update({"format": DATASET_FORMAT, "files": files})
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps(merged, indent=2))
        tmp.replace(out)
    print(f"[merge] {n_rows} rows, {len(seen_docs)} documents, {len(parts_used)} parts -> {out}"
          f"{'' if copy else ' (zero-copy manifest)'}")
    return out
