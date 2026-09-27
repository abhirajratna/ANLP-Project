"""Joining the shards of one collection run into a single trace.

A full sweep is 2,000 documents x ~466 forward passes per document on an 8B
model, which does not fit one job, so ``collect_shard`` splits it by
``doc_id % N`` and writes parts.  The merge is where silent mistakes would
enter: a shard collected at another precision, a revision that moved, a
document written twice after a resume.  So it refuses rather than repairs --
every shard meta must agree on the run's identity, every expected document must
be present exactly once, and no (state_id, position) pair may repeat.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .collect import IDENTITY_KEYS, SCHEMA_VERSION

_SHARD_META = re.compile(r"\.shard(\d+)of(\d+)\.meta\.json$")


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
            missing = sorted(set(m["doc_ids_expected"]) - set(m.get("doc_ids_done", [])))
            raise MergeError(f"shard {m['shard']} is incomplete: {len(missing)} documents "
                             f"missing (first: {missing[:5]}); resume it before merging")
    return {"num_shards": n, "commits": commits}


def merge(stem: str | Path, out: str | Path, allow_mixed_commits: bool = False) -> Path:
    """Merge every shard of ``stem`` into ``out`` (+ ``.meta.json``)."""
    import pyarrow.parquet as pq

    from ..provenance import collect as provenance, sha256_file

    stem, out = Path(stem), Path(out)
    meta_paths = find_shard_metas(stem)
    metas = [json.loads(p.read_text()) for p in meta_paths]
    summary = check_shards(metas, allow_mixed_commits)

    seen_docs: dict[int, str] = {}
    expected = sorted(d for m in metas for d in m["doc_ids_expected"])
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
                if writer is None:
                    writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
                elif table.schema != writer.schema:
                    raise MergeError(f"{part['file']}: schema differs from the first part")
                writer.write_table(table)
                n_rows += table.num_rows
                parts_used.append(part["file"])
        if writer is None:
            raise MergeError("the shards contain no parts")
        writer.close()
        writer = None
        if sorted(seen_docs) != expected:
            missing = sorted(set(expected) - set(seen_docs))
            raise MergeError(f"{len(missing)} expected documents missing (first: {missing[:5]})")
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
        "num_shards": summary["num_shards"], "parts": parts_used,
        "n_saturated": int(sum(m.get("stats", {}).get("n_saturated", 0) for m in metas)),
        "forward_passes": int(sum(m.get("stats", {}).get("forward_passes", 0) for m in metas)),
        "gpu_seconds": float(sum(m.get("stats", {}).get("seconds", 0.0) for m in metas)),
        "shard_provenance": [m.get("provenance") for m in metas],
        "sha256": sha256_file(out),
        "provenance": provenance(),
    })
    out.with_suffix(".meta.json").write_text(json.dumps(merged, indent=2))
    print(f"[merge] {n_rows} rows, {len(seen_docs)} documents, {len(parts_used)} parts -> {out}")
    return out
