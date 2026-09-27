"""Which documents the 110M model must never see, and proof that it did not.

The 110M arm's claims about the *level* of calibration error rest on the
evaluation documents being genuinely unseen (plan v2 T3).  A document is
excluded if it matches the manifest by any of three keys:

* its FineWeb record ``id``;
* a hash of its raw text (the proposal-era key, ``c3.data.fineweb.content_hash``);
* a hash of its first 256 GPT-2 tokens -- catches near-duplicates that carry a
  different id and slightly different trailing text but present the model with
  the same 256-token window the audit evaluates.

The manifest covers every document the study evaluates on: the 2,000 main
calibration documents, the pilot documents, the time-split set and the
validation documents.  After the shard build, :meth:`Exclusion.unmatched`
lists manifest entries that were never encountered in the stream; for entries
that were sampled *from* the stream (main and pilot), an unmatched entry means
the keys do not line up -- e.g. hashes computed on truncated text -- and the
build fails rather than silently training on the documents it claims to hold out.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TOKEN_WINDOW = 256
MANIFEST_VERSION = 2


def token_hash(ids, window: int = TOKEN_WINDOW) -> str:
    """blake2b of the first ``window`` token ids, as little-endian uint32."""
    arr = np.asarray(list(ids)[:window], dtype=np.uint32)
    return hashlib.blake2b(arr.astype("<u4").tobytes(), digest_size=16).hexdigest()


@dataclass
class ExclusionEntry:
    role: str                       # main | pilot | timesplit | val
    id: str | None = None
    content_hash: str | None = None
    token_hash: str | None = None
    from_stream: bool = True        # sampled from the training stream itself


@dataclass
class Exclusion:
    entries: list[ExclusionEntry]
    _by_id: dict = field(default_factory=dict, repr=False)
    _by_content: dict = field(default_factory=dict, repr=False)
    _by_token: dict = field(default_factory=dict, repr=False)
    matched: set = field(default_factory=set, repr=False)
    skipped_by: dict = field(default_factory=lambda: {"id": 0, "content": 0, "token": 0})

    def __post_init__(self) -> None:
        for i, e in enumerate(self.entries):
            if e.id:
                self._by_id.setdefault(str(e.id), []).append(i)
            if e.content_hash:
                self._by_content.setdefault(e.content_hash, []).append(i)
            if e.token_hash:
                self._by_token.setdefault(e.token_hash, []).append(i)
        if not self.entries:
            raise ValueError("exclusion manifest is empty; refusing to build a training "
                             "stream that may contain the evaluation documents")

    def check(self, rec_id=None, content: str | None = None, tokens: str | None = None) -> str | None:
        """The key that excludes this document, or None.  Records every match."""
        hit = None
        for key, table, val in (("id", self._by_id, None if rec_id is None else str(rec_id)),
                                ("content", self._by_content, content),
                                ("token", self._by_token, tokens)):
            if val is not None and val in table:
                self.matched.update(table[val])
                hit = hit or key
        if hit:
            self.skipped_by[hit] += 1
        return hit

    def unmatched(self, from_stream_only: bool = True) -> list[ExclusionEntry]:
        return [e for i, e in enumerate(self.entries)
                if i not in self.matched and (e.from_stream or not from_stream_only)]

    def summary(self) -> dict:
        roles: dict = {}
        for i, e in enumerate(self.entries):
            r = roles.setdefault(e.role, {"entries": 0, "matched": 0})
            r["entries"] += 1
            r["matched"] += int(i in self.matched)
        return {"entries": len(self.entries), "matched": len(self.matched),
                "skipped_by_key": dict(self.skipped_by), "by_role": roles}

    # ---------------------------------------------------------------- io

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "version": MANIFEST_VERSION, "token_window": TOKEN_WINDOW,
            "entries": [e.__dict__ for e in self.entries]}, indent=1))

    @classmethod
    def load(cls, *paths) -> "Exclusion":
        """Load v2 manifests, or the proposal-era manifest (content hashes only)."""
        entries: list[ExclusionEntry] = []
        for path in paths:
            raw = json.loads(Path(path).read_text())
            if isinstance(raw, dict) and raw.get("version") == MANIFEST_VERSION:
                entries += [ExclusionEntry(**e) for e in raw["entries"]]
            else:
                from ..data.fineweb import load_exclusion   # proposal-era format
                entries += [ExclusionEntry(role="main", content_hash=h)
                            for h in sorted(load_exclusion(path))]
        return cls(entries)


def entries_from_jsonl(path: str | Path, role: str, tokenizer, from_stream: bool = True,
                       content_hash_fn=None) -> list[ExclusionEntry]:
    """Manifest entries for every document in a JSONL corpus file.

    Each line needs a ``text`` field and, where available, the source ``id``
    (also accepted: ``source_id``, ``fineweb_id``).  ``content_hash_fn`` must be
    the same function the shard builder applies to raw text
    (``c3.data.fineweb.content_hash``).
    """
    if content_hash_fn is None:
        from ..data.fineweb import content_hash as content_hash_fn
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            text = rec.get("text", "")
            rid = rec.get("id") or rec.get("source_id") or rec.get("fineweb_id")
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            out.append(ExclusionEntry(role=role, id=None if rid is None else str(rid),
                                      content_hash=content_hash_fn(text),
                                      token_hash=token_hash(ids), from_stream=from_stream))
    return out
