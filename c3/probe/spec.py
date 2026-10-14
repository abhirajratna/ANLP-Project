"""Build the probe spec from SchemaStore schemas (plan v2 T9 steps 1-3).

1. **Schemas.**  Every schema in a pinned SchemaStore checkout
   (``src/schemas/json``) is read; versioned copies of one schema
   (``renovate-41``, ``renovate-42``) count once, keeping the one with most
   usable fields.  Simple fields are the top-level properties, and the
   properties of top-level objects one level down, that are an enum of strings,
   a boolean, a number or integer, or a plain string (no ``format``,
   ``pattern`` or ``const``).  Local ``$ref``s are followed; ``["T", "null"]``
   counts as T.  At most ``max_fields`` per schema, in document order.
2. **Values.**  Every value must be exactly one token, in context, under GPT-2
   (the 110M model), LLaDA-8B and Dream-7B.  Enum fields keep the enum values
   that are letters only and pass that check (at most ``max_enum``); numbers
   take the digits 0-9 inside the schema's bounds (both 8B tokenizers split
   numbers into single digits, so nothing longer is one token); booleans are
   ``true``/``false``; plain strings get 3-8 words drawn from a fixed pool of
   words that pass the check.  A field with fewer than two values is dropped.
3. **Distribution.**  Each field's marginal is a seeded Dirichlet(1) draw.
   Fields are paired in a seeded order, (leader, follower), an odd one out is
   a single; the follower's map spreads the leader's values evenly over a
   seeded permutation of the follower's values.

Schemas need at least ``min_fields`` fields and a distinct field layout (two
schemas with the same keys would be indistinguishable to a model), and every
instance must fit in ``max_tokens`` under each tokenizer.  The first
``n_schemas`` in a seeded order are kept.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from ..rng import generator
from . import MAX_TOKENS, N_SCHEMAS, PROBE_SEED, RHOS
from .instances import Field, Group, ProbeSpec, Schema, serialize
from .tokens import value_tokens

KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,24}$")
WORD = re.compile(r"^[A-Za-z]{1,16}$")

WORD_POOL = (
    "alpha beta gamma delta main test default local remote auto none basic simple custom "
    "public private shared static dynamic global primary internal external standard legacy "
    "stable latest next dev prod release debug info warn error fatal trace low medium high "
    "small large fast slow red green blue black white north south east west left right top "
    "bottom center inline block fixed relative absolute manual daily weekly monthly always "
    "never once open closed active inactive enabled disabled online offline admin user guest "
    "owner member reader writer app web api server client worker service cluster linux "
    "windows android python java ruby home office cloud mobile desktop light dark full empty "
    "first last new old hot cold soft hard strict loose single double master slave input "
    "output source target start stop pause play read write delete update create import "
    "export file folder image video audio text table list map tree graph"
).split()


# ---------------------------------------------------------------- schema walking

def _resolve(root, node, depth: int = 0):
    while (isinstance(node, dict) and isinstance(node.get("$ref"), str)
           and node["$ref"].startswith("#/") and depth < 8):
        target = root
        for part in node["$ref"][2:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            target = target.get(part) if isinstance(target, dict) else None
        if target is None:
            return None
        node, depth = target, depth + 1
    return node


def _kind(root, node) -> str | None:
    node = _resolve(root, node)
    if not isinstance(node, dict):
        return None
    if "enum" in node:
        return "enum" if sum(isinstance(e, str) for e in node["enum"]) >= 2 else None
    if "const" in node:
        return None
    t = node.get("type")
    if isinstance(t, list):
        t = [x for x in t if x != "null"]
        t = t[0] if len(t) == 1 else None
    if t == "boolean":
        return "bool"
    if t in ("integer", "number"):
        return "num"
    if t == "string" and not node.get("format") and not node.get("pattern"):
        return "str"
    if t == "object" or (t is None and isinstance(node.get("properties"), dict)):
        return "obj"
    return None


def simple_fields(schema: dict) -> list[dict]:
    """Candidate fields of one JSON Schema: path, kind and the resolved definition."""
    top = _resolve(schema, schema)
    props = top.get("properties") if isinstance(top, dict) else None
    if not isinstance(props, dict):
        return []
    out = []
    for key, node in props.items():
        if not KEY.match(key):
            continue
        kind = _kind(schema, node)
        if kind == "obj":
            sub = _resolve(schema, node).get("properties")
            if not isinstance(sub, dict):
                continue
            for k2, n2 in sub.items():
                k2kind = _kind(schema, n2)
                if KEY.match(k2) and k2kind not in (None, "obj"):
                    out.append({"path": (key, k2), "kind": k2kind, "node": _resolve(schema, n2)})
        elif kind is not None:
            out.append({"path": (key,), "kind": kind, "node": _resolve(schema, node)})
    return out


def stem(filename: str) -> str:
    """'renovate-42.json' -> 'renovate', 'project-1.0.0-rc2' -> 'project': one schema's versions."""
    return re.sub(r"(?:[-_.]?v?\d+(?:\.\d+)*|[-_.](?:rc|beta|alpha)\d*)+$", "", Path(filename).stem)


# ---------------------------------------------------------------- values

def candidate_values(cand: dict, rng: np.random.Generator, pool: list[str], max_enum: int
                     ) -> list[str]:
    node, kind = cand["node"], cand["kind"]
    if kind == "bool":
        return ["true", "false"]
    if kind == "num":
        def num(key):                  # draft-04 writes exclusiveMinimum: true
            v = node.get(key)
            return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None
        lo, hi = 0, 9
        if num("minimum") is not None:
            lo = max(lo, int(np.ceil(num("minimum"))))
        if num("exclusiveMinimum") is not None:
            lo = max(lo, int(np.floor(num("exclusiveMinimum"))) + 1)
        if num("maximum") is not None:
            hi = min(hi, int(np.floor(num("maximum"))))
        if num("exclusiveMaximum") is not None:
            hi = min(hi, int(np.ceil(num("exclusiveMaximum"))) - 1)
        return [str(d) for d in range(lo, hi + 1)]
    if kind == "enum":
        seen, vals = set(), []
        for v in node["enum"]:
            if isinstance(v, str) and WORD.match(v) and v not in seen:
                seen.add(v)
                vals.append(v)
        return vals[: 4 * max_enum]          # trimmed to max_enum after the token check
    m = int(rng.integers(3, 9))
    pick = rng.choice(len(pool), size=min(m, len(pool)), replace=False)
    return [pool[i] for i in sorted(pick)]


def _single_token_values(fields: list[Field], tokenizers: dict) -> list[list[bool]]:
    """ok[f][v]: value v of field f is one token in context under every tokenizer.

    Each value is checked in a filled-in instance (other fields at their first
    value; values are separated by punctuation, so the neighbours do not change
    the result -- tests/test_probe.py re-checks every evaluation instance).
    Records the token id of every value per tokenizer in ``Field.tokens``.
    """
    sch = Schema("check", fields, [])
    ok = [[True] * f.n for f in fields]
    for name, tok in tokenizers.items():
        for f in fields:
            f.tokens[name] = [None] * f.n
        width = max(f.n for f in fields)
        for v in range(width):
            vals = [min(v, f.n - 1) for f in fields]
            text, spans = serialize(sch, vals, RHOS[0])
            ids, which = value_tokens(tok, text, spans)
            for k, f in enumerate(fields):
                if v < f.n:
                    if which[k] is None:
                        ok[k][v] = False
                    else:
                        f.tokens[name][v] = int(ids[which[k]])
    return ok


def build_spec(schema_dir: str | Path, tokenizers: dict, n_schemas: int = N_SCHEMAS,
               min_fields: int = 6, max_fields: int = 16, max_enum: int = 12,
               max_tokens: int = MAX_TOKENS, seed: int = PROBE_SEED, log=print) -> ProbeSpec:
    """The probe spec from a SchemaStore checkout's ``src/schemas/json`` directory."""
    schema_dir = Path(schema_dir)
    # The word pool: words that are one token in context under every tokenizer.
    probe = [Field(("w",), "str", list(WORD_POOL), np.full(len(WORD_POOL), 1 / len(WORD_POOL)))]
    pool = [w for w, good in zip(WORD_POOL, _single_token_values(probe, tokenizers)[0]) if good]
    log(f"[spec] word pool: {len(pool)} of {len(WORD_POOL)} words are single tokens in context")

    by_stem: dict[str, list] = {}
    skipped = {"unreadable": 0, "few_fields": 0, "duplicate_layout": 0, "too_long": 0}
    for path in sorted(schema_dir.glob("*.json")):
        raw = path.read_bytes()
        try:
            doc = json.loads(raw)
        except ValueError:
            skipped["unreadable"] += 1
            continue
        if not isinstance(doc, dict):
            continue
        cands = simple_fields(doc)
        if len(cands) >= min_fields:
            by_stem.setdefault(stem(path.name), []).append((path, raw, cands))

    built = []
    for st in sorted(by_stem):
        best = None
        for path, raw, cands in by_stem[st]:
            rng = generator(seed, "probe-spec", st)
            fields = []
            for c in cands:
                vals = candidate_values(c, rng, pool, max_enum)
                if len(vals) >= 2:
                    fields.append(Field(tuple(c["path"]), c["kind"], vals, np.zeros(len(vals))))
            if not fields:
                continue
            ok = _single_token_values(fields, tokenizers)
            kept = []
            for f, good in zip(fields, ok):
                vals = [v for v, g in zip(f.values, good) if g]
                vals = vals[:max_enum] if f.kind == "enum" else vals
                if len(vals) >= 2:
                    kept.append(Field(f.path, f.kind, vals, np.zeros(len(vals))))
            kept = _contiguous(kept)[:max_fields]
            if len(kept) >= min_fields and (best is None or len(kept) >= len(best[2])):
                best = (path, raw, kept)
        if best is not None:
            built.append((st,) + best)
        else:
            skipped["few_fields"] += 1

    order = sorted(built, key=lambda b: hashlib.blake2b(f"{seed}:{b[0]}".encode(),
                                                       digest_size=8).hexdigest())
    schemas, layouts = [], set()
    for st, path, raw, fields in order:
        if len(schemas) == n_schemas:
            break
        layout = tuple(f.path for f in fields)
        if layout in layouts:
            skipped["duplicate_layout"] += 1
            continue
        _single_token_values(fields, tokenizers)          # records final token ids
        sch = _finish(st, fields, seed)
        lengths = {}
        for name, tok in tokenizers.items():
            text, _ = serialize(sch, [f.n - 1 for f in fields], RHOS[0])
            lengths[name] = len(tok(text, add_special_tokens=False)["input_ids"])
        if max(lengths.values()) > max_tokens:
            skipped["too_long"] += 1
            continue
        layouts.add(layout)
        sch.source = {"file": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
                      "n_tokens": lengths}
        schemas.append(sch)
    if len(schemas) < n_schemas:
        raise RuntimeError(f"only {len(schemas)} schemas qualify (need {n_schemas}); "
                           f"lower min_fields")
    log(f"[spec] {len(schemas)} schemas from {len(built)} candidates; skipped {skipped}")
    return ProbeSpec(schemas, RHOS, seed, {"word_pool": pool, "skipped": skipped,
                                           "params": {"min_fields": min_fields,
                                                      "max_fields": max_fields,
                                                      "max_enum": max_enum,
                                                      "max_tokens": max_tokens}})


def _contiguous(fields: list[Field]) -> list[Field]:
    """Keep each parent's nested fields together (one JSON object per parent)."""
    out, seen = [], {}
    for f in fields:
        key = f.path[0] if len(f.path) == 2 else None
        if key is None:
            out.append(f)
        elif key not in seen:
            seen[key] = len(out)
            out.append([f])
        else:
            out[seen[key]].append(f)
    flat = []
    for item in out:
        flat.extend(item if isinstance(item, list) else [item])
    return flat


def _finish(name: str, fields: list[Field], seed: int) -> Schema:
    """Marginals, pairs and follower maps, all seeded by the schema's name."""
    rng = generator(seed, "probe-dist", name)
    for f in fields:
        f.p = rng.dirichlet(np.ones(f.n))
    order = rng.permutation(len(fields))
    groups = []
    for j in range(0, len(order) - 1, 2):
        a, b = int(order[j]), int(order[j + 1])
        perm = rng.permutation(fields[b].n)
        groups.append(Group((a, b), perm[np.arange(fields[a].n) % fields[b].n]))
    if len(order) % 2:
        groups.append(Group((int(order[-1]),)))
    return Schema(name, fields, groups)
