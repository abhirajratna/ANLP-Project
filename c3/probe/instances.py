"""Probe instances: the spec in memory, seeded generation, compact-JSON text.

An instance is one schema's simple fields filled with values.  Fields are
grouped: a pair (leader, follower) or a single field.  At dependency level rho
the leader is drawn from its marginal; with probability rho the follower is the
leader's image under a fixed map, otherwise it is drawn from its own marginal.
Groups are independent of each other, which is what keeps every p* table small.

The text is compact JSON (no spaces), so every value sits between punctuation
and is its own pre-token under all three audited tokenizers.  The first member
is ``"_rho":"0.25"``: the dependency level is part of the observed context.
Without it rho would be a hidden variable shared by all groups of an instance,
the true conditional would be a mixture over rho that couples the groups, and
neither the small tables nor "(A) = 0 at rho = 0" would hold for a model that
cannot see rho.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..rng import generator
from . import PROBE_SEED, RHOS

QUOTED = ("enum", "str")
"""Kinds serialised as JSON strings; the others (num, bool) are bare literals."""


@dataclass
class Field:
    path: tuple[str, ...]            # ("key",) or ("parent", "key")
    kind: str                        # enum | str | num | bool
    values: list[str]                # surface form of each value: "debug", "5", "true"
    p: np.ndarray                    # marginal over values
    tokens: dict = field(default_factory=dict)   # tokenizer -> token id per value, in context

    @property
    def n(self) -> int:
        return len(self.values)


@dataclass
class Group:
    members: tuple[int, ...]         # (field,) or (leader, follower)
    fmap: np.ndarray | None = None   # follower value index for each leader value


@dataclass
class Schema:
    name: str
    fields: list[Field]
    groups: list[Group]
    source: dict = field(default_factory=dict)

    def group_of(self) -> np.ndarray:
        g = np.empty(len(self.fields), dtype=np.int64)
        for k, grp in enumerate(self.groups):
            g[list(grp.members)] = k
        return g


@dataclass
class ProbeSpec:
    schemas: list[Schema]
    rhos: tuple[float, ...] = RHOS
    seed: int = PROBE_SEED
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        return {"version": 1, "seed": self.seed, "rhos": list(self.rhos), **self.meta,
                "schemas": [{
                    "name": s.name, **s.source,
                    "fields": [{"path": list(f.path), "kind": f.kind, "values": f.values,
                                "p": [float(x) for x in f.p], "tokens": f.tokens}
                               for f in s.fields],
                    "groups": [{"members": list(g.members),
                                **({"map": [int(x) for x in g.fmap]} if g.fmap is not None else {})}
                               for g in s.groups]} for s in self.schemas]}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def from_dict(cls, d: dict) -> "ProbeSpec":
        schemas = []
        for s in d["schemas"]:
            fields = [Field(tuple(f["path"]), f["kind"], list(f["values"]),
                            np.asarray(f["p"], dtype=np.float64), dict(f.get("tokens", {})))
                      for f in s["fields"]]
            groups = [Group(tuple(g["members"]),
                            np.asarray(g["map"], dtype=np.int64) if "map" in g else None)
                      for g in s["groups"]]
            source = {k: v for k, v in s.items() if k not in ("name", "fields", "groups")}
            schemas.append(Schema(s["name"], fields, groups, source))
        meta = {k: v for k, v in d.items() if k not in ("version", "seed", "rhos", "schemas")}
        return cls(schemas, tuple(d["rhos"]), d["seed"], meta)

    @classmethod
    def load(cls, path: str | Path) -> "ProbeSpec":
        return cls.from_dict(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------- generation

def sample_values(schema: Schema, rho: float, rng: np.random.Generator) -> np.ndarray:
    """One draw of every field's value index.

    Every group consumes the same random numbers whatever rho is (the coin and
    the follower's independent draw are always made), so the stream does not
    depend on rho and instances stay reproducible field by field.
    """
    vals = np.empty(len(schema.fields), dtype=np.int64)
    for g in schema.groups:
        if len(g.members) == 1:
            (i,) = g.members
            vals[i] = rng.choice(schema.fields[i].n, p=schema.fields[i].p)
            continue
        a, b = g.members
        va = rng.choice(schema.fields[a].n, p=schema.fields[a].p)
        u = rng.random()
        vb_free = rng.choice(schema.fields[b].n, p=schema.fields[b].p)
        vals[a] = va
        vals[b] = g.fmap[va] if u < rho else vb_free
    return vals


@dataclass
class Instance:
    instance_id: int
    schema: int
    rho: float
    values: np.ndarray

    def to_dict(self) -> dict:
        return {"instance_id": self.instance_id, "schema": self.schema, "rho": self.rho,
                "values": [int(v) for v in self.values]}


def eval_instances(spec: ProbeSpec, per_rho: int | None = None, seed: int | None = None
                   ) -> list[Instance]:
    """The evaluation set: each schema once per rho (200 schemas x 5 rho = 1,000).

    ``per_rho`` instances per rho cycle through the schemas in order.  Each
    instance has its own generator, keyed by (schema, rho, repeat), so the set
    does not depend on generation order.
    """
    seed = spec.seed if seed is None else seed
    per_rho = len(spec.schemas) if per_rho is None else per_rho
    out = []
    for r, rho in enumerate(spec.rhos):
        for j in range(per_rho):
            s = j % len(spec.schemas)
            rng = generator(seed, "probe-eval", s, r, j // len(spec.schemas))
            out.append(Instance(len(out), s, float(rho), sample_values(spec.schemas[s], rho, rng)))
    return out


def train_instances(spec: ProbeSpec, n: int, label: str = "probe-train", start: int = 0,
                    seed: int | None = None):
    """Training (or validation, ``label="probe-val"``) instances ``start .. start+n-1``.

    Schema and rho are uniform; instance i depends only on (seed, label, i).
    """
    seed = spec.seed if seed is None else seed
    for i in range(start, start + n):
        rng = generator(seed, label, i)
        s = int(rng.integers(len(spec.schemas)))
        rho = float(spec.rhos[int(rng.integers(len(spec.rhos)))])
        yield Instance(i, s, rho, sample_values(spec.schemas[s], rho, rng))


# ---------------------------------------------------------------- text

def serialize(schema: Schema, values, rho: float) -> tuple[str, list[tuple[int, int]]]:
    """Compact JSON for one instance and the character span of each field's value.

    Fields with the same parent are contiguous in ``schema.fields`` and become
    one nested object.
    """
    out = ['{"_rho":"', f"{rho:.2f}", '"']
    n = len(out[0]) + len(out[1]) + 1
    spans: list[tuple[int, int]] = []
    parent = None

    def put(s: str):
        nonlocal n
        out.append(s)
        n += len(s)

    for i, f in enumerate(schema.fields):
        par = f.path[0] if len(f.path) == 2 else None
        if parent is not None and par != parent:
            put("}")
            parent = None
        first = False
        if par is not None and par != parent:
            put(f',"{par}":{{')
            parent, first = par, True
        put(("" if first else ",") + f'"{f.path[-1]}":')
        lit = f.values[int(values[i])]
        if f.kind in QUOTED:
            put('"')
            spans.append((n, n + len(lit)))
            put(lit)
            put('"')
        else:
            spans.append((n, n + len(lit)))
            put(lit)
    if parent is not None:
        put("}")
    put("}")
    return "".join(out), spans
