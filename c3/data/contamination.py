"""The contamination screen (plan v2 T4 step 1; proposal Sec. 6).

LLaDA and Dream were trained on web text that is not fully disclosed and may
include FineWeb-Edu documents.  A memorised document looks "too accurate" and
shifts calibration levels, so each document is probed directly: the model sees
its first 50 tokens with the next 50 masked, decodes them greedily, and the
longest run of consecutive positions where its continuation equals the true one
is recorded.  A run at or above a threshold fixed before looking flags the
document, and every endpoint is reported with and without flagged documents.

Greedy here means free-running argmax, one token per forward pass, committing
the most confident slot each time (LLaDA's low-confidence remasking with as many
steps as tokens, through ``c3.decode.loop``).  Nothing after the 100th token is
shown, so the continuation has to come from the prefix alone.

The run is position-aligned: a verbatim continuation shifted by one token would
not count.  That is the stricter, standard reading of "exact continuation".
"""

from __future__ import annotations

import numpy as np

PREFIX_TOKENS = 50
CONTINUATION_TOKENS = 50


def longest_run(a, b) -> int:
    """Longest run of consecutive positions where ``a`` and ``b`` agree."""
    eq = np.asarray(a) == np.asarray(b)
    best = cur = 0
    for v in eq:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return int(best)


def screen_document(model, ids, prefix: int = PREFIX_TOKENS,
                    continuation: int = CONTINUATION_TOKENS) -> dict:
    """Greedy continuation of one document and how much of it is verbatim."""
    from ..decode.loop import decode
    from ..decode.policies import TopKConfidence

    ids = np.asarray(ids, dtype=np.int64)
    if ids.size < prefix + continuation:
        raise ValueError(f"document has {ids.size} tokens, needs {prefix + continuation}")
    x = ids[:prefix + continuation].copy()
    x[prefix:] = model.mask_token_id
    res = decode(model, x, TopKConfidence(1), record=False)
    gen = res.tokens[prefix:prefix + continuation]
    ref = ids[prefix:prefix + continuation]
    eq = gen == ref
    return {"run_length": longest_run(gen, ref),
            "prefix_match": int(np.argmin(eq)) if not eq.all() else int(eq.size),
            "n_match": int(eq.sum()), "forward_passes": res.n_forward}


def flag(results: dict, threshold: int) -> dict:
    """Add ``flagged`` (run_length >= threshold) to every document's result."""
    return {d: {**r, "flagged": bool(r["run_length"] >= threshold)} for d, r in results.items()}


def excluded_doc_ids(path) -> set[int]:
    """Documents to drop, from a screen output ({"docs": {id: {flagged}}}) or a JSON list."""
    import json
    from pathlib import Path

    raw = json.loads(Path(path).read_text())
    if isinstance(raw, list):
        return {int(d) for d in raw}
    docs = raw["docs"] if "docs" in raw else raw
    if any("flagged" not in r for r in docs.values()):
        raise ValueError(f"{path}: documents without a 'flagged' field; was a threshold applied?")
    return {int(d) for d, r in docs.items() if r["flagged"]}
