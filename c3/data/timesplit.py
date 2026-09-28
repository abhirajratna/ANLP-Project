"""The time-split set: text written after both 8B model families were released (plan v2 T4 step 2).

LLaDA and Dream were trained on undisclosed web data, so the calibration corpus
may have been seen.  Text first published after their weights were released
cannot have been.  The later of the two releases sets the cutoff:

* LLaDA-8B (Nie et al., arXiv:2502.09992, submitted 2025-02-14): the Base and
  Instruct weight files on the hub were last changed on **2025-02-20**, including
  at the revisions pinned in ``c3.traces.models``;
* Dream-7B (Ye et al., arXiv:2508.15487, submitted 2025-08-21; weights published
  before the paper): the Base and Instruct weight files were last changed on
  **2025-04-03**, also at the pinned revisions.

(Dates read from the hub commit history of each weight file on 2026-09-28; later
commits touched only code and model cards.)  So documents must be first
published after 2025-04-03.

Source: English Wikipedia articles *created* in the main namespace after the
cutoff, from the page-creation log.  Sampling is spread over time -- seeded
random instants in [cutoff, as_of], taking the first acceptable creation after
each -- rather than the first 500 creations after the cutoff.  A candidate is
kept only if it is not a redirect, still exists, its first edit is not a split,
merge, move, copy or translation of older text (by edit summary), no page of the
same title was deleted before the cutoff (a recreation may restore old text),
and its plain text is long enough.

Text is the current revision's plain-text extract with section markers turned
into plain heading lines, recorded with its revision id so the document is
fixed.  Each document is truncated to 256 tokens at trace time, as the main
corpus is; ``min_tokens`` checks that every audited tokenizer reaches 256.

Limits worth stating: text added to a new article can still quote older
sources, and a copy not declared in the edit summary is not caught.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass

import numpy as np

CUTOFF = "2025-04-03T23:59:59Z"
"""End of the later release day (Dream-7B weights, 2025-04-03)."""

TIMESPLIT_ID_OFFSET = 100_000
"""Time-split doc_ids are OFFSET + i, disjoint from the main corpus's 0..1999."""

API = "https://en.wikipedia.org/w/api.php"
# Wikimedia's User-Agent policy asks for a way to reach the operator; the project
# repository is that contact, rather than anyone's personal address.
USER_AGENT = ("C3-timesplit/0.1 (+https://github.com/abhirajratna/ANLP-Project; "
              "academic research on diffusion LM calibration)")

_DERIVED = re.compile(r"split|merg|moved|copied|copy|translat|from \[\[|\bfrom\b.*\bwikipedia\b",
                      re.IGNORECASE)
# [ \t], not \s: \s would cross line breaks and swallow the blank line before a heading.
_HEADING = re.compile(r"^[ \t]*=+[ \t]*(.*?)[ \t]*=+[ \t]*$", re.MULTILINE)
# MediaWiki's automatic first-edit summary quotes the article's opening text,
# which must not be read as the editor's own description of the edit.
_AUTO_SUMMARY = re.compile(r"Created page with ['\"].*['\"]", re.DOTALL)


@dataclass
class TimeSplitSpec:
    n_documents: int = 500
    cutoff: str = CUTOFF
    as_of: str = "2026-09-01T00:00:00Z"   # fixed end of the window, for reproducibility
    seed: int = 20260917
    min_chars: int = 1500
    min_tokens: int = 256
    creations_per_probe: int = 20
    max_probes: int = 20_000
    pause: float = 1.0                    # seconds between page requests (rate limits)


def api(params: dict, retries: int = 8) -> dict:
    """One MediaWiki API call; backs off on 429/5xx, honouring Retry-After."""
    params = {**params, "format": "json", "formatversion": "2", "maxlag": 5}
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.load(r)
            if out.get("error", {}).get("code") == "maxlag":
                raise urllib.error.URLError("maxlag")
            return out
        except OSError as exc:
            if attempt == retries - 1:
                raise
            wait = 2.0 ** attempt
            if isinstance(exc, urllib.error.HTTPError) and exc.headers.get("Retry-After"):
                wait = max(wait, float(exc.headers["Retry-After"]))
            time.sleep(min(wait, 120.0))
    raise RuntimeError("unreachable")


def clean_text(extract: str) -> str:
    """Plain-text extract -> web-page-like text: '== History ==' becomes 'History'."""
    text = _HEADING.sub(lambda m: m.group(1), extract)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def creations_after(ts: str, limit: int) -> list[dict]:
    r = api({"action": "query", "list": "logevents", "letype": "create", "lenamespace": 0,
             "ledir": "newer", "lestart": ts, "lelimit": limit, "leprop": "title|ids|timestamp"})
    return [e for e in r["query"]["logevents"] if e.get("pageid")]


def page_detail(pageid: int) -> dict | None:
    """Current plain text, current revision id, and the first edit's time and summary."""
    r = api({"action": "query", "pageids": pageid, "prop": "info|extracts|revisions",
             "explaintext": 1, "rvdir": "newer", "rvlimit": 1,
             "rvprop": "timestamp|comment|ids"})
    pages = r["query"]["pages"]
    if not pages or pages[0].get("missing") or pages[0].get("redirect"):
        return None
    p = pages[0]
    first = (p.get("revisions") or [{}])[0]
    return {"title": p["title"], "text": p.get("extract") or "", "revid": p.get("lastrevid"),
            "first_edit": first.get("timestamp"), "first_comment": first.get("comment", "")}


def deleted_before(title: str, cutoff: str) -> bool:
    r = api({"action": "query", "list": "logevents", "letype": "delete", "letitle": title,
             "leend": cutoff, "ledir": "older", "lelimit": 1})
    return bool(r["query"]["logevents"])


def _probe_times(spec: TimeSplitSpec, rng: np.random.Generator):
    import datetime as dt

    def parse(s):
        return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)

    lo, hi = parse(spec.cutoff).timestamp(), parse(spec.as_of).timestamp()
    for _ in range(spec.max_probes):
        t = dt.datetime.fromtimestamp(rng.uniform(lo, hi), tz=dt.timezone.utc)
        yield t.strftime("%Y-%m-%dT%H:%M:%SZ")


def build(spec: TimeSplitSpec, token_counters: dict, log=print) -> tuple[list[dict], dict]:
    """Sample the set; ``token_counters`` maps tokenizer name -> fn(text) -> n tokens."""
    from .fineweb import content_hash

    rng = np.random.default_rng(spec.seed)
    seen: set[int] = set()
    docs: list[dict] = []
    rejected = {"redirect_or_missing": 0, "derived": 0, "recreated": 0, "short": 0,
                "too_few_tokens": 0, "outside_window": 0, "duplicate_text": 0}
    hashes: set[str] = set()
    probes = 0
    for ts in _probe_times(spec, rng):
        if len(docs) >= spec.n_documents:
            break
        probes += 1
        for e in creations_after(ts, spec.creations_per_probe):
            if e["pageid"] in seen:
                continue
            seen.add(e["pageid"])
            time.sleep(spec.pause)
            d = page_detail(e["pageid"])
            if d is None:
                rejected["redirect_or_missing"] += 1
                continue
            if not d["first_edit"] or d["first_edit"] <= spec.cutoff or d["first_edit"] > spec.as_of:
                rejected["outside_window"] += 1
                continue
            if _DERIVED.search(_AUTO_SUMMARY.sub("", d["first_comment"] or "")):
                rejected["derived"] += 1
                continue
            text = clean_text(d["text"])
            if len(text) < spec.min_chars:
                rejected["short"] += 1
                continue
            counts = {name: fn(text) for name, fn in token_counters.items()}
            if min(counts.values(), default=spec.min_tokens) < spec.min_tokens:
                rejected["too_few_tokens"] += 1
                continue
            if deleted_before(d["title"], spec.cutoff):
                rejected["recreated"] += 1
                continue
            h = content_hash(text)
            if h in hashes:
                rejected["duplicate_text"] += 1
                continue
            hashes.add(h)
            docs.append({"doc_id": TIMESPLIT_ID_OFFSET + len(docs),
                         "id": f"enwiki:{e['pageid']}", "text": text, "hash": h,
                         "title": d["title"], "created": d["first_edit"],
                         "revid": d["revid"], "tokens": counts, "source": "wikipedia"})
            if len(docs) % 50 == 0:
                log(f"[timesplit] {len(docs)}/{spec.n_documents} documents after {probes} probes")
            break                    # one document per probe: spread over the window
    if len(docs) < spec.n_documents:
        raise RuntimeError(f"only {len(docs)} documents after {probes} probes; raise max_probes")
    return docs, {"spec": asdict(spec), "probes": probes, "candidates_seen": len(seen),
                  "rejected": rejected}
