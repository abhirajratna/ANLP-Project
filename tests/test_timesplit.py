"""The time-split builder, offline: the Wikipedia calls are replaced by fakes."""
import pytest

import c3.data.timesplit as ts


def test_clean_text_turns_section_markers_into_headings():
    raw = "Intro line.\n\n\n== History ==\nOld.\n=== Early ===\nEarlier."
    assert ts.clean_text(raw) == "Intro line.\n\nHistory\nOld.\nEarly\nEarlier."


def test_cutoff_is_after_both_weight_releases():
    assert ts.CUTOFF.startswith("2025-04-03")      # Dream weights; LLaDA's were 2025-02-20
    assert ts.TIMESPLIT_ID_OFFSET >= 2000          # disjoint from the main corpus ids


def _fake_wiki(monkeypatch, pages, deleted=()):
    """pages: pageid -> page_detail() result (or None for a redirect)."""
    ids = sorted(pages)
    monkeypatch.setattr(ts, "creations_after",
                        lambda t, limit: [{"pageid": i, "title": str(i)} for i in ids][:limit])
    monkeypatch.setattr(ts, "page_detail", lambda pid: pages[pid])
    monkeypatch.setattr(ts, "deleted_before", lambda title, cutoff: title in deleted)


def _page(pid, created="2025-09-01T00:00:00Z", comment="Created page with 'text moved from x'",
          n=3000):
    return {"title": str(pid), "text": f"page{pid} " + "word " * (n // 5), "revid": pid * 10,
            "first_edit": created, "first_comment": comment}


def test_filters_and_ids(monkeypatch):
    pages = {1: None,                                              # redirect
             2: _page(2, created="2025-03-01T00:00:00Z"),          # before the cutoff
             3: _page(3, comment="Split from [[Old article]]"),    # derived text
             4: _page(4, n=500),                                   # too short
             5: _page(5),                                          # recreated (deleted before)
             6: _page(6),                                          # kept (auto summary is fine)
             7: _page(7)}                                          # kept
    _fake_wiki(monkeypatch, pages, deleted={"5"})
    spec = ts.TimeSplitSpec(n_documents=2, pause=0.0, max_probes=10)
    docs, info = ts.build(spec, {"words": lambda text: len(text.split())}, log=lambda *_: None)
    assert [d["id"] for d in docs] == ["enwiki:6", "enwiki:7"]
    assert [d["doc_id"] for d in docs] == [ts.TIMESPLIT_ID_OFFSET, ts.TIMESPLIT_ID_OFFSET + 1]
    r = info["rejected"]
    assert (r["redirect_or_missing"], r["outside_window"], r["derived"], r["short"],
            r["recreated"]) == (1, 1, 1, 1, 1)


def test_too_few_tokens_and_running_out(monkeypatch):
    _fake_wiki(monkeypatch, {1: _page(1)})
    spec = ts.TimeSplitSpec(n_documents=1, pause=0.0, max_probes=3)
    with pytest.raises(RuntimeError, match="only 0 documents"):
        ts.build(spec, {"tiny": lambda text: 10}, log=lambda *_: None)
