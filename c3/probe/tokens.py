"""Which tokens cover a field value, in context, under a given tokenizer.

p* is exact at the token level only if every value is exactly one token *where
it appears in the serialised instance*: a value that is one token on its own can
merge with neighbouring punctuation, or split, once it sits inside the JSON.  So
the check tokenises the whole instance and maps each value's character span to
the tokens that overlap it.

Fast tokenizers report character offsets.  Dream-7B's tokenizer is a slow one,
so offsets are rebuilt from the decoded tokens; that is exact for ASCII text
(every byte-level token then decodes to whole characters), and probe instances
are ASCII by construction.
"""

from __future__ import annotations


def token_offsets(tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """(ids, [(start, end) char offsets per token]) for ``text``, no special tokens."""
    if getattr(tokenizer, "is_fast", False):
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
    if not text.isascii():
        raise ValueError("offsets from decoded tokens need ASCII text")
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    offsets, pos = [], 0
    for i in ids:
        piece = tokenizer.decode([i], clean_up_tokenization_spaces=False)
        offsets.append((pos, pos + len(piece)))
        pos += len(piece)
    if pos != len(text):
        raise ValueError(f"decoded tokens do not reassemble the text ({pos} vs {len(text)} chars)")
    return list(ids), offsets


def value_tokens(tokenizer, text: str, spans) -> tuple[list[int], list[int | None]]:
    """(ids, per span: the index of the single token covering it, or None).

    None means the value is not exactly one token in context: it is split over
    several tokens, or its token also covers part of another value.
    """
    ids, offs = token_offsets(tokenizer, text)
    out: list[int | None] = []
    for k, (s, e) in enumerate(spans):
        hit = [t for t, (a, b) in enumerate(offs) if a < e and b > s]
        if len(hit) != 1:
            out.append(None)
            continue
        a, b = offs[hit[0]]
        others = any(a < e2 and b > s2 for j, (s2, e2) in enumerate(spans) if j != k)
        out.append(None if others else hit[0])
    return ids, out
