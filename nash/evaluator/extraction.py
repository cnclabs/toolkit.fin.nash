from __future__ import annotations

import re

from .types import NumericMention


_NUMERIC_TOKEN_RE = re.compile(
    r"""
    (?:
        (?P<currency>[$€£¥])?(?P<lead>\.)\d+(?:\.\d+)?
        (?:st|nd|rd|th)?
        (?:%|bp|bps|k|K|m|M|b|B)?
        (?!\w)
    )
    |
    (?:
        (?P<currency2>[$€£¥])?[+-]?
        (?:
            \d{1,3}(?:,\d{3})+(?:\.\d+)?
            |
            \d+(?:\.\d+)?
        )
        (?:st|nd|rd|th)?
        (?:%|bp|bps|k|K|m|M|b|B)?
        (?!\w)
    )
    """,
    flags=re.VERBOSE,
)


def _clean_numeric_string(token: str) -> str:
    cleaned = token.strip()
    cleaned = re.sub(r"(st|nd|rd|th)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace(",", "")
    if cleaned.startswith("."):
        cleaned = "0" + cleaned
    if cleaned.startswith("-."):
        cleaned = cleaned.replace("-.", "-0.", 1)
    if cleaned.startswith("+."):
        cleaned = cleaned.replace("+.", "+0.", 1)
    cleaned = re.sub(r"(%|bp|bps|k|K|m|M|b|B)\b", "", cleaned, flags=re.IGNORECASE)
    return cleaned


def extract_numeric_mentions(sentence: str) -> list[NumericMention]:
    mentions: list[NumericMention] = []

    for match in _NUMERIC_TOKEN_RE.finditer(sentence):
        token = match.group(0)
        # A compact suffix is unambiguous; normalise it rather than merely
        # stripping it.  Long written units are deliberately not consumed:
        # e.g. "1.5 million" remains 1.5, with no claim that it is 1.5M.
        suffix_match = re.search(r"(%|bps?|[kKmMbB])$", token)
        suffix = suffix_match.group(1) if suffix_match else None
        cleaned = _clean_numeric_string(token)
        cleaned = re.sub(r"[^\d\.\-\+]", "", cleaned)
        if cleaned.count(".") > 1:
            parts = cleaned.split(".")
            cleaned = parts[0] + "." + "".join(parts[1:])

        try:
            value = float(cleaned)
        except ValueError:
            continue

        start, end = match.span()
        multiplier = {"k": 1_000.0, "m": 1_000_000.0, "b": 1_000_000_000.0}.get((suffix or "").lower(), 1.0)
        mentions.append(
            NumericMention(
                text=token,
                value=value * multiplier,
                start=start,
                end=end,
                unit="percent" if suffix == "%" else ("basis_points" if suffix and suffix.lower() in {"bp", "bps"} else None),
                suffix=suffix,
            )
        )

    return mentions
