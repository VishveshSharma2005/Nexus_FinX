"""Text normalisation and content hashing.

The hash defined here is what makes re-uploading a revised agreement cheap: an
unchanged clause must hash identically across versions so it is never
re-embedded. That makes normalisation a correctness concern rather than a
cosmetic one.

Two failure modes to avoid, in opposite directions:

* Under-normalising -- a PDF that re-flows its line breaks, or uses a different
  quote glyph, would make an untouched clause look changed and we would pay to
  embed it again.
* Over-normalising -- lowercasing, or stripping digits and punctuation, would
  make a genuine change to a rate or a fee look unchanged, which is far worse:
  a v2 clause charging 4% would reuse the v1 embedding for 2%.

So: normalise presentation (whitespace, unicode form, quote and dash glyphs),
and preserve everything that carries meaning (case, digits, punctuation).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

# PDF extraction routinely yields these in place of ASCII equivalents; they are
# presentation, not content.
_GLYPH_REPLACEMENTS = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "–": "-",
    "—": "-",
    "−": "-",
    " ": " ",
    "​": "",
    "﻿": "",
    "�": "",
}

_WHITESPACE = re.compile(r"\s+")
_SOFT_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")


def clean_extracted_text(raw: str) -> str:
    """Tidy text straight out of a PDF while preserving line structure.

    Line structure survives because clause segmentation keys off line starts;
    only :func:`normalise_for_hash` flattens it.
    """
    text = unicodedata.normalize("NFKC", raw)
    for bad, good in _GLYPH_REPLACEMENTS.items():
        text = text.replace(bad, good)
    # Rejoin words split across a line by a typesetting hyphen.
    text = _SOFT_HYPHEN_BREAK.sub(r"\1\2", text)
    # Collapse runs of spaces/tabs but keep newlines.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def normalise_for_hash(text: str) -> str:
    """Reduce a clause to the form its content hash is taken over."""
    text = unicodedata.normalize("NFKC", text)
    for bad, good in _GLYPH_REPLACEMENTS.items():
        text = text.replace(bad, good)
    return _WHITESPACE.sub(" ", text).strip()


def content_hash(text: str) -> str:
    """Stable hash of a clause's meaningful content.

    Prefixed with the algorithm so a future change of hash function is visible
    in stored data instead of silently colliding with old rows.
    """
    digest = hashlib.sha256(normalise_for_hash(text).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def document_fingerprint(content: bytes) -> str:
    """Content-addressed id for a document, so parsing is reproducible.

    Deriving the id from the bytes rather than from a counter or a timestamp is
    what lets the parser satisfy its determinism invariant.
    """
    return hashlib.sha256(content).hexdigest()[:16]
