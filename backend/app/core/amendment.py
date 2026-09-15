"""Reading what an amendment says it does to the agreement it revises.

These helpers interpret the *language* of a revision -- which clauses it
deletes, which of its clauses merely assert that the rest still stands, which
sections it reprints only in part. That is domain knowledge about loan
documents, not knowledge about PDF or Word extraction, and both the parsers
and the versioning layer need it. It lives in core so that neither has to
import the other.

Recognising a document as an amendment in the first place stays with the
parsers, in :mod:`app.parsers.lite.amendment`, because it leans on the same
date and metadata reading the rest of parsing uses.
"""

from __future__ import annotations

import re

# --- What an amendment does to each clause it names -------------------------

# "Clause 11.2 ... is hereby DELETED in its entirety."
# "Clauses 11.4 and 11.5 ... stand superseded ... and are of no further effect."
_DELETION = re.compile(
    r"\b(?:is|are|stand|stands)\s+(?:hereby\s+)?(?:deleted|superseded|withdrawn|"
    r"of no further effect)\b",
    re.I,
)

# A deletion sentence usually names the clauses it removes, and they are not
# always the clause doing the talking: clause 11.3 of the amendment is what
# supersedes clauses 11.4 and 11.5 of the original.
_CLAUSE_REFERENCE = re.compile(
    r"clauses?\s+(?P<numbers>\d{1,2}(?:\.\d{1,2})*(?:\s*(?:,|and|&)\s*\d{1,2}(?:\.\d{1,2})*)*)",
    re.I,
)


def deleted_clause_numbers(text: str) -> set[str]:
    """Clause numbers this text says are removed from the base agreement.

    Read from the amendment's own words rather than inferred from absence,
    because in an amendment absence means "unchanged".
    """
    removed: set[str] = set()
    for sentence in re.split(r"(?<=[.;])\s+", text):
        if not _DELETION.search(sentence):
            continue
        for match in _CLAUSE_REFERENCE.finditer(sentence):
            for number in re.findall(r"\d{1,2}(?:\.\d{1,2})*", match.group("numbers")):
                removed.add(number)
    return removed


# An amendment does not only enact new terms. It also carries numbered
# *statements about* the original -- "9.2 All other provisions of Clause 9
# remain unchanged" -- which share the original's numbering but replace
# nothing. Treating one as a revision of the base clause it shares a number
# with silently deletes that clause: the real 9.2, forbidding capitalisation of
# penal charges, was lost exactly this way.
_CARRY_FORWARD_STATEMENT = re.compile(
    r"all other\b[^.]{0,200}?\b(?:remain|continue)s?\b[^.]{0,40}?"
    r"(?:unchanged|in full force|in effect|to apply)",
    re.I,
)

# A section that reprints only its changed rows leaves the rest of the base
# section standing, so replacing it wholesale would drop the unchanged fees.
_PARTIAL_RESTATEMENT = re.compile(
    r"only\s+(?:the\s+)?(?:rows?|items?|entries|charges)\s+(?:changed|revised|amended|shown)",
    re.I,
)


def is_carry_forward_statement(text: str) -> bool:
    """Whether this clause asserts that the rest of a section still stands.

    Such a clause enacts nothing. It must not replace the base clause whose
    number it borrows, and it is not worth indexing as a term of the loan.
    """
    return bool(_CARRY_FORWARD_STATEMENT.search(text))


def is_partial_restatement(text: str) -> bool:
    """Whether a section reprints only the rows it changed."""
    return bool(_PARTIAL_RESTATEMENT.search(text) or _CARRY_FORWARD_STATEMENT.search(text))
