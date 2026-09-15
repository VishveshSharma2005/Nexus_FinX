"""Resolving what a version of a document actually contains.

A full restatement needs nothing from this module: what you uploaded is what
the version holds. A delta amendment does, because it reprints only the clauses
it changes and leaves the rest standing. The effective version is therefore
assembled rather than read:

    resolved v2 = v1's clauses
                - the ones v2 says are deleted
                + the ones v2 replaces
                + the ones v2 adds

Carrying a clause forward is free. Its text is byte-identical to v1's, so its
content hash is identical, so the index reuses the stored embedding and merely
records that this clause is also in v2. Only genuinely changed clauses are
embedded again, which is the whole point of hashing clauses in the first place.

Two things this module refuses to do:

* Infer a deletion from absence. In an amendment, absence means "unchanged".
  A clause is only removed when the amendment says so in words.
* Silently merge a section that was only partly reprinted. A revised fee
  schedule showing "only rows changed from the original" cannot replace the
  original wholesale without dropping the unchanged fees, so both are kept and
  the ambiguity is reported rather than resolved by guesswork.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from app.core.amendment import (
    deleted_clause_numbers,
    is_carry_forward_statement,
    is_partial_restatement,
)
from app.core.contracts import Clause, DocumentKind, ParsedDocument


class ClauseOperation(StrEnum):
    """What a version did to one clause, relative to the version before it."""

    CARRIED_FORWARD = "carried_forward"
    REPLACED = "replaced"
    ADDED = "added"
    DELETED = "deleted"


@dataclass(frozen=True)
class ClauseChange:
    operation: ClauseOperation
    number: str | None
    content_hash: str | None
    previous_hash: str | None = None
    note: str | None = None


@dataclass
class ResolvedVersion:
    """The clauses in force for one version of one document."""

    document_id: str
    version: int
    clauses: tuple[Clause, ...]
    changes: tuple[ClauseChange, ...] = ()
    notes: list[str] = field(default_factory=list)
    base_version: int | None = None

    @property
    def reused_hashes(self) -> set[str]:
        """Clauses whose stored embedding can be reused as-is."""
        return {
            change.content_hash
            for change in self.changes
            if change.operation is ClauseOperation.CARRIED_FORWARD and change.content_hash
        }

    @property
    def embedding_required_hashes(self) -> set[str]:
        return {
            change.content_hash
            for change in self.changes
            if change.operation in {ClauseOperation.REPLACED, ClauseOperation.ADDED}
            and change.content_hash
        }


def _renumber(clauses: list[Clause], document_id: str, version: int) -> tuple[Clause, ...]:
    """Reassign ordinals and clause ids so the resolved version is addressable.

    A carried-forward clause keeps its text and therefore its content hash --
    that is what makes reuse possible -- but it gets an id belonging to this
    version, because a citation has to name the version it was read from.
    """
    return tuple(
        clause.model_copy(
            update={
                "ordinal": ordinal,
                "clause_id": f"{document_id}-v{version}-c{ordinal:04d}",
            }
        )
        for ordinal, clause in enumerate(clauses)
    )


def _sort_key(clause: Clause) -> tuple:
    """Order by printed clause number, falling back to original position."""
    number = clause.number or ""
    if number and number[0].isdigit():
        return (0, [int(part) for part in number.split(".")], clause.ordinal)
    return (1, [], clause.ordinal)


def resolve_standalone(document: ParsedDocument) -> ResolvedVersion:
    """A document that restates itself in full is already its own version."""
    return ResolvedVersion(
        document_id=document.document_id,
        version=document.version,
        clauses=document.clauses,
        changes=tuple(
            ClauseChange(ClauseOperation.ADDED, clause.number, clause.content_hash)
            for clause in document.clauses
        ),
    )


def resolve_amendment(base: ResolvedVersion, amendment: ParsedDocument) -> ResolvedVersion:
    """Assemble the clauses in force after an amendment.

    ``base`` is the resolved previous version, not the raw upload, so a chain
    of amendments composes: v3 is resolved against resolved v2.
    """
    if amendment.kind is not DocumentKind.AMENDMENT:
        raise ValueError(f"{amendment.document_id} is not an amendment")

    base_by_number: dict[str, Clause] = {}
    for clause in base.clauses:
        if clause.number:
            base_by_number.setdefault(clause.number, clause)

    amendment_text = "\n".join(clause.text for clause in amendment.clauses)
    deleted = deleted_clause_numbers(amendment_text)

    changes: list[ClauseChange] = []
    notes: list[str] = []
    resolved: list[Clause] = []
    consumed: set[str] = set()

    # 1. What the amendment itself states.
    for clause in amendment.clauses:
        number = clause.number

        # A clause whose only job is to announce a deletion adds no text to
        # the agreement, but the deletion it announces is recorded.
        if number and number in deleted and not _replaces_as_well_as_deletes(clause.text):
            consumed.add(number)
            previous = base_by_number.get(number)
            if previous is not None:
                changes.append(
                    ClauseChange(
                        ClauseOperation.DELETED,
                        number,
                        None,
                        previous_hash=previous.content_hash,
                        note="The amendment states this clause is deleted.",
                    )
                )
            continue

        # A clause that only asserts the rest of its section still stands
        # enacts nothing. It borrows the original's numbering, so treating it
        # as a revision would delete the base clause of the same number.
        if number and is_carry_forward_statement(clause.text):
            previous = base_by_number.get(number)
            if previous is not None:
                consumed.add(number)
                resolved.append(previous)
                changes.append(
                    ClauseChange(
                        ClauseOperation.CARRIED_FORWARD,
                        number,
                        previous.content_hash,
                        note="The amendment confirms this clause is unchanged.",
                    )
                )
            continue

        if number and number in base_by_number:
            previous = base_by_number[number]
            consumed.add(number)
            if previous.content_hash == clause.content_hash:
                changes.append(
                    ClauseChange(ClauseOperation.CARRIED_FORWARD, number, clause.content_hash)
                )
            else:
                changes.append(
                    ClauseChange(
                        ClauseOperation.REPLACED,
                        number,
                        clause.content_hash,
                        previous_hash=previous.content_hash,
                    )
                )
            resolved.append(clause)
            continue

        changes.append(ClauseChange(ClauseOperation.ADDED, number, clause.content_hash))
        resolved.append(clause)

    # 2. Deletions the amendment names but does not itself reprint.
    for number in sorted(deleted):
        if number in consumed or number not in base_by_number:
            continue
        consumed.add(number)
        changes.append(
            ClauseChange(
                ClauseOperation.DELETED,
                number,
                None,
                previous_hash=base_by_number[number].content_hash,
                note="Stated as deleted or superseded by the amendment.",
            )
        )

    # 3. Everything the amendment is silent about is still in force.
    for clause in base.clauses:
        if clause.number and clause.number in consumed:
            continue
        resolved.append(clause)
        changes.append(
            ClauseChange(ClauseOperation.CARRIED_FORWARD, clause.number, clause.content_hash)
        )

    # 4. Sections reprinted only in part leave the rest of the base standing.
    partial = [
        clause.number
        for clause in amendment.clauses
        if clause.number and is_partial_restatement(clause.text)
    ]
    if partial:
        notes.append(
            "The amendment reprints "
            + ", ".join(str(number) for number in partial)
            + " only in part, showing just the rows it changes. The unchanged rows of the "
            "original remain in force, so both versions of that section are retained and "
            "an answer drawn from it should say which rows it relied on."
        )
        for number in partial:
            original = base_by_number.get(number)
            if original is not None and original not in resolved:
                resolved.append(original)
                changes.append(
                    ClauseChange(
                        ClauseOperation.CARRIED_FORWARD,
                        number,
                        original.content_hash,
                        note="Retained because the amendment restated this section only in part.",
                    )
                )

    resolved.sort(key=_sort_key)

    return ResolvedVersion(
        document_id=base.document_id,
        version=amendment.version,
        clauses=_renumber(resolved, base.document_id, amendment.version),
        changes=tuple(changes),
        notes=notes,
        base_version=base.version,
    )


def _replaces_as_well_as_deletes(text: str) -> bool:
    """Whether a clause deletes something and puts something in its place.

    "Clause 11.3 ... is hereby DELETED and replaced as follows: no prepayment
    charge shall apply" both removes and enacts. Dropping it as a pure deletion
    would lose the new term.
    """
    lowered = text.lower()
    return "replaced as follows" in lowered or "substituted" in lowered


# --- The zero-token version diff --------------------------------------------


@dataclass(frozen=True)
class VersionDiff:
    """What changed between two resolved versions.

    A pure set operation over content hashes. No model is called, so asking
    "what changed?" costs nothing and cannot hallucinate a change that did not
    happen.
    """

    document_id: str
    from_version: int
    to_version: int
    added: tuple[Clause, ...]
    removed: tuple[Clause, ...]
    unchanged_count: int

    @property
    def is_empty(self) -> bool:
        return not self.added and not self.removed


def diff_versions(earlier: ResolvedVersion, later: ResolvedVersion) -> VersionDiff:
    before = {clause.content_hash: clause for clause in earlier.clauses}
    after = {clause.content_hash: clause for clause in later.clauses}

    return VersionDiff(
        document_id=later.document_id,
        from_version=earlier.version,
        to_version=later.version,
        added=tuple(after[h] for h in after.keys() - before.keys()),
        removed=tuple(before[h] for h in before.keys() - after.keys()),
        unchanged_count=len(before.keys() & after.keys()),
    )
