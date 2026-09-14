"""Interfaces and shared schemas that the rest of FinX depends on.

This module is the architectural spine of the project. Two rules follow from it:

1.  Nothing outside ``app.parsers`` may import a concrete parser. Callers depend
    on :class:`DocumentParser`, so a production-grade parser can replace the
    hackathon one without touching the RAG layer.
2.  Nothing in ``app.rag`` may import a vendor SDK. Callers depend on
    :class:`LLMProvider`, so model choice stays a single environment variable.

Everything here is declaration only -- no behaviour, no I/O, no business logic.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Vocabulary
#
# These enums are the shared language between a parsed document's metadata and
# an RBI corpus chunk's applicability record. Retrieval compares the two, so the
# two sides must never drift apart -- hence one definition, here.
# ---------------------------------------------------------------------------


class LoanType(StrEnum):
    PERSONAL = "personal"
    HOME = "home"
    VEHICLE = "vehicle"
    EDUCATION = "education"
    GOLD = "gold"
    BUSINESS = "business"
    MICROFINANCE = "microfinance"
    CREDIT_CARD = "credit_card"
    DIGITAL = "digital"
    OTHER = "other"


class BorrowerType(StrEnum):
    INDIVIDUAL = "individual"
    BUSINESS = "business"


class LenderClass(StrEnum):
    """Regulated-entity classes, because RBI directions bind them differently."""

    SCHEDULED_COMMERCIAL_BANK = "scheduled_commercial_bank"
    SMALL_FINANCE_BANK = "small_finance_bank"
    REGIONAL_RURAL_BANK = "regional_rural_bank"
    LOCAL_AREA_BANK = "local_area_bank"
    COOPERATIVE_BANK = "cooperative_bank"
    NBFC = "nbfc"
    NBFC_MFI = "nbfc_mfi"
    HOUSING_FINANCE_COMPANY = "housing_finance_company"
    ALL_INDIA_FINANCIAL_INSTITUTION = "all_india_financial_institution"
    DIGITAL_LENDING_APP = "digital_lending_app"
    ALL_REGULATED_ENTITIES = "all_regulated_entities"


class DocumentKind(StrEnum):
    LOAN_AGREEMENT = "loan_agreement"
    KEY_FACTS_STATEMENT = "key_facts_statement"
    SANCTION_LETTER = "sanction_letter"
    RBI_CIRCULAR = "rbi_circular"
    UNKNOWN = "unknown"


class SourceKind(StrEnum):
    """Which corpus a retrieved passage came from. Citations render differently."""

    USER_DOCUMENT = "user_document"
    RBI_CORPUS = "rbi_corpus"


# ---------------------------------------------------------------------------
# Applicability
# ---------------------------------------------------------------------------


class Applicability(BaseModel):
    """Scope conditions carried by every RBI corpus chunk.

    Empty collections mean "unrestricted on this axis". Retrieval treats an
    unrestricted axis as matching anything, and a populated axis as a hard
    filter applied *before* reranking.
    """

    model_config = ConfigDict(frozen=True)

    loan_types: tuple[LoanType, ...] = ()
    borrower_types: tuple[BorrowerType, ...] = ()
    lender_classes: tuple[LenderClass, ...] = ()
    purposes: tuple[str, ...] = ()
    effective_from: date | None = None
    effective_to: date | None = None


class DocumentMetadata(BaseModel):
    """What we know about the borrower's own document.

    These are the fields retrieval matches against :class:`Applicability`.
    Every field is optional: an unknown field must widen the candidate set
    rather than silently excluding governing regulation.
    """

    model_config = ConfigDict(frozen=True)

    loan_type: LoanType | None = None
    borrower_type: BorrowerType | None = None
    lender_class: LenderClass | None = None
    purpose: str | None = None
    agreement_date: date | None = None
    currency: str = "INR"
    language: str = "en"
    extra: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parsed documents
# ---------------------------------------------------------------------------


class Clause(BaseModel):
    """One addressable unit of a document -- the granularity FinX cites.

    ``content_hash`` is the basis of token-efficient versioning: an unchanged
    clause in a revised document keeps its embedding and merely gains a new
    version pointer.
    """

    model_config = ConfigDict(frozen=True)

    clause_id: str
    ordinal: int = Field(ge=0, description="0-based position within the document")
    heading: str | None = None
    number: str | None = Field(default=None, description="Printed clause number, e.g. 7.2")
    text: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    content_hash: str = Field(description="Stable hash of normalised clause text")
    is_ocr: bool = False


class ParseWarning(BaseModel):
    """A parser's honest account of what it could not do.

    Surfaced to the user rather than swallowed -- a silently empty page is the
    failure mode most likely to produce a confidently wrong answer.
    """

    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    page: int | None = None


class ParsedDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    version: int = Field(ge=1)
    kind: DocumentKind = DocumentKind.UNKNOWN
    title: str | None = None
    page_count: int = Field(ge=0)
    clauses: tuple[Clause, ...] = ()
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    parser_name: str = ""
    warnings: tuple[ParseWarning, ...] = ()
    parsed_at: datetime | None = None


class ParseSource(BaseModel):
    """Input to a parser: bytes plus whatever the caller already knows."""

    model_config = ConfigDict(frozen=True)

    content: bytes
    filename: str
    media_type: str = "application/pdf"
    document_id: str | None = None
    version: int = Field(default=1, ge=1)
    kind_hint: DocumentKind | None = None
    metadata_hint: DocumentMetadata | None = None


@runtime_checkable
class DocumentParser(Protocol):
    """Turns document bytes into an addressable, citable :class:`ParsedDocument`.

    Implementations must satisfy these invariants -- ``test_parser_contract.py``
    asserts them against any registered implementation:

    * ``parse`` is deterministic: equal input bytes yield an equal document,
      including identical ``content_hash`` values.
    * ``clauses`` are ordered by ``ordinal``, starting at 0 with no gaps.
    * ``1 <= page_start <= page_end <= page_count`` for every clause.
    * ``clause_id`` is unique within a document.
    * Clause text is non-empty after stripping.
    * A page yielding no extractable text produces a :class:`ParseWarning`
      rather than being dropped in silence.
    * ``parse`` never raises for well-formed input it declared it ``supports``;
      partial extraction plus warnings is preferred to an exception.
    """

    name: str

    def supports(self, source: ParseSource) -> bool:
        """Whether this parser can handle the source at all."""
        ...

    def parse(self, source: ParseSource) -> ParsedDocument:
        """Extract text, segment into clauses, and hash each clause."""
        ...


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """The provenance of one claim. Every grounded sentence binds to one."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    source_kind: SourceKind
    document_id: str
    version: int
    page: int | None = None
    clause_number: str | None = None
    title: str | None = None
    circular_id: str | None = None
    source_url: str | None = None


class Chunk(BaseModel):
    """An indexed, embeddable unit. One clause may become one or more chunks."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    version: int
    clause_id: str | None = None
    ordinal: int = Field(ge=0)
    text: str
    content_hash: str
    source_kind: SourceKind
    applicability: Applicability = Field(default_factory=Applicability)
    citation: Citation


class RetrievedPassage(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float
    dense_score: float | None = None
    lexical_score: float | None = None
    rerank_score: float | None = None


# ---------------------------------------------------------------------------
# Provider interfaces
# ---------------------------------------------------------------------------


class LLMMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: str  # "system" | "user" | "assistant"
    content: str


class LLMUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    model: str
    usage: LLMUsage = Field(default_factory=LLMUsage)
    finish_reason: str | None = None


@runtime_checkable
class LLMProvider(Protocol):
    """The only door between FinX and any language model.

    Implementations must never be imported by ``app.rag``; they arrive through
    dependency injection so that swapping models is an environment change.

    Implementations must not be asked to compute money -- that is
    ``app.risk``'s job, in deterministic Python.
    """

    name: str
    model: str
    supported_languages: frozenset[str]

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Return a full completion."""
        ...

    def stream(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        """Return an async iterator of text deltas."""
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Embeddings are separable from generation: retrieval may stay multilingual
    while generation routes to whichever model supports the user's language."""

    name: str
    model: str
    dimension: int

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class RetrievalFilter(BaseModel):
    """Hard constraints applied inside the index, before any reranking."""

    model_config = ConfigDict(frozen=True)

    document_id: str | None = None
    version: int | None = Field(
        default=None,
        description="Retrieval is scoped to one version; v1 must never leak into a v2 answer.",
    )
    source_kinds: tuple[SourceKind, ...] = ()
    metadata: DocumentMetadata | None = None
    as_of: date | None = None


@runtime_checkable
class VectorIndex(Protocol):
    """Storage for chunks and their embeddings.

    Two implementations exist behind this interface: pgvector (the
    production-shaped path, via docker-compose) and a local SQLite+numpy index
    so a live demo never depends on an external service being up.
    """

    name: str

    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    async def existing_hashes(self, content_hashes: list[str]) -> dict[str, str]:
        """Map ``content_hash -> chunk_id`` for chunks already embedded.

        This is what makes re-uploading a revised document cheap: a hit here
        means reuse the stored embedding and add a version pointer instead of
        paying to embed the clause again.
        """
        ...

    async def add_version_pointer(self, chunk_id: str, document_id: str, version: int) -> None: ...

    async def search(
        self,
        embedding: list[float],
        *,
        query_text: str,
        filters: RetrievalFilter,
        top_k: int,
    ) -> list[RetrievedPassage]: ...


@runtime_checkable
class LenderCatalog(Protocol):
    """Lookup for lender attributes (regulated class, published rates).

    Kept behind an interface and seeded from public data so no integration is
    wired into the codebase.
    """

    name: str

    def lender_class_for(self, lender_name: str) -> LenderClass | None: ...
