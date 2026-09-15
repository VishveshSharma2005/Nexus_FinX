"""Build the searchable index from the RBI corpus and the sample documents.

The Phase 2 deliverable you can run and look at. It parses, resolves versions,
chunks, embeds only what is new, and reports what it did -- in particular how
much of a revised agreement it did *not* have to embed again.

Usage:
    python scripts/build_index.py                  # corpus + samples
    python scripts/build_index.py --corpus-only
    python scripts/build_index.py --reset          # start from an empty index
    python scripts/build_index.py --search "is there a lock-in period?"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.config import get_settings  # noqa: E402
from app.core.contracts import ParseSource, RetrievalFilter  # noqa: E402
from app.core.deps import (  # noqa: E402
    get_embedding_provider,
    get_parser_for,
    get_vector_index,
)
from app.rag.ingest import MEDIA_TYPES, ingest_corpus, ingest_document  # noqa: E402

# The demo loan. Both versions share one logical document id, which is what
# makes them versions of one loan rather than two unrelated uploads.
DEMO_DOCUMENT_ID = "loan-KHFL-HL-2026-0041872"
DEMO_VERSIONS = [
    ("home_loan_agreement_A.docx", 1),
    ("home_loan_agreement_A_v2.docx", 2),
]
DEMO_KFS = ("home_loan_kfs_A.docx", "kfs-KHFL-HL-2026-0041872")


def _parser_for(source, settings):
    return get_parser_for(source, settings)


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()

    if args.reset and settings.local_index_path.exists():
        settings.local_index_path.unlink()
        print(f"removed {settings.local_index_path}")

    embedder = get_embedding_provider(settings)
    index = get_vector_index(settings)
    print(f"index      {index.name}  ({settings.local_index_path})")
    print(f"embeddings {embedder.name}  dim={embedder.dimension}\n")

    if not args.samples_only:
        print("RBI corpus")
        results = await ingest_corpus(
            corpus_dir=settings.rbi_corpus_dir,
            parser_for=lambda source: _parser_for(source, settings),
            embedder=embedder,
            index=index,
        )
        for result in results:
            print(
                f"  {result.document_id:<46} chunks={result.chunks_written:>4} "
                f"embedded={result.embeddings_computed:>4} reused={result.embeddings_reused:>4}"
            )
        print()

    if not args.corpus_only:
        print("Borrower documents")
        previous = None
        for filename, version in DEMO_VERSIONS:
            path = settings.sample_corpus_dir / filename
            if not path.is_file():
                print(f"  (skipping {filename}: not present)")
                continue
            source = ParseSource(
                content=b"",
                filename=path.name,
                media_type=MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"),
            )
            result = await ingest_document(
                path=path,
                parser=_parser_for(source, settings),
                embedder=embedder,
                index=index,
                document_id=DEMO_DOCUMENT_ID,
                version=version,
                base=previous,
            )
            previous = result.resolved
            print(
                f"  {filename:<34} v{version}  clauses={len(result.resolved.clauses):>3} "
                f"chunks={result.chunks_written:>3} embedded={result.embeddings_computed:>3} "
                f"reused={result.embeddings_reused:>3} ({result.reuse_ratio:.0%} reuse)"
            )
            for note in result.notes:
                print(f"       note: {note[:150]}")

        kfs_name, kfs_id = DEMO_KFS
        kfs_path = settings.sample_corpus_dir / kfs_name
        if kfs_path.is_file():
            source = ParseSource(
                content=b"",
                filename=kfs_path.name,
                media_type=MEDIA_TYPES[".docx"],
            )
            result = await ingest_document(
                path=kfs_path,
                parser=_parser_for(source, settings),
                embedder=embedder,
                index=index,
                document_id=kfs_id,
                version=1,
            )
            print(
                f"  {kfs_name:<34} v1  clauses={len(result.resolved.clauses):>3} "
                f"chunks={result.chunks_written:>3} embedded={result.embeddings_computed:>3}"
            )

    stats = index.stats()
    print(
        f"\nindex: {stats['chunks']} chunks over {stats['embeddings']} embeddings; "
        f"{stats['shared_embeddings']} embeddings shared between versions"
    )

    if args.search:
        await _search(index, embedder, args)

    return 0


async def _search(index, embedder, args: argparse.Namespace) -> None:
    query = await embedder.embed_query(args.search)
    filters = RetrievalFilter(
        document_id=DEMO_DOCUMENT_ID if not args.all_documents else None,
        version=args.version,
    )
    hits = await index.search(query, query_text=args.search, filters=filters, top_k=args.top_k)

    scope = "all documents" if args.all_documents else f"{DEMO_DOCUMENT_ID} v{args.version}"
    print(f"\nsearch ({scope}): {args.search!r}")
    if not hits:
        print("  no results")
        return
    for hit in hits:
        citation = hit.chunk.citation
        where = f"clause {citation.clause_number}" if citation.clause_number else f"p{citation.page}"
        print(f"  {hit.score:.3f}  v{hit.chunk.version} {where:<16} {_one_line(hit.chunk.text)}")


def _one_line(text: str, width: int = 96) -> str:
    collapsed = " ".join(text.split())
    return collapsed[:width] + ("..." if len(collapsed) > width else "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="Delete the index first.")
    parser.add_argument("--corpus-only", action="store_true")
    parser.add_argument("--samples-only", action="store_true")
    parser.add_argument("--search", help="Run a query against the index once built.")
    parser.add_argument("--version", type=int, default=2, help="Version to search (default 2).")
    parser.add_argument("--all-documents", action="store_true", help="Search the corpus too.")
    parser.add_argument("--top-k", type=int, default=6)
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
