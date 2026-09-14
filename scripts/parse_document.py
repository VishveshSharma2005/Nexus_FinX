"""Parse a document to ParsedDocument JSON.

The Phase 1 deliverable, and the debugging tool for every phase after it: when
an answer cites the wrong clause, this shows what the parser actually saw.

Usage:
    python scripts/parse_document.py corpus/samples/personal_loan_agreement_v1.pdf
    python scripts/parse_document.py <file.pdf> --out parsed.json
    python scripts/parse_document.py <file.pdf> --summary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.config import get_settings  # noqa: E402
from app.core.contracts import ParseSource  # noqa: E402
from app.core.deps import get_document_parser  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse a document into ParsedDocument JSON."
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--out", type=Path, help="Write JSON here instead of stdout.")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a human-readable clause table instead of JSON.",
    )
    args = parser.parse_args()

    if not args.path.is_file():
        print(f"No such file: {args.path}", file=sys.stderr)
        return 2

    settings = get_settings()
    document_parser = get_document_parser(settings)

    source = ParseSource(
        content=args.path.read_bytes(),
        filename=args.path.name,
        version=args.version,
    )

    if not document_parser.supports(source):
        print(
            f"Parser {document_parser.name!r} does not support {args.path.name}",
            file=sys.stderr,
        )
        return 2

    parsed = document_parser.parse(source)

    if args.summary:
        _print_summary(parsed)
        return 0

    payload = parsed.model_dump_json(indent=2)
    if args.out:
        args.out.write_text(payload, encoding="utf-8")
        print(f"wrote {args.out}  ({len(payload):,} bytes)")
    else:
        print(payload)
    return 0


def _print_summary(parsed) -> None:
    meta = parsed.metadata
    print(f"document_id   {parsed.document_id}  (version {parsed.version})")
    print(f"title         {parsed.title}")
    print(f"kind          {parsed.kind.value}")
    print(f"parser        {parsed.parser_name}")
    print(f"pages         {parsed.page_count}")
    print(f"clauses       {len(parsed.clauses)}")
    print(
        "metadata      "
        f"loan_type={_show(meta.loan_type)} borrower={_show(meta.borrower_type)} "
        f"lender={_show(meta.lender_class)} dated={_show(meta.agreement_date)}"
    )

    if parsed.warnings:
        print(f"\nwarnings ({len(parsed.warnings)}):")
        for warning in parsed.warnings:
            page = f" p{warning.page}" if warning.page else ""
            print(f"  [{warning.code}]{page} {warning.message}")
    else:
        print("\nwarnings      none")

    print(f"\n{'ord':>3}  {'no.':<8} {'pages':<7} {'chars':>6}  heading / first line")
    print("-" * 100)
    for clause in parsed.clauses:
        pages = (
            f"{clause.page_start}"
            if clause.page_start == clause.page_end
            else f"{clause.page_start}-{clause.page_end}"
        )
        label = clause.heading or clause.text.split("\n")[0]
        flag = " [ocr]" if clause.is_ocr else ""
        print(
            f"{clause.ordinal:>3}  {(clause.number or '-'):<8} {pages:<7} "
            f"{len(clause.text):>6}  {label[:60]}{flag}"
        )


def _show(value) -> str:
    if value is None:
        return "-"
    return getattr(value, "value", str(value))


if __name__ == "__main__":
    sys.exit(main())
