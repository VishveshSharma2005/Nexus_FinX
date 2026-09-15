"""Parser registry -- the only module that knows concrete parser classes exist.

Kept separate from ``app.core.deps`` so that the contract test can enumerate
every implementation and assert the interface's invariants against all of them,
rather than against whichever one happens to be configured.
"""

from __future__ import annotations

from collections.abc import Callable

from app.config import Settings
from app.core.contracts import DocumentParser
from app.parsers.docx.parser import DocxParser
from app.parsers.lite.parser import LiteParser

ParserFactory = Callable[[Settings], DocumentParser]

PARSERS: dict[str, ParserFactory] = {
    LiteParser.name: lambda settings: LiteParser(
        ocr_enabled=settings.ocr_enabled,
        tesseract_cmd=settings.tesseract_cmd,
    ),
    DocxParser.name: lambda _settings: DocxParser(),
}

DEFAULT_PARSER = LiteParser.name


def select_parser(source, settings: Settings) -> DocumentParser:
    """Pick the implementation that declares it can handle this source.

    Callers ask for "a parser for this document", never for a named class, so
    adding a format is a registry entry rather than a change at every call
    site.
    """
    for name in PARSERS:
        parser = build_parser(name, settings)
        if parser.supports(source):
            return parser
    raise ValueError(
        f"No registered parser supports {source.filename!r} "
        f"({source.media_type}). Available: {sorted(PARSERS)}"
    )


def build_parser(name: str, settings: Settings) -> DocumentParser:
    try:
        factory = PARSERS[name]
    except KeyError:
        raise ValueError(f"Unknown parser {name!r}. Available: {sorted(PARSERS)}") from None
    return factory(settings)
