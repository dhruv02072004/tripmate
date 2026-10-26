from __future__ import annotations
import logging, re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from tripmate.config import Settings, get_settings
from tripmate.logging_setup import log_event, setup_logging
from tripmate.rag.store import DestinationStore, normalize

logger = logging.getLogger("tripmate.ingest")

_TITLE_RE = re.compile(r"^DESTINATION GUIDE:\s*(?P<city>[^,]+),\s*(?P<country>.+)$")
_SECTION_RE = re.compile(r"^[A-Z][A-Z &/]+$")
_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)


def parse_destination_file(path: Path) -> tuple[list[Document], list[str]]:
    city = country = None
    sections: list[tuple[str, list[str]]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if (m := _TITLE_RE.match(line)):
            city, country = m["city"].title(), m["country"].title()
        elif _SECTION_RE.match(line):
            sections.append((line.title(), []))
        elif sections:                     
            sections[-1][1].append(line)
    if not city or not sections:
        raise ValueError(f"{path.name}: not a valid destination guide")

    slug = normalize(city)
    docs: list[Document] = []
    ids: list[str] = []
    for section, lines in sections:
        for i, part in enumerate(_splitter.split_text(" ".join(lines))):
            docs.append(Document(
                page_content=f"{city}, {country} — {section}: {part}",
                metadata={"city": slug, "city_display": city, "country": country,
                          "section": section, "source": path.name},
            ))
            ids.append(f"{slug}:{normalize(section).replace(' ', '_')}:{i}")
    return docs, ids


def ingest_directory(store: DestinationStore, data_dir: Path) -> int:
    total = 0
    for path in sorted(data_dir.glob("*.txt")):
        try:
            docs, ids = parse_destination_file(path)
        except ValueError as exc:
            log_event(logger, "ingest.skip", level=logging.WARNING, file=path.name, error=str(exc))
            continue
        store.upsert(docs, ids)
        total += len(docs)
        log_event(logger, "ingest.file", file=path.name, chunks=len(docs))
    return total


if __name__ == "__main__":
    s = get_settings()
    setup_logging(s.log_level, s.log_file, verbose=True)
    n = ingest_directory(DestinationStore(s), s.data_dir)
    print(f"Ingested {n} chunks.")