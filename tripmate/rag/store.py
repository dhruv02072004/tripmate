from __future__ import annotations
import difflib, unicodedata
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from tripmate.config import Settings


def normalize(text: str) -> str:
    """lowercase + strip accents: 'Reykjavík' -> 'reykjavik'."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c)).strip()


@dataclass(frozen=True)
class Hit:
    text: str
    city: str
    section: str
    distance: float  


class DestinationStore:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._emb = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
        self._vs = Chroma(
            collection_name=settings.collection_name,
            embedding_function=self._emb,
            persist_directory=str(settings.chroma_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )
        self._cities: dict[str, str] | None = None   

    # ---------- write ----------
    def upsert(self, docs: list[Document], ids: list[str]) -> None:
        self._vs.add_documents(docs, ids=ids)   
        self._cities = None

    def count(self) -> int:
        return self._vs._collection.count()

    # ---------- catalogue ----------
    def supported_cities(self) -> dict[str, str]:
        if self._cities is None:
            metas = self._vs.get(include=["metadatas"])["metadatas"] or []
            self._cities = {m["city"]: m["city_display"] for m in metas}
        return self._cities

    def supported_names(self) -> list[str]:
        return sorted(self.supported_cities().values())

    def resolve_city(self, raw: str | None) -> str | None:
        """Map user text ('Tokio', 'reykjavík, iceland') to a supported city slug."""
        if not raw:
            return None
        key = normalize(raw.split(",")[0])
        cities = self.supported_cities()
        if key in cities:
            return key
        close = difflib.get_close_matches(key, list(cities), n=1, cutoff=0.8)
        return close[0] if close else None

    def detect_city(self, text: str) -> str | None:
        t = normalize(text)
        return next((slug for slug in self.supported_cities() if slug in t), None)

    # ---------- read ----------
    def search(self, query: str, city_slug: str | None, k: int, max_distance: float) -> list[Hit]:
        flt = {"city": city_slug} if city_slug else None
        results = self._vs.similarity_search_with_score(query, k=k, filter=flt)
        hits = [
            Hit(d.page_content, d.metadata["city_display"], d.metadata["section"], float(dist))
            for d, dist in results
            if dist <= max_distance
        ]
        return sorted(hits, key=lambda h: h.distance)