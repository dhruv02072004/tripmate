import pytest
from tripmate.rag.ingest import parse_destination_file

pytestmark = pytest.mark.slow

def test_parse_produces_five_sections(settings):
    docs, ids = parse_destination_file(settings.data_dir / "tokyo.txt")
    assert len(docs) == 5 and len(set(ids)) == 5
    assert docs[0].page_content.startswith("Tokyo, Japan — ")

def test_retrieves_right_section(store, settings):
    hits = store.search("pickpocketing risks", "barcelona", 3, settings.rag_max_distance)
    assert hits and hits[0].section == "Safety & Health"

def test_city_typo_resolves(store):
    assert store.resolve_city("Tokio") == "tokyo"
    assert store.resolve_city("Reykjavík, Iceland") == "reykjavik"

def test_unknown_city_not_resolved(store):
    assert store.resolve_city("Paris") is None