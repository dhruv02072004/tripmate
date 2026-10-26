import pytest
from tripmate.config import Settings
from tripmate.rag.ingest import ingest_directory
from tripmate.rag.store import DestinationStore


@pytest.fixture(scope="session")
def settings(tmp_path_factory):
    return Settings(chroma_dir=tmp_path_factory.mktemp("chroma"), google_api_key="test")


@pytest.fixture(scope="session")
def store(settings):
    s = DestinationStore(settings)
    ingest_directory(s, settings.data_dir)
    return s