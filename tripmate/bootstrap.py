from __future__ import annotations
from langchain_google_genai import ChatGoogleGenerativeAI

from tripmate.agent.graph import TripMateAgent
from tripmate.config import Settings
from tripmate.rag.ingest import ingest_directory
from tripmate.rag.store import DestinationStore
from tripmate.services.weather import WeatherService
from tripmate.tools.factory import build_tools


def build_agent(settings: Settings) -> TripMateAgent:
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key.")
    store = DestinationStore(settings)
    if store.count() == 0:
        ingest_directory(store, settings.data_dir)
    weather = WeatherService(settings.weather_timeout_s, settings.weather_cache_ttl_s)
    tools = build_tools(settings, store, weather)
    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model, google_api_key=settings.google_api_key,
        timeout=settings.llm_timeout_s, max_retries=2,
    )
    return TripMateAgent(llm.bind_tools(tools), tools, settings)