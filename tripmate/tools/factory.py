from __future__ import annotations
from langchain_core.tools import StructuredTool

from tripmate.config import Settings
from tripmate.rag.store import DestinationStore
from tripmate.services.weather import WeatherService
from tripmate.tools.schemas import SearchGuideInput, WeatherInput
from tripmate.tools.tracing import traced


def build_tools(settings: Settings, store: DestinationStore, weather: WeatherService):
    cities = ", ".join(store.supported_names())

    @traced("search_destination_guide", on_error=lambda m: [f"TOOL_ERROR: {m}"])
    def search_destination_guide(query: str, city: str | None = None, reason: str = "") -> list[str]:
        query = (query or "").strip()
        if not query:
            return ["INVALID_INPUT: query must be a non-empty question."]
        slug = None
        if city:
            slug = store.resolve_city(city)
            if slug is None:
                return [f"UNSUPPORTED_DESTINATION: '{city}' is not in the knowledge base. "
                        f"Supported destinations: {cities}."]
        else:
            slug = store.detect_city(query)
        hits = store.search(query, slug, settings.rag_top_k, settings.rag_max_distance)
        if not hits:
            return ["NO_RELEVANT_INFORMATION: the knowledge base has nothing relevant to this question."]
        return [h.text for h in hits]

    @traced("get_weather_forecast",
            on_error=lambda m: {"status": "error", "error_code": "TOOL_FAILURE", "message": m})
    def get_weather_forecast(city: str, date_or_month: str, reason: str = "") -> dict:
        return weather.get_forecast(city, date_or_month)

    rag_tool = StructuredTool.from_function(
        func=search_destination_guide, name="search_destination_guide",
        args_schema=SearchGuideInput,
        description=(
            f"Search the TripMate destination knowledge base ({cities}) for visa/entry rules, best "
            "time to visit, local customs, packing tips, and safety & health. Use for destination "
            "facts and advice, NOT for weather. Returns text passages, or a marker "
            "(UNSUPPORTED_DESTINATION / NO_RELEVANT_INFORMATION / INVALID_INPUT) - when you receive "
            "a marker, never invent the information."),
    )
    weather_tool = StructuredTool.from_function(
        func=get_weather_forecast, name="get_weather_forecast", args_schema=WeatherInput,
        description=(
            "Get weather for a city and a date or month. Uses a live forecast when the date is within "
            "15 days, otherwise typical (multi-year average) conditions. Returns a dict with status "
            "'ok' (conditions, temp_range_c, rainy_day_pct, source) or status 'error' with error_code "
            "and message - on error, tell the user; never invent weather."),
    )
    return [rag_tool, weather_tool]