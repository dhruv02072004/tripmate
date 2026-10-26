import httpx, pytest
from datetime import date
from langchain_core.messages import AIMessage
from tripmate.agent.graph import TripMateAgent
from tripmate.services.weather import WeatherService
from tripmate.tools.factory import build_tools

pytestmark = pytest.mark.slow


class ScriptedLLM:
    def __init__(self, script):
        self.script = list(script)
    def bind_tools(self, tools): return self
    def invoke(self, messages): return self.script.pop(0)


def call(i, name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{i}"}])


def weather_service():
    def handler(req):
        if "geocoding" in str(req.url):
            return httpx.Response(200, json={"results": [{"name": "Reykjavik", "country": "Iceland",
                                                          "latitude": 64.1, "longitude": -21.9}]})
        return httpx.Response(200, json={"daily": {"temperature_2m_max": [3], "temperature_2m_min": [-3],
                                                   "precipitation_sum": [2.0], "snowfall_sum": [1.0]}})
    return WeatherService(client=httpx.Client(transport=httpx.MockTransport(handler)),
                          today_fn=lambda: date(2026, 10, 21))


def make_agent(script, settings, store):
    tools = build_tools(settings, store, weather_service())
    return TripMateAgent(ScriptedLLM(script), tools, settings)


def test_full_chained_flow(settings, store):
    agent = make_agent([
        call(1, "search_destination_guide", query="best time Northern Lights", city="Reykjavik"),
        call(2, "get_weather_forecast", city="Reykjavik", date_or_month="January"),
        AIMessage(content="Pack warm, waterproof layers."),
    ], settings, store)
    r = agent.ask("What should I pack to see the Northern Lights in Iceland?")
    assert [c["tool"] for c in r.tool_calls] == ["search_destination_guide", "get_weather_forecast"]
    assert "Reykjavik" in r.tool_calls[0]["result_preview"]
    assert "freezing" in r.tool_calls[1]["result_preview"] or "cold" in r.tool_calls[1]["result_preview"]
    assert r.status == "ok" and "waterproof" in r.answer


def test_no_tool_out_of_scope(settings, store):
    agent = make_agent([AIMessage(content="I can't book flights, but I can help plan.")], settings, store)
    r = agent.ask("Can you book my flight?")
    assert r.tool_calls == [] and "can't book" in r.answer


def test_empty_input_never_reaches_llm(settings, store):
    agent = make_agent([], settings, store)      # empty script: any LLM call would crash
    r = agent.ask("   ")
    assert r.status == "rejected"


def test_unsupported_destination_marker(settings, store):
    agent = make_agent([
        call(1, "search_destination_guide", query="customs", city="Paris"),
        AIMessage(content="I don't have Paris in my guides."),
    ], settings, store)
    r = agent.ask("Paris customs?")
    assert "UNSUPPORTED_DESTINATION" in r.tool_calls[0]["result_preview"]