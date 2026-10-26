import pytest
from tripmate.bootstrap import build_agent
from tripmate.config import get_settings

pytestmark = pytest.mark.live
CASES = [
    ("Do I need a visa to visit Tokyo from the US?", {"search_destination_guide"}),
    ("What's the weather in Bangkok in July?", {"get_weather_forecast"}),
    ("What should I pack for Reykjavik in January?", {"search_destination_guide", "get_weather_forecast"}),
    ("Can you book my flight to Tokyo?", set()),
    ("What is the capital of France?", set()),
]

@pytest.fixture(scope="module")
def agent():
    s = get_settings()
    if not s.google_api_key:
        pytest.skip("GOOGLE_API_KEY not set")
    return build_agent(s)

@pytest.mark.parametrize("query,expected", CASES)
def test_tool_selection(agent, query, expected):
    r = agent.ask(query, thread_id=query)
    assert {c["tool"] for c in r.tool_calls} == expected