from datetime import date
import httpx
from tripmate.services.weather import WeatherService

TODAY = date(2026, 10, 21)

def make_service(handler):
    return WeatherService(client=httpx.Client(transport=httpx.MockTransport(handler)),
                          today_fn=lambda: TODAY)

def geo(_req):
    return httpx.Response(200, json={"results": [{"name": "Reykjavik", "country": "Iceland",
                                                  "latitude": 64.1, "longitude": -21.9}]})

def test_climatology_for_far_month():
    def handler(req):
        if "geocoding" in str(req.url):
            return geo(req)
        return httpx.Response(200, json={"daily": {
            "temperature_2m_max": [3, 4], "temperature_2m_min": [-3, -2],
            "precipitation_sum": [2.0, 0.0], "snowfall_sum": [1.0, 0.0]}})
    r = make_service(handler).get_forecast("Reykjavik", "January")
    assert r["status"] == "ok" and r["source"] == "climatology"
    assert r["temp_range_c"][0] < r["temp_range_c"][1]

def test_city_not_found():
    svc = make_service(lambda req: httpx.Response(200, json={}))
    assert svc.get_forecast("Atlantis", "May")["error_code"] == "CITY_NOT_FOUND"

def test_api_failure_is_graceful():
    def handler(req):
        return geo(req) if "geocoding" in str(req.url) else httpx.Response(500)
    assert make_service(handler).get_forecast("Reykjavik", "May")["error_code"] == "WEATHER_SERVICE_UNAVAILABLE"

def test_incomplete_data():
    def handler(req):
        if "geocoding" in str(req.url):
            return geo(req)
        return httpx.Response(200, json={"daily": {"temperature_2m_max": [None], "temperature_2m_min": [None]}})
    assert make_service(handler).get_forecast("Reykjavik", "May")["error_code"] == "INCOMPLETE_DATA"

def test_invalid_date():
    assert make_service(geo).get_forecast("Reykjavik", "whenever")["error_code"] == "INVALID_DATE_REFERENCE"

def test_single_day_forecast_has_no_percentages():
    def handler(req):
        if "geocoding" in str(req.url):
            return geo(req)
        return httpx.Response(200, json={"daily": {
            "temperature_2m_max": [24], "temperature_2m_min": [16],
            "precipitation_sum": [5.0], "snowfall_sum": [0.0]}})
    r = make_service(handler).get_forecast("Reykjavik", "2026-10-25")
    assert r["source"] == "forecast" and "rainy_day_pct" not in r
    assert r["conditions"] == "mild, rain expected"