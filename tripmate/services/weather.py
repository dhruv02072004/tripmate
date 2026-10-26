from __future__ import annotations
import calendar, logging, statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from typing import Callable

import httpx
from cachetools import TTLCache
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from tripmate.logging_setup import log_event
from tripmate.services.period import Period, PeriodError, parse_period

logger = logging.getLogger("tripmate.weather")

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY = "temperature_2m_max,temperature_2m_min,precipitation_sum,snowfall_sum"
FORECAST_HORIZON_DAYS = 15
CLIMATE_YEARS = 3


class WeatherError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def _retryable(exc: BaseException) -> bool:
    return isinstance(exc, httpx.TransportError) or (
        isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500
    )


def _shift_year(d: date, year: int) -> date:
    try:
        return d.replace(year=year)
    except ValueError:                      # Feb 29
        return d.replace(year=year, day=28)


def _describe(avg_hi: float, avg_lo: float, wet_pct, snow_pct) -> str:
    mean = (avg_hi + avg_lo) / 2
    temp = ("freezing" if mean < 0 else "cold" if mean < 8 else "cool" if mean < 15
            else "mild" if mean < 22 else "warm" if mean < 28 else "hot")
    parts = [temp]
    if snow_pct is not None and snow_pct >= 15:
        parts.append("frequent snow" if snow_pct >= 40 else "occasional snow")
    elif wet_pct is not None:
        parts.append("frequent rain" if wet_pct >= 40 else "occasional rain" if wet_pct >= 15
                     else "mostly dry")
    return ", ".join(parts)

def _describe_day(hi: float, lo: float, precip_mm, snow_cm) -> str:
    mean = (hi + lo) / 2
    temp = ("freezing" if mean < 0 else "cold" if mean < 8 else "cool" if mean < 15
            else "mild" if mean < 22 else "warm" if mean < 28 else "hot")
    if snow_cm is not None and snow_cm >= 0.5:
        return f"{temp}, snow expected"
    if precip_mm is not None and precip_mm >= 1.0:
        return f"{temp}, rain expected"
    return f"{temp}, mostly dry"

class WeatherService:
    def __init__(self, timeout_s: float = 8.0, cache_ttl_s: int = 3600,
                 client: httpx.Client | None = None,
                 today_fn: Callable[[], date] = date.today):
        self._client = client or httpx.Client(timeout=timeout_s)
        self._cache: TTLCache = TTLCache(maxsize=512, ttl=cache_ttl_s)
        self._geo: dict[str, dict] = {}
        self._today = today_fn

    # ---------- public ----------
    def get_forecast(self, city: str, date_or_month: str) -> dict:
        try:
            period = parse_period(date_or_month, self._today())
            place = self._geocode(city)
            key = (place["lat"], place["lon"], period.kind, period.start, period.end)
            if key in self._cache:
                log_event(logger, "weather.cache_hit", city=place["name"], period=period.label)
                return self._cache[key]
            result = self._build(place, period)
            self._cache[key] = result
            return result
        except PeriodError as exc:
            return {"status": "error", "error_code": "INVALID_DATE_REFERENCE", "message": str(exc)}
        except WeatherError as exc:
            log_event(logger, "weather.error", level=logging.WARNING, code=exc.code, message=exc.message)
            return {"status": "error", "error_code": exc.code, "message": exc.message}

    # ---------- internals ----------
    def _geocode(self, city: str) -> dict:
        if not isinstance(city, str) or not city.strip():
            raise WeatherError("INVALID_CITY", "A city name is required.")
        key = city.strip().lower()
        if key in self._geo:
            return self._geo[key]
        data = self._get_json(GEOCODE_URL, {"name": city.strip(), "count": 1,
                                            "language": "en", "format": "json"})
        results = data.get("results") or []
        if not results:
            raise WeatherError("CITY_NOT_FOUND", f"I couldn't find a location called '{city}'.")
        r = results[0]
        place = {"name": r["name"], "country": r.get("country", ""),
                 "lat": r["latitude"], "lon": r["longitude"]}
        self._geo[key] = place
        return place

    def _build(self, place: dict, period: Period) -> dict:
        today = self._today()
        in_horizon = (period.kind == "date"
                      and today <= period.start <= today + timedelta(days=FORECAST_HORIZON_DAYS))
        if in_horizon:
            merged = self._daily(FORECAST_URL, place, period.start, period.end)
            return self._summarize(merged, place, period, "forecast", single_day=True)

        windows = self._climate_windows(period, today)
        with ThreadPoolExecutor(max_workers=len(windows)) as pool:
            chunks = list(pool.map(lambda w: self._daily(ARCHIVE_URL, place, *w), windows))
        merged = {k: [v for c in chunks for v in c.get(k, [])]
                  for k in ("temperature_2m_max", "temperature_2m_min",
                            "precipitation_sum", "snowfall_sum")}
        years = [today.year - k for k in range(1, CLIMATE_YEARS + 1)]
        return self._summarize(merged, place, period, "climatology", years=years)

    @staticmethod
    def _climate_windows(period: Period, today: date) -> list[tuple[date, date]]:
        out = []
        for k in range(1, CLIMATE_YEARS + 1):
            y = today.year - k
            if period.kind == "month":
                m = period.start.month
                out.append((date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1])))
            else:
                c = _shift_year(period.start, y)
                out.append((c - timedelta(days=3), c + timedelta(days=3)))
        return out

    def _daily(self, url: str, place: dict, start: date, end: date) -> dict:
        data = self._get_json(url, {
            "latitude": place["lat"], "longitude": place["lon"], "daily": DAILY,
            "start_date": start.isoformat(), "end_date": end.isoformat(), "timezone": "auto",
        })
        return data.get("daily") or {}

    
    @staticmethod
    def _summarize(d: dict, place: dict, period: Period, source: str,
                   single_day: bool = False, years: list[int] | None = None) -> dict:
        highs = [v for v in d.get("temperature_2m_max", []) if v is not None]
        lows = [v for v in d.get("temperature_2m_min", []) if v is not None]

        if not highs or not lows:
            raise WeatherError(
                "INCOMPLETE_DATA",
                f"Weather data for {place['name']} ({period.label}) is incomplete."
            )

        prec = [v for v in d.get("precipitation_sum", []) if v is not None]
        snow = [v for v in d.get("snowfall_sum", []) if v is not None]

        wet = round(100 * sum(v >= 1.0 for v in prec) / len(prec)) if prec else None
        snw = round(100 * sum(v >= 0.5 for v in snow) / len(snow)) if snow else None

        hi, lo = statistics.fmean(highs), statistics.fmean(lows)

        result = {
            "status": "ok",
            "city": place["name"],
            "country": place["country"],
            "period": period.label,
            "source": source,
            "temp_range_c": [round(lo), round(hi)],
        }

        if single_day:
            mm = prec[0] if prec else None
            cm = snow[0] if snow else None
            result["conditions"] = _describe_day(hi, lo, mm, cm)
            result["precipitation_mm"] = mm
        else:
            result["conditions"] = _describe(hi, lo, wet, snw)
            result["rainy_day_pct"] = wet
            result["snow_day_pct"] = snw

        if source == "climatology":
            result["note"] = (
                f"Typical conditions averaged over {min(years)}-{max(years)}; "
                "this is climate data, not a forecast."
            )

        return result


    def _get_json(self, url: str, params: dict) -> dict:
        try:
            return self._request(url, params)
        except (httpx.HTTPError, ValueError) as exc:
            log_event(logger, "weather.http_failure", level=logging.ERROR, url=url, error=repr(exc))
            raise WeatherError("WEATHER_SERVICE_UNAVAILABLE",
                               "The weather service is temporarily unavailable.") from exc

    @retry(retry=retry_if_exception(_retryable), stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=0.3, max=2), reraise=True)
    def _request(self, url: str, params: dict) -> dict:
        resp = self._client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()