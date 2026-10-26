from __future__ import annotations
import calendar, re
from dataclasses import dataclass
from datetime import date, timedelta


class PeriodError(ValueError):
    pass


_MONTHS: dict[str, int] = {}
for _i in range(1, 13):
    _MONTHS[calendar.month_name[_i].lower()] = _i
    _MONTHS[calendar.month_abbr[_i].lower()] = _i
_MONTHS["sept"] = 9


@dataclass(frozen=True)
class Period:
    kind: str        
    start: date
    end: date
    label: str


def _month(year: int, month: int) -> Period:
    last = calendar.monthrange(year, month)[1]
    return Period("month", date(year, month, 1), date(year, month, last),
                  f"{calendar.month_name[month]} {year}")


def _day(d: date) -> Period:
    return Period("date", d, d, d.isoformat())


def parse_period(text: str, today: date) -> Period:
    if not isinstance(text, str) or not text.strip():
        raise PeriodError("No date or month was provided.")
    t = re.sub(r"\s+", " ", text.replace(",", " ").strip().lower())

    if t in {"today", "now"}:
        return _day(today)
    if t == "tomorrow":
        return _day(today + timedelta(days=1))
    if t == "this month":
        return _month(today.year, today.month)
    if t == "next month":
        return _month(today.year + (today.month == 12), today.month % 12 + 1)

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t):
        try:
            return _day(date.fromisoformat(t))
        except ValueError as exc:
            raise PeriodError(f"'{text}' is not a valid calendar date.") from exc
    if (m := re.fullmatch(r"(\d{4})-(\d{2})", t)):
        y, mo = int(m[1]), int(m[2])
        if 1 <= mo <= 12:
            return _month(y, mo)
        raise PeriodError(f"'{text}' is not a valid month.")

    if (m := re.fullmatch(r"([a-z]+)(?: (\d{4}))?", t)) and m[1] in _MONTHS:
        mo = _MONTHS[m[1]]
        year = int(m[2]) if m[2] else (today.year if mo >= today.month else today.year + 1)
        return _month(year, mo)

    raise PeriodError(
        f"Could not understand the date reference '{text}'. Use an ISO date (YYYY-MM-DD) "
        "or a month name such as 'December' or 'December 2026'."
    )