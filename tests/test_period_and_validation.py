from datetime import date
import pytest
from tripmate.services.period import PeriodError, parse_period
from tripmate.validation import InputValidationError, validate_user_query

TODAY = date(2026, 10, 21)

def test_month_name_rolls_forward():
    assert parse_period("January", TODAY).start == date(2027, 1, 1)
    assert parse_period("December", TODAY).start == date(2026, 12, 1)

def test_iso_and_relative():
    assert parse_period("2026-10-25", TODAY).kind == "date"
    assert parse_period("next month", TODAY).start == date(2026, 11, 1)

@pytest.mark.parametrize("bad", ["", "   ", "soonish", "2026-02-31"])
def test_bad_period(bad):
    with pytest.raises(PeriodError):
        parse_period(bad, TODAY)

@pytest.mark.parametrize("bad", ["", "   ", "!!!", None, "x" * 5000])
def test_bad_queries(bad):
    with pytest.raises(InputValidationError):
        validate_user_query(bad, 1000)