from pydantic import BaseModel, Field


class SearchGuideInput(BaseModel):
    query: str = Field(..., description="The traveler's question or topic, e.g. 'tipping customs' or 'what to wear for temples'.")
    city: str | None = Field(None, description="Destination city the question is about. ALWAYS set this when a city is named or implied.")
    reason: str = Field("", description="One short sentence: why you are calling this tool.")


class WeatherInput(BaseModel):
    city: str = Field(..., description="City name, e.g. 'Reykjavik'.")
    date_or_month: str = Field(..., description="ISO date 'YYYY-MM-DD' or a month name like 'December' / 'December 2026'. Convert relative phrases ('this Christmas') to one of these.")
    reason: str = Field("", description="One short sentence: why you are calling this tool.")