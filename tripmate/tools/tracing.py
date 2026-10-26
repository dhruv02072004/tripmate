from __future__ import annotations
import functools, logging, time
from typing import Any, Callable

from tripmate.logging_setup import log_event

logger = logging.getLogger("tripmate.tools")


def _preview(obj: Any, limit: int = 3000) -> str:
    s = str(obj)
    return s if len(s) <= limit else s[:limit] + "…"


def traced(name: str, on_error: Callable[[str], Any]):
    """Log every call/result and guarantee the tool never raises into the LLM."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            log_event(logger, "tool.call", tool=name, args=kwargs)
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:  
                logger.exception("tool.exception")
                log_event(logger, "tool.error", level=logging.ERROR, tool=name, error=repr(exc))
                return on_error("An internal error occurred while running this tool.")
            log_event(logger, "tool.result", tool=name,
                      ms=round((time.perf_counter() - start) * 1000), result=_preview(result))
            return result
        return wrapper
    return deco