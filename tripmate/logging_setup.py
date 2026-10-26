from __future__ import annotations
import contextvars, json, logging, sys, threading
from collections import OrderedDict
from pathlib import Path

trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")
LOGGER_NAME = "tripmate"
class TraceCollector(logging.Handler):
    """Keeps recent requests' log events in memory, grouped by trace_id (for the UI)."""

    def __init__(self, max_traces: int = 200):
        super().__init__()
        self._events: OrderedDict[str, list[dict]] = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_traces

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tid = trace_id_var.get()
            if tid == "-":                      # not inside a request (e.g. ingest)
                return
            data = json.loads(json.dumps(getattr(record, "data", {}), default=str))
            event = {"t": record.created, "level": record.levelname,
                     "event": record.getMessage(), **data}
            with self._lock:
                self._events.setdefault(tid, []).append(event)
                while len(self._events) > self._max:
                    self._events.popitem(last=False)
        except Exception:
            self.handleError(record)

    def pop(self, trace_id: str) -> list[dict]:
        with self._lock:
            return self._events.pop(trace_id, [])


collector = TraceCollector()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "trace_id": trace_id_var.get(),
            "event": record.getMessage(),
            **getattr(record, "data", {}),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class HumanFormatter(logging.Formatter):
    """Readable live trace for the terminal / Loom demo."""
    def format(self, record: logging.LogRecord) -> str:
        data = getattr(record, "data", {})
        body = json.dumps(data, default=str, ensure_ascii=False)
        if len(body) > 400:
            body = body[:400] + "…"
        return f"[{trace_id_var.get()}] {record.getMessage():<18} {body}"


def setup_logging(level: str, log_file: Path, verbose: bool = False) -> None:
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.setLevel(level.upper())
    logger.propagate = False

    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(JsonFormatter())
    logger.addHandler(fh)
    logger.addHandler(collector)

    if verbose:
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(HumanFormatter())
        logger.addHandler(ch)

    for noisy in ("httpx", "httpcore", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **data) -> None:
    logger.log(level, event, extra={"data": data})