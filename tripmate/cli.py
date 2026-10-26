from __future__ import annotations
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import argparse

from tripmate.bootstrap import build_agent
from tripmate.config import get_settings
from tripmate.logging_setup import setup_logging

DEMO = [
    "Do I need a visa for Tokyo? I'm a US citizen.",                           # single tool (RAG)
    "What's the weather like in Bangkok in July?",                             # single tool (weather)
    "What should I pack for Reykjavik in January?",                            # multi-tool
    "I want to see the Northern Lights in Iceland. What should I pack?",       # chained: RAG -> weather
    "Can you book my flight to Barcelona?",                                    # out of scope
    "Tell me about Paris customs.",                                            # unsupported destination
]


def main() -> None:
    p = argparse.ArgumentParser(prog="tripmate")
    p.add_argument("query", nargs="?", help="Ask one question and exit")
    p.add_argument("-v", "--verbose", action="store_true", help="Show live reasoning trace")
    p.add_argument("--demo", action="store_true", help="Run the sample queries")
    args = p.parse_args()

    s = get_settings()
    setup_logging(s.log_level, s.log_file, verbose=args.verbose)
    agent = build_agent(s)

    def run(q: str, thread: str) -> None:
        res = agent.ask(q, thread_id=thread)
        print(f"\n> {q}\n{res.answer}")
        print(f"[tools: {', '.join(c['tool'] for c in res.tool_calls) or 'none'} | trace {res.trace_id}]")

    if args.demo:
        for i, q in enumerate(DEMO):
            run(q, f"demo-{i}")
    elif args.query:
        run(args.query, "once")
    else:
        print("TripMate - type a question ('quit' to exit)")
        while True:
            try:
                q = input("\nYou: ")
            except (EOFError, KeyboardInterrupt):
                break
            if q.strip().lower() in {"quit", "exit"}:
                break
            run(q, "interactive")