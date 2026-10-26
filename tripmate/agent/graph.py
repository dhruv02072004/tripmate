from __future__ import annotations
import logging, uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from tripmate.agent.prompts import build_system_prompt
from tripmate.config import Settings
from tripmate.logging_setup import log_event, trace_id_var
from tripmate.validation import InputValidationError, validate_user_query

logger = logging.getLogger("tripmate.agent")


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def message_text(msg: AnyMessage) -> str:
    """Gemini 3.x can return content as a list of blocks; normalise to text."""
    c = msg.content
    if isinstance(c, str):
        return c.strip()
    parts = []
    for block in c:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts).strip()


@dataclass
class AgentResult:
    answer: str
    status: str                     
    trace_id: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


def build_graph(llm_with_tools, tools):
    tool_node = ToolNode(tools, handle_tool_errors=True)

    def call_model(state: AgentState) -> dict:
        response = llm_with_tools.invoke([SystemMessage(content=build_system_prompt())] + state["messages"])
        calls = [{"tool": c["name"], "args": c["args"]} for c in (getattr(response, "tool_calls", None) or [])]
        log_event(logger, "agent.decision", text=message_text(response), tool_calls=calls,
                  decision="call_tools" if calls else "final_answer")
        return {"messages": [response]}

    g = StateGraph(AgentState)
    g.add_node("agent", call_model)
    g.add_node("tools", tool_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)  
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=MemorySaver())


class TripMateAgent:
    def __init__(self, llm_with_tools, tools, settings: Settings):
        self._graph = build_graph(llm_with_tools, tools)
        self._settings = settings
        self._epoch: dict[str, int] = defaultdict(int)

    def ask(self, query: str, thread_id: str = "default") -> AgentResult:
        trace_id = uuid.uuid4().hex[:8]
        token = trace_id_var.set(trace_id)
        try:
            try:
                cleaned = validate_user_query(query, self._settings.max_query_chars)
            except InputValidationError as exc:
                log_event(logger, "validation.rejected", level=logging.WARNING, reason=str(exc))
                return AgentResult(str(exc), "rejected", trace_id)

            log_event(logger, "request.received", query=cleaned, thread=thread_id)
            config = {
                "configurable": {"thread_id": f"{thread_id}:{self._epoch[thread_id]}"},
                "recursion_limit": self._settings.max_agent_steps * 2 + 1,
            }
            try:
                out = self._graph.invoke({"messages": [HumanMessage(content=cleaned)]}, config=config)
            except GraphRecursionError:
                self._epoch[thread_id] += 1       
                log_event(logger, "agent.step_limit", level=logging.ERROR)
                return AgentResult("I couldn't finish that request within my step limit. "
                                   "Could you rephrase or simplify it?", "error", trace_id)
            except Exception:
                self._epoch[thread_id] += 1
                logger.exception("agent.failure")
                return AgentResult("Sorry, I hit a problem reaching the AI service. "
                                   "Please try again in a moment.", "error", trace_id)

            msgs = out["messages"]
            last_human = max(i for i, m in enumerate(msgs) if isinstance(m, HumanMessage))
            turn = msgs[last_human + 1:]
            results = {m.tool_call_id: str(m.content)[:300] for m in turn if isinstance(m, ToolMessage)}
            calls = [{"tool": c["name"], "args": c["args"], "result_preview": results.get(c["id"], "")}
                     for m in turn if isinstance(m, AIMessage) for c in (m.tool_calls or [])]
            final = next((message_text(m) for m in reversed(turn) if isinstance(m, AIMessage)
                          and message_text(m)), "")
            if not final:
                final = "Sorry, I wasn't able to produce an answer. Please try rephrasing."
            log_event(logger, "agent.final", answer=final, tools_used=[c["tool"] for c in calls])
            return AgentResult(final, "ok", trace_id, calls)
        finally:
            trace_id_var.reset(token)