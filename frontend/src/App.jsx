import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import "./App.css";

const SUGGESTIONS = [
  "Do I need a visa for Tokyo? I'm a US citizen.",
  "What's the weather like in Bangkok in July?",
  "What should I pack for Reykjavik in January?",
  "I want to see the Northern Lights in Iceland. What should I pack?",
  "Can you book my flight to Barcelona?",
  "Tell me about Paris customs.",
];

const split = (args = {}) => {
  const { reason, ...rest } = args;
  return { reason, rest };
};

function EventBody({ e }) {
  switch (e.event) {
    case "request.received":
      return (<>
        <div className="ev-title">Request received</div>
        <div className="ev-text">{e.query}</div>
      </>);

    case "validation.rejected":
      return (<>
        <div className="ev-title">Input rejected (no model call was made)</div>
        <div className="ev-text">{e.reason}</div>
      </>);

    case "agent.decision":
      if (e.decision === "call_tools") {
        const calls = e.tool_calls || [];
        return (<>
          <div className="ev-title">
            Model decided to call {calls.length} tool{calls.length !== 1 ? "s" : ""}
          </div>
          {calls.map((c, i) => {
            const { reason, rest } = split(c.args);
            return (
              <div className="ev-sub" key={i}>
                <b>{c.tool}</b>
                {reason && <div className="ev-reason">Why: {reason}</div>}
                <pre>{JSON.stringify(rest, null, 2)}</pre>
              </div>
            );
          })}
        </>);
      }
      return (<>
        <div className="ev-title">Model decided to answer (no more tools needed)</div>
        <div className="ev-text">{e.text}</div>
      </>);

    case "tool.call":
      return (<>
        <div className="ev-title">Running tool: {e.tool}</div>
        <pre>{JSON.stringify(e.args, null, 2)}</pre>
      </>);

    case "tool.result":
      return (<>
        <div className="ev-title">Tool result: {e.tool} ({e.ms} ms)</div>
        <pre>{e.result}</pre>
      </>);

    case "tool.error":
      return (<>
        <div className="ev-title">Tool error: {e.tool}</div>
        <pre>{e.error}</pre>
      </>);

    case "weather.cache_hit":
      return <div className="ev-title">Weather cache hit: {e.city}, {e.period}</div>;

    case "weather.error":
      return (<>
        <div className="ev-title">Weather error: {e.code}</div>
        <div className="ev-text">{e.message}</div>
      </>);

    case "agent.final":
      return (<>
        <div className="ev-title">Final answer returned</div>
        <div className="ev-text">Tools used: {(e.tools_used || []).join(", ") || "none"}</div>
      </>);

    default: {
      const { t, level, event, ...rest } = e;
      return (<>
        <div className="ev-title">{event}</div>
        {Object.keys(rest).length > 0 && <pre>{JSON.stringify(rest, null, 2)}</pre>}
      </>);
    }
  }
}

function TracePanel({ events, traceId, open }) {
  if (!events?.length) return <div className="trace">No trace available · {traceId}</div>;
  const t0 = events[0].t;
  const total = (events[events.length - 1].t - t0).toFixed(2);
  const models = events.filter((e) => e.event === "agent.decision").length;
  const tools = events.filter((e) => e.event === "tool.call").length;
  return (
    <details className="trace" open={open}>
      <summary>
        Full trace {traceId} · {events.length} events · {total}s · {models} model call{models !== 1 ? "s" : ""} · {tools} tool call{tools !== 1 ? "s" : ""}
      </summary>
      <div className="timeline">
        {events.map((e, i) => (
          <div key={i} className={`ev ${e.level === "ERROR" ? "err" : e.level === "WARNING" ? "warn" : ""}`}>
            <div className="ev-head">
              <span className="ev-step">#{i + 1}</span>
              <span className="ev-time">+{Math.round((e.t - t0) * 1000)} ms</span>
            </div>
            <EventBody e={e} />
          </div>
        ))}
      </div>
    </details>
  );
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [threadId, setThreadId] = useState(() => crypto.randomUUID());
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function send(text) {
    const q = text.trim();
    if (!q || loading) return;
    setMessages((m) => [...m, { role: "user", text: q }]);
    setInput("");
    setLoading(true);
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: q, thread_id: threadId }),
      });
      if (!res.ok) throw new Error(`Server error (${res.status})`);
      const d = await res.json();
      setMessages((m) => [...m, {
        role: "assistant", text: d.answer, status: d.status,
        traceId: d.trace_id, events: d.events,
      }]);
    } catch {
      setMessages((m) => [...m, {
        role: "assistant", status: "error", events: [],
        text: "I couldn't reach the TripMate server. Is the backend running on port 8000?",
      }]);
    } finally {
      setLoading(false);
    }
  }

  function newChat() {
    setMessages([]);
    setThreadId(crypto.randomUUID());
  }

  return (
    <div className="app">
      <header>
        <h1>🧳 TripMate</h1>
        <button onClick={newChat} disabled={loading}>New chat</button>
      </header>

      <main>
        {messages.length === 0 && (
          <div className="welcome">
            <p>Ask about visas, weather, packing, customs or safety for Tokyo, Reykjavik, Bangkok or Barcelona.</p>
            <div className="chips">
              {SUGGESTIONS.map((s) => (
                <button key={s} onClick={() => send(s)}>{s}</button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role} ${m.status === "error" || m.status === "rejected" ? "warn" : ""}`}>
            {m.role === "user" ? m.text : <ReactMarkdown>{m.text}</ReactMarkdown>}
            {m.role === "assistant" && m.traceId && (
              <TracePanel events={m.events} traceId={m.traceId} open={i === messages.length - 1} />
            )}
          </div>
        ))}

        {loading && <div className="msg assistant typing">TripMate is thinking…</div>}
        <div ref={endRef} />
      </main>

      <form onSubmit={(e) => { e.preventDefault(); send(input); }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about a destination…"
          maxLength={1000}
          disabled={loading}
        />
        <button type="submit" disabled={loading || !input.trim()}>Send</button>
      </form>
    </div>
  );
}