# 01 · Architecture

## Layers

```mermaid
flowchart TB
    subgraph PACKS["Packs — change per use case"]
        P1[refunds]
        P2[it_helpdesk]
        P3[general]
    end
    subgraph ENGINE["Engine — promptlife/ (we write this)"]
        API[API + CLI]
        REG[Pack registry]
        RT[Router]
        ORC[Orchestrator]
        FAC[Agent factory]
        EXE[ToolExecutor + policy]
        CTX[Context builder]
        MEM[Memory + RAG]
        GW[Model gateway]
        SAFE[Guardrails]
        TEL[Telemetry]
    end
    subgraph LIBS["Libraries"]
        STR[Strands Agents]
        SDK[anthropic SDK]
        MCPSDK[MCP SDK]
    end
    CLAUDE[(Claude API)]
    MOCK[(Mock model)]

    PACKS --> REG
    FAC --> STR
    FAC --> EXE
    GW --> SDK --> CLAUDE
    GW --> MOCK
    EXE --> MCPSDK
```

Analogy for platform engineers: **packs are Deployments** (a declarative spec), **the
engine is Kubernetes** (reads specs, wires config, secrets, policy, observability), and
**Strands / the raw loop is the container runtime** (actually runs the workload).

## Component view

```mermaid
flowchart LR
    U([User / client]) -->|HTTPS| API

    subgraph REQ["Request path"]
        API[API service<br/>FastAPI · SSE] --> GIN[Input guardrails]
        GIN --> ROUTER[Router<br/>rules → embeddings → LLM]
        ROUTER -->|single domain| AG[Pack agent]
        ROUTER -->|multi-part| ORC[Orchestrator]
        ORC -->|one run per pack| AG
        AG --> RUN[Agent runtime<br/>backend: raw or strands]
        RUN -->|messages + tools| GW[Model gateway]
        RUN -->|tool call| EXE[ToolExecutor]
        RUN -->|reply| GOUT[Output guardrails]
        GOUT -->|SSE stream| API
    end

    subgraph State
        SES[(Session store)]
        MEMS[(Memory store)]
        VEC[(Vector + keyword index)]
        APR[(Approvals)]
    end

    subgraph Tools
        PY[Python tools]
        MCP[MCP servers]
        OAS[OpenAPI APIs]
        SUB[Agents as tools]
    end

    REG[Pack registry] --> ROUTER
    REG --> AG
    API <-->|load / save| SES
    CTX[Context builder] --> RUN
    MEMS --> CTX
    VEC --> CTX
    EXE -->|policy check| POL[Policy rules]
    EXE -->|needs approval| APR
    EXE --> PY & MCP & OAS & SUB
    GW --> ANT[(Anthropic API)]
    GW --> MOCKP[(Mock model)]
```

## One request, end to end

Maya asks *"Where's my refund for order #4417?"*

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant A as API
    participant G as Guardrails
    participant R as Router
    participant F as Agent factory
    participant L as Agent loop
    participant W as Model gateway
    participant M as Claude
    participant X as ToolExecutor
    participant T as get_order tool
    participant S as Session store

    C->>A: POST /v1/chat {message, session_id}
    A->>S: load session (messages + state)
    A->>G: check input (injection, PII)
    G-->>A: clean input
    A->>R: route(question)
    R-->>A: {pack: refunds, complexity: simple, method: embeddings}
    A->>F: get_agent(refunds)
    F-->>A: configured agent (cached)
    A->>L: run(messages)
    L->>W: complete(alias=default, system, messages, tools)
    W->>M: messages.create(model=claude-sonnet-5-5, ...)
    M-->>W: stop_reason=tool_use, get_order(order_id=4417)
    W-->>L: response + usage + cost
    L->>X: execute(get_order, {order_id: 4417})
    X->>X: policy: allowed, read-only, retry ≤ 3
    X->>T: call
    T-->>X: {status: refunded, date: ...}
    X-->>L: tool_result
    L->>W: complete(... + tool_result)
    W->>M: messages.create(...)
    M-->>W: stop_reason=end_turn, text
    W-->>L: final answer
    L-->>A: answer + events
    A->>G: check output (PII, policy)
    A->>S: save session
    A-->>C: SSE: route, tool_call, tool_result, text, done
```

Every arrow above is also an OpenTelemetry span (see [09](09-observability-and-evals.md)).

## Engine modules and their contracts

| Module | Responsibility | Key interface |
|---|---|---|
| `packs.registry` | Discover, validate, hot-reload packs | `Registry.get(name) -> Pack`, `Registry.cards() -> list[PackCard]` |
| `packs.factory` | Turn a `Pack` into a runnable agent | `AgentFactory.build(pack, backend) -> Agent` |
| `routing.router` | Pick pack + complexity | `Router.route(question, session) -> Route` |
| `orchestration` | Plan, fan out, merge | `Orchestrator.run(question, route) -> Answer` |
| `backends` | Run the agent loop | `Backend.run(agent_spec, messages, events) -> RunResult` |
| `tools.executor` | Every tool call goes through here | `ToolExecutor.call(tool, input, ctx) -> ToolResult` |
| `policy` | Allow / deny / needs approval / retry class | `Policy.evaluate(tool, input, ctx) -> Decision` |
| `context` | Build system prompt and message window | `ContextBuilder.build(pack, session, question) -> Context` |
| `memory`, `rag` | Long-term facts, retrieval | `Memory.read/write`, `Retriever.search(query, k)` |
| `gateway` | Model calls with alias resolution, retries, fallback, cost | `Gateway.complete(alias, request) -> ModelResponse` |
| `safety` | Input and output checks | `Guardrails.check_input/check_output` |
| `telemetry` | Spans, metrics, cost | `span(name, **attrs)`, `metrics.record(...)` |

All interfaces use the Anthropic Messages content-block shape (`text`, `tool_use`,
`tool_result`) as the internal message format. It is the format the raw loop speaks
natively, and the mock provider emits the same blocks.

## Data stores

| Store | Holds | Local | Production |
|---|---|---|---|
| Sessions | messages (append-only), agent state, checkpoint | SQLite | Postgres |
| Approvals | pending tool calls awaiting a human | SQLite | Postgres |
| Memory | long-term facts per user per pack | SQLite | Postgres |
| RAG index | chunks, embeddings, keyword index | SQLite + in-process index | Postgres + pgvector |
| Offloaded blobs | large tool outputs | `./data/blobs` | object storage (S3/R2) |
| Traces | spans | Jaeger (Compose) | OTel Collector → Jaeger/Tempo |

## Failure modes and responses

| Failure | Response |
|---|---|
| Model 429 / 5xx | Gateway retries with backoff, then falls back to the alias's fallback model |
| Model refusal (`stop_reason: refusal`) | Server-side fallback when enabled; otherwise a safe message and a `refusal` event |
| Tool timeout (read-only) | Retry up to the pack's `read_only` budget |
| Tool timeout (side effect) | Never retried; surfaced to the agent as an error result |
| Loop runs too long | `max_turns` stops it; optional escalation to the `deep` alias |
| Replica dies mid-run | Session checkpoint lets any replica resume |
| Approval needed | Run pauses with an `interrupt`; resumes on approve/reject |
