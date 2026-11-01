# 11 · HTTP API

Base path `/v1`. JSON in and out. Auth via `Authorization: Bearer <token>`.

## Chat

### `POST /v1/chat`

```json
{
  "message": "Where's my refund for order #4417?",
  "session_id": "s_01H...",          // optional; created if absent
  "pack": "refunds",                 // optional; skips routing
  "stream": true,                    // SSE when true
  "options": {"backend": "raw"}      // dev only
}
```

Non-streaming response:

```json
{
  "session_id": "s_01H...",
  "answer": "Your refund of $349.00 for order #4417 was issued on 12 September...",
  "route": {"pack": "refunds", "method": "embeddings", "confidence": 0.91, "complexity": "simple"},
  "tool_calls": [{"name": "get_order", "status": "ok", "duration_ms": 120}],
  "interrupt": null,
  "usage": {"input_tokens": 2310, "output_tokens": 190, "cache_read_tokens": 1800, "cost_usd": 0.0071},
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736"
}
```

### Streaming events (SSE)

| Event | Data |
|---|---|
| `route` | route object |
| `tool_call` | `{id, name, input}` (input redacted) |
| `tool_result` | `{id, name, status, summary}` |
| `text` | `{delta}` |
| `interrupt` | `{approval_id, tool, input, reason}` |
| `usage` | usage object |
| `error` | `{code, message}` |
| `done` | `{session_id, trace_id}` |

## Sessions

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/sessions/{id}` | messages (redacted), state, pending approvals |
| `DELETE` | `/v1/sessions/{id}` | delete a session |

## Approvals

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/approvals?status=pending&pack=refunds` | approval queue |
| `POST` | `/v1/approvals/{id}` | `{"decision": "approve" \| "reject", "note": "..."}`; resumes the run |

## Packs

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/packs` | pack cards and versions |
| `GET` | `/v1/packs/{name}` | manifest (secrets masked), tool list, status |
| `POST` | `/v1/packs/{name}/reload` | dev only |

## Memory

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/memories?pack=refunds` | the caller's memories |
| `DELETE` | `/v1/memories/{id}` | forget one |

## Operations

| Path | Purpose |
|---|---|
| `/healthz` | liveness |
| `/readyz` | readiness: packs loaded, store reachable |
| `/metrics` | Prometheus metrics |

## Errors

```json
{"error": {"code": "policy_denied", "message": "Refunds over $500 must go through finance.", "trace_id": "..."}}
```

| HTTP | `code` |
|---|---|
| 400 | `invalid_request` |
| 401 / 403 | `unauthenticated` / `forbidden` |
| 404 | `not_found` (session, pack, approval) |
| 409 | `approval_already_decided` |
| 413 | `input_too_large` |
| 429 | `rate_limited` (per user) |
| 502 | `model_unavailable` (all fallbacks failed) |

## CLI equivalents

```
promptlife serve
promptlife chat [--pack refunds] [--backend raw]
promptlife ask "Where's my refund for #4417?"
promptlife packs list | validate | new <name>
promptlife approvals list | approve <id> | reject <id>
promptlife evals run [--pack refunds] [--live]
promptlife mcp serve --pack refunds
promptlife a2a serve
```
