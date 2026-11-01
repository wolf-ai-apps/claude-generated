# 00 · Overview

## Problem

We want one codebase that teaches every layer of an LLM application, as described in
*The Life of a Prompt*, and that is built the way a production team would build it. It
must not be a refund bot: refunds are the first example, and any other use case should
plug in without changing the platform.

## Goals

1. **Generic engine, pluggable use cases.** Use cases ("packs") are declared as data plus
   small code modules. The engine has zero domain knowledge.
2. **Learn the mechanics, then the abstractions.** Every pack can run on a hand-written
   agent loop (`raw`) or on Strands Agents (`strands`), so the difference is visible.
3. **Production shape.** Stateless API, durable sessions, policy enforced in code,
   model gateway with fallback, guardrails, OpenTelemetry tracing, evals in CI,
   container and Kubernetes deployment.
4. **Runs offline.** A deterministic mock model makes the app, the tests and the evals
   run with no API key and no cost.
5. **One chapter, one component.** Each of the 16 chapters maps to a concrete module.

## Non-goals (for now)

- Multi-tenant SaaS (per-tenant isolation, billing). Noted in [Deployment](10-deployment.md) as future work.
- A polished end-user UI. We ship a minimal chat page and a CLI.
- Training or fine-tuning models.
- Supporting every LLM vendor. The provider interface allows it; we implement Anthropic and Mock.

## Requirements

### Functional

| ID | Requirement |
|---|---|
| F1 | Chat over HTTP with streamed replies (SSE) and over a CLI |
| F2 | Discover packs from `packs/*/pack.yaml`, validate them, hot-reload in dev |
| F3 | Route a question to the right pack (rules → embeddings → LLM) |
| F4 | Orchestrate multi-part questions across several packs |
| F5 | Tools from Python functions, MCP servers (stdio and HTTP), OpenAPI specs, and other agents |
| F6 | Declarative policies: deny rules, approval-required tools, retry classes |
| F7 | Human approval queue; paused runs resume after approve/reject |
| F8 | Long-term memory and RAG over each pack's knowledge folder |
| F9 | Model aliases (`fast`, `default`, `deep`), escalation and fallback |
| F10 | Input/output guardrails: prompt injection, PII redaction |
| F11 | Traces, metrics and cost per request, per pack, per model |
| F12 | Eval runner: answer quality, tool-use correctness, routing accuracy |
| F13 | Expose any pack as an MCP server |

### Non-functional

| Concern | Target |
|---|---|
| Latency | Router decision p95 < 500 ms; first streamed token p95 < 2 s (real model) |
| Availability | API replicas are stateless; any replica can resume any session |
| Safety | No side-effect tool runs without passing policy; money-moving tools never auto-retry |
| Cost | Cost is recorded per request; cheap models for routing, compaction, guardrails |
| Observability | Every hop emits an OpenTelemetry span with `pack`, `model`, `tokens`, `cost` |
| Testability | Full test suite runs on the mock provider in < 30 s |
| Extensibility | A new pack requires no engine change; a new provider/backend is one class |

## Chapter → component map

| Ch | Chapter | Component | Built with |
|---|---|---|---|
| 01 | Application | `api/` chat endpoint + SSE, CLI, minimal chat page | FastAPI |
| 02 | Harness | `policy/`, approvals, sandbox runner, skills loader, checkpoints | engine |
| 03 | Agent loop | `backends/raw.py`, `backends/strands.py`, sessions, interrupts, retries | engine + Strands |
| 04 | Context | `context/` prompt templates, injection, compaction, conversation manager | engine (+ Strands manager) |
| 05 | Memory | `memory/` long-term memory tool, `rag/` chunking + hybrid search | engine |
| 06 | Tools | `tools/` Python, MCP, OpenAPI, agent-as-tool, ToolExecutor | engine + MCP SDK |
| 07 | Multi-agent | `routing/`, `orchestration/` | engine + Strands (swarm/graph) |
| 08 | Client SDK | `gateway/providers/anthropic.py`, `backends/raw.py` | anthropic SDK |
| 09 | Gateway | `gateway/` aliases, rate limits, fallback, cost | engine |
| 10 | Provider | `gateway/providers/` anthropic, mock | engine |
| 11 | Model | model capabilities, token counting, stop reasons | anthropic SDK |
| 12 | Caching | prompt-cache placement, response cache for deterministic calls | API + engine |
| 13 | Safety | `safety/` guardrails in and out | engine |
| 14 | Observability | `telemetry/` OTel spans, metrics, cost | OpenTelemetry |
| 15 | Evals | `evals/` golden sets, judges, routing accuracy | engine |
| 16 | Deployment | `deploy/` Dockerfile, Compose, Kubernetes | Docker, K8s |

## Key decisions (see [ADRs](../adr/))

1. [ADR-0001](../adr/0001-engine-owns-the-platform.md) — We write the engine; agent SDKs are pluggable backends.
2. [ADR-0002](../adr/0002-packs-as-declarative-config.md) — Use cases are declarative packs.
3. [ADR-0003](../adr/0003-two-backends-raw-and-strands.md) — Two backends: raw loop and Strands.
4. [ADR-0004](../adr/0004-policy-at-the-tool-boundary.md) — Policy is enforced at the tool boundary by the engine.
5. [ADR-0005](../adr/0005-model-aliases-and-gateway.md) — Packs reference model aliases, never model IDs.
6. [ADR-0006](../adr/0006-state-outside-the-process.md) — State lives outside the process (SQLite locally, Postgres in K8s).
7. [ADR-0007](../adr/0007-mock-provider-first.md) — A deterministic mock provider is a first-class provider.
8. [ADR-0008](../adr/0008-mcp-as-language-neutral-tool-boundary.md) — MCP is the language-neutral tool boundary.
9. [ADR-0009](../adr/0009-a2a-only-across-ownership-boundaries.md) — A2A only across ownership boundaries.
