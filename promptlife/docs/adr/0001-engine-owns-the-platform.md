# ADR-0001 · The engine owns the platform; agent SDKs are backends

- **Status:** Accepted
- **Date:** 2026-10-02

## Context
Agent SDKs (Strands, Claude Agent SDK, LangGraph, ADK) each bundle a loop, tools, hooks
and state. Building directly on one couples every use case to that SDK's concepts and
release cycle, and hides the mechanics we want to learn.

## Decision
We write a thin platform ("engine") that owns configuration, routing, policy, the model
gateway, safety, telemetry and evals. Agent loops are pluggable **backends** behind a
small interface.

## Alternatives considered
- **Build everything on Strands:** fastest start, but policy and telemetry would be
  Strands-specific, and swapping frameworks would mean rewriting packs.
- **Claude Agent SDK as the base:** excellent assembled harness (built-in tools,
  permissions), but it is Claude Code's harness; less suited to learning the loop and to
  model-agnostic routing.

## Consequences
- More code to own (engine), in exchange for portability and visibility.
- Backends must pass conformance tests to keep behavior identical.
