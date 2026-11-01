# ADR-0003 · Two backends: raw loop and Strands

- **Status:** Accepted
- **Date:** 2026-10-02

## Context
The project's purpose includes learning. A hand-written loop shows the mechanics; an
SDK shows what abstractions buy you (multi-agent, conversation managers, tracing).

## Decision
Ship a `raw` backend (our loop over the anthropic SDK via the gateway) and a `strands`
backend. Packs choose with `backend:`; both pass the same conformance tests.

## Alternatives considered
- Raw only: least dependencies, but reimplements multi-agent patterns.
- Strands only: less to learn about the loop itself.

## Consequences
- Two code paths to keep consistent; mitigated by routing model calls through the
  gateway and tool calls through the ToolExecutor in both.
