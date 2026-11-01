# ADR-0004 · Policy is enforced at the tool boundary

- **Status:** Accepted
- **Date:** 2026-10-02

## Context
Prompts can be ignored or attacked. Rules like "no refunds over $500" must hold
regardless of model behavior and regardless of backend.

## Decision
Every tool call from every backend goes through the engine's `ToolExecutor`, which
applies schema validation, deny rules, approvals, retry classes, timeouts, redaction,
offloading and telemetry. Unknown tools are denied by default.

## Alternatives considered
- Backend-native hooks only (e.g. Strands hooks): ties safety to one backend.
- Prompt-only rules: not enforceable.

## Consequences
- One place to audit safety. Backends are not trusted for enforcement.
- Slight overhead per call; negligible next to model latency.
