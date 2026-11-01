# ADR-0006 · State lives outside the process

- **Status:** Accepted
- **Date:** 2026-10-02

## Context
Replicas must be interchangeable for scaling, deploys and crash recovery; runs can pause
for human approval for hours.

## Decision
Sessions, checkpoints, approvals, memory and RAG indexes live in a store behind
interfaces: SQLite for single-container local use, Postgres (+ pgvector) in Kubernetes.
Large blobs go to object storage.

## Alternatives considered
- In-memory sessions with sticky load balancing: breaks on deploys and crashes.
- Redis for sessions: fast, but we need durable, queryable approvals and memory anyway.

## Consequences
- Any replica can resume any run. Store latency is on the request path; keep writes to
  one per tool round.
