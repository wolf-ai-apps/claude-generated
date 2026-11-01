# ADR-0005 · Packs reference model aliases through a gateway

- **Status:** Accepted
- **Date:** 2026-10-02

## Context
Model IDs change often; cost, fallback, rate limits and caching should be uniform.

## Decision
Packs and engine roles reference aliases (`fast`, `default`, `deep`). The gateway
resolves aliases to provider + model ID and owns retries, fallback, rate limits,
response caching and cost accounting.

## Alternatives considered
- Model IDs in packs: every model upgrade touches every pack.
- A third-party gateway (LiteLLM, Portkey, Cloudflare AI Gateway): viable in production;
  we build a small one to learn, behind an interface that can delegate to one later.

## Consequences
- One-line model upgrades; consistent cost data.
- The gateway must track provider-specific API rules (see doc 04).
