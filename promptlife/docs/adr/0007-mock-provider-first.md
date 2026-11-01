# ADR-0007 · A deterministic mock provider is first-class

- **Status:** Accepted
- **Date:** 2026-10-02

## Context
Tests, CI and learners without an API key must be able to run everything. Live model
calls cost money and are non-deterministic.

## Decision
The gateway ships a `mock` provider that emits real Messages-API-shaped responses
(tool calls, text, usage). `provider_mode: auto` uses it when no API key is set.

## Alternatives considered
- Recorded cassettes of live calls: realistic but brittle as prompts change.
- Mocking at the HTTP layer: couples tests to SDK internals.

## Consequences
- Fast, free, deterministic tests. Mock quality limits what CI proves; nightly live
  evals cover real model behavior.
