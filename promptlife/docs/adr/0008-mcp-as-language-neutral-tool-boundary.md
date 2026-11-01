# ADR-0008 · MCP is the language-neutral tool boundary

- **Status:** Accepted
- **Date:** 2026-10-02

## Context
Pack tools may be written in Go, TypeScript, Java or other languages, and may bring
heavy or conflicting dependencies.

## Decision
Python tools run in-process as a convenience. Tools in any other language, or with
heavy dependencies, are MCP servers (stdio or HTTP), optionally in their own container.
Existing REST services integrate through OpenAPI.

## Alternatives considered
- gRPC plugin protocol of our own: more work, no ecosystem.
- Language-specific embedding (e.g. calling Node from Python): fragile.

## Consequences
- Any language with an MCP SDK can provide tools; servers are reusable outside Promptlife.
- The engine manages MCP process lifecycle, allowlists and pooling (doc 13).
