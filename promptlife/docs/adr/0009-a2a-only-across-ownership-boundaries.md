# ADR-0009 · A2A only across ownership boundaries

- **Status:** Accepted
- **Date:** 2026-10-02

## Context
A2A lets independent agents delegate tasks to each other. Inside one deployment we
already have in-process multi-agent patterns.

## Decision
Use in-process calls between our own packs and agents. Use A2A only when crossing an
ownership or trust boundary: as a client (remote agents become a tool source) and as a
server (opted-in packs publish Agent Cards).

## Alternatives considered
- A2A between all packs: uniform, but adds network hops, serialization and split traces
  for no benefit inside one deployment.
- No A2A: blocks interoperability with other teams' agents.

## Consequences
- Remote agents are treated as side-effecting and untrusted by default.
- A2A task states map onto existing interrupts and checkpoints (doc 14).
