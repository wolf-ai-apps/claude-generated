# ADR-0002 · Use cases are declarative packs

- **Status:** Accepted
- **Date:** 2026-10-02

## Context
The platform must serve refunds, IT helpdesk, docs Q&A and future use cases without
engine changes.

## Decision
Each use case is a folder with `pack.yaml` (validated schema), a prompt template,
optional tools, knowledge, skills and evals. The engine discovers and loads packs.

## Alternatives considered
- **One Python class per use case:** flexible but makes safety rules code, not data,
  and hard to review.
- **Database-stored configs edited in a UI:** useful later; harder to version and review
  in Git now.

## Consequences
- New use cases are reviewable diffs; policies are visible in YAML.
- The schema must evolve carefully (`apiVersion`), with validation in CI.
