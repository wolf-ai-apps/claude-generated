# Guide · Adding a pack

A new use case is a new folder under `packs/`. You should not need to change engine code.
This guide builds a small `docs_qa` pack (answers questions from a folder of Markdown).

## 1. Scaffold

```bash
promptlife packs new docs_qa
```

Creates:

```
packs/docs_qa/
  pack.yaml
  prompt.md
  tools.py
  knowledge/.gitkeep
  evals.jsonl
```

## 2. Write the card first

The card decides whether the router ever sends traffic to you. Be specific.

```yaml
apiVersion: promptlife/v1
name: docs_qa
version: 0.1.0
description: Answers questions about the internal engineering handbook (on-call, deploys, code review).
examples:
  - how do I request on-call swap
  - what is our deploy freeze policy
  - who approves database migrations
```

Tips:

- Write `description` from the user's point of view, not the system's.
- 5–15 `examples` that cover different phrasings. Add the ones that get misrouted later.
- Check overlap with other packs: `promptlife packs overlap docs_qa` prints the nearest
  examples from other packs.

## 3. Pick models, limits and backend

```yaml
model: { alias: default, escalate_to: deep }
backend: raw
limits: { max_turns: 6, max_tool_calls: 8, max_cost_usd: 0.10 }
```

Start cheap and tight; loosen only when evals show the need.

## 4. Prompt

`prompt.md`:

```markdown
You answer questions about {{ company }}'s engineering handbook.
Use the knowledge.search tool before answering. Cite the source file and heading.
If the handbook does not cover it, say so and suggest who to ask.
```

Keep it stable (no dates or per-user data); injected context is added separately and
keeps the prompt cacheable.

## 5. Knowledge, tools, policies

```yaml
knowledge: { path: knowledge/, search: hybrid, top_k: 4 }
tools: []                  # retrieval tool is built in when knowledge is set
policies: {}               # no side effects in this pack
guardrails: { output: [grounding] }
```

For tools, see [Adding tools](adding-tools.md). Any tool that changes something must be
listed in `side_effect_tools`, and should usually be in `require_approval`.

## 6. Evals

Add at least 10 cases before shipping, including:

- 6+ normal questions with expected facts (`contains`);
- 2 questions the handbook does not answer (expect "not covered");
- 2 questions that belong to another pack (expect routing elsewhere).

```json
{"id": "freeze-001", "input": "Can I deploy on Friday after 3pm?", "expect": {"pack": "docs_qa", "tools_called": ["knowledge.search"], "contains": ["freeze"]}}
```

## 7. Validate and run

```bash
promptlife packs validate docs_qa
promptlife ask --pack docs_qa "what is the deploy freeze policy?"
promptlife evals run --pack docs_qa            # mock provider
promptlife evals run --pack docs_qa --live     # real models
```

## Checklist

- [ ] Card has a clear description and varied examples
- [ ] No secrets in YAML; all `${VAR}` documented in the pack README
- [ ] Side-effect tools listed and gated
- [ ] Limits set
- [ ] ≥ 10 eval cases, including out-of-scope ones
- [ ] `packs validate` and evals pass in CI
