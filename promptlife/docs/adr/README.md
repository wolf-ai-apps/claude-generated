# Architecture decision records

Each ADR records one decision: context, the decision, alternatives, and consequences.
New ADRs copy [template.md](template.md) and take the next number. ADRs are never
edited after acceptance; a later ADR supersedes an earlier one.

| ADR | Title | Status |
|---|---|---|
| [0001](0001-engine-owns-the-platform.md) | The engine owns the platform; agent SDKs are backends | Accepted |
| [0002](0002-packs-as-declarative-config.md) | Use cases are declarative packs | Accepted |
| [0003](0003-two-backends-raw-and-strands.md) | Two backends: raw loop and Strands | Accepted |
| [0004](0004-policy-at-the-tool-boundary.md) | Policy is enforced at the tool boundary | Accepted |
| [0005](0005-model-aliases-and-gateway.md) | Packs reference model aliases through a gateway | Accepted |
| [0006](0006-state-outside-the-process.md) | State lives outside the process | Accepted |
| [0007](0007-mock-provider-first.md) | A deterministic mock provider is first-class | Accepted |
| [0008](0008-mcp-as-language-neutral-tool-boundary.md) | MCP is the language-neutral tool boundary | Accepted |
| [0009](0009-a2a-only-across-ownership-boundaries.md) | A2A only across ownership boundaries | Accepted |
