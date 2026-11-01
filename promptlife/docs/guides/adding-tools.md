# Guide · Adding tools

Five ways to give a pack tools. All of them end up as the same `Tool` object and go
through the same `ToolExecutor` (policy, approvals, retries, redaction, telemetry).

## Python functions

```python
# packs/it_helpdesk/tools.py
from promptlife.tools import tool, ToolContext

@tool(side_effect=False)
def service_health(service: str, ctx: ToolContext) -> dict:
    """Return health status, error rate and p95 latency for a service over the last 15 minutes."""
    return ctx.http.get(f"{ctx.secret('STATUS_API')}/services/{service}").json()

@tool(side_effect=True)
def restart_deployment(namespace: str, name: str, ctx: ToolContext) -> dict:
    """Rolling-restart a Kubernetes deployment. Requires approval."""
    ...
```

- The JSON schema comes from type hints; the description from the docstring. Write the
  docstring for the model: what it does, when to use it, what it returns.
- `ctx` is injected by the engine and is not part of the schema. It carries `user_id`,
  secrets, an HTTP client with tracing, and the blob store.
- Return JSON-serializable data. Large results are offloaded automatically.
- Raise `ToolError("message", retryable=True)` for transient failures.

## MCP servers

```yaml
tools:
  - mcp:
      name: github
      command: npx
      args: [-y, "@modelcontextprotocol/server-github"]
      env: { GITHUB_TOKEN: ${GITHUB_TOKEN} }
      include: [search_issues, get_issue]   # optional allowlist
  - mcp:
      name: grafana
      url: https://mcp.internal.example.com/mcp
      auth: { bearer: GRAFANA_MCP_TOKEN }
```

- Tool names are prefixed with the server name: `github__search_issues`.
- Use `include` to expose only what the pack needs; fewer tools means better tool choice.
- Mark MCP tools that change things in `policies.side_effect_tools`.
- Only allowlisted commands/URLs are permitted in production (`engine.yaml: mcp.allow`).
- Tools written in Go, TypeScript, Java and other languages plug in this way; see
  [Polyglot packs](../design/13-polyglot-packs.md).

## OpenAPI

```yaml
tools:
  - openapi:
      spec: https://status.example.com/openapi.json
      operations: [getIncidents, getIncident]
      auth: { header: Authorization, prefix: "Bearer ", secret: STATUS_TOKEN }
      base_url: https://status.example.com
```

Each listed operation becomes a tool; parameters and request body become the input
schema; the operation summary becomes the description (override with `descriptions:`).

## Agents as tools

```yaml
tools:
  - agent: docs_qa        # another pack
```

Exposes `ask_docs_qa(question)`. The child run has its own policy, limits and spans,
linked to the parent trace. Use this when the parent should keep control; use handoff
when control should pass (see [Adding agents](adding-agents.md)).

## Remote agents over A2A

```yaml
tools:
  - a2a:
      name: security_agent
      card_url: https://security-agent.internal.example.com/.well-known/agent-card.json
      skills: [triage_alert]
```

For agents owned by another team or vendor. See
[Interop: MCP and A2A](../design/14-interop-mcp-a2a.md).

## Testing tools

```python
from promptlife.testing import run_tool, fake_context

def test_service_health_ok():
    out = run_tool("it_helpdesk", "service_health", {"service": "checkout"}, ctx=fake_context())
    assert out["status"] in {"healthy", "degraded", "down"}
```

`run_tool` goes through the real `ToolExecutor`, so policy is tested too.
