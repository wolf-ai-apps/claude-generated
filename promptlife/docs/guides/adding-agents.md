# Guide · Adding agents and multi-agent flows

Pick the simplest option that works. In order:

1. **One pack, one agent, better tools.** Most problems end here.
2. **Sub-agent as a tool** inside a pack: a specialist the main agent calls.
3. **Handoff (swarm)** inside a pack: control passes between specialists.
4. **Graph** inside a pack: fixed steps with model-chosen branches (e.g. draft → approve → execute).
5. **Cross-pack orchestration**: the engine's orchestrator splits a multi-part question
   across packs. No pack config needed; it uses the router.
6. **Remote agents over A2A**: only when the agent belongs to another team or vendor
   (see [Interop: MCP and A2A](../design/14-interop-mcp-a2a.md)).

## Sub-agent as a tool

```yaml
agents:
  log_analyst:
    description: Reads logs and summarizes errors for a service and time window.
    prompt: prompts/log_analyst.md
    model: { alias: fast }
    tools: [fetch_logs]
    limits: { max_turns: 4 }

tools:
  - python: tools.py
  - agent: log_analyst        # exposed as ask_log_analyst(question)
```

Good for reading-heavy work: the sub-agent reads 2 MB of logs with a cheap model and
returns a short summary, keeping the main agent's context small.

## Handoff

```yaml
orchestration:
  type: handoff
  entry: triage
  agents:
    triage:      { prompt: prompts/triage.md, model: { alias: fast }, handoffs: [incident, access] }
    incident:    { prompt: prompts/incident.md, tools: [service_health, fetch_logs, restart_deployment] }
    access:      { prompt: prompts/access.md, tools: [request_access] }
```

Each agent gets an engine-provided `handoff_to(agent, reason)` tool limited to its
`handoffs` list. Maximum handoffs per run is capped (`limits.max_handoffs`, default 3).
On the Strands backend this maps to a Strands swarm; on the raw backend the engine
switches the active agent spec.

## Graph

```yaml
orchestration:
  type: graph
  nodes:
    diagnose: { agent: incident }
    propose:  { agent: incident, prompt: prompts/propose_fix.md }
    approve:  { human: true }
    execute:  { agent: incident, tools: [restart_deployment] }
  edges:
    - diagnose -> propose
    - propose -> approve
    - approve -> execute: when decision == "approve"
```

## Custom hooks for an agent

```yaml
hooks:
  - promptlife.hooks.builtin:AuditLog
  - packs.it_helpdesk.hooks:BlockProdOutsideHours
```

```python
from promptlife.hooks import Hook, BeforeToolCall, Decision

class BlockProdOutsideHours(Hook):
    def before_tool_call(self, event: BeforeToolCall) -> Decision | None:
        if event.tool.name == "restart_deployment" and event.input["namespace"] == "prod" \
                and not event.clock.is_business_hours():
            return Decision.deny("Prod restarts outside business hours need the on-call lead.")
        return None
```

## Testing multi-agent flows

- Assert on the event stream: which agents ran, in what order, which handoffs happened.
- Add eval cases with `expect.agents: ["triage", "incident"]`.
- Check cost: multi-agent runs multiply tokens; set `limits.max_cost_usd`.
