# Guide · Extending the engine

Packs cover use cases. Engine extensions cover new *capabilities*. Each extension point
is a small interface plus a registry entry.

| Extension | Interface | Registered in |
|---|---|---|
| Model provider | `gateway.Provider` | `engine.yaml: providers` |
| Agent backend | `backends.Backend` | `engine.yaml: backends` |
| Router stage | `routing.RouterStage` | `engine.yaml: routing.stages` |
| Guardrail | `safety.Guardrail` | `engine.yaml: guardrails` |
| Tool source | `tools.ToolSource` | `engine.yaml: tool_sources` |
| Context injector | `context.Injector` | `engine.yaml: injectors` |
| Embedder | `rag.Embedder` | `engine.yaml: rag.embedder` |
| Session store | `sessions.SessionStore` | `engine.yaml: storage` |
| Eval scorer | `evals.Scorer` | `engine.yaml: evals.scorers` |

Registration uses import paths, so extensions can live outside the repo:

```yaml
providers:
  bedrock: mycompany.promptlife_ext.bedrock:BedrockProvider
```

## New model provider

```python
class MyProvider:
    name = "myvendor"

    def available(self) -> bool:
        return bool(os.environ.get("MYVENDOR_API_KEY"))

    def complete(self, req: ModelRequest, on_text=None) -> ModelResponse:
        # 1. translate Messages-API content blocks to the vendor format
        # 2. call the vendor
        # 3. translate back: text / tool_use blocks, stop_reason, usage
        # 4. raise ProviderError(retryable=...) for transient failures
        ...

    def count_tokens(self, req: ModelRequest) -> int: ...
```

Must pass the provider conformance tests: `pytest tests/conformance/providers -k myvendor`
(text reply, single tool call, parallel tool calls, error mapping, usage reporting).

## New backend

Implement `run` and `resume` (see [05](../design/05-agent-runtime.md)). Rules:

- Call the model only through the gateway.
- Call tools only through the `ToolExecutor` you are given.
- Emit the standard events (`tool_call`, `tool_result`, `text`, `interrupt`, `usage`).
- Pass the backend conformance tests, which run the example packs against the mock provider.

## New router stage

```python
class RouterStage(Protocol):
    name: str
    def route(self, question: str, cards: list[PackCard], session: Session) -> Route | None: ...
```

Return `None` to pass to the next stage. Add it to `routing.stages` in order.

## New guardrail

See the interface in [08](../design/08-safety.md). Add unit tests with both positive and
negative examples, and an eval case that proves it fires end to end.

## Conventions for engine code

- No domain words in `promptlife/` (CI greps for pack names).
- Every public interface has a mock or fake for tests.
- Every new behavior emits a span or attribute.
- Configuration lives in `engine.yaml` with defaults in code; secrets only from env.
