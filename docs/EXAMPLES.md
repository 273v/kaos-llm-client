# kaos-llm-client Advanced Examples

Advanced patterns for production use: failover, cost tracking, concurrency control, output validation, multi-turn tool use, and the OpenAI Responses API.

## FallbackClient -- Provider Failover

`FallbackClient` tries providers in order. If the primary fails with a retryable error (transport, provider, or retry-exhausted), it falls through to the next client.

```python
from kaos_llm_client import create_client
from kaos_llm_client.providers.fallback import FallbackClient

openai = create_client("openai:gpt-5.4-nano")
anthropic = create_client("anthropic:claude-haiku-4-5")

client = FallbackClient([openai, anthropic])
response = client.chat(
    messages=[{"role": "user", "content": "What is the capital of France?"}]
)
print(f"Answered by: {response.provider}:{response.model}")
print(response.text)
```

### Custom fallback conditions

By default, `FallbackClient` falls back on `KaosLLMProviderError`, `KaosLLMTransportError`, and `KaosLLMRetryExhaustedError`. You can customize this:

```python
from kaos_llm_client.errors import KaosLLMProviderError, KaosLLMTransportError
from kaos_llm_client.providers.fallback import FallbackClient

# Only fall back on transport errors (not 4xx provider errors)
client = FallbackClient(
    [openai, anthropic],
    fallback_on=(KaosLLMTransportError,),
)
```

### Three-provider chain

```python
from kaos_llm_client import create_client
from kaos_llm_client.providers.fallback import FallbackClient

client = FallbackClient([
    create_client("openai:gpt-5.4-nano"),
    create_client("anthropic:claude-haiku-4-5"),
    create_client("google:gemini-2.5-flash"),
])
response = client.chat(
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.text)
```

## InstrumentedClient -- Cost Tracking

`InstrumentedClient` wraps any client and accumulates request count, token usage, and estimated cost.

```python
from kaos_llm_client import create_client
from kaos_llm_client.providers.instrumented import InstrumentedClient

inner = create_client("openai:gpt-5.4-nano")
client = InstrumentedClient(
    inner,
    cost_per_input_token=0.000000150,   # $0.15 per 1M input tokens
    cost_per_output_token=0.000000600,  # $0.60 per 1M output tokens
)

# Make several requests
for question in ["What is 2+2?", "What is the speed of light?", "Define entropy."]:
    response = client.chat(messages=[{"role": "user", "content": question}])
    print(f"Q: {question} -> {response.text[:50]}...")

# Check accumulated metrics
print(f"\nTotal requests: {client.total_requests}")
print(f"Total input tokens: {client.total_input_tokens}")
print(f"Total output tokens: {client.total_output_tokens}")
print(f"Estimated cost: ${client.total_cost:.6f}")

# Reset counters for a new batch
client.reset_counters()
```

## ConcurrencyLimitedClient -- Rate Limiting

`ConcurrencyLimitedClient` wraps a client with an asyncio semaphore to cap parallel requests. Useful for respecting provider rate limits in async code.

```python
import asyncio
from kaos_llm_client import create_client
from kaos_llm_client.providers.concurrency import ConcurrencyLimitedClient

inner = create_client("openai:gpt-5.4-nano")
client = ConcurrencyLimitedClient(inner, limit=5)  # max 5 concurrent requests

async def ask(question: str) -> str:
    response = await client.chat_async(
        messages=[{"role": "user", "content": question}]
    )
    return response.text

async def main():
    questions = [
        "What is photosynthesis?",
        "Explain gravity.",
        "What causes rain?",
        "How do computers work?",
        "What is DNA?",
        "Why is the sky blue?",
        "How do planes fly?",
        "What is electricity?",
        "Define evolution.",
        "What is a black hole?",
    ]
    # All 10 fire concurrently, but only 5 run at a time
    results = await asyncio.gather(*[ask(q) for q in questions])
    for q, a in zip(questions, results):
        print(f"Q: {q}\nA: {a[:80]}...\n")

asyncio.run(main())
```

## Composing Wrappers

Wrappers compose naturally. Outer wrappers see the interface of the inner wrapper.

```python
from kaos_llm_client import create_client
from kaos_llm_client.providers.concurrency import ConcurrencyLimitedClient
from kaos_llm_client.providers.fallback import FallbackClient
from kaos_llm_client.providers.instrumented import InstrumentedClient

# Build inner clients with rate limiting
openai = ConcurrencyLimitedClient(
    create_client("openai:gpt-5.4-nano"),
    limit=5,
)
anthropic = create_client("anthropic:claude-haiku-4-5")

# Failover from rate-limited OpenAI to Anthropic
fallback = FallbackClient([openai, anthropic])

# Track cost across the entire chain
client = InstrumentedClient(
    fallback,
    cost_per_input_token=0.000000150,
    cost_per_output_token=0.000000600,
)

response = client.chat(
    messages=[{"role": "user", "content": "Explain recursion."}]
)
print(response.text)
print(f"Cost so far: ${client.total_cost:.6f}")
```

## Output Validation with Retry

The `pydantic()` method supports an `output_validator` callback and `max_validation_retries`. When validation fails, the error is appended to the conversation and the model is asked to self-correct.

```python
from pydantic import BaseModel
from kaos_llm_client import create_client


class Recipe(BaseModel):
    name: str
    ingredients: list[str]
    prep_time_minutes: int
    steps: list[str]


def validate_recipe(recipe: Recipe) -> Recipe:
    """Custom validator: ensure the recipe is reasonable."""
    if recipe.prep_time_minutes <= 0:
        raise ValueError("prep_time_minutes must be positive")
    if len(recipe.ingredients) < 2:
        raise ValueError("A recipe needs at least 2 ingredients")
    if len(recipe.steps) < 2:
        raise ValueError("A recipe needs at least 2 steps")
    return recipe


client = create_client("openai:gpt-5.4-nano")
recipe = client.pydantic(
    messages=[{"role": "user", "content": "Give me a recipe for chocolate chip cookies."}],
    output_type=Recipe,
    output_validator=validate_recipe,
    max_validation_retries=2,
)
print(f"{recipe.name} ({recipe.prep_time_minutes} min)")
print(f"Ingredients: {', '.join(recipe.ingredients)}")
for i, step in enumerate(recipe.steps, 1):
    print(f"  {i}. {step}")
```

### Accessing the raw response after pydantic()

The original `ProviderResponse` is attached to the result object for introspection:

```python
recipe = client.pydantic(
    messages=[{"role": "user", "content": "Give me a recipe for pasta."}],
    output_type=Recipe,
)
raw_response = recipe._response  # type: ignore[attr-defined]
print(f"Tokens used: {raw_response.usage.total_tokens}")
```

## Multi-Turn Tool Use (Full Example)

A complete working example of the tool-calling loop: define tools, call the model, execute tools locally, send results back, and get the final answer.

### OpenAI

```python
import json
from kaos_llm_client import (
    create_client,
    AssistantMessage,
    ToolResultMessage,
    ToolDefinition,
)

# Define tools
calculator = ToolDefinition(
    name="calculator",
    description="Evaluate a mathematical expression and return the result.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A mathematical expression, e.g. '2 + 3 * 4'",
            },
        },
        "required": ["expression"],
    },
)

# Simulated tool execution.
#
# IMPORTANT: never call ``eval`` on model-supplied input — that lets the
# model execute arbitrary Python in your process. The snippet below uses
# ``ast.parse`` + node-type allowlist to evaluate arithmetic safely.
import ast
import operator

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_arith(expression: str) -> float:
    """Evaluate a numeric arithmetic expression safely (no eval)."""
    tree = ast.parse(expression, mode="eval")

    def _walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](_walk(node.left), _walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](_walk(node.operand))
        raise ValueError(f"unsupported expression: {ast.dump(node)}")

    return _walk(tree)


def execute_tool(name: str, args: dict) -> str:
    if name == "calculator":
        try:
            result = safe_arith(args["expression"])
            return json.dumps({"result": result})
        except (SyntaxError, ValueError, ZeroDivisionError) as e:
            return json.dumps({"error": str(e)})
    return json.dumps({"error": f"Unknown tool: {name}"})

# Conversation loop
client = create_client("openai:gpt-5.4-nano")
messages = [
    {"role": "system", "content": "You are a helpful math assistant. Use the calculator tool."},
    {"role": "user", "content": "What is (17 * 23) + (45 / 9)?"},
]

while True:
    response = client.chat(messages, tools=[calculator])

    if not response.tool_calls:
        # No tool calls -- model is done
        print(f"Answer: {response.text}")
        break

    # Process all tool calls
    messages.append(AssistantMessage.from_response(response))
    for tc in response.tool_calls:
        print(f"Calling {tc.name}({tc.arguments})")
        result = execute_tool(tc.name, tc.arguments)
        messages.append(ToolResultMessage(tc.id, result, name=tc.name))
```

### Anthropic

The same pattern works for Anthropic. `AssistantMessage.from_response()` automatically handles the Anthropic content block format (including thinking block replay for extended thinking models).

```python
from kaos_llm_client import (
    create_client,
    AssistantMessage,
    ToolResultMessage,
    ToolDefinition,
)

lookup_city = ToolDefinition(
    name="lookup_population",
    description="Look up the population of a city.",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "country": {"type": "string"},
        },
        "required": ["city", "country"],
    },
)

# Simulated database
POPULATIONS = {
    ("Tokyo", "Japan"): 13_960_000,
    ("Delhi", "India"): 32_941_000,
    ("Shanghai", "China"): 28_517_000,
}

client = create_client("anthropic:claude-haiku-4-5")
messages = [
    {"role": "user", "content": "What is the population of Tokyo, Japan?"},
]

response = client.chat(messages, tools=[lookup_city], max_tokens=1024)

if response.tool_calls:
    tc = response.tool_calls[0]
    city = tc.arguments.get("city", "")
    country = tc.arguments.get("country", "")
    pop = POPULATIONS.get((city, country), "unknown")

    messages.append(AssistantMessage.from_response(response))
    messages.append(ToolResultMessage(tc.id, {"population": pop}, name=tc.name))

    final = client.chat(messages, tools=[lookup_city], max_tokens=1024)
    print(final.text)
```

### Google

```python
from kaos_llm_client import (
    create_client,
    AssistantMessage,
    ToolResultMessage,
    ToolDefinition,
)

convert_unit = ToolDefinition(
    name="convert_temperature",
    description="Convert a temperature between Celsius and Fahrenheit.",
    parameters={
        "type": "object",
        "properties": {
            "value": {"type": "number"},
            "from_unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            "to_unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["value", "from_unit", "to_unit"],
    },
)

def convert(value: float, from_unit: str, to_unit: str) -> float:
    if from_unit == "celsius" and to_unit == "fahrenheit":
        return value * 9 / 5 + 32
    elif from_unit == "fahrenheit" and to_unit == "celsius":
        return (value - 32) * 5 / 9
    return value

client = create_client("google:gemini-2.5-flash")
messages = [
    {"role": "user", "content": "Convert 100 degrees Celsius to Fahrenheit."},
]

response = client.chat(messages, tools=[convert_unit])

if response.tool_calls:
    tc = response.tool_calls[0]
    result = convert(
        tc.arguments["value"],
        tc.arguments["from_unit"],
        tc.arguments["to_unit"],
    )

    messages.append(AssistantMessage.from_response(response))
    messages.append(ToolResultMessage(tc.id, {"result": result}, name=tc.name))

    final = client.chat(messages, tools=[convert_unit])
    print(final.text)
    # 100 degrees Celsius is 212 degrees Fahrenheit.
```

## Typed Messages

Typed message classes (`SystemMessage`, `UserMessage`, `AssistantMessage`, `ToolResultMessage`, `CachePoint`) are optional but catch errors at construction time and provide better IDE support.

```python
from kaos_llm_client import (
    create_client,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolResultMessage,
    CachePoint,
    image_from_path,
)

client = create_client("anthropic:claude-haiku-4-5")

# Typed messages and raw dicts can be mixed freely
messages = [
    SystemMessage("You are a concise assistant."),
    CachePoint(),  # Anthropic caches the system prompt
    {"role": "user", "content": "What is 2+2?"},  # raw dict still works
]

response = client.chat(messages, max_tokens=256)
print(response.text)

# Multimodal user message with typed helper
messages = [
    SystemMessage("Describe images accurately."),
    UserMessage([
        "What do you see in this image?",
        image_from_path("photo.jpg"),
    ]),
]
response = client.chat(messages, max_tokens=1024)
print(response.text)

# Multi-turn with typed assistant message
messages = [
    UserMessage("Tell me a joke."),
]
response = client.chat(messages, max_tokens=256)
messages.append(AssistantMessage.from_response(response))
messages.append(UserMessage("Now explain why it's funny."))
response = client.chat(messages, max_tokens=512)
print(response.text)
```

## OpenAI Responses API

The Responses API (`/v1/responses`) is a separate wire format from Chat Completions. It supports builtin tools (web search, code interpreter), reasoning summaries, and stateful conversations via `previous_response_id`.

### Basic usage

```python
from kaos_llm_client import create_client

client = create_client("openai-responses:gpt-5.4-nano")
response = client.chat(
    messages=[{"role": "user", "content": "What is the weather in San Francisco?"}]
)
print(response.text)
print(f"Response ID: {response.response_id}")
```

### Builtin tools

```python
from kaos_llm_client import create_client

client = create_client("openai-responses:gpt-5.4-nano")
response = client.chat(
    messages=[{"role": "user", "content": "Search the web for recent news about AI regulation."}],
    builtin_tools=[{"type": "web_search"}],
)
print(response.text)
```

### Reasoning (thinking) with Responses API

```python
from kaos_llm_client import create_client

client = create_client("openai-responses:o4-mini")
response = client.chat(
    messages=[{"role": "user", "content": "How many r's are in 'strawberry'?"}],
    reasoning={"effort": "high", "summary": "auto"},
)
if response.thinking:
    print(f"Reasoning: {response.thinking[:200]}...")
print(f"Answer: {response.text}")
```

### Stateful conversations with previous_response_id

```python
from kaos_llm_client import create_client

client = create_client("openai-responses:gpt-5.4-nano")

# First turn
response1 = client.chat(
    messages=[{"role": "user", "content": "My name is Alice."}]
)
print(response1.text)

# Second turn -- references the first response by ID
response2 = client.chat(
    messages=[{"role": "user", "content": "What is my name?"}],
    previous_response_id=response1.response_id,
)
print(response2.text)
# Your name is Alice.
```

## FileCache for Development

Enable response caching to avoid re-calling the API during development and test loops. Cached responses are stored as gzipped JSON files at `~/.cache/kaos/llm/` (or a custom path).

### Via settings

```python
from kaos_llm_client import create_client
from kaos_llm_client.settings import KaosLLMSettings

settings = KaosLLMSettings(
    cache_enabled=True,
    cache_path="/tmp/kaos-llm-cache",
)
client = create_client("openai:gpt-5.4-nano", settings=settings)

# First call hits the API
response1 = client.chat(messages=[{"role": "user", "content": "What is 2+2?"}])
print(response1.text)

# Second call with identical messages returns cached response instantly
response2 = client.chat(messages=[{"role": "user", "content": "What is 2+2?"}])
print(response2.text)  # same result, no API call
```

### Via environment variables

```bash
export KAOS_LLM_CACHE_ENABLED=true
export KAOS_LLM_CACHE_PATH=/tmp/kaos-llm-cache
```

```python
from kaos_llm_client import create_client

# Settings auto-load from env vars
client = create_client("openai:gpt-5.4-nano")
# All requests are now cached to /tmp/kaos-llm-cache
```

### Injecting a cache directly

```python
from kaos_llm_client import create_client, FileCache

cache = FileCache("/tmp/my-project-cache")
client = create_client("openai:gpt-5.4-nano", cache=cache)

response = client.chat(messages=[{"role": "user", "content": "Hello!"}])
print(response.text)

# Clear the cache when needed
cache.clear()
```

### Per-request cache control

```python
from kaos_llm_client import create_client, RequestOptions
from kaos_llm_client.types import CachePolicy
from kaos_llm_client.settings import KaosLLMSettings

settings = KaosLLMSettings(cache_enabled=True)
client = create_client("openai:gpt-5.4-nano", settings=settings)

# Skip cache for this specific request
response = client.request(
    messages=[{"role": "user", "content": "What time is it?"}],
    options=RequestOptions(cache_policy=CachePolicy.SKIP),
)
```

## Request Lifecycle Hooks

`RequestHooks` lets you observe requests and responses without modifying behavior.

```python
from kaos_llm_client import create_client, RequestHooks

def on_request(request):
    print(f"-> {request.provider}:{request.model} [{request.endpoint}]")

def on_response(request, response):
    print(f"<- {response.usage.total_tokens} tokens, stop={response.stop_reason}")

def on_error(request, error):
    print(f"!! Error: {error}")

client = create_client(
    "openai:gpt-5.4-nano",
    hooks=RequestHooks(
        on_request=on_request,
        on_response=on_response,
        on_error=on_error,
    ),
)

response = client.chat(messages=[{"role": "user", "content": "Hello!"}])
# -> openai:gpt-5.4-nano [/v1/chat/completions]
# <- 28 tokens, stop=stop
```

## OpenAI-Compatible Endpoints

Use the `openai-compatible` provider with any OpenAI-compatible API (vLLM, Ollama, LiteLLM, etc.).

```python
from kaos_llm_client import create_client

# vLLM local server
client = create_client(
    "openai-compatible:meta-llama/Llama-3.3-70B-Instruct",
    base_url="http://localhost:8000",
    api_key="not-needed",
)
response = client.chat(
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.text)

# Ollama
client = create_client(
    "openai-compatible:llama3.3",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)
response = client.chat(
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.text)
```

## Structured Output Modes

The `json()` and `pydantic()` methods auto-select the structured output strategy from the model profile. You can override it explicitly.

```python
from kaos_llm_client import create_client
from kaos_llm_client.profiles import StructuredOutputMode
from pydantic import BaseModel


class City(BaseModel):
    name: str
    country: str
    population: int


client = create_client("openai:gpt-5.4-nano")

# Native: uses OpenAI's response_format with JSON schema (default for OpenAI)
city = client.pydantic(
    messages=[{"role": "user", "content": "Facts about Paris."}],
    output_type=City,
    output_mode=StructuredOutputMode.NATIVE,
)
print(city)

# Tool: defines a return_output tool with the schema
city = client.pydantic(
    messages=[{"role": "user", "content": "Facts about Paris."}],
    output_type=City,
    output_mode=StructuredOutputMode.TOOL,
)
print(city)

# Prompted: adds schema instructions to the prompt (works with any model)
city = client.pydantic(
    messages=[{"role": "user", "content": "Facts about Paris."}],
    output_type=City,
    output_mode=StructuredOutputMode.PROMPTED,
)
print(city)
```
