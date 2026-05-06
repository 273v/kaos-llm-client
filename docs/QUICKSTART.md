# kaos-llm-client Quickstart

Thin, provider-native LLM client for direct model calls within the KAOS ecosystem.
One interface across OpenAI, Anthropic, Google, xAI, Groq, Mistral, and any OpenAI-compatible endpoint.

## Installation

```bash
pip install kaos-llm-client
```

## Configuration

Set API keys as environment variables. The `KAOS_LLM_` prefix is canonical; standard names (`OPENAI_API_KEY`, etc.) are also accepted as fallbacks.

```bash
# Canonical KAOS prefix (recommended)
export KAOS_LLM_OPENAI_API_KEY=sk-...
export KAOS_LLM_ANTHROPIC_API_KEY=sk-ant-...
export KAOS_LLM_GOOGLE_API_KEY=AI...
export KAOS_LLM_XAI_API_KEY=xai-...

# Standard names (also work)
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=AI...
```

Or use a `.env` file in your project root:

```ini
# .env
KAOS_LLM_OPENAI_API_KEY=sk-...
KAOS_LLM_ANTHROPIC_API_KEY=sk-ant-...
KAOS_LLM_GOOGLE_API_KEY=AI...
KAOS_LLM_XAI_API_KEY=xai-...

# Optional: override base URLs for proxies or local models
# KAOS_LLM_OPENAI_BASE_URL=http://localhost:8080
# KAOS_LLM_DEFAULT_TIMEOUT=60.0
```

## Basic Chat

Use `create_client` with a `provider:model` string. The response has `.text`, `.usage`, `.tool_calls`, and other structured accessors.

### OpenAI

```python
from kaos_llm_client import create_client

client = create_client("openai:gpt-5.4-nano")
response = client.chat(
    messages=[{"role": "user", "content": "What is the capital of France?"}]
)
print(response.text)
# Paris is the capital of France.

print(f"Tokens used: {response.usage.input_tokens} in, {response.usage.output_tokens} out")
```

### Anthropic

```python
from kaos_llm_client import create_client

client = create_client("anthropic:claude-haiku-4-5")
response = client.chat(
    messages=[{"role": "user", "content": "Explain quantum computing in one sentence."}]
)
print(response.text)
```

### Google

```python
from kaos_llm_client import create_client

client = create_client("google:gemini-2.5-flash")
response = client.chat(
    messages=[{"role": "user", "content": "What are the planets in our solar system?"}]
)
print(response.text)
```

### Provider inference

If the model name is unambiguous, you can omit the provider prefix:

```python
client = create_client("gpt-5.4-nano")       # infers openai
client = create_client("claude-haiku-4-5")    # infers anthropic
client = create_client("gemini-2.5-flash")    # infers google
client = create_client("grok-3")              # infers xai
```

### Context manager

Clients hold an httpx connection pool. Use a context manager to ensure cleanup:

```python
from kaos_llm_client import create_client

with create_client("openai:gpt-5.4-nano") as client:
    response = client.chat(
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.text)
# connection pool closed automatically
```

## Streaming

Use `chat_stream_async` to receive tokens as they arrive. Each chunk has a `.type` (`text_delta`, `thinking_delta`, `tool_call_delta`, `usage`, `done`) and optional `.text`.

```python
import asyncio
from kaos_llm_client import create_client

async def main():
    client = create_client("openai:gpt-5.4-nano")
    async for chunk in client.chat_stream_async(
        messages=[{"role": "user", "content": "Write a haiku about programming."}]
    ):
        if chunk.type == "text_delta" and chunk.text:
            print(chunk.text, end="", flush=True)
        elif chunk.type == "done":
            print()  # newline at end
    client.close()

asyncio.run(main())
```

## Structured Output

### JSON with schema

The `json()` method requests JSON output. Pass a JSON Schema to constrain the structure. The output strategy (native schema enforcement, tool-based, or prompted) is selected automatically based on the model profile.

```python
from kaos_llm_client import create_client

client = create_client("openai:gpt-5.4-nano")
response = client.json(
    messages=[{"role": "user", "content": "List the 3 largest countries by area."}],
    schema={
        "type": "object",
        "properties": {
            "countries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "area_km2": {"type": "number"},
                    },
                    "required": ["name", "area_km2"],
                },
            }
        },
        "required": ["countries"],
    },
)
print(response.output_json)
# {'countries': [{'name': 'Russia', 'area_km2': 17098242}, ...]}
```

### Pydantic models

The `pydantic()` method validates the response against a Pydantic BaseModel and returns a typed instance.

```python
from pydantic import BaseModel
from kaos_llm_client import create_client


class Country(BaseModel):
    name: str
    capital: str
    population: int


client = create_client("openai:gpt-5.4-nano")
country = client.pydantic(
    messages=[{"role": "user", "content": "Give me facts about Japan."}],
    output_type=Country,
)
print(f"{country.name}: capital={country.capital}, pop={country.population}")
# Japan: capital=Tokyo, pop=125000000
```

## Tool Calling

Define tools with `ToolDefinition`, send them with `chat()`, process `response.tool_calls`, then continue the conversation with `AssistantMessage.from_response()` and `ToolResultMessage`.

```python
import json
from kaos_llm_client import (
    create_client,
    AssistantMessage,
    ToolResultMessage,
    ToolDefinition,
)

# 1. Define a tool
get_weather = ToolDefinition(
    name="get_weather",
    description="Get the current weather for a city.",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
        },
        "required": ["city"],
    },
)

# 2. Send the request with tools
client = create_client("openai:gpt-5.4-nano")
messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]
response = client.chat(messages, tools=[get_weather])

# 3. Process tool calls
if response.tool_calls:
    tc = response.tool_calls[0]
    print(f"Model wants to call: {tc.name}({tc.arguments})")

    # Execute the tool (your application logic)
    result = {"temperature": "22C", "condition": "sunny"}

    # 4. Build continuation messages
    messages.append(AssistantMessage.from_response(response))
    messages.append(ToolResultMessage(tc.id, result, name=tc.name))

    # 5. Get final answer
    final = client.chat(messages, tools=[get_weather])
    print(final.text)
    # The weather in Tokyo is 22C and sunny.
```

### Controlling tool choice

```python
from kaos_llm_client import ToolChoice

# Force the model to use a specific tool
response = client.chat(
    messages,
    tools=[get_weather],
    tool_choice=ToolChoice(type="specific", name="get_weather"),
)

# Let the model decide (default)
response = client.chat(messages, tools=[get_weather], tool_choice=ToolChoice(type="auto"))

# Require a tool call (any tool)
response = client.chat(messages, tools=[get_weather], tool_choice=ToolChoice(type="required"))

# Prevent tool use
response = client.chat(messages, tools=[get_weather], tool_choice=ToolChoice(type="none"))
```

## Multimodal

### Images

```python
from kaos_llm_client import create_client, UserMessage, image_from_path, image_url

client = create_client("openai:gpt-5.4-nano")

# From a local file
response = client.chat(messages=[
    UserMessage([
        "What is in this image?",
        image_from_path("photo.jpg"),
    ])
])
print(response.text)

# From a URL
response = client.chat(messages=[
    UserMessage([
        "Describe this image.",
        image_url("https://example.com/photo.jpg"),
    ])
])
```

### Documents (PDFs)

```python
from kaos_llm_client import create_client, UserMessage, document_from_path

client = create_client("anthropic:claude-haiku-4-5")
response = client.chat(messages=[
    UserMessage([
        "Summarize this document.",
        document_from_path("report.pdf"),
    ])
])
print(response.text)
```

### Audio

```python
from kaos_llm_client import create_client, UserMessage, audio_from_path

client = create_client("openai:gpt-5.4-nano")
response = client.chat(messages=[
    UserMessage([
        "Transcribe this audio.",
        audio_from_path("recording.wav"),
    ])
])
print(response.text)
```

## Thinking Mode

### Anthropic extended thinking

```python
from kaos_llm_client import create_client

client = create_client("anthropic:claude-sonnet-4-6")
response = client.chat(
    messages=[{"role": "user", "content": "Prove that the square root of 2 is irrational."}],
    thinking=True,
    max_tokens=16384,
)
print("Thinking:", response.thinking[:200], "...")
print("Answer:", response.text)
```

### OpenAI reasoning effort

```python
from kaos_llm_client import create_client

client = create_client("openai:o4-mini")
response = client.chat(
    messages=[{"role": "user", "content": "How many r's in 'strawberry'?"}],
    reasoning_effort="high",
)
print(response.text)
```

### Google thinking

```python
from kaos_llm_client import create_client

client = create_client("google:gemini-2.5-flash")
response = client.chat(
    messages=[{"role": "user", "content": "Solve: integrate x^2 * e^x dx"}],
    thinking=True,
)
if response.thinking:
    print("Thought process:", response.thinking[:200], "...")
print("Answer:", response.text)
```

## Embeddings

Embeddings are supported by providers that offer embedding endpoints (OpenAI, Mistral).

```python
from kaos_llm_client import create_client

client = create_client("openai:text-embedding-3-small")
result = client.embed("The quick brown fox jumps over the lazy dog.")
print(f"Dimensions: {len(result.embedding)}")
print(f"First 5 values: {result.embedding[:5]}")

# Batch embedding
result = client.embed([
    "First document about machine learning.",
    "Second document about cooking.",
    "Third document about astronomy.",
])
print(f"Got {len(result.embeddings)} vectors")
# result.embeddings[0], result.embeddings[1], result.embeddings[2]
```

### With custom dimensions

```python
result = client.embed(
    "The quick brown fox",
    model="text-embedding-3-large",
    dimensions=256,
)
print(f"Dimensions: {len(result.embedding)}")  # 256
```

## Typed Messages

You can use raw dicts (OpenAI format) or typed message classes interchangeably. Typed messages catch errors at construction time.

```python
from kaos_llm_client import (
    create_client,
    SystemMessage,
    UserMessage,
    CachePoint,
    image_from_path,
)

client = create_client("anthropic:claude-haiku-4-5")
response = client.chat(messages=[
    SystemMessage("You are a helpful assistant. Be concise."),
    CachePoint(),  # Anthropic caches everything before this point
    UserMessage("What is 2 + 2?"),
])
print(response.text)
```

## Error Handling

All errors inherit from `KaosLLMError` with structured `**details` for agent-friendly messages.

```python
from kaos_llm_client import create_client
from kaos_llm_client.errors import (
    KaosLLMAuthError,
    KaosLLMProviderError,
    KaosLLMTransportError,
    KaosLLMRetryExhaustedError,
)

client = create_client("openai:gpt-5.4-nano")
try:
    response = client.chat(messages=[{"role": "user", "content": "Hello"}])
except KaosLLMAuthError as e:
    print(f"Auth failed: {e}")
    # Set KAOS_LLM_OPENAI_API_KEY or pass api_key= to create_client
except KaosLLMProviderError as e:
    print(f"Provider error {e.status_code}: {e}")
except KaosLLMRetryExhaustedError as e:
    print(f"All {e.attempts} retries failed: {e}")
except KaosLLMTransportError as e:
    print(f"Network error: {e}")
```

## Transport Options

### Per-request overrides

```python
from kaos_llm_client import create_client, RequestOptions

client = create_client("openai:gpt-5.4-nano")
response = client.request(
    messages=[{"role": "user", "content": "Hello"}],
    options=RequestOptions(
        timeout=30.0,
        max_retries=5,
        extra_headers={"X-Custom-Header": "value"},
    ),
)
```

### Custom retry policy

```python
from kaos_llm_client import create_client

client = create_client(
    "openai:gpt-5.4-nano",
    timeout=60.0,
    max_retries=5,
)
```

## Async API

Every method has a sync and async variant. The sync version wraps the async one internally.

```python
import asyncio
from kaos_llm_client import create_client

async def main():
    client = create_client("openai:gpt-5.4-nano")
    response = await client.chat_async(
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.text)
    await client.aclose()

asyncio.run(main())
```

## Supported Providers

| Provider | Prefix | Models |
|----------|--------|--------|
| OpenAI | `openai:` | gpt-5.4-nano, gpt-5, gpt-4.1, o4-mini, o3 |
| Anthropic | `anthropic:` | claude-haiku-4-5, claude-sonnet-4-6, claude-opus-4-6 |
| Google | `google:` | gemini-2.5-flash, gemini-2.5-pro |
| xAI | `xai:` | grok-3, grok-4 |
| Groq | `groq:` | llama-3.3-70b |
| Mistral | `mistral:` | mistral-large-latest |
| OpenRouter | `openrouter:` | any model slug |
| OpenAI-compatible | `openai-compatible:` | any model (requires `base_url`) |
| OpenAI Responses | `openai-responses:` | gpt-5.4-nano (Responses API) |

See [EXAMPLES.md](EXAMPLES.md) for advanced patterns: fallback chains, cost tracking, concurrency limiting, output validation, and multi-turn tool use.
