"""KaosLLMCostEstimateTool — extracted from tools.py per audit-01 KLC-03."""

from __future__ import annotations

from typing import Any

from kaos_core import KaosContext, KaosTool, ToolMetadata, ToolResult
from kaos_core.types.enums import ToolCapability, ToolCategory
from kaos_core.types.parameters import ParameterSchema

from kaos_llm_client.tools._common import (
    _LOCAL_ANNOTATIONS,
    _MODEL_PRICING,
    _MODULE,
    _VERSION,
    _lookup_pricing,
    _tool_log_extra,
    logger,
)
from kaos_llm_client.tools.embed import _estimate_tokens


class KaosLLMCostEstimateTool(KaosTool):
    """Estimate token count and cost for an LLM request."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-llm-cost-estimate",
            display_name="LLM Cost Estimate",
            description=(
                "Estimate the token count and cost for an LLM request. "
                "Uses approximate tokenization (~4 chars/token) and known model pricing. "
                "Does NOT make API calls. Useful for cost planning before making requests. "
                "Pricing data covers major models from OpenAI, Anthropic, Google, and xAI. "
                "For actual usage tracking, check the 'usage' field in tool responses."
            ),
            category=ToolCategory.INTEGRATION,
            capability=ToolCapability.ANALYZE,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_LOCAL_ANNOTATIONS,
            input_schema=[
                ParameterSchema(
                    name="model",
                    type="string",
                    description=(
                        "Model name to estimate cost for. "
                        "Use bare model name (e.g., 'gpt-5') or provider:model format "
                        "(e.g., 'openai:gpt-5'). Provider prefix is stripped for pricing lookup."
                    ),
                ),
                ParameterSchema(
                    name="input_text",
                    type="string",
                    description="The input text to estimate token count for.",
                ),
                ParameterSchema(
                    name="max_output_tokens",
                    type="integer",
                    description=(
                        "Expected maximum output tokens (default: 1000). "
                        "Used to estimate output cost."
                    ),
                    required=False,
                    constraints={"minimum": 1},
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        model = inputs.get("model")
        if not model:
            return ToolResult.create_error(
                "Missing required parameter 'model'. "
                "Provide a model name like 'gpt-5' or 'openai:gpt-5'."
            )

        input_text = inputs.get("input_text")
        if not input_text:
            return ToolResult.create_error(
                "Missing required parameter 'input_text'. "
                "Provide the text you plan to send to the model."
            )

        max_output_tokens = inputs.get("max_output_tokens", 1000)

        # Strip provider prefix for pricing lookup
        model_name = model.split(":", 1)[1] if ":" in model else model

        estimated_input_tokens = _estimate_tokens(input_text)
        pricing = _lookup_pricing(model_name)

        if pricing is None:
            known_models = ", ".join(sorted(_MODEL_PRICING.keys()))
            return ToolResult.create_error(
                f"Unknown model '{model_name}' -- no pricing data available. "
                f"Known models: {known_models}. "
                "For models not in the pricing table, check the provider's pricing page."
            )

        input_cost = (estimated_input_tokens / 1_000_000) * pricing["input"]
        output_cost = (max_output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost

        result_data = {
            "model": model_name,
            "estimated_input_tokens": estimated_input_tokens,
            "max_output_tokens": max_output_tokens,
            "estimated_cost_usd": round(total_cost, 6),
            "pricing_per_1m": {
                "input": pricing["input"],
                "output": pricing["output"],
            },
        }

        summary = (
            f"Estimated cost for {model_name}: ~{estimated_input_tokens} input tokens + "
            f"{max_output_tokens} max output tokens = ${total_cost:.6f} USD"
        )

        logger.info(
            "Cost estimate: model=%s, input_tokens=%d, cost=$%.6f",
            model_name,
            estimated_input_tokens,
            total_cost,
            extra=_tool_log_extra(
                context,
                tool_name="kaos-llm-cost-estimate",
                model=model_name,
                input_tokens=estimated_input_tokens,
                output_tokens=max_output_tokens,
                estimated_usd=round(total_cost, 6),
            ),
        )

        return ToolResult.create_success(output=result_data, summary=summary)
