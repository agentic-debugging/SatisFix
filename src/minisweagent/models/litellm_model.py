import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import litellm
from pydantic import BaseModel

from minisweagent.exceptions import FormatError
from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.actions_toolcall import (
    BASH_TOOL,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)
from minisweagent.models.utils.anthropic_utils import _reorder_anthropic_thinking_blocks
from minisweagent.models.utils.cache_control import set_cache_control
from minisweagent.models.utils.openai_multimodal import expand_multimodal_content
from minisweagent.models.utils.retry import retry

logger = logging.getLogger("litellm_model")


class LitellmModelConfig(BaseModel):
    model_name: str
    """Model name. Highly recommended to include the provider in the model name, e.g., `anthropic/claude-sonnet-4-5-20250929`."""
    model_kwargs: dict[str, Any] = {}
    """Additional arguments passed to the API."""
    litellm_model_registry: Path | str | None = os.getenv("LITELLM_MODEL_REGISTRY_PATH")
    """Model registry for cost tracking and model metadata. See the local model guide (https://mini-swe-agent.com/latest/models/local_models/) for more details."""
    set_cache_control: Literal["default_end"] | None = None
    """Set explicit cache control markers, for example for Anthropic models"""
    cost_tracking: Literal["default", "ignore_errors"] = os.getenv("MSWEA_COST_TRACKING", "default")
    """Cost tracking mode for this model. Can be "default" or "ignore_errors" (ignore errors/missing cost info)"""
    format_error_template: str = "{{ error }}"
    """Template used when the LM's output is not in the expected format."""
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    """Template used to render the observation after executing an action."""
    multimodal_regex: str = ""
    """Regex to extract multimodal content. Empty string disables multimodal processing."""


class LitellmModel:
    abort_exceptions: list[type[Exception]] = [
        litellm.exceptions.UnsupportedParamsError,
        litellm.exceptions.NotFoundError,
        litellm.exceptions.PermissionDeniedError,
        litellm.exceptions.ContextWindowExceededError,
        litellm.exceptions.AuthenticationError,
        KeyboardInterrupt,
    ]

    def __init__(self, *, config_class: Callable = LitellmModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        if self.config.litellm_model_registry and Path(self.config.litellm_model_registry).is_file():
            litellm.utils.register_model(json.loads(Path(self.config.litellm_model_registry).read_text()))

    def _api_kwargs(self, kwargs: dict) -> dict:
        """`model_kwargs` overlaid with per-call kwargs."""
        return self.config.model_kwargs | kwargs

    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            return litellm.completion(
                model=self.config.model_name,
                messages=messages,
                tools=kwargs.pop("tools", [BASH_TOOL]),
                **self._api_kwargs(kwargs),
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        prepared = [{k: v for k, v in msg.items() if k != "extra"} for msg in messages]
        prepared = _reorder_anthropic_thinking_blocks(prepared)
        return set_cache_control(prepared, mode=self.config.set_cache_control)

    def _call(self, messages: list[dict[str, str]], **kwargs):
        """Send one request (with retries) and return the raw response and its cost, already counted globally."""
        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                response = self._query(self._prepare_messages_for_api(messages), **kwargs)
        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        return response, cost_output

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        response, cost_output = self._call(messages, **kwargs)
        # Note: all model.query() implementations must persist the response and cost on FormatError.
        try:
            actions = self._parse_actions(response)
        except FormatError as e:
            self._persist_on_format_error(e, response, cost_output)
            raise
        message = response.choices[0].message.model_dump()
        message["extra"] = {
            "actions": actions,
            "response": response.model_dump(),
            **cost_output,
            "timestamp": time.time(),
        }
        return message

    def _persist_on_format_error(self, e: FormatError, response, cost_output: dict) -> None:
        e.messages[0]["extra"].update(cost_output)
        try:
            e.messages[0]["extra"]["response"] = response.model_dump(mode="json")
        except Exception:
            # model_dump failed (e.g. unserializable object); fall back to repr
            # so the spec contract ("response MUST be persisted") holds unconditionally.
            e.messages[0]["extra"]["response"] = repr(response)

    def query_report(
        self, messages: list[dict], name: str, description: str, parameters: dict, *, require: bool = True, **kwargs
    ) -> dict:
        """Query with the extra tool `name` beside `bash`; the same tool list in every request, so caching holds.

        With `require`, `tool_choice` forces that tool and nothing is parsed as an action. Without it, the
        `bash` calls are parsed as in `query`; a reply with only calls of tool `name` is a FormatError whose
        messages start with the reply, so that its calls can be answered.
        The returned message carries the calls of tool `name` in `extra["report_calls"]` as
        `{"id", "arguments"}` (the raw arguments string), empty if the model did not call it, and
        `extra["truncated"]`, whether the reply was cut off at the output limit.
        """
        tool = {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}
        if require:
            kwargs["tool_choice"] = {"type": "function", "function": {"name": name}}
        response, cost_output = self._call(messages, tools=[BASH_TOOL, tool], **kwargs)
        tool_calls = response.choices[0].message.tool_calls or []
        message = response.choices[0].message.model_dump()
        message["extra"] = {
            "actions": [],
            "report_calls": [
                {"id": c.id, "arguments": c.function.arguments} for c in tool_calls if c.function.name == name
            ],
            "truncated": response.choices[0].finish_reason == "length",
            "response": response.model_dump(),
            **cost_output,
            "timestamp": time.time(),
        }
        if not require:
            try:
                message["extra"]["actions"] = self._parse_actions(
                    response, [c for c in tool_calls if c.function.name != name]
                )
            except FormatError as e:
                if message["extra"]["report_calls"] and len(message["extra"]["report_calls"]) == len(tool_calls):
                    e.messages = (message, *e.messages)
                else:
                    self._persist_on_format_error(e, response, cost_output)
                raise
        return message

    def format_tool_result(self, call_id: str, content: str) -> dict:
        """The message that answers a tool call outside of command execution."""
        return {"role": "tool", "tool_call_id": call_id, "content": content, "extra": {}}

    def _calculate_cost(self, response) -> dict[str, float]:
        try:
            cost = litellm.cost_calculator.completion_cost(response, model=self.config.model_name)
            if cost <= 0.0:
                raise ValueError(f"Cost must be > 0.0, got {cost}")
        except Exception as e:
            cost = 0.0
            if self.config.cost_tracking != "ignore_errors":
                msg = (
                    f"Error calculating cost for model {self.config.model_name}: {e}, perhaps it's not registered? "
                    "You can ignore this issue from your config file with cost_tracking: 'ignore_errors' or "
                    "globally with export MSWEA_COST_TRACKING='ignore_errors'. "
                    "Alternatively check the 'Cost tracking' section in the documentation at "
                    "https://klieret.short.gy/mini-local-models. "
                    " Still stuck? Please open a github issue at https://github.com/SWE-agent/mini-swe-agent/issues/new/choose!"
                )
                logger.critical(msg)
                raise RuntimeError(msg) from e
        return {"cost": cost}

    def _parse_actions(self, response, tool_calls: list | None = None) -> list[dict]:
        """Parse tool calls from the response. Raises FormatError if unknown tool."""
        if tool_calls is None:
            tool_calls = response.choices[0].message.tool_calls or []
        return parse_toolcall_actions(
            tool_calls,
            format_error_template=self.config.format_error_template,
            template_kwargs={"finish_reason": response.choices[0].finish_reason},
        )

    def format_message(self, **kwargs) -> dict:
        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Format execution outputs into tool result messages."""
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }
