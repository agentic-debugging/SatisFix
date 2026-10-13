import logging
import time
from collections.abc import Callable

import litellm

from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_toolcall_response import (
    BASH_TOOL_RESPONSE_API,
    finish_reason_from_responses_api,
    format_toolcall_observation_messages,
    parse_toolcall_actions_response,
)

logger = logging.getLogger("litellm_response_model")


class LitellmResponseModelConfig(LitellmModelConfig):
    pass


class LitellmResponseModel(LitellmModel):
    def __init__(self, *, config_class: Callable = LitellmResponseModelConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        """Flatten response objects into their output items for stateless API calls."""
        result = []
        for msg in messages:
            if msg.get("object") == "response":
                for item in msg.get("output", []):
                    result.append({k: v for k, v in item.items() if k != "extra"})
            else:
                result.append({k: v for k, v in msg.items() if k != "extra"})
        return result

    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            return litellm.responses(
                model=self.config.model_name,
                input=messages,
                tools=kwargs.pop("tools", [BASH_TOOL_RESPONSE_API]),
                **self._api_kwargs(kwargs),
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        response, cost_output = self._call(messages, **kwargs)
        try:
            actions = self._parse_actions(response)
        except FormatError as e:
            self._persist_on_format_error(e, response, cost_output)
            raise
        message = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        message["extra"] = {
            "actions": actions,
            **cost_output,
            "timestamp": time.time(),
        }
        return message

    def _persist_on_format_error(self, e: FormatError, response, cost_output: dict) -> None:
        e.messages[0]["extra"].update(cost_output)
        # hasattr guard: litellm.responses() returns a pydantic object, but tests
        # may inject a plain dict; dict(response) is the correct fallback in that case.
        # Inner try: if serialization itself fails, repr() guarantees the key is always set.
        try:
            e.messages[0]["extra"]["response"] = (
                response.model_dump(mode="json") if hasattr(response, "model_dump") else dict(response)
            )
        except Exception:
            e.messages[0]["extra"]["response"] = repr(response)

    def query_report(
        self, messages: list[dict], name: str, description: str, parameters: dict, *, require: bool = True, **kwargs
    ) -> dict:
        """See `LitellmModel.query_report`; same contract in the Responses API format."""
        tool = {"type": "function", "name": name, "description": description, "parameters": parameters}
        if require:
            kwargs["tool_choice"] = {"type": "function", "name": name}
        response, cost_output = self._call(messages, tools=[BASH_TOOL_RESPONSE_API, tool], **kwargs)
        message = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        output = message.get("output") or []
        reports = [i for i in output if i.get("type") == "function_call" and i.get("name") == name]
        others = [i for i in output if i not in reports]
        message["extra"] = {
            "actions": [],
            "report_calls": [{"id": i.get("call_id"), "arguments": i.get("arguments")} for i in reports],
            "truncated": message.get("status") == "incomplete",
            **cost_output,
            "timestamp": time.time(),
        }
        if not require:
            try:
                message["extra"]["actions"] = self._parse_actions(response, others)
            except FormatError as e:
                if reports and not any(i.get("type") == "function_call" for i in others):
                    e.messages = (message, *e.messages)
                else:
                    self._persist_on_format_error(e, response, cost_output)
                raise
        return message

    def format_tool_result(self, call_id: str, content: str) -> dict:
        return {"type": "function_call_output", "call_id": call_id, "output": content, "extra": {}}

    def _parse_actions(self, response, output: list | None = None) -> list[dict]:
        return parse_toolcall_actions_response(
            getattr(response, "output", []) if output is None else output,
            format_error_template=self.config.format_error_template,
            template_kwargs={"finish_reason": finish_reason_from_responses_api(response)},
        )

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
