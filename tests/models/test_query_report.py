import json
from unittest.mock import MagicMock, patch

import pytest

from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.litellm_response_model import LitellmResponseModel
from minisweagent.models.utils.actions_toolcall import BASH_TOOL
from minisweagent.models.utils.actions_toolcall_response import BASH_TOOL_RESPONSE_API

PARAMETERS = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}


def _chat_response(tool_calls):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.tool_calls = tool_calls
    response.choices[0].message.model_dump.return_value = {"role": "assistant", "content": None}
    response.model_dump.return_value = {}
    return response


def _call(name, call_id, arguments):
    call = MagicMock()
    call.id, call.function.name, call.function.arguments = call_id, name, arguments
    return call


@patch("minisweagent.models.litellm_model.litellm.completion")
@patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
def test_chat_query_report_forces_the_tool_and_lists_its_calls(mock_cost, mock_completion):
    mock_completion.return_value = _chat_response(
        [_call("report", "r1", json.dumps({"text": "yes"})), _call("bash", "b1", "{}"), _call("report", "r2", "{}")]
    )
    mock_cost.return_value = 0.002
    model = LitellmModel(model_name="gpt-4o")
    message = model.query_report([{"role": "user", "content": "q"}], "report", "Send.", PARAMETERS)
    kwargs = mock_completion.call_args.kwargs
    assert kwargs["tools"] == [
        BASH_TOOL,
        {"type": "function", "function": {"name": "report", "description": "Send.", "parameters": PARAMETERS}},
    ]
    assert kwargs["tool_choice"] == {"type": "function", "function": {"name": "report"}}
    assert message["extra"]["actions"] == [] and message["extra"]["cost"] == 0.002
    assert message["extra"]["report_calls"] == [
        {"id": "r1", "arguments": json.dumps({"text": "yes"})},
        {"id": "r2", "arguments": "{}"},
    ]
    assert model.format_tool_result("r1", "Recorded.") == {
        "role": "tool",
        "tool_call_id": "r1",
        "content": "Recorded.",
        "extra": {},
    }


@patch("minisweagent.models.litellm_model.litellm.completion")
@patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
def test_chat_query_report_without_a_call(mock_cost, mock_completion):
    mock_completion.return_value = _chat_response(None)
    mock_completion.return_value.choices[0].finish_reason = "length"
    mock_cost.return_value = 0.001
    message = LitellmModel(model_name="gpt-4o").query_report([], "report", "Send.", PARAMETERS)
    assert message["extra"]["report_calls"] == [] and message["extra"]["truncated"] is True


@patch("minisweagent.models.litellm_response_model.litellm.responses")
@patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
def test_responses_query_report(mock_cost, mock_responses):
    response = MagicMock()
    response.model_dump.return_value = {
        "object": "response",
        "output": [
            {"type": "reasoning", "summary": []},
            {"type": "function_call", "call_id": "r1", "name": "report", "arguments": json.dumps({"text": "no"})},
            {"type": "function_call", "call_id": "b1", "name": "bash", "arguments": "{}"},
        ],
    }
    mock_responses.return_value = response
    mock_cost.return_value = 0.003
    model = LitellmResponseModel(model_name="gpt-5")
    message = model.query_report([{"role": "user", "content": "q"}], "report", "Send.", PARAMETERS)
    kwargs = mock_responses.call_args.kwargs
    assert kwargs["tools"] == [
        BASH_TOOL_RESPONSE_API,
        {"type": "function", "name": "report", "description": "Send.", "parameters": PARAMETERS},
    ]
    assert kwargs["tool_choice"] == {"type": "function", "name": "report"}
    assert message["object"] == "response" and message["extra"]["actions"] == []
    assert message["extra"]["report_calls"] == [{"id": "r1", "arguments": json.dumps({"text": "no"})}]
    assert model.format_tool_result("r1", "Recorded.") == {
        "type": "function_call_output",
        "call_id": "r1",
        "output": "Recorded.",
        "extra": {},
    }


@patch("minisweagent.models.litellm_response_model.litellm.responses")
@patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
def test_response_query_report_marks_a_cut_off_reply(mock_cost, mock_responses):
    response = MagicMock()
    response.model_dump.return_value = {
        "object": "response",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [{"type": "function_call", "call_id": "r1", "name": "report", "arguments": '{"answer": "f(" '}],
    }
    mock_responses.return_value = response
    mock_cost.return_value = 0.001
    message = LitellmResponseModel(model_name="gpt-5").query_report([], "report", "Send.", PARAMETERS)
    assert message["extra"]["truncated"] is True and message["extra"]["report_calls"][0]["id"] == "r1"


@patch("minisweagent.models.litellm_model.litellm.completion")
@patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
def test_chat_resident_tool_sends_the_same_tools(mock_cost, mock_completion):
    mock_completion.return_value = _chat_response(
        [_call("report", "r1", json.dumps({"text": "hi"})), _call("bash", "b1", json.dumps({"command": "ls"}))]
    )
    mock_cost.return_value = 0.002
    model = LitellmModel(model_name="gpt-4o")
    normal = model.query_report([{"role": "user", "content": "q"}], "report", "Send.", PARAMETERS, require=False)
    normal_kwargs = mock_completion.call_args.kwargs
    assert normal["extra"]["actions"] == [{"command": "ls", "tool_call_id": "b1"}]
    assert normal["extra"]["report_calls"] == [{"id": "r1", "arguments": json.dumps({"text": "hi"})}]
    assert normal["extra"]["cost"] == 0.002
    model.query_report([{"role": "user", "content": "q"}], "report", "Send.", PARAMETERS)
    forced_kwargs = mock_completion.call_args.kwargs
    assert normal_kwargs["tools"] == forced_kwargs["tools"] and [
        t["function"]["name"] for t in normal_kwargs["tools"]
    ] == ["bash", "report"]
    assert "tool_choice" not in normal_kwargs and forced_kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "report"},
    }

    mock_completion.return_value = _chat_response([_call("report", "r1", "{}")])
    with pytest.raises(FormatError) as info:
        model.query_report([], "report", "Send.", PARAMETERS, require=False)
    reply, error = info.value.messages
    assert reply["extra"]["report_calls"] == [{"id": "r1", "arguments": "{}"}] and reply["extra"]["cost"] == 0.002
    assert error["extra"]["interrupt_type"] == "FormatError" and "No tool calls" in error["content"]

    mock_completion.return_value = _chat_response([_call("report", "r1", "{}"), _call("bash", "b1", "{}")])
    with pytest.raises(FormatError) as info:
        model.query_report([], "report", "Send.", PARAMETERS, require=False)
    (error,) = info.value.messages
    assert "Missing 'command'" in error["content"] and error["extra"]["cost"] == 0.002


@patch("minisweagent.models.litellm_response_model.litellm.responses")
@patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
def test_responses_resident_tool_sends_the_same_tools(mock_cost, mock_responses):
    def response(output):
        r = MagicMock()
        r.output = output
        r.model_dump.return_value = {"object": "response", "output": output}
        return r

    report = {"type": "function_call", "call_id": "r1", "name": "report", "arguments": json.dumps({"text": "hi"})}
    bash = {"type": "function_call", "call_id": "b1", "name": "bash", "arguments": json.dumps({"command": "ls"})}
    mock_responses.return_value = response([{"type": "reasoning", "summary": []}, report, bash])
    mock_cost.return_value = 0.003
    model = LitellmResponseModel(model_name="gpt-5")
    normal = model.query_report([], "report", "Send.", PARAMETERS, require=False)
    normal_kwargs = mock_responses.call_args.kwargs
    assert normal["extra"]["actions"] == [{"command": "ls", "tool_call_id": "b1"}]
    assert normal["extra"]["report_calls"] == [{"id": "r1", "arguments": json.dumps({"text": "hi"})}]
    model.query_report([], "report", "Send.", PARAMETERS)
    forced_kwargs = mock_responses.call_args.kwargs
    assert normal_kwargs["tools"] == forced_kwargs["tools"] and [t["name"] for t in normal_kwargs["tools"]] == [
        "bash",
        "report",
    ]
    assert "tool_choice" not in normal_kwargs and forced_kwargs["tool_choice"] == {"type": "function", "name": "report"}

    mock_responses.return_value = response([report])
    with pytest.raises(FormatError) as info:
        model.query_report([], "report", "Send.", PARAMETERS, require=False)
    reply, error = info.value.messages
    assert (
        reply["object"] == "response"
        and reply["extra"]["report_calls"][0]["id"] == "r1"
        and reply["extra"]["cost"] == 0.003
    )
    assert error["extra"]["interrupt_type"] == "FormatError" and error["type"] == "message"
