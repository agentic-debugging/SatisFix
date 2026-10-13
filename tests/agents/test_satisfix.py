import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from minisweagent.agents.satisfix import SatisFixAgent, check_action, check_has_items, check_text
from minisweagent.config import get_config_from_spec
from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import FormatError
from minisweagent.models.test_models import (
    DeterministicResponseAPIToolcallModel,
    DeterministicToolcallModel,
    make_response_api_output,
    make_toolcall_output,
)
from minisweagent.utils.serialize import recursive_merge

SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
YES = json.dumps({"answer": "yes", "clauses": ["Make x equal 2"]})
NO = json.dumps({"answer": "no", "clauses": []})
FISH = "I left x at 1 instead of 2: renumbering takes extra work, so I set it aside for now."
HAS = "The report lists at least one place."
NONE_LINE = "The report is empty, or says there is nothing to report."
REFLECT = (
    "x must be 2: I assumed renumbering is costly; alternatives: keep x at 1, set x to 2. I will look at pkg/a.py."
)
RELEASE_START = "Now continue the task. Start with what you said you would look at"


def agent_config() -> dict:
    config = recursive_merge(get_config_from_spec("mini"), get_config_from_spec("satisfix"))["agent"]
    config.pop("agent_class")
    return config | {"mode": "yolo", "confirm_exit": False, "cost_limit": 0.0}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(
        "git init -q -b main . && git config user.email a@b && git config user.name a && mkdir pkg && "
        "echo 'x=1' > pkg/a.py && git add -A && git commit -qm init",
        shell=True,
        cwd=root,
        check=True,
    )
    return root


# --- replies in the two tool-call formats ---


def tc_bash(command: str, call_id: str = "c") -> dict:
    tool_calls = [
        {"id": call_id, "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": command})}}
    ]
    return make_toolcall_output(None, tool_calls, [{"command": command, "tool_call_id": call_id}])


def tc_report(text: str | None, call_id: str = "r", extra_calls: int = 0, command: str | None = None) -> dict:
    """A reply calling `report` with `answer` = `text` (none if None), `extra_calls` more times, and `bash` with `command`."""
    calls = (
        []
        if text is None
        else [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "report", "arguments": json.dumps({"answer": text})},
            }
        ]
    )
    calls += [
        {"id": f"{call_id}{i}", "type": "function", "function": {"name": "report", "arguments": "{}"}}
        for i in range(extra_calls)
    ]
    if command is not None:
        calls.append(
            {"id": "c", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": command})}}
        )
    actions = [] if command is None else [{"command": command, "tool_call_id": "c"}]
    return make_toolcall_output("thinking" if text is None else None, calls, actions)


def ra_bash(command: str, call_id: str = "c") -> dict:
    return make_response_api_output(None, [{"command": command, "tool_call_id": call_id}])


def ra_report(text: str | None, call_id: str = "r", command: str | None = None) -> dict:
    output = (
        []
        if text is None
        else [
            {"type": "function_call", "call_id": call_id, "name": "report", "arguments": json.dumps({"answer": text})}
        ]
    )
    reply = (
        ra_bash(command) if command is not None else {"extra": {"actions": [], "cost": 1.0, "timestamp": time.time()}}
    )
    return reply | {"object": "response", "output": output + reply.get("output", [])}


def cut_off(reply: dict) -> dict:
    """The reply as cut off at the output limit, in either format."""
    return reply | {"status": "incomplete", "extra": reply["extra"] | {"truncated": True}}


def run(repo: Path, model, tmp_path: Path, **overrides) -> SatisFixAgent:
    env = LocalEnvironment(cwd=str(repo))
    agent = SatisFixAgent(model, env, **agent_config() | {"snapshot_dir": str(tmp_path / "snapshots")} | overrides)
    agent.run("Make x equal 2 in pkg/a.py.")
    return agent


def kinds(messages: list[dict]) -> list[str]:
    """Message roles regardless of format: user / assistant / tool / system / exit."""
    out = []
    for m in messages:
        if m.get("object") == "response" or m.get("role") == "assistant":
            out.append("assistant")
        elif m.get("type") == "function_call_output" or m.get("role") == "tool":
            out.append("tool")
        else:
            out.append(m["role"])
    return out


def user_texts(messages: list[dict]) -> list[str]:
    """The text of the user messages, in either format (a string, or a list of `input_text` parts)."""
    out = []
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            out.append("".join(part.get("text", "") for part in content if isinstance(part, dict)))
    return out


# --- checks ---


def test_checks():
    answer = {"answer": " `Yes`. ", "clauses": ["a", "b"]}
    assert check_action(json.dumps({"answer": json.dumps(answer)})) == {"answer": "yes", "clauses": ["a", "b"]}
    assert check_action(json.dumps({"answer": json.dumps({"answer": "no", "clauses": []})})) == {
        "answer": "no",
        "clauses": [],
    }
    assert check_action(json.dumps({"answer": json.dumps({"answer": "no"})})) == {"answer": "no", "clauses": []}
    with pytest.raises(ValueError, match="`answer` must be `yes` or `no`"):
        check_action(json.dumps({"answer": json.dumps({"answer": "maybe", "clauses": []})}))
    with pytest.raises(ValueError, match="`clauses` must be a list of strings"):
        check_action(json.dumps({"answer": json.dumps({"answer": "yes", "clauses": "x"})}))
    with pytest.raises(ValueError, match="`clauses` must be a list of strings"):
        check_action(json.dumps({"answer": json.dumps({"answer": "yes", "clauses": [1]})}))
    with pytest.raises(ValueError, match='JSON object {"answer"'):
        check_action(json.dumps({"answer": "yes"}))
    with pytest.raises(ValueError, match="not valid JSON"):
        check_action("nope")
    with pytest.raises(ValueError, match="`answer`"):
        check_action(json.dumps({"text": "yes"}))  # the argument is `answer`
    assert check_text(json.dumps({"answer": FISH})) == FISH and check_text(json.dumps({"answer": ""})) == ""
    with pytest.raises(ValueError, match="`answer`"):
        check_text(json.dumps({"answer": 3}))
    assert check_has_items(json.dumps({"answer": HAS})) is True
    assert check_has_items(json.dumps({"answer": " - `The report lists at least one place` "})) is True
    assert check_has_items(json.dumps({"answer": NONE_LINE})) is False
    assert check_has_items(json.dumps({"answer": "the report is empty, or says there is nothing to report"})) is False
    with pytest.raises(ValueError, match="exactly one of the two lines"):
        check_has_items(json.dumps({"answer": "yes"}))
    with pytest.raises(ValueError, match="exactly one of the two lines"):
        check_has_items(json.dumps({"answer": ""}))


# --- agent flow ---


def test_no_change_never_asks(repo: Path, tmp_path: Path):
    model = DeterministicToolcallModel(outputs=[tc_bash("cat pkg/a.py"), tc_bash(SUBMIT)])
    agent = run(repo, model, tmp_path)
    assert model.current_index == 1 and agent.satisfix["status"] == "armed"
    assert [s["changed"] for s in agent.satisfix["steps"]] == [False]
    assert agent.cost == 2.0


def test_non_implementation_write_continues(repo: Path, tmp_path: Path):
    model = DeterministicToolcallModel(outputs=[tc_bash("echo draft > notes.txt"), tc_report(NO), tc_bash(SUBMIT)])
    agent = run(repo, model, tmp_path)
    assert (repo / "notes.txt").exists() and agent.satisfix["status"] == "armed"
    assert [(e["answer"], e["clauses"]) for e in agent.satisfix["events"]] == [("no", [])]
    assert kinds(agent.messages) == ["system", "user", "assistant", "tool", "assistant", "tool", "exit"]


@pytest.mark.parametrize(
    ("model_class", "bash", "report"),
    [(DeterministicToolcallModel, tc_bash, tc_report), (DeterministicResponseAPIToolcallModel, ra_bash, ra_report)],
)
def test_intervention(repo: Path, tmp_path: Path, model_class, bash, report):
    a_py = repo / "pkg/a.py"
    before = os.stat(a_py).st_mtime_ns
    outputs = [
        bash("echo draft > notes.txt"),
        report(NO),
        bash("echo 'x=2' > pkg/a.py && git checkout -q -b feature && git commit -qam step"),
        report(YES),
        report(FISH),
        report(HAS),
        report(REFLECT),
        bash(SUBMIT),
    ]
    agent = run(repo, model_class(outputs=outputs), tmp_path, intercept=True)

    assert agent.satisfix["status"] == "released"
    assert a_py.read_text() == "x=1\n" and os.stat(a_py).st_mtime_ns == before
    git = subprocess.run(
        "git symbolic-ref HEAD && git log --oneline | wc -l", shell=True, cwd=repo, capture_output=True, text=True
    )
    assert git.stdout == "refs/heads/main\n1\n"
    assert (repo / "notes.txt").exists()  # the step before the intervention stays
    rollback = next(e for e in agent.satisfix["events"] if e["event"] == "rollback")
    assert rollback["ok"] and rollback["tree_match"] and rollback["status_match"]
    assert kinds(agent.satisfix["intercepted"]) == ["assistant", "tool"]
    assert agent.satisfix["events"][1]["clauses"] == ["Make x equal 2"]
    assert agent.satisfix["get_satisficing"]["text"] == FISH
    sorting = agent.satisfix["has_items"]
    assert sorting["answer"] is True and sorting["error"] is None
    assert kinds(sorting["messages"]) == ["user", "assistant", "tool"]  # asked, answered, receipt
    assert agent.satisfix["reflection"]["text"] == REFLECT and agent.satisfix["reflection"]["error"] is None

    conversation = json.dumps(agent.messages)
    assert "A quick check on the report" not in conversation and HAS not in conversation  # the sorting is not kept
    reply, placeholder = agent.messages[4], agent.messages[5]  # the rolled-back step stays, its output does not
    assert "x=2" in json.dumps(reply)
    assert reply == agent.satisfix["intercepted"][0]
    call_id = reply["extra"]["actions"][0]["tool_call_id"]
    assert placeholder == agent.model.format_tool_result(call_id, agent.config.rolled_back)
    assert agent.satisfix["intercepted"][1] != placeholder and agent.config.rolled_back not in json.dumps(
        agent.satisfix["intercepted"]
    )
    assert kinds(agent.messages) == [
        *["system", "user", "assistant", "tool"],  # the kept step
        *["assistant", "tool"],  # the rolled-back step and its placeholder result
        *["user", "assistant", "tool"],  # get_satisficing: request, report, receipt
        *["user", "assistant", "tool"],  # reflection: request, report, receipt
        "user",  # release
        *["assistant", "tool", "exit"],  # the submission
    ]
    texts = user_texts(agent.messages)
    assert texts[1].startswith("A message from the maintainers")
    assert texts[2].startswith("[Important]") and "<task_description>" in texts[2]
    assert texts[3].startswith(RELEASE_START) and "<task_description>" not in texts[3]
    assert agent.messages[-3]["extra"]["actions"][0]["command"] == SUBMIT
    assert agent.cost == 8.0 and agent.n_calls == 3


@pytest.mark.parametrize(
    ("model_class", "bash", "report"),
    [(DeterministicToolcallModel, tc_bash, tc_report), (DeterministicResponseAPIToolcallModel, ra_bash, ra_report)],
)
def test_no_intercept_keeps_step(repo: Path, tmp_path: Path, model_class, bash, report):
    outputs = [
        bash("echo 'x=2' > pkg/a.py && git checkout -q -b feature && git commit -qam step"),
        report(YES),
        report(FISH),
        report(HAS),
        report(REFLECT),
        bash(SUBMIT),
    ]
    agent = run(repo, model_class(outputs=outputs), tmp_path, intercept=False)
    assert agent.satisfix["status"] == "released"
    assert (repo / "pkg/a.py").read_text() == "x=2\n"
    assert (
        subprocess.run("git symbolic-ref HEAD", shell=True, cwd=repo, capture_output=True, text=True).stdout
        == "refs/heads/feature\n"
    )
    assert [e["event"] for e in agent.satisfix["events"]] == ["action_question"]
    assert "intercepted" not in agent.satisfix and [s["changed"] for s in agent.satisfix["steps"]] == [True]
    assert agent.satisfix["get_satisficing"]["text"] == FISH and agent.satisfix["reflection"]["text"] == REFLECT
    assert kinds(agent.messages) == [
        *["system", "user", "assistant", "tool"],  # the triggering step stays
        *["user", "assistant", "tool"],  # get_satisficing
        *["user", "assistant", "tool"],  # reflection
        "user",  # release
        *["assistant", "tool", "exit"],
    ]
    assert "x=2" in json.dumps(agent.messages[2])
    assert user_texts(agent.messages)[-1].startswith(RELEASE_START)
    assert agent.cost == 6.0 and agent.n_calls == 2


@pytest.mark.parametrize(
    ("model_class", "bash", "report"),
    [(DeterministicToolcallModel, tc_bash, tc_report), (DeterministicResponseAPIToolcallModel, ra_bash, ra_report)],
)
def test_unrequested_report_is_answered(repo: Path, tmp_path: Path, model_class, bash, report):
    outputs = [report("hello", command="echo draft > notes.txt"), report(NO), bash(SUBMIT)]
    agent = run(repo, model_class(outputs=outputs), tmp_path)
    assert (repo / "notes.txt").read_text() == "draft\n"
    assert kinds(agent.messages) == ["system", "user", "assistant", "tool", "tool", "assistant", "tool", "exit"]
    receipt, observation = agent.messages[3], agent.messages[4]
    assert receipt.get("tool_call_id", receipt.get("call_id")) == "r"
    assert (
        agent.config.report_unrequested in json.dumps(receipt)
        and "No report was requested" in agent.config.report_unrequested
    )
    assert observation.get(
        "tool_call_id", observation.get("call_id")
    ) == "c" and "<returncode>0</returncode>" in json.dumps(observation)
    assert [e["answer"] for e in agent.satisfix["events"]] == ["no"] and agent.satisfix["status"] == "armed"
    assert agent.cost == 3.0 and agent.n_calls == 2 and agent.n_consecutive_format_errors == 0


@pytest.mark.parametrize(
    ("model_class", "bash", "report"),
    [(DeterministicToolcallModel, tc_bash, tc_report), (DeterministicResponseAPIToolcallModel, ra_bash, ra_report)],
)
def test_report_only_step_is_a_format_error(repo: Path, tmp_path: Path, model_class, bash, report):
    outputs = [report("hello"), bash(SUBMIT)]
    agent = run(repo, model_class(outputs=outputs), tmp_path)
    assert kinds(agent.messages) == ["system", "user", "assistant", "tool", "user", "assistant", "tool", "exit"]
    assert agent.messages[3].get("tool_call_id", agent.messages[3].get("call_id")) == "r"
    assert agent.config.report_unrequested in json.dumps(agent.messages[3])
    assert agent.messages[4]["extra"] == {"interrupt_type": "FormatError"} and "No tool calls" in json.dumps(
        agent.messages[4]
    )
    assert agent.satisfix["status"] == "armed" and agent.satisfix["events"] == []
    assert agent.cost == 2.0 and agent.n_calls == 2
    with pytest.raises(FormatError) as info:
        model_class(outputs=[report("x", call_id="q")]).query_report([], "report", "d", {}, require=False)
    assert (
        kinds(info.value.messages) == ["assistant", "user"]
        and info.value.messages[0]["extra"]["report_calls"][0]["id"] == "q"
    )


def test_blank_report_skips_sorting_and_reflection(repo: Path, tmp_path: Path):
    model = DeterministicToolcallModel(
        outputs=[tc_bash("echo 'x=2' > pkg/a.py"), tc_report(YES), tc_report("  \n"), tc_bash(SUBMIT)]
    )
    agent = run(repo, model, tmp_path, intercept=True)
    assert agent.satisfix["status"] == "released" and "reflection" not in agent.satisfix
    assert agent.satisfix["has_items"] == {"answer": False, "messages": [], "error": None}
    assert kinds(agent.messages) == [
        *["system", "user", "assistant", "tool"],  # the rolled-back step and its placeholder result
        *["user", "assistant", "tool", "user", "assistant", "tool", "exit"],
    ]
    release = user_texts(agent.messages)[-1]
    assert release.startswith("[Important]") and "<task_description>" in release
    assert (repo / "pkg/a.py").read_text() == "x=1\n" and agent.cost == 4.0


def test_nothing_to_report_skips_reflection(repo: Path, tmp_path: Path):
    outputs = [
        tc_bash("echo 'x=2' > pkg/a.py"),
        tc_report(YES),
        tc_report("I went over the requirements again and have nothing to report."),
        tc_report(NONE_LINE),
        tc_bash(SUBMIT),
    ]
    agent = run(repo, DeterministicToolcallModel(outputs=outputs), tmp_path, intercept=False)
    assert agent.satisfix["status"] == "released" and "reflection" not in agent.satisfix
    sorting = agent.satisfix["has_items"]
    assert sorting["answer"] is False and kinds(sorting["messages"]) == ["user", "assistant", "tool"]
    assert "A quick check on the report" not in json.dumps(agent.messages)
    assert kinds(agent.messages) == [
        *["system", "user", "assistant", "tool"],
        *["user", "assistant", "tool"],  # get_satisficing
        "user",  # release for a report with nothing in it
        *["assistant", "tool", "exit"],
    ]
    release = user_texts(agent.messages)[-1]
    assert release.startswith("[Important]") and agent.cost == 5.0


def test_long_report_skips_sorting(repo: Path, tmp_path: Path):
    long_fish = " ".join(["word"] * 31)  # over the limit: taken to list a place, no side question
    outputs = [tc_bash("echo 'x=2' > pkg/a.py"), tc_report(YES), tc_report(long_fish), tc_report(REFLECT)]
    outputs += [tc_bash(SUBMIT)]
    agent = run(repo, DeterministicToolcallModel(outputs=outputs), tmp_path, intercept=False)
    assert agent.satisfix["has_items"] == {"answer": True, "messages": [], "error": None}
    assert agent.satisfix["reflection"]["text"] == REFLECT and agent.satisfix["status"] == "released"
    assert kinds(agent.messages) == [
        *["system", "user", "assistant", "tool"],
        *["user", "assistant", "tool"],  # get_satisficing
        *["user", "assistant", "tool"],  # reflection
        "user",
        *["assistant", "tool", "exit"],
    ]
    assert agent.cost == 5.0


def test_unsorted_report_is_reflected_on(repo: Path, tmp_path: Path):
    outputs = [tc_bash("echo 'x=2' > pkg/a.py"), tc_report(YES), tc_report(FISH)]
    outputs += [tc_report("yes")] * 4 + [tc_report(REFLECT), tc_bash(SUBMIT)]  # the sorting never gets a valid line
    agent = run(repo, DeterministicToolcallModel(outputs=outputs), tmp_path, report_retries=3)
    sorting = agent.satisfix["has_items"]
    assert sorting["answer"] is None and sorting["error"].startswith("report not accepted: the report must be exactly")
    assert agent.satisfix["reflection"]["text"] == REFLECT and agent.satisfix["status"] == "released"
    assert "could not be accepted" not in json.dumps(agent.messages)


def test_reasks_then_accepts(repo: Path, tmp_path: Path):
    outputs = [
        tc_bash("echo 'x=2' > pkg/a.py"),
        tc_report(None),  # no call -> asked again
        tc_report("perhaps", extra_calls=1),  # rejected, extra call answered
        tc_report(YES),
        tc_report(""),
        tc_bash(SUBMIT),
    ]
    agent = run(repo, DeterministicToolcallModel(outputs=outputs), tmp_path)
    event = agent.satisfix["events"][0]
    assert event["answer"] == "yes" and event["error"] is None
    assert kinds(event["messages"]) == ["user", "assistant", "user", "assistant", "tool", "tool", "assistant", "tool"]
    assert agent.satisfix["status"] == "released"


def test_accepted_report_drops_failed_attempts(repo: Path, tmp_path: Path):
    outputs = [
        tc_bash("echo 'x=2' > pkg/a.py"),
        tc_report(YES),
        tc_report(None),  # no call -> asked again
        tc_report(FISH, call_id="ok", extra_calls=1),  # accepted, extra call answered
        tc_report(HAS),
        tc_report(REFLECT),
        tc_bash(SUBMIT),
    ]
    agent = run(repo, DeterministicToolcallModel(outputs=outputs), tmp_path, intercept=False)
    record = agent.satisfix["get_satisficing"]
    assert kinds(record["messages"]) == ["user", "assistant", "user", "assistant", "tool", "tool"]
    assert "No call to the tool `report` was received" in json.dumps(record["messages"]) and record["error"] is None
    assert kinds(agent.messages) == [
        *["system", "user", "assistant", "tool"],
        *["user", "assistant", "tool", "tool"],  # get_satisficing: request, accepted report, two receipts
        *["user", "assistant", "tool"],  # reflection
        "user",
        *["assistant", "tool", "exit"],
    ]
    assert "No call to the tool `report` was received" not in json.dumps(agent.messages)
    assert agent.messages[7] == {"role": "tool", "tool_call_id": "ok", "content": "Recorded.", "extra": {}}
    assert agent.satisfix["status"] == "released" and agent.cost == 7.0


@pytest.mark.parametrize(
    ("model_class", "bash", "report"),
    [(DeterministicToolcallModel, tc_bash, tc_report), (DeterministicResponseAPIToolcallModel, ra_bash, ra_report)],
)
def test_cut_off_report_is_dropped(repo: Path, tmp_path: Path, model_class, bash, report):
    class Recording(model_class):
        seen = []

        def query_report(self, messages, *args, **kwargs):
            self.seen.append(messages)
            return super().query_report(messages, *args, **kwargs)

    outputs = [
        bash("echo 'x=2' > pkg/a.py"),
        report(YES),
        cut_off(report("RUNAWAY `f(", call_id="zzcut")),
        report(FISH, call_id="ok"),
        report(HAS),
        report(REFLECT),
        bash(SUBMIT),
    ]
    model = Recording(outputs=outputs)
    agent = run(repo, model, tmp_path, intercept=False)
    record = agent.satisfix["get_satisficing"]
    assert record["text"] == FISH and record["error"] is None
    assert kinds(record["messages"]) == ["user", "assistant", "user", "assistant", "tool"]  # no result for the cut-off
    assert record["messages"][1]["extra"]["dropped"] and "dropped" not in record["messages"][3]["extra"]
    retry = model.seen[3]  # normal steps are asked through `query_report` too
    assert all(m is not record["messages"][1] for m in retry) and "zzcut" not in json.dumps(retry)
    assert user_texts(retry)[-1].startswith("Your last reply was cut off at the output limit")
    conversation = json.dumps(agent.messages)
    assert "cut off at the output limit" not in conversation and "RUNAWAY" not in conversation
    assert agent.satisfix["status"] == "released"


def test_cut_off_on_every_attempt_fails(repo: Path, tmp_path: Path):
    outputs = [tc_bash("echo 'x=2' > pkg/a.py"), tc_report(YES), *[cut_off(tc_report(FISH))] * 4, tc_bash(SUBMIT)]
    agent = run(repo, DeterministicToolcallModel(outputs=outputs), tmp_path, report_retries=3, intercept=False)
    assert agent.satisfix["status"] == "intervention_failed: report cut off at the output limit"
    assert kinds(agent.satisfix["get_satisficing"]["messages"]) == ["user", *["assistant", "user"] * 3, "assistant"]


def test_intervention_failure_keeps_step_intercepted(repo: Path, tmp_path: Path):
    outputs = [tc_bash("echo 'x=2' > pkg/a.py"), tc_report(YES), *[tc_report(None)] * 4, tc_bash(SUBMIT)]
    agent = run(repo, DeterministicToolcallModel(outputs=outputs), tmp_path, report_retries=3, intercept=True)
    assert agent.satisfix["status"] == "intervention_failed: no call to tool `report`"
    assert (repo / "pkg/a.py").read_text() == "x=1\n"
    assert kinds(agent.satisfix["intercepted"]) == ["assistant", "tool"]
    assert kinds(agent.messages) == ["system", "user", *["assistant", "tool"] * 2, "exit"]
    assert json.dumps(agent.messages[3]).count(agent.config.rolled_back[:30]) == 1


def test_reflection_failure(repo: Path, tmp_path: Path):
    outputs = [tc_bash("echo 'x=2' > pkg/a.py"), tc_report(YES), tc_report(FISH), tc_report(HAS)]
    outputs += [tc_report(None)] * 4 + [tc_bash(SUBMIT)]
    agent = run(repo, DeterministicToolcallModel(outputs=outputs), tmp_path, report_retries=3, intercept=False)
    assert agent.satisfix["status"] == "intervention_failed: no call to tool `report`"
    assert agent.satisfix["reflection"]["text"] is None and agent.satisfix["reflection"]["error"] is not None
    assert kinds(agent.messages) == ["system", "user", "assistant", "tool", "assistant", "tool", "exit"]


def test_no_repository_is_inert(tmp_path: Path):
    plain = tmp_path / "plain"
    plain.mkdir()
    model = DeterministicToolcallModel(outputs=[tc_bash("echo 'x=2' > a.py"), tc_bash(SUBMIT)])
    agent = run(plain, model, tmp_path)
    assert agent.satisfix["status"] == "no_repository" and model.current_index == 1
    assert (plain / "a.py").read_text() == "x=2\n"


def test_serialize_carries_record(repo: Path, tmp_path: Path):
    outputs = [tc_bash("echo 'x=2' > pkg/a.py"), tc_report(YES), tc_report(""), tc_bash(SUBMIT)]
    agent = run(repo, DeterministicToolcallModel(outputs=outputs), tmp_path)
    data = agent.serialize()
    assert data["satisfix"]["status"] == "released" and data["info"]["config"]["agent_type"].endswith("SatisFixAgent")
    assert json.dumps(data)  # json-serializable
