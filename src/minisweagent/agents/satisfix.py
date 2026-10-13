"""The interactive agent plus SatisFix: one intervention at the first step that writes implementation code.

Before every model query the agent snapshots the repository. When a step changed it, the agent asks
the model (in a side conversation that is not kept) whether the code of that step gives behavior to
the task description. The first `yes` is the intervention point: with `intercept` on (the default), the
repository is rolled back to the snapshot before that step; the step's reply stays in the conversation and
its observations are replaced by a placeholder tool result (`rolled_back`), the originals being kept in the
record; with it off, the step stays as it was. Either way the
model is then asked, in plain language, where it concluded that a requirement need not be met as the
description says (`get_satisficing`). A side question that is not kept sorts the report: one that
lists a place leads to a reflection (`reflection`), a report that is blank or says there is nothing
to report leads straight to the release. The kept exchange becomes part of the conversation, a
message asks the model to continue the task, and the agent runs on as the interactive agent with no
further intervention.

All text shown to the model comes from the config (see `config/extra/satisfix.yaml`). Every request
of a run sends the same two tools, `bash` and `report`, so prompt caching holds across the side
conversations. Every `report` request follows one path: the model must call tool `report`; a
missing call or a report that fails its check is answered with the configured message and asked
again, up to `report_retries` times; a reply cut off at the output limit is instead marked dropped,
left out of the requests that follow, and followed by `report_truncated_template`; an accepted report
is answered with `report_recorded`, and the failed attempts are then left out of the conversation (the
record keeps them). A `report` call in a normal step is answered with `report_unrequested`; a normal
step with only such calls is a format error.
"""

import json
import re
from collections.abc import Callable

from jinja2 import StrictUndefined, Template

from minisweagent.agents.interactive import InteractiveAgent, InteractiveAgentConfig
from minisweagent.agents.utils.snapshot import RepoSnapshots, Snapshot
from minisweagent.exceptions import FormatError
from minisweagent.utils.serialize import recursive_merge

REPORT_PARAMETERS = {
    "type": "object",
    "properties": {"answer": {"type": "string", "description": "The answer the message asks for."}},
    "required": ["answer"],
}


class SatisFixAgentConfig(InteractiveAgentConfig):
    repo_dir: str = ""
    """Repository to snapshot and roll back; the environment's working directory by default."""
    snapshot_dir: str = ""
    """Where snapshots are kept inside the environment; a fresh directory under `/var/tmp` by default."""
    report_retries: int = 3
    """How often a missing or rejected report is asked for again."""
    intercept: bool = True
    """Roll back the triggering step and replace its observations with `rolled_back` before intervening; off, it stays."""
    has_items_max_words: int = 30
    """A report of up to this many words is sorted by the side question; a longer one is taken to list a place."""
    report_tool_description: str
    rolled_back: str
    """Placeholder tool result for each command of the rolled-back step, in place of its output."""
    report_unrequested: str
    """Answer to a `report` call in a normal step, where no report was asked for."""
    report_recorded: str
    """Answer to an accepted report."""
    report_extra_call: str
    """Answer to a second `report` call in the same reply."""
    report_missing_template: str
    """User message when a reply has no `report` call."""
    report_rejected_template: str
    """Answer to a report that fails its check; `error` names the problem."""
    report_truncated_template: str
    """User message after a reply cut off at the output limit, which is dropped from the requests."""
    action_template: str
    """Did the code of the last reply give behavior to the task description? Answered with `{"answer", "clauses"}`."""
    get_satisficing_template: str
    """Asks, as the maintainers, where the model concluded a requirement need not be met as described."""
    has_items_template: str
    """Side question, not kept: does the report list a place, or is it blank / nothing to report?"""
    reflection_template: str
    """Reveals the framing and asks for the reflection; `task` is the task text."""
    continue_template: str
    """Reveals the framing and asks the model to continue when the report had nothing; `task` is the task text."""
    continue_after_reflection_template: str
    """Asks the model to continue the task after a reflection."""


def _report_text(arguments: str) -> str:
    try:
        args = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        raise ValueError("the tool arguments are not valid JSON")
    if not isinstance(args, dict) or not isinstance(args.get("answer"), str):
        raise ValueError("the tool argument `answer` must be a string")
    return args["answer"]


def check_action(arguments: str) -> dict:
    """The answer to the action question: `{"answer": "yes" | "no", "clauses": [...]}`, normalised."""
    try:
        report = json.loads(_report_text(arguments))
    except json.JSONDecodeError:
        report = None
    if not isinstance(report, dict) or not isinstance(report.get("answer"), str):
        raise ValueError('the report must be a JSON object {"answer": "yes" | "no", "clauses": ["..."]}')
    answer = re.sub(r"[\s`'\".]", "", report["answer"]).lower()
    if answer not in ("yes", "no"):
        raise ValueError("`answer` must be `yes` or `no`")
    clauses = report.get("clauses", [])  # a `no` may leave them out
    if not isinstance(clauses, list) or not all(isinstance(c, str) for c in clauses):
        raise ValueError("`clauses` must be a list of strings")
    return {"answer": answer, "clauses": clauses}


def check_text(arguments: str) -> str:
    """A free-text report: the `answer` argument as it is."""
    return _report_text(arguments)


HAS_ITEMS = "the report lists at least one place"
NO_ITEMS = "the report is empty, or says there is nothing to report"


def check_has_items(arguments: str) -> bool:
    """The answer to the sorting question: True for the line that says the report lists a place."""
    line = re.sub(r"^[\s`'\"*-]+|[\s`'\"*.]+$", "", _report_text(arguments)).lower()
    if line.startswith(HAS_ITEMS):
        return True
    if line.startswith(NO_ITEMS):
        return False
    raise ValueError(f"the report must be exactly one of the two lines: `{HAS_ITEMS}.` or `{NO_ITEMS}.`")


class SatisFixAgent(InteractiveAgent):
    def __init__(self, *args, config_class=SatisFixAgentConfig, **kwargs):
        super().__init__(*args, config_class=config_class, **kwargs)
        self.snapshots = RepoSnapshots(self.env, self.config.repo_dir, self.config.snapshot_dir)
        self.last_snapshot: Snapshot | None = None
        self.satisfix: dict = {"status": "armed", "steps": [], "events": []}

    def serialize(self, *extra_dicts) -> dict:
        return recursive_merge(super().serialize(*extra_dicts), {"satisfix": self.satisfix})

    def _render(self, template: str, **kwargs) -> str:
        return Template(template, undefined=StrictUndefined).render(**self.get_template_vars(**kwargs))

    def query(self) -> dict:
        if self.satisfix["status"] == "armed":
            self._check_commit_point()
        message = super().query()
        self.add_messages(*self._unrequested(message.get("extra", {}).get("report_calls", [])))
        return message

    def _model_query(self) -> dict:
        """The normal step with tool `report` resident; its calls are answered as unrequested."""
        try:
            return self.model.query_report(
                self.messages, "report", self.config.report_tool_description, REPORT_PARAMETERS, require=False
            )
        except FormatError as e:  # a reply with only `report` calls: the reply stays, so its calls are answered
            calls = e.messages[0].get("extra", {}).get("report_calls", [])
            e.messages = (e.messages[0], *self._unrequested(calls), *e.messages[1:])
            raise

    def _unrequested(self, calls: list[dict]) -> list[dict]:
        return [self.model.format_tool_result(c["id"], self.config.report_unrequested) for c in calls]

    def _check_commit_point(self) -> None:
        snapshot = self.snapshots.take()
        if snapshot is None:
            self.satisfix["status"] = "no_repository" if self.last_snapshot is None else "snapshot_failed"
            return
        if self.last_snapshot is None:  # baseline, before the first query
            self.last_snapshot = snapshot
            self.satisfix["baseline"] = snapshot.tree
            return
        changed = snapshot.tree != self.last_snapshot.tree
        self.satisfix["steps"].append({"n_calls": self.n_calls, "tree": snapshot.tree, "changed": changed})
        if not changed:  # same tree, but the status and the saved metadata are the newer ones
            self.last_snapshot = snapshot
            return
        report, _, exchange, error = self._ask(self.messages, [self._render(self.config.action_template)], check_action)
        report = report or {"answer": None, "clauses": None}
        self.satisfix["events"].append(
            {"event": "action_question", "n_calls": self.n_calls, **report, "error": error, "messages": exchange}
        )
        if report["answer"] != "yes":
            self.last_snapshot = snapshot
            return
        if self.config.intercept:
            ok, details = self.snapshots.rollback(self.last_snapshot)
            self.satisfix["events"].append({"event": "rollback", "n_calls": self.n_calls, "ok": ok, **details})
            if not ok:
                self.satisfix["status"] = "rollback_failed"
                return
            self.satisfix["intercepted"] = self._intercept_last_step()
        added, error = self._intervene()
        if added is not None:
            self.add_messages(*added)
        self.satisfix["status"] = "released" if error is None else f"intervention_failed: {error}"

    def _intercept_last_step(self) -> list[dict]:
        """Keep the last reply, replace the observations of its commands with `rolled_back`; return the reply and
        the replaced observations."""
        start = max(
            i for i, m in enumerate(self.messages) if m.get("role") == "assistant" or m.get("object") == "response"
        )
        reply, after = self.messages[start], self.messages[start + 1 :]
        ids = [a["tool_call_id"] for a in reply.get("extra", {}).get("actions", [])]
        replaced = [m for m in after if (m.get("tool_call_id") or m.get("call_id")) in ids]
        kept = [m for m in after if m not in replaced]
        self.messages[start + 1 :] = [*(self.model.format_tool_result(i, self.config.rolled_back) for i in ids), *kept]
        return [reply, *replaced]

    def _intervene(self) -> tuple[list[dict] | None, str | None]:
        text, added, exchange, error = self._ask(
            self.messages, [self._render(self.config.get_satisficing_template)], check_text
        )
        self.satisfix["get_satisficing"] = {"text": text, "messages": exchange, "error": error}
        if error is not None:
            return None, error
        words = len(text.split())
        if 0 < words <= self.config.has_items_max_words:  # short: sorted by the model, in a side conversation not kept
            answer, _, exchange, error = self._ask(
                self.messages + added, [self._render(self.config.has_items_template)], check_has_items
            )
            self.satisfix["has_items"] = {"answer": answer, "messages": exchange, "error": error}
            has_items = True if error is not None else answer  # unsorted: reflect, the safe side
        else:  # blank, or long enough to list a place
            has_items = words > 0
            self.satisfix["has_items"] = {"answer": has_items, "messages": [], "error": None}
        if has_items:
            reflection, more, exchange, error = self._ask(
                self.messages + added, [self._render(self.config.reflection_template)], check_text
            )
            self.satisfix["reflection"] = {"text": reflection, "messages": exchange, "error": error}
            if error is not None:
                return None, error
            added += more
        template = self.config.continue_after_reflection_template if has_items else self.config.continue_template
        added.append(self.model.format_message(role="user", content=self._render(template)))
        return added, None

    def _ask(
        self, history: list[dict], questions: list[str], check: Callable[[str], object]
    ) -> tuple[object, list[dict], list[dict], str | None]:
        """Ask with tool `report` required; returns (checked value, messages to keep, all messages exchanged, error).

        While retrying, the model sees the failed attempts, except replies cut off at the output limit; once a
        report is accepted, the kept messages are the questions, the accepted reply and its receipts only.
        """
        asked = [self.model.format_message(role="user", content=q) for q in questions]
        exchange = list(asked)
        for attempt in range(self.config.report_retries + 1):
            reply = self.model.query_report(
                history + [m for m in exchange if not m.get("extra", {}).get("dropped")],
                "report",
                self.config.report_tool_description,
                REPORT_PARAMETERS,
            )
            self.cost += reply.get("extra", {}).get("cost", 0.0)
            exchange.append(reply)
            if reply["extra"].get("truncated"):
                reply["extra"]["dropped"] = True
                if attempt == self.config.report_retries:
                    return None, [], exchange, "report cut off at the output limit"
                exchange.append(
                    self.model.format_message(role="user", content=self._render(self.config.report_truncated_template))
                )
                continue
            calls = reply["extra"]["report_calls"]
            if not calls:
                if attempt == self.config.report_retries:
                    return None, [], exchange, "no call to tool `report`"
                exchange.append(
                    self.model.format_message(role="user", content=self._render(self.config.report_missing_template))
                )
                continue
            receipts = [self.model.format_tool_result(c["id"], self.config.report_extra_call) for c in calls[1:]]
            try:
                value = check(calls[0]["arguments"])
            except ValueError as e:
                exchange += receipts
                if attempt == self.config.report_retries:
                    return None, [], exchange, f"report not accepted: {e}"
                exchange.append(
                    self.model.format_tool_result(
                        calls[0]["id"], self._render(self.config.report_rejected_template, error=str(e))
                    )
                )
                continue
            receipts.append(self.model.format_tool_result(calls[0]["id"], self.config.report_recorded))
            exchange += receipts
            return value, asked + [reply, *receipts], exchange, None
        raise AssertionError("unreachable")
