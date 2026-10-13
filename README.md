# SatisFix

SatisFix is a coding agent built on [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent); the original README is kept in `README-mini-swe-agent.md`.

The agent intervenes once per run, at the first step that implements behavior from the task description:

1. **Detect.** The repository is snapshotted before every model query. Each time a step changes the repository, the model is asked whether the new code implements behavior the task specifies. The first "yes" marks the intervention point.
2. **Roll back.** The repository is restored to its state before that step. The step stays in the conversation, and the output of its commands is replaced by a placeholder saying the step was rolled back.
3. **Report.** The model reports where it decided that a requirement did not need to be met exactly as written.
4. **Reflect.** If the report lists anything, the model turns each of those decisions back into an open question and says what it must check before settling it.
5. **Continue.** The agent resumes the task. There is no further intervention.

## Usage

```bash
pip install -e .
mini -c mini.yaml -c satisfix.yaml -m <model> -t "<task>"
```

`satisfix.yaml` is located in `src/minisweagent/config/extra/` and contains all SatisFix prompts. Set `-c agent.intercept=false` to skip the roll-back and keep the intervened step.

Saved trajectories include a `satisfix` record with the intervention events, the rolled-back step with its original output, the report and the reflection.

## License

MIT, see `LICENSE.md`.
