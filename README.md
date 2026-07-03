# Document Intelligence Crew (CrewAI)

Three collaborating agents process a business document end to end: **Extractor** pulls structured facts, **Summarizer** writes an executive summary from those facts, and **QA Checker** cross-verifies the summary against the extraction to catch hallucinations or omissions before anything ships.

## Why three agents instead of one prompt

A single "summarize this contract" prompt conflates two different jobs that need different failure modes:

- **Extraction** needs to be conservative — it must never invent a number or date that isn't in the source.
- **Summarization** needs to be readable — it has to compress and prioritize, which is inherently lossy.

Doing both in one pass means a model under pressure to "write a nice summary" can quietly smooth over an ambiguity it should have flagged. Splitting them into separate agents with separate goals, and adding a **third agent whose only job is catching disagreements between the first two**, makes that failure visible instead of silent.

```
   sample_contract.txt
           │
           ▼
   ┌───────────────────┐
   │ Extractor Agent    │  -> structured facts (dates, $, %, obligations)
   └─────────┬──────────┘
             │ (context)
             ▼
   ┌───────────────────┐
   │ Summarizer Agent   │  -> executive summary, built ONLY from extracted facts
   └─────────┬──────────┘
             │ (context: both prior outputs)
             ▼
   ┌───────────────────┐
   │ QA Reviewer Agent  │  -> PASS/FAIL + list of unsupported claims or omissions
   └───────────────────┘
```

The QA agent never sees the original document — only the other two agents' outputs. That's deliberate: it forces it to catch *inconsistency between two AI-generated artifacts*, which is closer to how review actually has to work when you can't always have a human re-read the full source at scale.

## Running it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=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

python main.py                              # runs on the included sample contract
python main.py path/to/your_document.txt    # or your own document
```

Output prints all three stages: extracted facts, the executive summary, and the QA report (with explicit PASS/FAIL and any flagged unsupported claims).

## Running the tests

```bash
pytest tests/ -v
```

8 tests covering agent and task **construction** — roles, delegation settings, and the `context=` links between tasks that determine what each agent can and can't see. These run with zero API calls, since construction is pure Python object assembly; only `crew.kickoff()` (not exercised in CI) makes real LLM calls.

## Project structure

```
data/
  sample_contract.txt   # example vendor agreement to analyze
src/
  agents.py              # 3 agent definitions (role, goal, backstory)
  tasks.py               # 3 task definitions, wired via context=
  crew.py                # Crew assembly + run_on_document() entrypoint
tests/
  test_agents_and_tasks.py
main.py                  # CLI
```

## Stack

Python · CrewAI · Anthropic Claude · pytest
