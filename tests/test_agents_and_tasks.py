"""
Tests for agent/task/crew construction.

CrewAI agents make real LLM calls only when crew.kickoff() runs -- agent
and task *construction* (role, goal, backstory, context wiring) is pure
Python object assembly and can be tested without any API key or network
call. These tests catch the class of bug that's easy to introduce when
refactoring agent definitions: a typo in a role name, a missing context=
link between tasks, etc.
"""

from __future__ import annotations

from src.agents import (
    build_llm,
    build_extractor_agent,
    build_summarizer_agent,
    build_qa_agent,
)
from src.tasks import build_extraction_task, build_summary_task, build_qa_task
from src.crew import build_crew


def test_build_llm_uses_claude_model():
    llm = build_llm()
    assert "claude" in llm.model.lower()


def test_extractor_agent_has_no_delegation():
    """Extraction must be a single agent's independent judgment -- it
    shouldn't delegate to another agent and dilute accountability for
    fidelity to the source text."""
    llm = build_llm()
    agent = build_extractor_agent(llm)
    assert agent.allow_delegation is False
    assert "Extraction" in agent.role


def test_summarizer_agent_role_and_no_delegation():
    llm = build_llm()
    agent = build_summarizer_agent(llm)
    assert agent.allow_delegation is False
    assert "Summary" in agent.role


def test_qa_agent_role_and_no_delegation():
    llm = build_llm()
    agent = build_qa_agent(llm)
    assert agent.allow_delegation is False
    assert "QA" in agent.role or "Quality" in agent.role


def test_summary_task_has_extraction_as_context():
    """The summarizer must only see the extractor's output, not the raw
    document directly -- this is what we test here via the context= link."""
    llm = build_llm()
    extractor = build_extractor_agent(llm)
    summarizer = build_summarizer_agent(llm)

    extraction_task = build_extraction_task(extractor, "sample doc")
    summary_task = build_summary_task(summarizer, extraction_task)

    assert extraction_task in summary_task.context


def test_qa_task_has_both_prior_tasks_as_context():
    """The QA reviewer must see BOTH the extraction and the summary -- it's
    the only agent comparing the two against each other."""
    llm = build_llm()
    extractor = build_extractor_agent(llm)
    summarizer = build_summarizer_agent(llm)
    qa = build_qa_agent(llm)

    extraction_task = build_extraction_task(extractor, "sample doc")
    summary_task = build_summary_task(summarizer, extraction_task)
    qa_task = build_qa_task(qa, extraction_task, summary_task)

    assert extraction_task in qa_task.context
    assert summary_task in qa_task.context


def test_build_crew_assembles_three_agents_and_tasks_sequentially():
    crew = build_crew("Sample document text.")

    assert len(crew.agents) == 3
    assert len(crew.tasks) == 3
    # Sequential process means task order matters: extract, then summarize,
    # then QA -- and each later task must depend on earlier ones via context.
    assert crew.tasks[0].agent.role == "Document Extraction Specialist"
    assert crew.tasks[1].agent.role == "Executive Summary Writer"
    assert crew.tasks[2].agent.role == "Quality Assurance Reviewer"
    assert crew.tasks[0] in crew.tasks[1].context
    assert crew.tasks[0] in crew.tasks[2].context
    assert crew.tasks[1] in crew.tasks[2].context


def test_extraction_task_embeds_document_text():
    llm = build_llm()
    extractor = build_extractor_agent(llm)
    task = build_extraction_task(extractor, "UNIQUE_MARKER_TEXT_12345")

    assert "UNIQUE_MARKER_TEXT_12345" in task.description
