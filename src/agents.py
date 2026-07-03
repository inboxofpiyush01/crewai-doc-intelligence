"""
Agent definitions for the Document Intelligence Crew.

Three agents with deliberately narrow, non-overlapping responsibilities:
  1. Extractor   -- pulls structured facts out of raw document text
  2. Summarizer  -- writes a human-readable executive summary FROM those facts
  3. QA Checker  -- verifies the summary doesn't contradict or invent
                    anything beyond what the Extractor actually found

The QA Checker is the most important design choice here: it's a built-in
hallucination guardrail, not just an extra agent for show. It only sees the
Extractor's output and the Summarizer's output -- never the original
document -- which forces it to catch *inconsistencies between two AI
outputs* rather than re-reading the source itself, mirroring how a real
review/audit step works when you can't always re-verify against ground
truth at scale.
"""

from __future__ import annotations

from crewai import Agent, LLM


def build_llm() -> LLM:
    """CrewAI's LLM wrapper around litellm -- model string controls backend.
    Using Claude here since document-intelligence quality (especially the
    QA/verification step) benefits from a stronger model than a free local
    one would provide."""
    return LLM(model="claude-sonnet-4-6", temperature=0.2)


def build_extractor_agent(llm: LLM) -> Agent:
    return Agent(
        role="Document Extraction Specialist",
        goal=(
            "Extract every concrete fact, figure, date, obligation, and "
            "defined term from the source document with perfect fidelity -- "
            "nothing invented, nothing omitted."
        ),
        backstory=(
            "You are a meticulous contract analyst with 15 years of experience "
            "in vendor agreement review. You never paraphrase numbers or dates -- "
            "you extract them exactly as written. You flag anything ambiguous "
            "rather than guessing at its meaning."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def build_summarizer_agent(llm: LLM) -> Agent:
    return Agent(
        role="Executive Summary Writer",
        goal=(
            "Turn extracted facts into a clear, well-organized executive "
            "summary that a non-technical business stakeholder could read in "
            "under two minutes and understand the key risks and obligations."
        ),
        backstory=(
            "You write for busy executives. You never add information that "
            "wasn't in the facts you were given -- if something is unclear or "
            "missing, you say so explicitly rather than filling the gap with "
            "an assumption."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def build_qa_agent(llm: LLM) -> Agent:
    return Agent(
        role="Quality Assurance Reviewer",
        goal=(
            "Cross-check the executive summary against the extracted facts and "
            "flag ANY claim in the summary that is not directly supported by "
            "those facts, plus any important extracted fact the summary left out."
        ),
        backstory=(
            "You are a skeptical auditor. Your only job is catching errors -- "
            "you assume the summary may contain mistakes until you've verified "
            "every claim line by line against the source facts."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
