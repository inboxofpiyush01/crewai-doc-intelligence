"""
Task definitions for the Document Intelligence Crew.

Tasks are wired sequentially: extract -> summarize -> qa_check. CrewAI
passes each task's output into the next task's context automatically when
using Process.sequential (see crew.py), so the Summarizer task only ever
sees the Extractor's structured output, and the QA task only ever sees
both prior outputs -- never the raw document directly. This is what makes
the QA step a genuine cross-check rather than just "another opinion."
"""

from __future__ import annotations

from crewai import Task, Agent


def build_extraction_task(agent: Agent, document_text: str) -> Task:
    return Task(
        description=(
            "Read the following document and extract every concrete fact: "
            "dates, monetary amounts, percentages, named parties, defined "
            "terms, obligations, and deadlines. Organize them under clear "
            "headings matching the document's sections. Do not summarize or "
            "interpret -- extract verbatim facts only.\n\n"
            f"DOCUMENT:\n{document_text}"
        ),
        expected_output=(
            "A structured list of facts grouped by section heading, with "
            "exact figures, dates, and defined terms preserved verbatim."
        ),
        agent=agent,
    )


def build_summary_task(agent: Agent, extraction_task: Task) -> Task:
    return Task(
        description=(
            "Using ONLY the extracted facts provided to you (not any "
            "outside knowledge), write a clear executive summary covering: "
            "what the agreement is for, the key financial terms, the "
            "main risks or obligations for the Client, and how the "
            "agreement can end. Keep it under 250 words. If a fact needed "
            "for a complete picture wasn't extracted, explicitly note the gap "
            "rather than inferring it."
        ),
        expected_output=(
            "A well-organized executive summary under 250 words, covering "
            "purpose, financial terms, key risks/obligations, and termination."
        ),
        agent=agent,
        context=[extraction_task],
    )


def build_qa_task(
    agent: Agent, extraction_task: Task, summary_task: Task
) -> Task:
    return Task(
        description=(
            "Compare the executive summary against the extracted facts. "
            "List: (1) any claim in the summary NOT directly supported by "
            "the extracted facts, (2) any important extracted fact the "
            "summary omitted, (3) an overall verdict of PASS or FAIL, where "
            "FAIL means at least one unsupported claim was found."
        ),
        expected_output=(
            "A structured QA report with three sections: Unsupported Claims, "
            "Omitted Facts, and Verdict (PASS/FAIL)."
        ),
        agent=agent,
        context=[extraction_task, summary_task],
    )
