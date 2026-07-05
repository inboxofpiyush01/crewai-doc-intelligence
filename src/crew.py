"""
Crew assembly: wires the three agents and three tasks into a CrewAI Crew
running Process.sequential (extract -> summarize -> qa_check, in order,
each task's output feeding the next).
"""

from __future__ import annotations

from crewai import Crew, Process

from src.agents import build_llm, build_extractor_agent, build_summarizer_agent, build_qa_agent
from src.tasks import build_extraction_task, build_summary_task, build_qa_task


def build_crew(document_text: str) -> Crew:
    llm = build_llm()

    extractor = build_extractor_agent(llm)
    summarizer = build_summarizer_agent(llm)
    qa_reviewer = build_qa_agent(llm)

    extraction_task = build_extraction_task(extractor, document_text)
    summary_task = build_summary_task(summarizer, extraction_task)
    qa_task = build_qa_task(qa_reviewer, extraction_task, summary_task)

    return Crew(
        agents=[extractor, summarizer, qa_reviewer],
        tasks=[extraction_task, summary_task, qa_task],
        process=Process.sequential,
        verbose=True,
    )


def run_on_document(document_path: str) -> dict:
    """Convenience entrypoint: read a document file and run the full crew."""
    with open(document_path, "r") as f:
        document_text = f.read()

    crew = build_crew(document_text)
    result = crew.kickoff()

    # crew.kickoff() returns a CrewOutput; .tasks_output gives us each
    # individual task's result so callers can inspect the extraction and
    # summary separately from the final QA verdict, not just the last output.
    return {
        "extraction": str(result.tasks_output[0].raw),
        "summary": str(result.tasks_output[1].raw),
        "qa_report": str(result.tasks_output[2].raw),
        "final_output": str(result.raw),
    }
