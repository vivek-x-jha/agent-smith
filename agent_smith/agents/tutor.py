"""Tutor agent that creates formative assessments."""

from __future__ import annotations

import re
from typing import Iterable

from ..models import LearningGoal, PlanItem
from .base import Agent


class TutorAgent(Agent):
    """Generates quiz questions and lightweight coaching snippets."""

    def __init__(self, llm=None) -> None:
        super().__init__(
            name="tutor",
            system_prompt=(
                "You are a friendly coach. Create short formative questions and include the answer key."
            ),
            llm=llm,
        )

    def run(
        self,
        goal: LearningGoal,
        plan_items: Iterable[PlanItem],
        curated_summary: str,
        num_questions: int = 3,
    ) -> list[dict[str, str]]:
        plan_brief = "\n".join(f"- {item.task}" for item in plan_items)
        user_prompt = (
            f"Goal: {goal.title}. Craft {num_questions} quick questions that test the study plan.\n"
            f"Plan:\n{plan_brief}\nSummary:\n{curated_summary}\n"
            "Respond as numbered question + answer pairs."
        )
        response = self.call_llm(self.build_prompt(user_prompt))
        quizzes = self._parse_questions(response)
        if not quizzes:
            quizzes = [
                {
                    "question": "Explain today's concept in your own words.",
                    "answer": curated_summary[:150],
                    "difficulty": "medium",
                }
            ]
        return quizzes[:num_questions]

    @staticmethod
    def _parse_questions(raw: str) -> list[dict[str, str]]:
        entries: list[dict[str, str]] = []
        pattern = re.compile(r"^(\d+)[\.)\-]*\s*(.+?)\s*Answer[:\-]\s*(.+)$", re.IGNORECASE)
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            match = pattern.match(line)
            if match:
                entries.append(
                    {
                        "question": match.group(2).strip(),
                        "answer": match.group(3).strip(),
                        "difficulty": "medium",
                    }
                )
        return entries

    @staticmethod
    def evaluate_answer(expected: str, given: str) -> tuple[bool, str]:
        """Approximate scoring by overlap heuristics."""

        expected_tokens = set(expected.lower().split())
        given_tokens = set(given.lower().split())
        overlap = expected_tokens & given_tokens
        score = len(overlap) / max(1, len(expected_tokens))
        is_correct = score >= 0.4
        feedback = (
            "Great job!" if is_correct else f"Focus on covering: {' '.join(sorted(expected_tokens - given_tokens))[:120]}"
        )
        return is_correct, feedback


__all__ = ["TutorAgent"]
