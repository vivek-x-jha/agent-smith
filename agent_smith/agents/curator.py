"""Curator agent that ranks and summarizes candidate resources."""

from __future__ import annotations

from typing import Iterable

from ..models import LearningGoal, PlanItem, Resource
from .base import Agent


class CuratorAgent(Agent):
    """Evaluates fetched resources and produces a concise study brief."""

    def __init__(self, llm=None) -> None:
        super().__init__(
            name="curator",
            system_prompt=(
                "You read resource snippets and summarize the most actionable study plan. "
                "Return highlights emphasizing why they help the learner."
            ),
            llm=llm,
        )

    def run(
        self,
        goal: LearningGoal,
        plan_items: Iterable[PlanItem],
        resources: Iterable[Resource],
    ) -> dict[str, object]:
        plan_summary = "\n".join(f"- {item.task}" for item in plan_items)
        resource_lines = [
            f"- {res.title}: {res.snippet or res.content or ''}" for res in resources
        ]
        if not resource_lines:
            resource_lines = ["No resources were found. Recommend next steps."]
        user_prompt = (
            f"Goal: {goal.title}\nPlan:\n{plan_summary}\n\nResources:\n"
            + "\n".join(resource_lines[:8])
            + "\n\nProvide a summary and recommended shortlist."
        )
        response = self.call_llm(self.build_prompt(user_prompt))
        scored_resources = self._score_resources(resources)
        return {"summary": response.strip(), "resources": scored_resources}

    @staticmethod
    def _score_resources(resources: Iterable[Resource]) -> list[Resource]:
        scored: list[Resource] = []
        for idx, resource in enumerate(resources):
            resource.relevance_score = max(0.1, 1.0 - (idx * 0.1))
            scored.append(resource)
        return scored


__all__ = ["CuratorAgent"]
