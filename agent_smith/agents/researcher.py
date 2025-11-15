"""Researcher agent that fetches resources from the open web."""

from __future__ import annotations

from typing import Iterable

from ..models import LearningGoal, PlanItem, Resource
from ..tools.vector import upsert_resources
from ..tools.web import SearchResult, arxiv_search, duckduckgo_search, wikipedia_search
from .base import Agent


class ResearcherAgent(Agent):
    """Turns plan items into search queries and persists curated resources."""

    def __init__(self, llm=None) -> None:
        super().__init__(
            name="researcher",
            system_prompt="Find practical, current resources that align with the learner's plan.",
            llm=llm,
        )

    def run(self, goal: LearningGoal, plan_items: Iterable[PlanItem]) -> list[Resource]:
        if goal.id is None:
            raise ValueError("Goal must be persisted before running researcher agent")

        resources: list[Resource] = []
        for item in plan_items:
            goal_id = item.goal_id or goal.id
            query = f"{goal.title} {item.task}".strip()
            hits: list[SearchResult] = []
            hits.extend(duckduckgo_search(query, max_results=3))
            hits.extend(wikipedia_search(goal.title))
            if any(keyword in item.task.lower() for keyword in ("paper", "theory", "research")):
                hits.extend(arxiv_search(goal.title, max_results=1))

            for hit in hits:
                resources.append(
                    Resource(
                        goal_id=goal_id,
                        plan_item_id=item.id,
                        title=hit.title,
                        url=hit.url,
                        snippet=hit.snippet,
                        content=hit.snippet,
                        source=hit.source,
                    )
                )

        upsert_resources(resources)
        return resources


__all__ = ["ResearcherAgent"]
