"""Coordinates agent pipeline executions and persists results."""

from __future__ import annotations

from datetime import datetime
from sqlmodel import select

from .agents.curator import CuratorAgent
from .agents.planner import PlannerAgent
from .agents.researcher import ResearcherAgent
from .agents.tutor import TutorAgent
from .db import get_session
from .logging_config import get_logger
from .models import (
    Episode,
    GoalStatus,
    LearningGoal,
    PlanItem,
    PlanStatus,
    QuizItem,
    QuizStatus,
)

logger = get_logger(__name__)


class AgentOrchestrator:
    """High-level façade used by the FastAPI layer."""

    def __init__(self) -> None:
        self.planner = PlannerAgent()
        self.researcher = ResearcherAgent()
        self.curator = CuratorAgent()
        self.tutor = TutorAgent()

    # Goal helpers ---------------------------------------------------------
    def create_goal(
        self,
        title: str,
        description: str | None = None,
        learner_profile: str | None = None,
        target_days: int | None = None,
    ) -> LearningGoal:
        with get_session() as session:
            goal = LearningGoal(
                title=title,
                description=description,
                learner_profile=learner_profile,
                target_days=target_days,
                status=GoalStatus.NEW,
            )
            session.add(goal)
            session.commit()
            session.refresh(goal)
            return goal

    def get_goal(self, goal_id: int) -> LearningGoal:
        with get_session() as session:
            goal = session.get(LearningGoal, goal_id)
            if goal is None:
                raise ValueError(f"Goal {goal_id} not found")
            return goal

    def get_plan(self, goal_id: int, day_number: int | None = None) -> list[PlanItem]:
        with get_session() as session:
            statement = select(PlanItem).where(PlanItem.goal_id == goal_id).order_by(PlanItem.day_number, PlanItem.sequence)
            if day_number is not None:
                statement = statement.where(PlanItem.day_number == day_number)
            return list(session.exec(statement))

    # Pipeline -------------------------------------------------------------
    def run_day(self, goal_id: int, day_number: int) -> Episode:
        logger.info("run_day", goal_id=goal_id, day_number=day_number)
        with get_session() as session:
            goal = session.get(LearningGoal, goal_id)
            if goal is None:
                raise ValueError(f"Goal {goal_id} not found")

            previous_items = list(
                session.exec(
                    select(PlanItem).where(PlanItem.goal_id == goal_id, PlanItem.day_number == day_number)
                )
            )
            plan_payloads = self.planner.run(goal, day_number, previous_items=previous_items)
            plan_items = [
                PlanItem(
                    goal_id=goal_id,
                    day_number=day_number,
                    sequence=payload["sequence"],
                    task=str(payload["task"]),
                    notes=str(payload.get("notes", "")),
                    status=PlanStatus.PENDING,
                )
                for payload in plan_payloads
            ]
            session.add_all(plan_items)
            goal.status = GoalStatus.ACTIVE
            goal.updated_at = datetime.utcnow()
            session.add(goal)
            session.commit()
            for item in plan_items:
                session.refresh(item)
            logger.info("planner_completed", plan_count=len(plan_items))

            resources = self.researcher.run(goal, plan_items)
            session.add_all(resources)
            session.commit()
            for resource in resources:
                session.refresh(resource)
            logger.info("researcher_completed", resource_count=len(resources))

            curation = self.curator.run(goal, plan_items, resources)
            curated_resources = curation["resources"]
            session.add_all(curated_resources)
            session.commit()
            logger.info("curator_completed", summary_length=len(str(curation["summary"])))

            quiz_payloads = self.tutor.run(goal, plan_items, str(curation["summary"]))
            quiz_items = [
                QuizItem(
                    goal_id=goal_id,
                    day_number=day_number,
                    question=payload["question"],
                    answer=payload["answer"],
                    difficulty=payload.get("difficulty"),
                    status=QuizStatus.DELIVERED,
                )
                for payload in quiz_payloads
            ]
            session.add_all(quiz_items)
            session.commit()
            for quiz in quiz_items:
                session.refresh(quiz)
            logger.info("tutor_completed", quiz_count=len(quiz_items))

            reflection = self._generate_reflection(goal, plan_items, str(curation["summary"]))
            self._rewrite_future_plan_items(session, goal_id, day_number, reflection)
            logger.info("reflection_generated", excerpt=reflection[:120])

            episode = Episode(
                goal_id=goal_id,
                day_number=day_number,
                planner_summary="\n".join(f"{item.sequence}. {item.task}" for item in plan_items),
                researcher_summary=f"Curated {len(resources)} resources",
                curator_summary=str(curation["summary"]),
                tutor_summary="; ".join(q.question for q in quiz_items),
                reflection=reflection,
            )
            session.add(episode)
            session.commit()
            session.refresh(episode)
            return episode

    def get_quiz_for_day(self, goal_id: int, day_number: int) -> list[QuizItem]:
        with get_session() as session:
            statement = select(QuizItem).where(QuizItem.goal_id == goal_id, QuizItem.day_number == day_number)
            return list(session.exec(statement))

    def submit_quiz_answer(self, quiz_id: int, answer: str) -> QuizItem:
        with get_session() as session:
            quiz = session.get(QuizItem, quiz_id)
            if quiz is None:
                raise ValueError(f"Quiz item {quiz_id} not found")
            quiz.learner_answer = answer
            is_correct, feedback = TutorAgent.evaluate_answer(quiz.answer, answer)
            quiz.is_correct = is_correct
            quiz.feedback = feedback
            quiz.status = QuizStatus.ANSWERED
            session.add(quiz)
            session.commit()
            session.refresh(quiz)
            return quiz

    def _generate_reflection(
        self,
        goal: LearningGoal,
        plan_items: list[PlanItem],
        curated_summary: str,
    ) -> str:
        """Use the planner LLM prompt to produce a reflection."""

        user_prompt = (
            f"Goal: {goal.title}\n"
            f"Plan items completed/queued:\n"
            + "\n".join(f"- {item.task}" for item in plan_items)
            + "\nSummary:\n"
            + curated_summary
            + "\nProvide a reflection with two sentences: what worked + what to adjust."
        )
        return self.planner.call_llm(self.planner.build_prompt(user_prompt)).strip()

    def _rewrite_future_plan_items(
        self,
        session,
        goal_id: int,
        day_number: int,
        reflection: str,
    ) -> None:
        """Append reflection hints to future plan items."""

        future_items = list(
            session.exec(
                select(PlanItem).where(PlanItem.goal_id == goal_id, PlanItem.day_number > day_number)
            )
        )
        if not future_items or not reflection:
            return

        first_sentence = reflection.split(".")[0].strip()
        for item in future_items:
            if first_sentence:
                item.task = f"{item.task} (Focus: {first_sentence[:60]})"
            note = f"Reflection applied on day {day_number}: {reflection[:200]}"
            item.notes = f"{(item.notes or '').strip()}\n{note}".strip()
            session.add(item)
        session.commit()


__all__ = ["AgentOrchestrator"]
