"""SQLModel ORM definitions for Agent Smith."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Optional

from sqlmodel import Field, Relationship, SQLModel


class GoalStatus(StrEnum):
    """Lifecycle states for a learning goal."""

    NEW = "new"
    PLANNING = "planning"
    ACTIVE = "active"
    COMPLETE = "complete"


class PlanStatus(StrEnum):
    """Execution states for individual plan items."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    BLOCKED = "blocked"


class QuizStatus(StrEnum):
    """Possible states for a quiz item."""

    DRAFT = "draft"
    DELIVERED = "delivered"
    ANSWERED = "answered"


class LearningGoal(SQLModel, table=True):
    """Primary entity describing what a learner wants to achieve."""

    __tablename__ = "learning_goals"

    id: Optional[int] = Field(default=None, primary_key=True)
    title: str = Field(index=True, nullable=False)
    description: Optional[str] = None
    learner_profile: Optional[str] = None
    status: GoalStatus = Field(default=GoalStatus.NEW)
    target_days: Optional[int] = Field(default=None, ge=1)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

    plan_items: list["PlanItem"] = Relationship(back_populates="goal")
    resources: list["Resource"] = Relationship(back_populates="goal")
    quiz_items: list["QuizItem"] = Relationship(back_populates="goal")
    episodes: list["Episode"] = Relationship(back_populates="goal")


class PlanItem(SQLModel, table=True):
    """Represents a single actionable task produced by the planner."""

    __tablename__ = "plan_items"

    id: Optional[int] = Field(default=None, primary_key=True)
    goal_id: int = Field(foreign_key="learning_goals.id")
    day_number: int = Field(default=1, ge=1, index=True)
    sequence: int = Field(default=1, ge=1)
    task: str
    status: PlanStatus = Field(default=PlanStatus.PENDING)
    notes: Optional[str] = None
    reflection: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

    goal: LearningGoal = Relationship(back_populates="plan_items")
    resources: list["Resource"] = Relationship(back_populates="plan_item")


class Resource(SQLModel, table=True):
    """External reference material curated for a plan item."""

    __tablename__ = "resources"

    id: Optional[int] = Field(default=None, primary_key=True)
    goal_id: int = Field(foreign_key="learning_goals.id")
    plan_item_id: Optional[int] = Field(default=None, foreign_key="plan_items.id")
    title: str
    url: Optional[str] = None
    snippet: Optional[str] = None
    content: Optional[str] = None
    source: Optional[str] = None
    vector_id: Optional[str] = Field(default=None, index=True)
    relevance_score: Optional[float] = Field(default=None, ge=0)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

    goal: LearningGoal = Relationship(back_populates="resources")
    plan_item: Optional[PlanItem] = Relationship(back_populates="resources")


class QuizItem(SQLModel, table=True):
    """Generated formative assessment linked to a learning goal/day."""

    __tablename__ = "quiz_items"

    id: Optional[int] = Field(default=None, primary_key=True)
    goal_id: int = Field(foreign_key="learning_goals.id")
    day_number: int = Field(default=1, ge=1)
    question: str
    answer: str
    difficulty: Optional[str] = None
    learner_answer: Optional[str] = None
    is_correct: Optional[bool] = None
    feedback: Optional[str] = None
    status: QuizStatus = Field(default=QuizStatus.DRAFT)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

    goal: LearningGoal = Relationship(back_populates="quiz_items")


class Episode(SQLModel, table=True):
    """Daily orchestration run storing multi-agent summaries and reflections."""

    __tablename__ = "episodes"

    id: Optional[int] = Field(default=None, primary_key=True)
    goal_id: int = Field(foreign_key="learning_goals.id")
    day_number: int = Field(default=1, ge=1, index=True)
    planner_summary: Optional[str] = None
    researcher_summary: Optional[str] = None
    curator_summary: Optional[str] = None
    tutor_summary: Optional[str] = None
    reflection: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

    goal: LearningGoal = Relationship(back_populates="episodes")


__all__ = [
    "Episode",
    "GoalStatus",
    "LearningGoal",
    "PlanItem",
    "PlanStatus",
    "QuizItem",
    "QuizStatus",
    "Resource",
]
