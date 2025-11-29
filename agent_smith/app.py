"""FastAPI application exposing the Agent Smith orchestrator."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from .db import init_db
from .models import Episode, LearningGoal, PlanItem, QuizItem
from .orchestrator import AgentOrchestrator

app = FastAPI(title='Agent Smith', version='0.1.0')
orchestrator = AgentOrchestrator()


class GoalRequest(BaseModel):
    title: str
    description: str | None = None
    learner_profile: str | None = None
    target_days: int | None = None


class AnswerRequest(BaseModel):
    answer: str


@app.on_event('startup')
def on_startup() -> None:
    init_db()


@app.get('/health')
async def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.get('/', response_class=PlainTextResponse)
async def root_banner() -> str:
    """Friendly splash screen when visiting the service root."""

    return '+-----------------------------------+\n| I am a FastAPI webpage...so Fast! |\n+-----------------------------------+\n'


@app.post('/goals', response_model=LearningGoal)
async def create_goal(payload: GoalRequest) -> LearningGoal:
    return orchestrator.create_goal(
        title=payload.title,
        description=payload.description,
        learner_profile=payload.learner_profile,
        target_days=payload.target_days,
    )


@app.get('/goals/{goal_id}', response_model=LearningGoal)
async def get_goal(goal_id: int) -> LearningGoal:
    try:
        return orchestrator.get_goal(goal_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get('/goals/{goal_id}/plan', response_model=list[PlanItem])
async def get_plan(goal_id: int, day: int | None = None) -> list[PlanItem]:
    return orchestrator.get_plan(goal_id, day_number=day)


@app.post('/goals/{goal_id}/run/{day}', response_model=Episode)
async def run_day(goal_id: int, day: int) -> Episode:
    try:
        return orchestrator.run_day(goal_id, day)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get('/goals/{goal_id}/quiz/{day}', response_model=list[QuizItem])
async def get_quiz(goal_id: int, day: int) -> list[QuizItem]:
    return orchestrator.get_quiz_for_day(goal_id, day)


@app.post('/quiz/{quiz_id}/answer', response_model=QuizItem)
async def submit_answer(quiz_id: int, payload: AnswerRequest) -> QuizItem:
    try:
        return orchestrator.submit_quiz_answer(quiz_id, payload.answer)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
