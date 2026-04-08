"""
main.py
=======
FastAPI application — all OpenEnv required + bonus endpoints.

Required:
  POST /reset
  POST /step
  GET  /state
  GET  /health

Bonus (hackathon spec):
  GET  /tasks
  POST /grader
  POST /baseline
"""

from __future__ import annotations
import re
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import Action, Observation, Reward
from environment import CustomerSupportEnvironment
from graders import grade_episode
from knowledge_base import get_tickets_for_task


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

app = FastAPI(
    title="Customer Support OpenEnv",
    description="AI agent handles support tickets — classify, lookup, respond, resolve.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = CustomerSupportEnvironment()


# ─────────────────────────────────────────────
# REQUEST SCHEMAS
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "easy"
    ticket_index: int | None = None


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any]


class GraderRequest(BaseModel):
    task_id: str
    state: dict[str, Any]


# ─────────────────────────────────────────────
# REQUIRED ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "environment": "customer-support-v1"}


@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest):
    if request.task_id not in ("easy", "medium", "hard"):
        raise HTTPException(400, detail=f"task_id must be easy/medium/hard. Got: '{request.task_id}'")
    return env.reset(task_id=request.task_id, ticket_index=request.ticket_index)


@app.post("/step", response_model=StepResponse)
def step(action: Action):
    try:
        obs, reward = env.step(action)
    except RuntimeError as e:
        raise HTTPException(400, detail=str(e))
    return StepResponse(observation=obs, reward=reward, done=reward.done, info=reward.info)


@app.get("/state")
def state():
    return env.state()


# ─────────────────────────────────────────────
# BONUS ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {
                "task_id": "easy",
                "description": "Classify the support ticket into the correct category.",
                "difficulty": "easy",
                "max_steps": 10,
                "tickets": [t["ticket_id"] for t in get_tickets_for_task("easy")],
                "grading": {"classify_correct": 1.0},
            },
            {
                "task_id": "medium",
                "description": "Classify, lookup data, respond correctly, make correct escalation decision.",
                "difficulty": "medium",
                "max_steps": 10,
                "tickets": [t["ticket_id"] for t in get_tickets_for_task("medium")],
                "grading": {
                    "classify_correct": 0.25,
                    "lookup_performed": 0.25,
                    "response_quality": 0.25,
                    "escalation_correct": 0.25,
                },
            },
            {
                "task_id": "hard",
                "description": "Full pipeline — ticket type hidden. Classify, lookup, respond, escalate or close correctly.",
                "difficulty": "hard",
                "max_steps": 10,
                "tickets": [t["ticket_id"] for t in get_tickets_for_task("hard")],
                "grading": {
                    "classify_correct": 0.20,
                    "lookup_performed": 0.15,
                    "response_quality": 0.20,
                    "escalation_correct": 0.20,
                    "resolution_correct": 0.15,
                    "efficiency_bonus": 0.10,
                    "loop_penalty": "up to -0.20",
                },
            },
        ],
        "action_schema": {
            "action_type": {
                "required": True,
                "type": "string",
                "options": ["classify", "lookup_order", "lookup_customer", "respond", "close", "escalate"],
            },
            "category": {
                "required_when": "action_type == 'classify'",
                "options": ["shipping", "billing", "technical", "general"],
            },
            "order_id":    {"required_when": "action_type == 'lookup_order'",    "example": "ORD-001"},
            "customer_id": {"required_when": "action_type == 'lookup_customer'", "example": "CUST-A1"},
            "content":     {"required_when": "action_type in [respond, close, escalate]"},
            "resolution": {
                "required_when": "action_type == 'close'",
                "options": [
                    "send_tracking_update", "initiate_refund",
                    "retry_payment_or_escalate", "send_troubleshooting_steps",
                    "send_policy_info", "no_action_needed",
                ],
            },
        },
    }


@app.post("/grader")
def grader(request: GraderRequest):
    try:
        return grade_episode(request.task_id, request.state)
    except (KeyError, ValueError) as e:
        raise HTTPException(400, detail=str(e))


@app.post("/baseline")
def baseline():
    """
    Deterministic rule-based baseline — no API key needed.
    Judges use this to verify scores without running the LLM script.
    """
    results = []
    for task_id in ["easy", "medium", "hard"]:
        tickets     = get_tickets_for_task(task_id)
        task_scores = []
        for i, ticket in enumerate(tickets):
            obs = env.reset(task_id=task_id, ticket_index=i)
            _rule_agent(task_id, obs)
            grade = grade_episode(task_id, env.state())
            task_scores.append(grade["final_score"])
        avg = round(sum(task_scores) / len(task_scores), 3)
        results.append({"task_id": task_id, "avg_score": avg, "ticket_scores": task_scores})
    return {"agent": "rule-based-baseline", "results": results}


# ─────────────────────────────────────────────
# RULE-BASED BASELINE AGENT
# ─────────────────────────────────────────────

def _rule_agent(task_id: str, obs: Observation):
    """Simple keyword agent used by /baseline endpoint."""
    text = obs.ticket_text.lower()

    # 1. Classify
    if any(w in text for w in ["delayed", "arrived", "tracking", "shipping"]):
        cat = "shipping"
    elif any(w in text for w in ["refund", "payment", "billing", "charge"]):
        cat = "billing"
    elif any(w in text for w in ["crash", "app", "bug", "not working"]):
        cat = "technical"
    else:
        cat = "general"
    try:
        env.step(Action(action_type="classify", category=cat))
    except RuntimeError:
        return

    # 2. Lookup
    raw = obs.ticket_text
    order_match = re.search(r"ORD-\d+", raw)
    cust_match  = re.search(r"CUST-\w+", raw)
    try:
        if order_match:
            env.step(Action(action_type="lookup_order", order_id=order_match.group()))
        elif cust_match:
            env.step(Action(action_type="lookup_customer", customer_id=cust_match.group()))
    except RuntimeError:
        return

    # 3. Respond
    templates = {
        "shipping":  "We apologize for the delay. Your tracking information has been updated.",
        "billing":   "We are sorry. Your refund is processed and will arrive in 3-5 business days. Please retry your payment and contact your bank if needed.",
        "technical": "Please restart the app and update to the latest version.",
        "general":   "We are happy to help. Please let us know what you need.",
    }
    try:
        env.step(Action(action_type="respond", content=templates[cat]))
    except RuntimeError:
        return

    # 4. Close
    resolutions = {
        "shipping":  "send_tracking_update",
        "billing":   "initiate_refund",
        "technical": "send_troubleshooting_steps",
        "general":   "send_policy_info",
    }
    try:
        env.step(Action(action_type="close", content="Resolved.", resolution=resolutions[cat]))
    except RuntimeError:
        return