"""
models.py
=========
Pydantic data models for the Customer Support OpenEnv environment.

Three core types every OpenEnv must define:
  - Observation  (what the agent sees after every step)
  - Action       (what the agent sends to step())
  - Reward       (what the agent receives back)
"""

from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# ACTION
# ─────────────────────────────────────────────

class Action(BaseModel):
    """What the agent can do. Six action types, tightly typed."""

    action_type: Literal[
        "classify",
        "lookup_order",
        "lookup_customer",
        "respond",
        "close",
        "escalate",
    ]

    # classify
    category: Optional[Literal["shipping", "billing", "technical", "general"]] = None

    # lookup_order
    order_id: Optional[str] = None

    # lookup_customer
    customer_id: Optional[str] = None

    # respond / close / escalate
    content: Optional[str] = None

    # close only
    resolution: Optional[Literal[
        "send_tracking_update",
        "initiate_refund",
        "retry_payment_or_escalate",
        "send_troubleshooting_steps",
        "send_policy_info",
        "no_action_needed",
    ]] = None


# ─────────────────────────────────────────────
# OBSERVATION
# ─────────────────────────────────────────────

class OrderRecord(BaseModel):
    order_id: str
    customer_id: str
    customer_name: str
    status: str
    carrier: Optional[str] = None
    tracking_number: Optional[str] = None
    delay_days: Optional[int] = None
    refund_age_days: Optional[int] = None
    payment_attempts: Optional[int] = None
    amount: Optional[float] = None


class CustomerRecord(BaseModel):
    customer_id: str
    name: str
    email: str
    tier: Literal["standard", "premium", "enterprise"]
    open_tickets: int
    account_age_days: int


class Observation(BaseModel):
    """Everything the agent can see at any point in the episode."""

    ticket_id: str
    ticket_text: str
    ticket_type: str          # hidden from agent in hard task

    history: list[dict[str, Any]] = Field(default_factory=list)
    status: Literal["open", "in_progress", "escalated", "closed"] = "open"

    order_record: Optional[OrderRecord] = None
    customer_record: Optional[CustomerRecord] = None

    step_count: int = 0
    max_steps: int = 10
    task_id: str

    instructions: str = (
        "You are a customer support agent. "
        "Classify the ticket, look up relevant data, respond helpfully, "
        "then close or escalate with the correct resolution."
    )


# ─────────────────────────────────────────────
# REWARD
# ─────────────────────────────────────────────

class Reward(BaseModel):
    """Partial reward breakdown — signal at every step, not just the end."""

    value: float = Field(..., ge=-1.0, le=1.0)
    partial_scores: dict[str, float] = Field(default_factory=dict)
    done: bool = False
    info: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# EPISODE RESULT  (returned by /grader)
# ─────────────────────────────────────────────

class EpisodeResult(BaseModel):
    task_id: str
    ticket_id: str
    final_score: float = Field(..., ge=0.0, le=1.0)
    component_scores: dict[str, float]
    steps_taken: int
    outcome: Literal["resolved", "escalated", "timeout", "failed"]
    feedback: str