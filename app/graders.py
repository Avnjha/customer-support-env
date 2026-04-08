"""
graders.py
==========
Three fully deterministic graders — no LLM, no fuzzy matching.
The same episode state always returns the same score.

TASK SCORES:
  easy   → 1 component  → max 1.0
  medium → 4 components → max 1.0  (0.25 each)
  hard   → 6 components → max 1.0  (with loop penalty)
"""

from __future__ import annotations
from typing import Any
from knowledge_base import get_policy, should_escalate


# ─────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────

def _actions_of_type(history: list[dict], action_type: str) -> list[dict]:
    return [h for h in history if h.get("action_type") == action_type]


def _classify_correct(history: list[dict], correct_category: str) -> bool:
    actions = _actions_of_type(history, "classify")
    if not actions:
        return False
    return actions[-1].get("category") == correct_category


def _lookup_performed(history: list[dict], required_lookup: str | None) -> bool:
    if required_lookup is None:
        return True
    return any(h.get("action_type") == required_lookup for h in history)


def _keyword_ratio(history: list[dict], keywords: list[str]) -> float:
    actions = _actions_of_type(history, "respond")
    if not actions:
        return 0.0
    text = (actions[-1].get("content") or "").lower()
    hits = sum(1 for kw in keywords if kw.lower() in text)
    return round(hits / len(keywords), 2) if keywords else 1.0


def _close_resolution(history: list[dict]) -> str | None:
    actions = _actions_of_type(history, "close")
    return actions[-1].get("resolution") if actions else None


def _agent_escalated(history: list[dict]) -> bool:
    return bool(_actions_of_type(history, "escalate"))


# ─────────────────────────────────────────────
# TASK 1 — EASY
# ─────────────────────────────────────────────

def grade_easy(state: dict[str, Any]) -> dict[str, Any]:
    """
    Score: 1.0 if classified correctly, 0.0 otherwise.
    Binary and intentional — easy task only tests classification.
    """
    policy  = get_policy(state["ticket_type"])
    history = state.get("history", [])
    correct = _classify_correct(history, policy["classify_as"])

    score = 1.0 if correct else 0.0
    return {
        "final_score": score,
        "component_scores": {"classify_correct": score},
        "feedback": (
            f"Correct — classified as '{policy['classify_as']}'."
            if correct else
            f"Wrong classification. Expected '{policy['classify_as']}', "
            f"got '{(_actions_of_type(history,'classify') or [{}])[-1].get('category','none')}'."
        ),
        "outcome": "resolved" if correct else "failed",
    }


# ─────────────────────────────────────────────
# TASK 2 — MEDIUM
# ─────────────────────────────────────────────

def grade_medium(state: dict[str, Any]) -> dict[str, Any]:
    """
    Four components, 0.25 each:
      1. Classified correctly
      2. Performed required lookup
      3. Response keyword coverage  (partial credit)
      4. Correct escalation decision
    """
    policy          = get_policy(state["ticket_type"])
    history         = state.get("history", [])
    order_record    = state.get("order_record") or {}
    customer_record = state.get("customer_record") or {}
    combined        = {**order_record, **customer_record}

    scores   = {}
    feedback = []

    # 1. Classify
    if _classify_correct(history, policy["classify_as"]):
        scores["classify_correct"] = 0.25
        feedback.append("Classified correctly.")
    else:
        scores["classify_correct"] = 0.0
        feedback.append(f"Wrong classification — expected '{policy['classify_as']}'.")

    # 2. Lookup
    if _lookup_performed(history, policy["required_lookup"]):
        scores["lookup_performed"] = 0.25
        feedback.append("Required lookup performed.")
    else:
        scores["lookup_performed"] = 0.0
        feedback.append(f"Missing lookup '{policy['required_lookup']}'.")

    # 3. Response keywords (partial)
    ratio = _keyword_ratio(history, policy["response_keywords"])
    scores["response_quality"] = round(0.25 * ratio, 3)
    feedback.append(f"Response keywords: {ratio:.0%}.")

    # 4. Escalation decision
    needs_escalation  = should_escalate(state["ticket_type"], combined)
    agent_escalated   = _agent_escalated(history)
    escalation_ok     = needs_escalation == agent_escalated
    scores["escalation_correct"] = 0.25 if escalation_ok else 0.0
    feedback.append(
        "Escalation decision correct." if escalation_ok else
        ("Should have escalated." if needs_escalation else "Unnecessary escalation.")
    )

    final   = round(sum(scores.values()), 3)
    outcome = "resolved" if final >= 0.75 else ("escalated" if agent_escalated else "failed")

    return {
        "final_score": final,
        "component_scores": scores,
        "feedback": " ".join(feedback),
        "outcome": outcome,
    }


# ─────────────────────────────────────────────
# TASK 3 — HARD
# ─────────────────────────────────────────────

def grade_hard(state: dict[str, Any]) -> dict[str, Any]:
    """
    Six components:
      1. Classify correct          0.20
      2. Lookup performed          0.15
      3. Response keywords         0.20
      4. Escalation decision       0.20
      5. Correct resolution code   0.15
      6. Efficiency bonus (≤4 steps) 0.10
      Loop penalty: -0.05 per repeated action (capped at -0.20)
    """
    policy          = get_policy(state["ticket_type"])
    history         = state.get("history", [])
    order_record    = state.get("order_record") or {}
    customer_record = state.get("customer_record") or {}
    combined        = {**order_record, **customer_record}
    step_count      = state.get("step_count", len(history))

    scores   = {}
    feedback = []

    # 1. Classify (0.20)
    if _classify_correct(history, policy["classify_as"]):
        scores["classify_correct"] = 0.20
        feedback.append("Classified correctly.")
    else:
        scores["classify_correct"] = 0.0
        feedback.append(f"Wrong classification — expected '{policy['classify_as']}'.")

    # 2. Lookup (0.15)
    if _lookup_performed(history, policy["required_lookup"]):
        scores["lookup_performed"] = 0.15
        feedback.append("Lookup performed.")
    else:
        scores["lookup_performed"] = 0.0
        feedback.append(f"Missing lookup '{policy['required_lookup']}'.")

    # 3. Response keywords (0.20)
    ratio = _keyword_ratio(history, policy["response_keywords"])
    scores["response_quality"] = round(0.20 * ratio, 3)
    feedback.append(f"Response keywords: {ratio:.0%}.")

    # 4. Escalation (0.20)
    needs_escalation = should_escalate(state["ticket_type"], combined)
    agent_escalated  = _agent_escalated(history)
    escalation_ok    = needs_escalation == agent_escalated
    scores["escalation_correct"] = 0.20 if escalation_ok else 0.0
    feedback.append("Escalation correct." if escalation_ok else "Escalation wrong.")

    # 5. Resolution code (0.15)
    expected_resolution = policy["correct_resolution"]
    agent_resolution    = _close_resolution(history)
    resolution_ok       = agent_resolution == expected_resolution
    scores["resolution_correct"] = 0.15 if resolution_ok else 0.0
    feedback.append(
        f"Correct resolution '{expected_resolution}'." if resolution_ok else
        f"Wrong resolution — expected '{expected_resolution}', got '{agent_resolution}'."
    )

    # 6. Efficiency bonus (0.10)
    is_done = state.get("status") in ("closed", "escalated")
    if is_done and step_count <= 4:
        scores["efficiency_bonus"] = 0.10
        feedback.append(f"Efficiency bonus: resolved in {step_count} steps.")
    else:
        scores["efficiency_bonus"] = 0.0
        feedback.append(f"No efficiency bonus ({step_count} steps).")

    # Loop penalty
    seen      = set()
    loop_hits = 0
    for h in history:
        fp = f"{h.get('action_type')}|{h.get('content','')}|{h.get('category','')}"
        if fp in seen:
            loop_hits += 1
        seen.add(fp)

    penalty = min(0.20, loop_hits * 0.05)
    scores["loop_penalty"] = -penalty
    if penalty > 0:
        feedback.append(f"Loop penalty: -{penalty:.2f}.")

    final   = max(0.0, round(sum(scores.values()), 3))
    outcome = (
        "resolved"  if state.get("status") == "closed" else
        "escalated" if agent_escalated else
        "timeout"   if step_count >= state.get("max_steps", 10) else
        "failed"
    )

    return {
        "final_score": final,
        "component_scores": scores,
        "feedback": " ".join(feedback),
        "outcome": outcome,
    }


# ─────────────────────────────────────────────
# UNIFIED ENTRY POINT
# ─────────────────────────────────────────────

def grade_episode(task_id: str, state: dict[str, Any]) -> dict[str, Any]:
    """Route to the correct grader. Used by /grader endpoint."""
    graders = {"easy": grade_easy, "medium": grade_medium, "hard": grade_hard}
    if task_id not in graders:
        raise ValueError(f"Unknown task_id '{task_id}'.")
    result = graders[task_id](state)
    result["task_id"]     = task_id
    result["ticket_id"]   = state.get("ticket_id", "unknown")
    result["steps_taken"] = state.get("step_count", 0)
    return result