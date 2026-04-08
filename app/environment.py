"""
environment.py
==============
Core episode logic.

  reset(task_id)  →  Observation
  step(action)    →  (Observation, Reward)
  state()         →  dict

One episode = reset() + N × step() until done=True.
"""

from __future__ import annotations
import random
from typing import Any

from models import Action, Observation, Reward, OrderRecord, CustomerRecord
from knowledge_base import (
    get_policy,
    get_tickets_for_task,
    lookup_order,
    lookup_customer,
    should_escalate,
)


# ─────────────────────────────────────────────
# REWARD WEIGHTS  (per-step signals)
# ─────────────────────────────────────────────

REWARDS = {
    "classify_correct":    +0.30,
    "classify_wrong":      -0.10,
    "lookup_bonus":        +0.10,
    "respond_per_keyword": +0.10,   # max 0.50 total
    "close_correct":       +0.20,
    "close_wrong":          0.00,
    "escalate_correct":    +0.20,
    "escalate_wrong":      -0.10,
    "repeated_action":     -0.10,
}

MAX_STEPS = 10


class CustomerSupportEnvironment:

    def __init__(self):
        self._state: dict[str, Any] = {}
        self._active = False

    # ─────────────────────────────────────────
    # reset()
    # ─────────────────────────────────────────

    def reset(self, task_id: str = "easy", ticket_index: int | None = None) -> Observation:
        tickets = get_tickets_for_task(task_id)
        ticket  = tickets[ticket_index % len(tickets)] if ticket_index is not None \
                  else random.choice(tickets)

        self._state = {
            "ticket_id":       ticket["ticket_id"],
            "ticket_text":     ticket["ticket_text"],
            "ticket_type":     ticket["ticket_type"],
            "hint_order_id":   ticket.get("hint_order_id"),
            "hint_customer_id":ticket.get("hint_customer_id"),
            "task_id":         task_id,
            "status":          "open",
            "history":         [],
            "order_record":    None,
            "customer_record": None,
            "step_count":      0,
            "max_steps":       MAX_STEPS,
        }
        self._active = True
        return self._obs()

    # ─────────────────────────────────────────
    # step()
    # ─────────────────────────────────────────

    def step(self, action: Action) -> tuple[Observation, Reward]:
        if not self._active:
            raise RuntimeError("Call reset() before step().")

        s       = self._state
        history = s["history"]
        policy  = get_policy(s["ticket_type"])

        s["step_count"] += 1
        done            = False
        partial: dict[str, float] = {}
        reward_val      = 0.0
        info: dict[str, Any] = {"action_type": action.action_type}

        # fingerprint for loop detection
        fp = f"{action.action_type}|{action.content or ''}|{action.category or ''}|{action.resolution or ''}"
        if any(h.get("_fp") == fp for h in history):
            reward_val += REWARDS["repeated_action"]
            partial["repeated_action"] = REWARDS["repeated_action"]
            info["warning"] = "Repeated action — penalty applied."

        # ── classify ──────────────────────────
        if action.action_type == "classify":
            if action.category == policy["classify_as"]:
                reward_val += REWARDS["classify_correct"]
                partial["classify_correct"] = REWARDS["classify_correct"]
                info["result"] = f"Correct: '{action.category}'"
            else:
                reward_val += REWARDS["classify_wrong"]
                partial["classify_wrong"] = REWARDS["classify_wrong"]
                info["result"] = f"Wrong: '{action.category}' (expected '{policy['classify_as']}')"

        # ── lookup_order ──────────────────────
        elif action.action_type == "lookup_order":
            oid    = action.order_id or s.get("hint_order_id")
            record = lookup_order(oid) if oid else None
            if record:
                s["order_record"] = record
                reward_val += REWARDS["lookup_bonus"]
                partial["lookup_bonus"] = REWARDS["lookup_bonus"]
                info["result"]       = f"Found order {oid}"
                info["order_record"] = record
            else:
                info["result"] = f"Order not found: {oid}"

        # ── lookup_customer ───────────────────
        elif action.action_type == "lookup_customer":
            cid    = action.customer_id or s.get("hint_customer_id")
            record = lookup_customer(cid) if cid else None
            if record:
                s["customer_record"] = record
                reward_val += REWARDS["lookup_bonus"]
                partial["lookup_bonus"] = REWARDS["lookup_bonus"]
                info["result"]          = f"Found customer {cid}"
                info["customer_record"] = record
            else:
                info["result"] = f"Customer not found: {cid}"

        # ── respond ───────────────────────────
        elif action.action_type == "respond":
            s["status"] = "in_progress"
            text  = (action.content or "").lower()
            hits  = [kw for kw in policy["response_keywords"] if kw.lower() in text]
            kw_r  = min(0.50, len(hits) * REWARDS["respond_per_keyword"])
            reward_val += kw_r
            partial["response_quality"] = kw_r
            info["result"] = f"Keywords matched: {hits}"

        # ── escalate ──────────────────────────
        elif action.action_type == "escalate":
            combined = {**(s["order_record"] or {}), **(s["customer_record"] or {})}
            if should_escalate(s["ticket_type"], combined):
                reward_val += REWARDS["escalate_correct"]
                partial["escalate_correct"] = REWARDS["escalate_correct"]
                info["result"] = "Escalation correct."
            else:
                reward_val += REWARDS["escalate_wrong"]
                partial["escalate_wrong"] = REWARDS["escalate_wrong"]
                info["result"] = "Unnecessary escalation."
            s["status"] = "escalated"
            done = True

        # ── close ─────────────────────────────
        elif action.action_type == "close":
            combined = {**(s["order_record"] or {}), **(s["customer_record"] or {})}
            expected = policy["correct_resolution"]
            if action.resolution == expected:
                reward_val += REWARDS["close_correct"]
                partial["close_correct"] = REWARDS["close_correct"]
                info["result"] = f"Correct resolution: '{action.resolution}'"
            else:
                reward_val += REWARDS["close_wrong"]
                partial["close_wrong"] = REWARDS["close_wrong"]
                info["result"] = f"Wrong resolution: '{action.resolution}' (expected '{expected}')"
            s["status"] = "closed"
            done = True

        # ── timeout ───────────────────────────
        if s["step_count"] >= MAX_STEPS:
            done = True
            info["timeout"] = True

        if done:
            self._active = False

        history.append({
            "step":        s["step_count"],
            "action_type": action.action_type,
            "category":    action.category,
            "order_id":    action.order_id,
            "customer_id": action.customer_id,
            "content":     action.content,
            "resolution":  action.resolution,
            "step_reward": round(reward_val, 3),
            "_fp":         fp,
        })

        return self._obs(), Reward(
            value=round(max(-1.0, min(1.0, reward_val)), 3),
            partial_scores=partial,
            done=done,
            info=info,
        )

    # ─────────────────────────────────────────
    # state()
    # ─────────────────────────────────────────

    def state(self) -> dict[str, Any]:
        if not self._state:
            return {"error": "No active episode. Call /reset first."}
        clean = [{k: v for k, v in h.items() if k != "_fp"} for h in self._state.get("history", [])]
        return {**self._state, "history": clean}

    # ─────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────

    def _obs(self) -> Observation:
        s  = self._state
        or_ = s.get("order_record")
        cr_ = s.get("customer_record")
        return Observation(
            ticket_id      = s["ticket_id"],
            ticket_text    = s["ticket_text"],
            ticket_type    = s["ticket_type"] if s["task_id"] != "hard" else "unknown",
            history        = [{k: v for k, v in h.items() if k != "_fp"} for h in s["history"]],
            status         = s["status"],
            order_record   = OrderRecord(**or_) if or_ else None,
            customer_record= CustomerRecord(**cr_) if cr_ else None,
            step_count     = s["step_count"],
            max_steps      = s["max_steps"],
            task_id        = s["task_id"],
        )