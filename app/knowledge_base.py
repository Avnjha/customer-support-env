"""
knowledge_base.py
=================
Policy rules + synthetic database.

This file is what makes graders deterministic.
Instead of using LLM-as-judge or text similarity,
every grader does a simple dictionary lookup here.

CONTENTS:
  POLICIES      maps ticket_type → correct actions + escalation rules
  ORDERS_DB     synthetic order records
  CUSTOMERS_DB  synthetic customer records
  TICKETS       pre-built scenarios per difficulty level
"""

from __future__ import annotations
from typing import Any


# ─────────────────────────────────────────────
# POLICIES
# ─────────────────────────────────────────────

POLICIES: dict[str, dict[str, Any]] = {

    "order_delayed": {
        "classify_as":        "shipping",
        "required_lookup":    "lookup_order",
        "response_keywords":  ["apologize", "tracking", "delay"],
        "escalate_condition": "delay_days > 7",
        "correct_resolution": "send_tracking_update",
    },

    "refund_not_received": {
        "classify_as":        "billing",
        "required_lookup":    "lookup_order",
        "response_keywords":  ["refund", "business days", "processed"],
        "escalate_condition": "refund_age_days > 10",
        "correct_resolution": "initiate_refund",
    },

    "app_crashing": {
        "classify_as":        "technical",
        "required_lookup":    "lookup_customer",
        "response_keywords":  ["restart", "update", "version"],
        "escalate_condition": "open_tickets > 3",
        "correct_resolution": "send_troubleshooting_steps",
    },

    "payment_failed": {
        "classify_as":        "billing",
        "required_lookup":    "lookup_order",
        "response_keywords":  ["payment", "retry", "bank"],
        "escalate_condition": "payment_attempts > 3",
        "correct_resolution": "retry_payment_or_escalate",
    },

    "general_inquiry": {
        "classify_as":        "general",
        "required_lookup":    None,
        "response_keywords":  ["happy to help", "please", "let us know"],
        "escalate_condition": "False",
        "correct_resolution": "send_policy_info",
    },
}


# ─────────────────────────────────────────────
# ORDERS DATABASE
# ─────────────────────────────────────────────

ORDERS_DB: dict[str, dict[str, Any]] = {

    "ORD-001": {
        "order_id": "ORD-001", "customer_id": "CUST-A1",
        "customer_name": "Alice Sharma", "status": "delayed",
        "carrier": "FedEx", "tracking_number": "FX991234567",
        "delay_days": 3,          # does NOT trigger escalation (rule: > 7)
        "refund_age_days": None, "payment_attempts": None, "amount": 149.99,
    },
    "ORD-002": {
        "order_id": "ORD-002", "customer_id": "CUST-B2",
        "customer_name": "Bob Patel", "status": "delayed",
        "carrier": "UPS", "tracking_number": "UP887654321",
        "delay_days": 10,         # TRIGGERS escalation (rule: > 7)
        "refund_age_days": None, "payment_attempts": None, "amount": 89.50,
    },
    "ORD-003": {
        "order_id": "ORD-003", "customer_id": "CUST-C3",
        "customer_name": "Carol Zhang", "status": "refund_pending",
        "carrier": None, "tracking_number": None,
        "delay_days": None,
        "refund_age_days": 5,     # does NOT trigger escalation (rule: > 10)
        "payment_attempts": None, "amount": 220.00,
    },
    "ORD-004": {
        "order_id": "ORD-004", "customer_id": "CUST-D4",
        "customer_name": "David Osei", "status": "refund_pending",
        "carrier": None, "tracking_number": None,
        "delay_days": None,
        "refund_age_days": 14,    # TRIGGERS escalation (rule: > 10)
        "payment_attempts": None, "amount": 75.00,
    },
    "ORD-005": {
        "order_id": "ORD-005", "customer_id": "CUST-E5",
        "customer_name": "Eva Rossi", "status": "payment_failed",
        "carrier": None, "tracking_number": None,
        "delay_days": None, "refund_age_days": None,
        "payment_attempts": 2,    # does NOT trigger escalation (rule: > 3)
        "amount": 310.00,
    },
    "ORD-006": {
        "order_id": "ORD-006", "customer_id": "CUST-F6",
        "customer_name": "Frank Müller", "status": "payment_failed",
        "carrier": None, "tracking_number": None,
        "delay_days": None, "refund_age_days": None,
        "payment_attempts": 5,    # TRIGGERS escalation (rule: > 3)
        "amount": 199.00,
    },
}


# ─────────────────────────────────────────────
# CUSTOMERS DATABASE
# ─────────────────────────────────────────────

CUSTOMERS_DB: dict[str, dict[str, Any]] = {

    "CUST-A1": {
        "customer_id": "CUST-A1", "name": "Alice Sharma",
        "email": "alice@example.com", "tier": "standard",
        "open_tickets": 1, "account_age_days": 180,
    },
    "CUST-B2": {
        "customer_id": "CUST-B2", "name": "Bob Patel",
        "email": "bob@example.com", "tier": "premium",
        "open_tickets": 2, "account_age_days": 540,
    },
    "CUST-C3": {
        "customer_id": "CUST-C3", "name": "Carol Zhang",
        "email": "carol@example.com", "tier": "standard",
        "open_tickets": 1, "account_age_days": 90,
    },
    "CUST-D4": {
        "customer_id": "CUST-D4", "name": "David Osei",
        "email": "david@example.com", "tier": "standard",
        "open_tickets": 2, "account_age_days": 200,
    },
    "CUST-E5": {
        "customer_id": "CUST-E5", "name": "Eva Rossi",
        "email": "eva@example.com", "tier": "premium",
        "open_tickets": 1, "account_age_days": 365,
    },
    "CUST-F6": {
        "customer_id": "CUST-F6", "name": "Frank Müller",
        "email": "frank@example.com", "tier": "standard",
        "open_tickets": 3, "account_age_days": 120,
    },
    "CUST-G7": {
        "customer_id": "CUST-G7", "name": "Grace Kim",
        "email": "grace@example.com", "tier": "enterprise",
        "open_tickets": 5,          # TRIGGERS escalation (rule: > 3)
        "account_age_days": 1200,
    },
    "CUST-H8": {
        "customer_id": "CUST-H8", "name": "Henry Okafor",
        "email": "henry@example.com", "tier": "standard",
        "open_tickets": 1,          # does NOT trigger escalation
        "account_age_days": 60,
    },
}


# ─────────────────────────────────────────────
# TICKETS  (pre-built scenarios per task)
# ─────────────────────────────────────────────

TICKETS: dict[str, list[dict[str, Any]]] = {

    # ── EASY: classify only ──────────────────
    "easy": [
        {
            "ticket_id": "TKT-E001",
            "ticket_text": "Hi, my order ORD-001 hasn't arrived yet and it was supposed to be here 3 days ago. Can you help?",
            "ticket_type": "order_delayed",
            "hint_order_id": "ORD-001",
            "hint_customer_id": "CUST-A1",
        },
        {
            "ticket_id": "TKT-E002",
            "ticket_text": "I returned my item two weeks ago but still haven't received my refund for order ORD-003.",
            "ticket_type": "refund_not_received",
            "hint_order_id": "ORD-003",
            "hint_customer_id": "CUST-C3",
        },
        {
            "ticket_id": "TKT-E003",
            "ticket_text": "Your app keeps crashing every time I try to open it. I am customer CUST-H8.",
            "ticket_type": "app_crashing",
            "hint_order_id": None,
            "hint_customer_id": "CUST-H8",
        },
        {
            "ticket_id": "TKT-E004",
            "ticket_text": "My payment failed when I tried to check out order ORD-005. Please help.",
            "ticket_type": "payment_failed",
            "hint_order_id": "ORD-005",
            "hint_customer_id": "CUST-E5",
        },
    ],

    # ── MEDIUM: lookup + respond ─────────────
    "medium": [
        {
            "ticket_id": "TKT-M001",
            "ticket_text": "My order ORD-001 is delayed. It was due 3 days ago. What is going on?",
            "ticket_type": "order_delayed",
            "hint_order_id": "ORD-001",
            "hint_customer_id": "CUST-A1",
        },
        {
            "ticket_id": "TKT-M002",
            "ticket_text": "I still have not gotten my refund for ORD-004. It has been two weeks!",
            "ticket_type": "refund_not_received",
            "hint_order_id": "ORD-004",
            "hint_customer_id": "CUST-D4",
        },
        {
            "ticket_id": "TKT-M003",
            "ticket_text": "The app crashes on startup. I use it daily for work. Customer ID: CUST-G7.",
            "ticket_type": "app_crashing",
            "hint_order_id": None,
            "hint_customer_id": "CUST-G7",
        },
    ],

    # ── HARD: full pipeline, ticket_type hidden ──
    "hard": [
        {
            "ticket_id": "TKT-H001",
            "ticket_text": "Order ORD-002 was supposed to arrive 10 days ago. I need this urgently.",
            "ticket_type": "order_delayed",
            "hint_order_id": "ORD-002",
            "hint_customer_id": "CUST-B2",
        },
        {
            "ticket_id": "TKT-H002",
            "ticket_text": "My payment for ORD-006 has failed 5 times now. I really need to complete this purchase.",
            "ticket_type": "payment_failed",
            "hint_order_id": "ORD-006",
            "hint_customer_id": "CUST-F6",
        },
        {
            "ticket_id": "TKT-H003",
            "ticket_text": "Refund for order ORD-004 still not received after 14 days. This is unacceptable.",
            "ticket_type": "refund_not_received",
            "hint_order_id": "ORD-004",
            "hint_customer_id": "CUST-D4",
        },
    ],
}


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def get_policy(ticket_type: str) -> dict[str, Any]:
    if ticket_type not in POLICIES:
        raise KeyError(f"No policy for ticket_type='{ticket_type}'")
    return POLICIES[ticket_type]


def lookup_order(order_id: str) -> dict[str, Any] | None:
    return ORDERS_DB.get(order_id)


def lookup_customer(customer_id: str) -> dict[str, Any] | None:
    return CUSTOMERS_DB.get(customer_id)


def should_escalate(ticket_type: str, record: dict[str, Any]) -> bool:
    """Evaluate the escalation condition against a merged order+customer record."""
    condition = POLICIES[ticket_type]["escalate_condition"]
    try:
        return bool(eval(condition, {"__builtins__": {}}, record))
    except Exception:
        return False


def get_tickets_for_task(task_id: str) -> list[dict[str, Any]]:
    if task_id not in TICKETS:
        raise KeyError(f"No tickets for task_id='{task_id}'")
    return TICKETS[task_id]