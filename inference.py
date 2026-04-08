"""
inference.py
============
OpenEnv compliant inference script.

MANDATORY ENVIRONMENT VARIABLES:
  API_BASE_URL       LLM endpoint
  MODEL_NAME         Model identifier
  HF_TOKEN           Hugging Face / API key
  ENV_URL            Environment server  (default: http://localhost:7860)

STDOUT FORMAT:
  [START] task=<task> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>
"""

from __future__ import annotations
import json
import os
import sys
import requests
from openai import OpenAI


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")

BENCHMARK = "customer-support-env"
MAX_STEPS = 10

client = OpenAI(
    api_key=HF_TOKEN or "no-key",
    base_url=API_BASE_URL,
)


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a customer support agent inside an OpenEnv environment.

At every step you receive a JSON observation. Reply with ONE JSON action only.

AVAILABLE ACTIONS:
  {"action_type": "classify",        "category": "shipping|billing|technical|general"}
  {"action_type": "lookup_order",    "order_id": "ORD-XXX"}
  {"action_type": "lookup_customer", "customer_id": "CUST-XXX"}
  {"action_type": "respond",         "content": "your message to customer"}
  {"action_type": "escalate",        "content": "reason for escalation"}
  {"action_type": "close",           "content": "summary", "resolution": "CODE"}

RESOLUTION CODES (use with close):
  send_tracking_update
  initiate_refund
  retry_payment_or_escalate
  send_troubleshooting_steps
  send_policy_info
  no_action_needed

STRATEGY — follow this order every episode:
  1. classify   — read the ticket and pick the right category
  2. lookup     — look up the order (ORD-XXX) or customer (CUST-XXX) mentioned in the ticket
  3. respond    — write a helpful reply that includes: apology/acknowledgment + specific details + next steps
  4. escalate   — if delay > 7 days, refund > 10 days, or payment attempts > 3
     OR
     close      — with the correct resolution code if no escalation needed

IMPORTANT:
  - Always lookup before responding — you need the data to respond correctly
  - Response must mention relevant keywords: tracking/delay for shipping, refund/business days for billing, restart/update/version for technical
  - Output ONLY valid JSON. No markdown. No explanation."""


# ─────────────────────────────────────────────
# STDOUT LOGGERS  (exact spec format)
# ─────────────────────────────────────────────

def log_start(task: str) -> None:
    print(
        f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}",
        flush=True,
    )


def log_step(step: int, action: dict, reward: float, done: bool, error: str | None) -> None:
    action_str = json.dumps(action, separators=(",", ":"))
    error_str  = error if error else "null"
    done_str   = "true" if done else "false"
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_str = "true" if success else "false"
    print(
        f"[END] success={success_str} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────────
# LLM DECISION
# ─────────────────────────────────────────────

def get_action(observation: dict, messages: list) -> tuple[dict, list]:
    """Ask the LLM for the next action. Returns (action_dict, updated_messages)."""

    # Build a clean view of what the agent can see
    obs_summary = json.dumps({
        "ticket_text":     observation.get("ticket_text"),
        "status":          observation.get("status"),
        "step_count":      observation.get("step_count"),
        "order_record":    observation.get("order_record"),
        "customer_record": observation.get("customer_record"),
        "recent_history": [
            {
                "action": h.get("action_type"),
                "result_reward": h.get("step_reward"),
            }
            for h in observation.get("history", [])[-3:]  # last 3 steps only
        ],
    }, indent=2)

    messages.append({"role": "user", "content": obs_summary})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,                          # deterministic — required for reproducible scores
        response_format={"type": "json_object"},  # guarantees valid JSON output
    )

    raw = response.choices[0].message.content
    messages.append({"role": "assistant", "content": raw})

    action = json.loads(raw)
    return action, messages


# ─────────────────────────────────────────────
# EPISODE RUNNER
# ─────────────────────────────────────────────

def run_episode(task_id: str, ticket_index: int) -> tuple[bool, int, list[float]]:
    """
    Run one full episode.
    Returns (success, steps_taken, list_of_rewards).
    Emits [START], [STEP]×N, [END] to stdout.
    """
    log_start(task_id)

    # ── reset ────────────────────────────────
    try:
        reset_resp = requests.post(
            f"{ENV_URL}/reset",
            json={"task_id": task_id, "ticket_index": ticket_index},
            timeout=30,
        )
        reset_resp.raise_for_status()
        observation = reset_resp.json()
    except Exception as e:
        log_end(False, 0, [])
        print(f"[ERROR] reset failed: {e}", file=sys.stderr)
        return False, 0, []

    messages    = [{"role": "system", "content": SYSTEM_PROMPT}]
    rewards     = []
    steps_taken = 0
    success     = False

    # ── step loop ────────────────────────────
    for step_num in range(1, MAX_STEPS + 1):

        # Get action from LLM
        try:
            action, messages = get_action(observation, messages)
            last_error = None
        except Exception as e:
            last_error = str(e)
            log_step(step_num, {}, 0.0, False, last_error)
            print(f"[ERROR] LLM call failed at step {step_num}: {e}", file=sys.stderr)
            break

        # Send action to environment
        try:
            step_resp = requests.post(
                f"{ENV_URL}/step",
                json=action,
                timeout=30,
            )
            step_resp.raise_for_status()
            step_data = step_resp.json()
        except Exception as e:
            last_error = str(e)
            log_step(step_num, action, 0.0, False, last_error)
            print(f"[ERROR] step() failed at step {step_num}: {e}", file=sys.stderr)
            break

        observation  = step_data["observation"]
        reward_value = float(step_data["reward"]["value"])
        done         = bool(step_data["done"])
        warning      = step_data["reward"]["info"].get("warning")  # loop penalty warning

        rewards.append(reward_value)
        steps_taken = step_num

        # Emit [STEP] line — required immediately after env.step() returns
        log_step(step_num, action, reward_value, done, warning)

        if done:
            final_status = observation.get("status", "")
            success = final_status in ("closed", "escalated")
            break

    # ── end ──────────────────────────────────
    log_end(success, steps_taken, rewards)
    return success, steps_taken, rewards


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Verify environment is reachable
    try:
        health = requests.get(f"{ENV_URL}/health", timeout=10)
        health.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Cannot reach environment at {ENV_URL}: {e}", file=sys.stderr)
        sys.exit(1)

    # Get task list
    try:
        tasks_resp = requests.get(f"{ENV_URL}/tasks", timeout=10)
        tasks_resp.raise_for_status()
        all_tasks = tasks_resp.json()["tasks"]
    except Exception as e:
        print(f"[ERROR] Cannot fetch tasks: {e}", file=sys.stderr)
        sys.exit(1)

    summary: dict[str, list[float]] = {}

    for task_info in all_tasks:
        task_id    = task_info["task_id"]
        num_tickets = len(task_info["tickets"])
        task_rewards: list[float] = []

        for i in range(num_tickets):
            _, _, rewards = run_episode(task_id, ticket_index=i)
            task_rewards.extend(rewards)

        summary[task_id] = task_rewards

    # Summary goes to stderr — stdout must only contain [START]/[STEP]/[END]
    print("\n─── BASELINE SUMMARY ───", file=sys.stderr)
    for task_id, rewards in summary.items():
        avg = sum(rewards) / len(rewards) if rewards else 0.0
        bar = "█" * int(avg * 20)
        print(f"  {task_id:8s}  {avg:.2f}  {bar}", file=sys.stderr)
    print("────────────────────────", file=sys.stderr)