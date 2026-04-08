import json
import os
import sys
import requests

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

BENCHMARK = "customer-support-env"
MAX_STEPS = 10


# ─────────────────────────────
# LOGGING (same format)
# ─────────────────────────────

def log_start(task):
    print(f"[START] task={task} env={BENCHMARK} model=rule-based", flush=True)


def log_step(step, action, reward, done, error):
    action_str = json.dumps(action)
    error_str = error if error else "null"
    done_str = "true" if done else "false"

    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_str = "true" if success else "false"

    print(
        f"[END] success={success_str} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────
# RULE-BASED AGENT
# ─────────────────────────────

def get_action(observation):
    text = observation.get("ticket_text", "").lower()
    step = observation.get("step_count", 0)

    # Step 1 → classify
    if step == 0:
        if "refund" in text or "payment" in text:
            return {"action_type": "classify", "category": "billing"}
        elif "order" in text or "delivery" in text:
            return {"action_type": "classify", "category": "shipping"}
        elif "error" in text or "not working" in text:
            return {"action_type": "classify", "category": "technical"}
        else:
            return {"action_type": "classify", "category": "general"}

    # Step 2 → lookup
    if step == 1:
        return {"action_type": "lookup_order", "order_id": "ORD-001"}

    # Step 3 → respond
    if step == 2:
        return {
            "action_type": "respond",
            "content": "We are checking your issue and will update you shortly."
        }

    # Step 4 → close
    return {
        "action_type": "close",
        "content": "Issue resolved successfully",
        "resolution": "no_action_needed"
    }


# ─────────────────────────────
# EPISODE RUNNER
# ─────────────────────────────

def run_episode(task_id, ticket_index):
    log_start(task_id)

    try:
        reset_resp = requests.post(
            f"{ENV_URL}/reset",
            json={"task_id": task_id, "ticket_index": ticket_index},
        )
        observation = reset_resp.json()
    except Exception as e:
        log_end(False, 0, [])
        print(f"[ERROR] reset failed: {e}", file=sys.stderr)
        return False, 0, []

    rewards = []
    steps_taken = 0
    success = False

    for step_num in range(1, MAX_STEPS + 1):

        try:
            action = get_action(observation)
            last_error = None
        except Exception as e:
            log_step(step_num, {}, 0.0, False, str(e))
            break

        try:
            step_resp = requests.post(f"{ENV_URL}/step", json=action)
            step_data = step_resp.json()
        except Exception as e:
            log_step(step_num, action, 0.0, False, str(e))
            break

        observation = step_data["observation"]
        reward_value = float(step_data["reward"]["value"])
        done = step_data["done"]

        rewards.append(reward_value)
        steps_taken = step_num

        log_step(step_num, action, reward_value, done, None)

        if done:
            success = True
            break

    log_end(success, steps_taken, rewards)
    return success, steps_taken, rewards


# ─────────────────────────────
# MAIN
# ─────────────────────────────

if __name__ == "__main__":

    # Check server
    try:
        requests.get(f"{ENV_URL}/health")
    except Exception as e:
        print(f"[ERROR] Cannot reach environment: {e}")
        sys.exit(1)

    # Get tasks
    tasks_resp = requests.get(f"{ENV_URL}/tasks")
    all_tasks = tasks_resp.json()["tasks"]

    for task_info in all_tasks:
        task_id = task_info["task_id"]

        for i in range(len(task_info["tickets"])):
            run_episode(task_id, i)