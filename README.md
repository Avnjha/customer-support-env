# Customer Support OpenEnv

An OpenEnv-compliant environment where an AI agent handles customer support tickets by classifying issues, looking up order/customer data, responding correctly, and resolving or escalating — all graded deterministically against a policy knowledge base.

---

## Environment Description

The agent receives a customer support ticket (e.g. "My order ORD-001 hasn't arrived") and must work through a multi-step resolution process:

1. **Classify** the ticket into a category (shipping / billing / technical / general)
2. **Lookup** the relevant order or customer record from the database
3. **Respond** with a message that covers the required information
4. **Escalate** if the situation meets the escalation threshold, or **Close** with the correct resolution code

Every grader is fully deterministic — no LLM-as-judge, no text similarity. Scores are computed by checking actions against a policy knowledge base.

---

## Tasks

| Task | Difficulty | What the agent must do | Max score |
|------|------------|------------------------|-----------|
| `easy` | Easy | Classify the ticket correctly | 1.0 |
| `medium` | Medium | Classify + lookup + respond + correct escalation decision | 1.0 |
| `hard` | Hard | Full pipeline with hidden ticket type + efficiency bonus | 1.0 |

### Reward breakdown

**Easy**
- `classify_correct`: 1.0

**Medium** (0.25 each)
- `classify_correct`, `lookup_performed`, `response_quality`, `escalation_correct`

**Hard**
- `classify_correct`: 0.20
- `lookup_performed`: 0.15
- `response_quality`: 0.20
- `escalation_correct`: 0.20
- `resolution_correct`: 0.15
- `efficiency_bonus` (≤4 steps): 0.10
- `loop_penalty`: up to -0.20

---

## Action Space

| action_type | Required fields | Description |
|-------------|----------------|-------------|
| `classify` | `category` | Label the ticket: shipping / billing / technical / general |
| `lookup_order` | `order_id` | Fetch order record from database |
| `lookup_customer` | `customer_id` | Fetch customer record from database |
| `respond` | `content` | Send a response message to the customer |
| `escalate` | `content` | Escalate to a human agent |
| `close` | `content`, `resolution` | Close the ticket with a resolution code |

---

## Observation Space

```json
{
  "ticket_id": "TKT-E001",
  "ticket_text": "My order ORD-001 has not arrived...",
  "ticket_type": "order_delayed",
  "status": "open",
  "history": [],
  "order_record": null,
  "customer_record": null,
  "step_count": 0,
  "max_steps": 10,
  "task_id": "easy"
}
```

---

## Setup & Usage

### Local

```bash
pip install -r requirements.txt
cd app && uvicorn main:app --reload --port 7860
```

### Docker

```bash
docker build -t customer-support-env .
docker run -p 7860:7860 customer-support-env
```

### Run inference

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="sk-your-key"
export ENV_URL="http://localhost:7860"

python inference.py
```

### API endpoints

```bash
# Health check
curl http://localhost:7860/health

# Start episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "classify", "category": "shipping"}'

# Get current state
curl http://localhost:7860/state

# List tasks + action schema
curl http://localhost:7860/tasks

# Run rule-based baseline (no API key needed)
curl -X POST http://localhost:7860/baseline

# Score a completed episode
curl -X POST http://localhost:7860/grader \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "state": <state_from_/state>}'
```

---

## Expected Baseline Scores

| Task | Rule-based agent | GPT-4o-mini |
|------|-----------------|-------------|
| easy | ~0.90 | ~0.95 |
| medium | ~0.55 | ~0.65 |
| hard | ~0.30 | ~0.45 |

---

## Project Structure

```
customer-support-env/
├── inference.py          # OpenEnv compliant inference script (root, mandatory)
├── Dockerfile
├── openenv.yaml
├── requirements.txt
├── README.md
└── app/
    ├── main.py           # FastAPI — all endpoints
    ├── environment.py    # reset / step / state
    ├── models.py         # Pydantic types
    ├── graders.py        # Deterministic graders
    └── knowledge_base.py # Policies + synthetic DB
```