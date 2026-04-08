# 📜 Contract Review Environment

<div align="center">

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   An AI agent learns to review legal contracts like a      │
│   trained attorney — extracting clauses, flagging risks,   │
│   and producing structured redline reports.                 │
│                                                             │
│          Real task. Real value. Real reward signal.         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

[![OpenEnv](https://img.shields.io/badge/OpenEnv-1.0.0-6366f1?style=flat-square)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3b82f6?style=flat-square)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-10b981?style=flat-square)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-f59e0b?style=flat-square)](LICENSE)
[![Tasks](https://img.shields.io/badge/Tasks-3%20(Easy%E2%86%92Hard)-ef4444?style=flat-square)](#9-the-three-tasks)

</div>

---

## Table of Contents

1. [What Is This?](#1-what-is-this)
2. [Why Contract Review?](#2-why-contract-review)
3. [How It Works — Plain English](#3-how-it-works--plain-english)
4. [Architecture](#4-architecture)
5. [File Structure](#5-file-structure)
6. [Action Space](#6-action-space--what-the-agent-can-do)
7. [Observation Space](#7-observation-space--what-the-agent-sees)
8. [Reward Design](#8-reward-design)
9. [The Three Tasks](#9-the-three-tasks)
10. [The Graders](#10-the-graders--how-scoring-works)
11. [Anti-Exploit Design](#11-anti-exploit-design)
12. [Dataset](#12-dataset)
13. [API Reference](#13-api-reference)
14. [Quickstart](#14-quickstart)
15. [Running Inference](#15-running-inference)
16. [Baseline Scores](#16-baseline-scores)
17. [Integrity Tests](#17-integrity-tests)
18. [Episode Flow Walkthrough](#18-episode-flow-walkthrough)
19. [Metrics and Analytics](#19-metrics-and-analytics)
20. [Environment Variables](#20-environment-variables)
21. [Full Reward Reference](#21-full-reward-reference)

---

## 1. What Is This?

**Contract Review Environment** is an [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compliant reinforcement learning environment where AI agents learn to perform legal contract review — one of the most common, high-value knowledge-work tasks in existence.

Every company, every day, reviews contracts. NDAs before partnerships. SaaS agreements before buying software. Employment agreements before hiring. Legal teams charge $300–$800/hour to do this. An agent trained in this environment learns to:

- **Extract** specific clause types and their exact section locations
- **Flag risks** by severity — from minor (low) to deal-breakers (blocking)
- **Approve** clean sections or **request revision** on problematic ones
- **Submit** a structured review report with a measurable quality score

This is not a toy. This is not a game. This is a simulation of real legal work that organizations pay millions of dollars for every year.

---

## 2. Why Contract Review?

### The real-world gap

| Problem | Scale |
|---------|-------|
| Average NDA review by a lawyer | 2–4 hours, $500–$2,000 |
| Fortune 500 companies review | 20,000–100,000 contracts/year |
| SMBs that skip review due to cost | ~60% |
| Legal disputes from bad contracts | $300B+/year in the US |

An AI agent trained here could meaningfully reduce that cost. That is not hypothetical — it is already happening in legal tech. This environment fills a direct gap in the RL/agent evaluation ecosystem.

### Why it is novel in OpenEnv

- **No existing OpenEnv environment covers legal document analysis** — zero. This domain is completely new.
- Unlike games or simulations, every task maps 1:1 to a task a real human performs professionally.
- The reward function encodes genuine legal reasoning: missing a "blocking" risk in an employment contract is categorically worse than missing a "low" stylistic issue.

---

## 3. How It Works — Plain English

Imagine you are an AI agent and you receive a contract to review. Here is exactly what happens step by step:

```
STEP 1 — Start a fresh episode
  You call: reset(task_id="task1")
  You get:  Full contract text + empty review state

STEP 2 — Read and act
  You read: "Section 2 contains the confidentiality obligations"
  You call: step(action_type="extract_clause",
                 target_section="2",
                 content="confidentiality")
  You get:  reward=+0.17, message="Correct! Exact section match."

STEP 3 — Spot a risk
  You read: "Section 5.1 has a worldwide non-compete for 2 years after termination"
  You call: step(action_type="flag_risk",
                 target_section="5.1",
                 content="worldwide_noncompete",
                 severity="blocking")
  You get:  reward=+0.34, message="Risk correctly identified as blocking."

STEP 4 — Keep going
  Extract more clauses, flag more risks, approve clean sections...
  Each step gives immediate reward feedback.

STEP 5 — Submit when done
  You call: step(action_type="submit_review")
  You get:  final_score=0.82, reward=+0.49, done=True
```

The agent's goal is to maximise the final score by doing genuinely good legal review work — not by gaming the reward function.

---

## 4. Architecture

### System overview

```
╔══════════════════════════════════════════════════════════════════╗
║                         YOUR AGENT                               ║
║                                                                  ║
║   obs = env.reset(task_id="task1")                              ║
║   obs, reward = env.step(ContractAction(...))                   ║
║                                                                  ║
╚═══════════════════════╦══════════════════════════════════════════╝
                        ║
              HTTP/JSON  or  WebSocket /ws
                        ║
╔═══════════════════════▼══════════════════════════════════════════╗
║                     FASTAPI SERVER                               ║
║                    (server/app.py)                               ║
║                                                                  ║
║   POST /reset   POST /step   GET /state   WS /ws                ║
╚═══════════════════════╦══════════════════════════════════════════╝
                        ║
╔═══════════════════════▼══════════════════════════════════════════╗
║           ENVIRONMENT CORE  (server/environment.py)              ║
║                                                                  ║
║   reset()  → loads contract, initialises ContractState          ║
║   step()   → validates action                                   ║
║            → computes step reward (shaped, dense)               ║
║            → updates ContractState                              ║
║            → checks budget / done                               ║
║            → returns (ContractObservation, ContractReward)      ║
║   state()  → returns internal snapshot                          ║
╚═══════════════════════╦══════════════════════════════════════════╝
                        ║
╔═══════════════════════▼══════════════════════════════════════════╗
║              GRADERS  (server/graders.py)                        ║
║                                                                  ║
║   grade_task1() — F1 clause extraction vs ground truth          ║
║   grade_task2() — Severity-weighted risk recall minus FP        ║
║   grade_task3() — Clauses + risks + disposition + blocking gate ║
╚═══════════════════════╦══════════════════════════════════════════╝
                        ║
╔═══════════════════════▼══════════════════════════════════════════╗
║              DATASET  (data/contracts.py)                        ║
║                                                                  ║
║   4 synthetic contracts with embedded ground-truth annotations   ║
║   NDA x2  |  SaaS Agreement x1  |  Employment Agreement x1     ║
╚══════════════════════════════════════════════════════════════════╝
```

### Component responsibilities

| Component | File | Responsibility |
|-----------|------|----------------|
| Server | `server/app.py` | HTTP + WebSocket endpoints, request routing |
| Environment | `server/environment.py` | Episode logic, state, reward shaping |
| Graders | `server/graders.py` | Deterministic scoring vs ground truth |
| Models | `models/action.py`, `models/observation.py` | Pydantic type contracts |
| Dataset | `data/contracts.py` | Synthetic contracts + annotations |
| Inference | `inference.py` | Baseline LLM agent script |
| Tests | `integrity_tests.py` | 22 integrity and anti-exploit checks |

---

## 5. File Structure

```
contract-review-env/
│
├── inference.py          ← Baseline inference script (judges run this)
├── openenv.yaml          ← OpenEnv manifest (spec compliance)
├── Dockerfile            ← Container deployment
├── requirements.txt      ← Python dependencies
├── README.md             ← This file
├── integrity_tests.py    ← 22 genuineness verification checks
│
├── server/
│   ├── app.py            ← FastAPI: HTTP + WebSocket /ws
│   ├── environment.py    ← Core reset/step/state + reward shaping
│   └── graders.py        ← Deterministic task graders (no LLM)
│
├── models/
│   ├── action.py         ← ContractAction Pydantic model
│   └── observation.py    ← ContractObservation, ContractState, ContractReward
│
└── data/
    └── contracts.py      ← 4 synthetic contracts with ground-truth annotations
```

---

## 6. Action Space — What the Agent Can Do

The agent has **6 action types**. Every action is a `ContractAction` Pydantic model:

```python
class ContractAction(BaseModel):
    action_type: Literal[
        "extract_clause",    # Identify a clause type and its section number
        "flag_risk",         # Raise a risk with severity
        "annotate",          # Add a free-text note (neutral — no reward impact)
        "approve_section",   # Mark a section as reviewed and acceptable
        "request_revision",  # Mark a section as needing redline changes
        "submit_review",     # End episode and receive final grade
    ]
    target_section: str      # e.g. "2", "3.2", "5.1"
    content: str             # Clause type or risk type identifier
    severity: Optional[Literal["low", "medium", "high", "blocking"]]
```

### Action examples

```json
// Extract a confidentiality clause from section 2
{
  "action_type": "extract_clause",
  "target_section": "2",
  "content": "confidentiality",
  "severity": null
}

// Flag a blocking risk in section 5.1
{
  "action_type": "flag_risk",
  "target_section": "5.1",
  "content": "worldwide_noncompete",
  "severity": "blocking"
}

// Approve a clean section
{
  "action_type": "approve_section",
  "target_section": "1",
  "content": "",
  "severity": null
}

// End the episode
{
  "action_type": "submit_review",
  "target_section": "",
  "content": "",
  "severity": null
}
```

### Valid clause type identifiers

```
confidentiality         term                    governing_law
dispute_resolution      ip_assignment           non_compete
non_solicitation        termination             compensation
benefits                limitation_of_liability warranty
indemnification         entire_agreement        ip_ownership
remedies                subscription_rights     payment_terms
data_processing         non_disparagement       position_duties
```

### Valid risk type identifiers

```
overbroad_ip_assignment                  worldwide_noncompete
no_limitation_of_liability               data_used_for_ml_training
inadequate_liability_cap                 perpetual_non_disparagement
shortened_statute_of_limitations         unilateral_indemnification
termination_for_convenience_provider_only
perpetual_feedback_license               no_severance
prior_inventions_sweep                   unilateral_agreement
```

---

## 7. Observation Space — What the Agent Sees

Every call to `reset()` or `step()` returns a `ContractObservation`:

```python
class ContractObservation(BaseModel):
    # The document
    document_text: str           # Full contract text — agent reads this
    document_type: str           # "NDA" | "SaaS Subscription Agreement" | ...
    document_title: str

    # Episode progress
    step_count: int              # Steps taken so far
    max_steps: int               # Budget (15, 20, or 25 by task)
    steps_remaining: int         # Budget left before auto-submit

    # Agent's work so far (grows each step)
    clauses_extracted: List[ExtractedClause]
    flags_raised: List[RaisedFlag]
    approved_sections: List[str]
    revision_requests: List[str]

    # In-context RL feedback
    last_action_reward: float    # Reward for the last action
    cumulative_reward: float     # Total reward accumulated this episode

    # Status
    done: bool
    task_id: str
    contract_id: str
    message: str                 # Human-readable feedback on last action
    available_actions: List[str]
```

### Sample observation (mid-episode, task 3)

```json
{
  "document_type": "Employment Agreement",
  "document_title": "Senior Engineer Employment Agreement — MegaCorp",
  "step_count": 3,
  "max_steps": 25,
  "steps_remaining": 22,
  "clauses_extracted": [
    {"clause_type": "ip_assignment", "section": "4", "step_extracted": 1},
    {"clause_type": "non_compete",   "section": "5", "step_extracted": 2}
  ],
  "flags_raised": [
    {"risk_type": "overbroad_ip_assignment", "section": "4.1",
     "severity": "blocking", "step_raised": 3}
  ],
  "approved_sections": [],
  "revision_requests": [],
  "last_action_reward": 0.3420,
  "cumulative_reward": 0.6760,
  "done": false,
  "task_id": "task3",
  "message": "Risk 'overbroad_ip_assignment' correctly identified as blocking."
}
```

---

## 8. Reward Design

The reward function is **dense** (fires every step), **shaped** (partial credit for partial work), and **exploit-resistant** (all cheating strategies score worse than genuine review).

### Reward formula per step

```
reward = base_step_cost + action_bonus - penalties

base_step_cost = -0.008   (small cost per step — efficiency incentive)
```

### Reward by action type

| Action | Condition | Reward |
|--------|-----------|--------|
| `extract_clause` | Correct type + exact section | **+0.172** |
| `extract_clause` | Correct type + approx section | +0.072 |
| `extract_clause` | Correct type + wrong section | +0.032 |
| `extract_clause` | Wrong type (FP) | −0.048 |
| `flag_risk` (blocking) | Correct type + severity | **+0.342** |
| `flag_risk` (high) | Correct type + severity | +0.272 |
| `flag_risk` (medium) | Correct type + severity | +0.167 |
| `flag_risk` (low) | Correct type + severity | +0.080 |
| `flag_risk` any | Wrong severity (partial) | 50% of above |
| `flag_risk` any | Wrong type (FP) | **−0.128** |
| `approve_section` | Clean section | +0.072 |
| `approve_section` | Risky section | **−0.128** |
| `request_revision` | Risky section | +0.092 |
| `request_revision` | Clean section | −0.068 |
| `annotate` | Any | −0.008 (step cost only) |
| `submit_review` | — | **+0.6 × final_score** |
| Any (duplicate) | Already seen | **−0.050** |

### Terminal reward

When `submit_review` is called the grader runs and adds:

```
terminal_reward = final_score × 0.60   (normal submit)
terminal_reward = final_score × 0.40   (auto-submit after budget exhaustion)
```

### Severity weights for risk identification

```
blocking  →  1.00  (structural defect — contract cannot proceed as-is)
high      →  0.80  (serious risk — likely to cause harm if signed)
medium    →  0.50  (notable issue — warrants redline)
low       →  0.25  (minor issue — good practice to address)
```

### Score distribution by agent quality

```
Spam-flag agent    → 0.00   (20 FP flags × −0.15 = −3.0 penalty dominates)
Random agent       → ~0.00  (step costs exceed random correct hits)
Empty submit       → 0.00   (no work done before submit)
Weak LLM baseline  → 0.15–0.25
Moderate LLM       → 0.30–0.50
Strong LLM agent   → 0.50–0.70
GRPO-trained agent → 0.70–0.90
Oracle (scripted)  → 0.88 mean (1.0 on tasks 1–2, 0.67 on task 3)
```

---

## 9. The Three Tasks

### Task 1 — Clause Extraction (Easy)

```
Document:   Standard Mutual NDA — TechCorp & AlphaCo
Goal:       Extract all 5 clause types with correct section references
Max steps:  15

Target clauses:
  confidentiality    → section 2
  term               → section 3  (2 years, 30-day notice)
  governing_law      → section 5  (Delaware)
  dispute_resolution → section 6  (AAA arbitration, Wilmington DE)
  entire_agreement   → section 7

Planted risks:
  no_limitation_of_liability → severity: medium
```

**What a high-scoring agent does:**
1. Reads the NDA (approx. 500 words, clearly structured)
2. Identifies each clause type from section headers and content
3. Calls `extract_clause` for each with the exact section number
4. Flags the missing limitation of liability clause
5. Approves the clean sections (1, 4, 7)
6. Submits

**Grader:** F1-style — credit for type match, bonus for exact section reference.

---

### Task 2 — Risk Identification (Medium)

```
Document:   Master SaaS Subscription Agreement — CloudBase & Acme Corp
Goal:       Identify all 5 planted risks with correct severity levels
Max steps:  20

Planted risks (all must be found):
  data_used_for_ml_training          → high    (section 3.2)
  inadequate_liability_cap           → high    (section 5.2, cap = $1,000)
  termination_for_convenience_...    → medium  (section 7.2)
  unilateral_indemnification         → medium  (section 9.1)
  perpetual_feedback_license         → low     (section 4.2)
```

**What makes this medium difficulty:**
- The document is longer (approx. 800 words)
- Risks are embedded in legal boilerplate — easy to miss
- Severity must be correct for full credit (wrong severity = 50% of score)
- False positives are penalised (−0.15 per fake flag)
- The agent must read carefully, not spam flags

**Grader:** Severity-weighted precision-recall minus false-positive penalty.

---

### Task 3 — Full Contract Redline (Hard)

```
Document:   Senior Engineer Employment Agreement — MegaCorp
Goal:       Complete review of 12 sections
Max steps:  25

Clauses to extract: 12 (sections 1 through 12)

Risks to find:
  overbroad_ip_assignment  → BLOCKING  ← miss this: score capped at 0.15
  worldwide_noncompete     → BLOCKING  ← miss this: score capped at 0.15
  shortened_statute_of_limitations → high
  perpetual_non_disparagement      → high
  no_severance                     → medium
  prior_inventions_sweep           → medium

Sections to approve:         1, 2, 3, 6, 7, 11, 12 (clean)
Sections to flag for revision: 4, 5, 8, 9, 10 (problematic)
```

**The blocking risk gate:**

> If either `overbroad_ip_assignment` or `worldwide_noncompete` is missed, the final score is **capped at 0.15** regardless of all other work.

This mirrors real legal practice: some issues are so severe that missing them invalidates the entire review. A worldwide non-compete that is unenforceable in most US states cannot be signed — no amount of other good work compensates.

**Scoring breakdown:**

```
Clause coverage score   × 0.30
Risk identification     × 0.45
Section disposition     × 0.15   (correct approve/revise decisions)
Efficiency bonus        × 0.10   (steps saved)
─────────────────────────────────
Total                   → up to 1.00
```

---

## 10. The Graders — How Scoring Works

All graders are **100% deterministic** — no LLM calls, no randomness, pure comparison against embedded ground truth. The same episode always produces the same score.

### Task 1 grader — clause extraction

```python
def grade_task1(agent_clauses, ground_truth_clauses) -> float:
    """
    For each GT clause, find the agent's best matching extraction.

    Per-clause match score:
      Type correct + exact section    → 1.0
      Type correct + same top level   → 0.6  (e.g. "4" when GT is "4.1")
      Type correct + wrong section    → 0.4
      Type not matched                → 0.0

    Final = mean over all GT clauses. Clamped to [0, 1].
    """
```

Example:

| GT clause | Agent extracted | Match score |
|-----------|----------------|-------------|
| confidentiality, section 2 | confidentiality, section 2 | 1.00 |
| term, section 3 | term, section 99 | 0.40 |
| governing_law, section 5 | (nothing) | 0.00 |

### Task 2 grader — weighted risk recall

```python
def grade_task2(agent_flags, ground_truth_risks) -> float:
    """
    For each GT risk, check if agent flagged it:
      Correct type + correct severity  → full severity_weight
      Correct type + wrong severity    → 0.5 × severity_weight
      False positive (no GT match)     → -0.15 penalty

    score = (earned - penalties) / max_possible
    Clamped to [0.0, 1.0]
    """
```

Example calculation for Task 2:

```
GT risks max_possible = 0.8 + 0.8 + 0.5 + 0.5 + 0.25 = 2.85

Agent flags:
  data_used_for_ml_training (high, correct)    → +0.80
  inadequate_liability_cap  (high, correct)    → +0.80
  termination_... (medium, correct)            → +0.50
  perpetual_feedback_license (says "high")     → +0.125  (partial: 0.25 × 0.5)
  totally_fake_risk (FP)                       → −0.150

score = (0.80 + 0.80 + 0.50 + 0.125 − 0.15) / 2.85 = 0.729
```

### Task 3 grader — composite with gate

```python
def grade_task3(...) -> float:

    # 1. Blocking gate — must catch all blocking risks
    if any blocking risk is MISSED:
        partial = blocking_caught / blocking_total
        return 0.05 + 0.10 × partial   # max 0.15

    # 2. Component scores
    clause_score      = grade_task1(...)   # 0-1
    risk_score        = grade_task2(...)   # 0-1
    disposition_score = correct_dispositions / total_sections  # 0-1
    efficiency_bonus  = 0.5 + 0.5 × (1 - steps_used / max_steps)  # 0.5-1.0

    return (
        clause_score      × 0.30 +
        risk_score        × 0.45 +
        disposition_score × 0.15 +
        efficiency_bonus  × 0.10
    )
```

---

## 11. Anti-Exploit Design

Every naive cheating strategy scores worse than genuine review. All verified by automated integrity tests.

### Strategy: spam all flags

```
Spam 20 random risk types
Each FP: −0.15 penalty → total = −3.00
No TPs  → earned = 0.00
Score = max(0, −3.00 / 2.85) = 0.00   FAILS
```

### Strategy: approve everything

```
Approve all 12 sections
5 risky sections × −0.128 each = −0.64
7 clean sections × +0.072 each = +0.504
Net from approvals = −0.136
Result: negative net reward   FAILS
```

### Strategy: loop the same action

```
extract_clause("confidentiality") × 15 times
First: +0.172
Duplicates (14 times): −0.050 × 14 = −0.700
Net = −0.528   FAILS
```

### Strategy: submit immediately

```
submit_review with zero work
task1 score = 0.0 → terminal bonus = 0.0
task2 score = 0.0 → terminal bonus = 0.0
task3 score = 0.05 → terminal bonus = 0.03
Result: near-zero   FAILS
```

### Strategy: genuine expert review

```
Extract all clauses (correct type + section)
Flag all risks (correct type + severity)
Approve clean sections, flag risky ones
Submit after thorough work

task1: score = 1.00
task2: score = 1.00
task3: score = 0.67+
Result: high scores   SUCCEEDS
```

---

## 12. Dataset

Four synthetic contracts, procedurally generated with embedded ground truth. No copyright issues, no external API calls needed by the grader.

| Contract ID | Document Type | Task | Difficulty | Clauses | Risks |
|-------------|---------------|------|------------|---------|-------|
| `nda_tech_001` | Mutual NDA | task1 | Easy | 5 | 1 |
| `nda_saas_002` | Unilateral NDA | task1 | Easy | 6 | 2 |
| `saas_subscription_001` | SaaS Agreement | task2 | Medium | 9 | 5 |
| `employment_senior_001` | Employment Agreement | task3 | Hard | 12 | 6 |

Each contract in `data/contracts.py` has this structure:

```python
"employment_senior_001": {
    "type": "Employment Agreement",
    "title": "Senior Engineer Employment Agreement",
    "difficulty": "hard",
    "task_ids": ["task3"],
    "text": "... full contract text ...",
    "ground_truth": {
        "clauses": [
            {"type": "ip_assignment", "section": "4"},
            # ... 12 total
        ],
        "risks": [
            {"type": "overbroad_ip_assignment", "section": "4.1",
             "severity": "blocking",
             "description": "Section 4.1 assigns ALL inventions ..."},
            # ... 6 total
        ],
        "approved_sections": ["1", "2", "3", "6", "7", "11", "12"]
    }
}
```

Ground truth is used **only inside the graders** — it is never exposed in the observation. The agent must discover it by reading the contract text.

---

## 13. API Reference

The server exposes HTTP and WebSocket endpoints. The WebSocket endpoint (`/ws`) is the primary interface used by the Python client.

### HTTP Endpoints

#### `GET /health`

```json
{"status": "healthy", "environment": "contract-review-env", "version": "1.0.0"}
```

#### `POST /reset`

Request:
```json
{"task_id": "task1", "contract_id": null}
```

Response:
```json
{
  "observation": {
    "document_text": "MUTUAL NON-DISCLOSURE AGREEMENT...",
    "document_type": "NDA",
    "step_count": 0,
    "max_steps": 15,
    "steps_remaining": 15,
    "clauses_extracted": [],
    "flags_raised": [],
    "done": false,
    "task_id": "task1"
  },
  "done": false,
  "reward": 0.0,
  "info": {}
}
```

#### `POST /step`

Request:
```json
{
  "action": {
    "action_type": "extract_clause",
    "target_section": "2",
    "content": "confidentiality",
    "severity": null
  }
}
```

Response:
```json
{
  "observation": { "step_count": 1, "clauses_extracted": [...], ... },
  "reward": 0.172,
  "done": false,
  "info": {
    "reason": "Correct clause 'confidentiality' extracted from correct section 2.",
    "is_terminal": false,
    "final_score": null,
    "cumulative_reward": 0.172
  }
}
```

#### `GET /state`

Returns the current internal `ContractState` snapshot (useful for debugging).

#### `GET /web`

Opens the interactive browser UI for manual testing — no code needed.

#### `GET /docs`

Auto-generated OpenAPI documentation.

---

### WebSocket — `WS /ws`

Each WebSocket connection gets its own isolated environment instance.

**Client sends:**
```json
{"type": "reset", "task_id": "task1"}
{"type": "step",  "action": {"action_type": "...", ...}}
{"type": "state"}
```

**Server responds:**
```json
{"type": "reset", "data": {"observation": {...}, "done": false}}
{"type": "step",  "data": {"observation": {...}, "reward": 0.172, "done": false, "info": {...}}}
{"type": "state", "data": {"episode_id": "...", "step_count": 3}}
{"type": "error", "data": "Error message"}
```

---

## 14. Quickstart

### Option A — Local with Uvicorn

```bash
# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# Health check
curl http://localhost:7860/health

# Reset an episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1"}'

# Send an action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "extract_clause",
                   "target_section": "2",
                   "content": "confidentiality",
                   "severity": null}}'

# Open Web UI
open http://localhost:7860/web
```

### Option B — Docker

```bash
docker build -t contract-review-env .
docker run -d -p 7860:7860 contract-review-env
curl http://localhost:7860/health
```

### Option C — Python client loop

```python
import httpx

base = "http://localhost:7860"
client = httpx.Client()

# Reset
result = client.post(f"{base}/reset", json={"task_id": "task3"}).json()
obs = result["observation"]
print(f"Document: {obs['document_type']} | Budget: {obs['max_steps']} steps")

# Episode loop
while not obs["done"]:
    action = your_agent(obs)   # your agent decides here
    result = client.post(f"{base}/step", json={"action": action}).json()
    obs = result["observation"]
    print(f"Step {obs['step_count']:2d}: {result['reward']:+.4f} | {result['info']['reason']}")

print(f"Final score: {result['info']['final_score']:.3f}")
```

---

## 15. Running Inference

The inference script uses the OpenAI-compatible Python client and works with Mistral AI (free), Google Gemini (free), or OpenAI.

### Required environment variables

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM API endpoint |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` | API key for the LLM |
| `ENV_BASE_URL` | Environment URL (default: `http://localhost:7860`) |
| `PROVIDER` | Shortcut: `mistral` or `gemini` (sets defaults automatically) |

### With Mistral AI (free tier)

```bash
# Get key: console.mistral.ai
export PROVIDER=mistral
export HF_TOKEN=your_mistral_api_key

# Start environment in a separate terminal
uvicorn server.app:app --port 7860

# Run inference
python inference.py
```

### With Google Gemini (free tier)

```bash
# Get key: aistudio.google.com
export PROVIDER=gemini
export HF_TOKEN=AIza_your_gemini_key
export API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
export MODEL_NAME=gemini-1.5-flash

python inference.py
```

### Expected output format

```
[START] task=task1 env=contract-review-env model=mistral-small-latest
[STEP] step=1 action='{"action_type": "extract_clause", ...}' reward=0.1720 done=False error=None
[STEP] step=2 action='{"action_type": "extract_clause", ...}' reward=0.1720 done=False error=None
[STEP] step=3 action='{"action_type": "flag_risk", ...}' reward=0.1670 done=False error=None
[STEP] step=4 action='{"action_type": "submit_review", ...}' reward=0.2400 done=True error=None
[END] success=True steps=4 score=0.4000 rewards=[0.172, 0.172, 0.167, 0.24]
```

The `[START]`, `[STEP]`, `[END]` format is parsed by the automated evaluation system — field names and ordering are exact.

---

## 16. Baseline Scores

### Scripted oracle baseline (no LLM — deterministic)

Verifies the environment is scorable and graders are correct.

| Task | Score | Steps used | Max steps |
|------|-------|-----------|-----------|
| task1 — Clause Extraction | **1.000** | 9 | 15 |
| task2 — Risk Identification | **1.000** | 6 | 20 |
| task3 — Full Contract Redline | **0.666** | 16 | 25 |
| Mean | **0.889** | | |

Task 3 scores 0.666 (not 1.0) because the scripted baseline extracts only 5 of 12 clauses within its 16 steps — a trained RL agent that learns to prioritise high-value actions can score higher.

### LLM baseline estimates

| Task | GPT-4o-mini | Mistral-small | Random agent |
|------|-------------|---------------|-------------|
| task1 | ~0.45 | ~0.40 | ~0.00 |
| task2 | ~0.35 | ~0.30 | ~0.00 |
| task3 | ~0.22 | ~0.18 | ~0.05 |

These scores leave substantial headroom for GRPO or RL training improvement — which is precisely the point of this environment.

---

## 17. Integrity Tests

Run the full suite to verify the environment is genuine:

```bash
python integrity_tests.py           # unit tests, no server needed
python integrity_tests.py --live    # also tests the running HTTP server
```

### 22 checks across 7 categories

| Category | Tests | What it verifies |
|----------|-------|-----------------|
| Score variance | 5 | Graders produce different scores for different quality agents |
| Anti-exploit | 5 | Every cheating strategy scores worse than genuine review |
| Determinism | 3 | Same inputs always give the same reward (reproducible) |
| Grader independence | 2 | Step reward does not depend on future actions |
| Episode isolation | 3 | Episodes are independent — no state leakage between resets |
| Boundary enforcement | 2 | All rewards in [−1, 1], all scores in [0, 1] |
| Full audit | 3 | Cumulative reward equals sum of steps; traces are identical |

### Sample passing output

```
  CATEGORY 1: Score variance
  PASS  task1 grader: 4 quality levels → 4 distinct scores  (scores=[1.0, 0.8, 0.4, 0.0])
  PASS  task2 grader: spam 20 fake risks → 0.0
  PASS  task3 grader: missing blocking → score ≤ 0.15

  CATEGORY 2: Anti-exploit
  PASS  smart agent beats spam-flag agent  (smart=0.561 > spam=0.000)
  PASS  duplicate action → -0.05 penalty
  PASS  approving risky section → negative reward

  CATEGORY 7: Full audit
  PASS  cumulative_reward = sum of all step rewards   (0.7510 == 0.7510)
  PASS  correct > wrong_severity > false_positive     (+0.272 > +0.112 > −0.128)
  PASS  3 identical runs → byte-for-byte identical traces

  INTEGRITY REPORT: 22/22 passed
  Environment is genuine, exploit-proof, tamper-free.
```

---

## 18. Episode Flow Walkthrough

A complete episode on Task 3 (Full Contract Redline), traced step by step:

```
══════════════════════════════════════════════════════════════
EPISODE START
reset(task_id="task3")
  Loads:  employment_senior_001  (12 sections, 6 risks)
  State:  step=0, done=False, flags=[], clauses=[], reward=0.0
══════════════════════════════════════════════════════════════

STEP 1
  Action:  flag_risk, "4.1", "overbroad_ip_assignment", "blocking"
  Check:   In ground truth? YES | Severity match? YES
  Bonus:   1.00 × 0.35 = 0.350 | Base: −0.008
  Reward:  +0.342
  State:   flags=[{overbroad_ip_assignment, blocking}], cumulative=0.342

STEP 2
  Action:  flag_risk, "5.1", "worldwide_noncompete", "blocking"
  Reward:  +0.342
  State:   flags=[..., {worldwide_noncompete, blocking}], cumulative=0.684
  Note:    BOTH BLOCKING RISKS CAUGHT — gate is now open

STEP 3
  Action:  flag_risk, "9.2", "shortened_statute_of_limitations", "high"
  Reward:  +0.272   (high weight=0.80)
  cumulative: 0.956

STEP 4
  Action:  flag_risk, "10.1", "perpetual_non_disparagement", "high"
  Reward:  +0.272
  cumulative: 1.228

STEP 5
  Action:  flag_risk, "8.2", "no_severance", "medium"
  Reward:  +0.167
  cumulative: 1.395

STEP 6
  Action:  flag_risk, "4.3", "prior_inventions_sweep", "medium"
  Reward:  +0.167
  cumulative: 1.562

STEPS 7–11 — Extract clauses
  extract_clause ip_assignment,    "4"  → +0.172
  extract_clause non_compete,      "5"  → +0.172
  extract_clause confidentiality,  "7"  → +0.172
  extract_clause dispute_resolution,"9" → +0.172
  extract_clause governing_law,   "11"  → +0.172
  cumulative: 2.422

STEPS 12–15 — Approve and flag sections
  approve_section  "1"  (clean)        → +0.072
  approve_section  "2"  (clean)        → +0.072
  request_revision "4"  (has 4.1,4.3)  → +0.092
  request_revision "5"  (has 5.1)      → +0.092
  cumulative: 2.750

STEP 16 — Submit
  Action:  submit_review
  Grader runs:
    Blocking gate?    PASSED (both caught)
    Clause score:     5/12 = 0.42 → × 0.30 = 0.126
    Risk score:       6/6  = 1.00 → × 0.45 = 0.450
    Disposition:      4/12 correct → × 0.15 = 0.050
    Efficiency:       16/25 steps → bonus=0.82 → × 0.10 = 0.082
    ─────────────────────────────────────────────────
    Final score = 0.126 + 0.450 + 0.050 + 0.082 = 0.708

  Terminal reward = 0.708 × 0.6 = 0.425
  Total episode reward = 2.750 + 0.425 = 3.175

══════════════════════════════════════════════════════════════
EPISODE END
  final_score = 0.708
  steps_used  = 16 / 25
  done        = True
══════════════════════════════════════════════════════════════
```

---

## 19. Metrics and Analytics

### Per-step metrics in observation

| Metric | Type | Description |
|--------|------|-------------|
| `last_action_reward` | float | Reward for the most recent action |
| `cumulative_reward` | float | Sum of all rewards this episode |
| `step_count` | int | Steps taken so far |
| `steps_remaining` | int | Budget remaining |
| `len(clauses_extracted)` | int | Clause coverage so far |
| `len(flags_raised)` | int | Risk recall so far |

### Terminal metrics (when `done=True`)

| Metric | Source | Description |
|--------|--------|-------------|
| `final_score` | `info["final_score"]` | Graded quality score 0.0–1.0 |
| `is_terminal` | `info["is_terminal"]` | True on last step |
| `cumulative_reward` | `obs["cumulative_reward"]` | Total episode reward |

### Tracking agent improvement

```python
metrics = {
    "episode": episode_number,
    "task": task_id,
    "final_score": result["info"]["final_score"],
    "total_reward": obs["cumulative_reward"],
    "steps_used": obs["step_count"],
    "efficiency": 1.0 - (obs["step_count"] / obs["max_steps"]),
    "clauses_found": len(obs["clauses_extracted"]),
    "risks_found": len(obs["flags_raised"]),
}
```

### Expected improvement curve over RL training

```
Episode    0:  score ≈ 0.05   (random — near zero)
Episode   50:  score ≈ 0.15   (learns to flag something relevant)
Episode  200:  score ≈ 0.30   (learns clause vocabulary)
Episode  500:  score ≈ 0.50   (learns severity weighting)
Episode 1000:  score ≈ 0.65   (learns blocking risk priority)
Episode 2000:  score ≈ 0.75+  (near-expert performance)
```

The sharp score signal on blocking risks (gate at 0.15 if missed) makes the agent learn to prioritise the most critical issues first — which is exactly the correct legal reasoning strategy.

---

## 20. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `7860` | Server port |
| `WORKERS` | `2` | Uvicorn worker processes |
| `MAX_CONCURRENT_ENVS` | `100` | Max WebSocket sessions per worker |
| `API_BASE_URL` | — | LLM API endpoint (inference.py) |
| `MODEL_NAME` | — | LLM model identifier (inference.py) |
| `HF_TOKEN` | — | API key for LLM (inference.py) |
| `ENV_BASE_URL` | `http://localhost:7860` | Environment base URL (inference.py) |
| `PROVIDER` | `mistral` | Shortcut: `mistral` or `gemini` |

---

## 21. Full Reward Reference

```
Action: extract_clause
  Correct type + exact section:           base + 0.180 = +0.172
  Correct type + same top-level section:  base + 0.080 = +0.072
  Correct type + wrong section:           base + 0.040 = +0.032
  Wrong type (false positive):            base − 0.040 = −0.048

Action: flag_risk
  Severity weights: blocking=1.0, high=0.8, medium=0.5, low=0.25

  Correct type + correct severity:        base + weight × 0.35
    blocking correct: 0.35 × 1.0 = +0.342
    high correct:     0.35 × 0.8 = +0.272
    medium correct:   0.35 × 0.5 = +0.167
    low correct:      0.35 × 0.25= +0.080

  Correct type + wrong severity:          base + weight × 0.15
  False positive (no GT match):           base − 0.120 = −0.128

Action: approve_section
  Section is clean (GT approved_sections): base + 0.080 = +0.072
  Section has risks (any subsection):      base − 0.120 = −0.128
  Section is neutral:                      base + 0.020 = +0.012

Action: request_revision
  Section has risks (any subsection):     base + 0.100 = +0.092
  Section is clean:                       base − 0.060 = −0.068
  Section is neutral:                     base + 0.020 = +0.012

Action: annotate
  Any:                                    base + 0.000 = −0.008

Action: submit_review
  Terminal bonus (normal):                final_score × 0.600
  Terminal bonus (budget exhausted):      final_score × 0.400

Duplicate action (seen before in this episode):       −0.050 always

Base per-step cost:                       −0.008
Reward clamped:                           [−1.0, +1.0]
Final score clamped:                      [0.0, 1.0]
```

---

<div align="center">

**Built for the OpenEnv Hackathon — Meta PyTorch × Hugging Face 2024**

*A real-world environment for a real-world skill.*

[openenv.yaml](openenv.yaml) · [inference.py](inference.py) · [integrity_tests.py](integrity_tests.py)

</div>
