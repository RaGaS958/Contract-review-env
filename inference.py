"""
inference.py — OpenEnv baseline inference script.

CRITICAL: stdout log format is parsed by automated evaluation.
Do NOT change [START], [STEP], [END] field names or ordering.

Usage:
    python inference.py

Required environment variables:
    API_BASE_URL  — LLM API endpoint (OpenAI-compatible)
    MODEL_NAME    — model identifier
    HF_TOKEN      — API key / HuggingFace token
"""

import os
import sys
import json
import time
import httpx
from typing import List, Optional

from openai import OpenAI

# ─── Provider selection ────────────────────────────────────────────────────────
# Set PROVIDER=mistral (default) or PROVIDER=gemini
# Both use the OpenAI-compatible client — just different base URLs.
PROVIDER: str = os.environ.get("PROVIDER", "mistral").lower()

_PROVIDER_DEFAULTS = {
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "model":    "mistral-small-latest",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model":    "gemini-1.5-flash",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model":    "gpt-4o-mini",
    },
}

_defaults = _PROVIDER_DEFAULTS.get(PROVIDER, _PROVIDER_DEFAULTS["mistral"])
API_BASE_URL: str = os.environ.get("API_BASE_URL", _defaults["base_url"])
MODEL_NAME: str   = os.environ.get("MODEL_NAME",   _defaults["model"])
API_KEY: str      = os.environ.get("HF_TOKEN",
                      os.environ.get("MISTRAL_API_KEY",
                        os.environ.get("GEMINI_API_KEY",
                          os.environ.get("OPENAI_API_KEY", "dummy"))))

ENV_BASE_URL: str = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK: str    = "contract-review-env"
MAX_STEPS: int    = 20
SUCCESS_THRESHOLD: float = 0.5

TASKS = ["task1", "task2", "task3"]
TASK_MAX_REWARDS = {"task1": 3.0, "task2": 3.5, "task3": 5.0}


# ─── Exact log format (DO NOT MODIFY) ─────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    print(
        f"[STEP] step={step} action={action!r} reward={reward:.4f} done={done} error={error}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(
        f"[END] success={success} steps={steps} score={score:.4f} rewards={rewards}",
        flush=True,
    )


# ─── LLM interaction ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert contract review attorney with deep knowledge of
commercial law, employment law, and SaaS agreements.

You are reviewing a legal contract. Your job is to:
1. Extract all key clauses (identify clause type and section number)
2. Flag any risks with appropriate severity (blocking/high/medium/low)
3. Approve clean sections or request revision on problematic ones
4. Submit your review when complete

You must respond ONLY with a JSON object (no markdown, no explanation):
{
  "action_type": "<one of: extract_clause, flag_risk, annotate, approve_section, request_revision, submit_review>",
  "target_section": "<section number like '2' or '3.2'>",
  "content": "<clause type or risk type>",
  "severity": "<blocking|high|medium|low or null>"
}

Common clause types: confidentiality, term, governing_law, dispute_resolution, 
  ip_assignment, non_compete, non_solicitation, termination, compensation, benefits,
  limitation_of_liability, warranty, indemnification, entire_agreement

Common risk types: overbroad_ip_assignment, worldwide_noncompete, no_limitation_of_liability,
  data_used_for_ml_training, inadequate_liability_cap, perpetual_non_disparagement,
  shortened_statute_of_limitations, unilateral_indemnification, termination_for_convenience_provider_only

When you have extracted the main clauses and flagged key risks, call submit_review."""


def get_action_from_llm(
    client: OpenAI,
    observation: dict,
    history: List[str],
) -> dict:
    """Ask LLM for next action. Returns action dict."""
    doc_text = observation.get("document_text", "")[:3000]  # truncate for context
    clauses = observation.get("clauses_extracted", [])
    flags = observation.get("flags_raised", [])
    step = observation.get("step_count", 0)
    steps_left = observation.get("steps_remaining", 10)
    last_msg = observation.get("message", "")
    last_reward = observation.get("last_action_reward", 0.0)

    user_content = f"""CONTRACT DOCUMENT:
{doc_text}

--- CURRENT STATE ---
Step: {step} | Steps remaining: {steps_left}
Last action reward: {last_reward:+.4f}
Last message: {last_msg}

Clauses extracted so far: {json.dumps([{"type": c["clause_type"], "section": c["section"]} for c in clauses])}
Risks flagged so far: {json.dumps([{"type": f["risk_type"], "severity": f["severity"]} for f in flags])}

Recent history:
{chr(10).join(history[-5:])}

What is your next action? Respond with JSON only."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=200,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        action = json.loads(raw.strip())
        return action
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        # Fallback: safe default action
        return {
            "action_type": "submit_review",
            "target_section": "",
            "content": "",
            "severity": None,
        }


# ─── Environment HTTP client helpers ──────────────────────────────────────────

def env_reset(task_id: str) -> dict:
    r = httpx.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def env_step(action: dict) -> dict:
    r = httpx.post(
        f"{ENV_BASE_URL}/step",
        json={"action": action},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# ─── Episode runner ────────────────────────────────────────────────────────────

def run_episode(client: OpenAI, task_id: str) -> float:
    """Run one full episode. Returns score in [0, 1]."""
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken: int = 0
    score: float = 0.0
    success: bool = False

    try:
        # Reset
        result = env_reset(task_id)
        observation = result["observation"]

        for step in range(1, MAX_STEPS + 1):
            if observation.get("done", False):
                break

            # Get action from LLM
            action = get_action_from_llm(client, observation, history)

            # Step environment
            step_result = env_step(action)
            observation = step_result["observation"]
            reward = step_result.get("reward", 0.0)
            done = step_result.get("done", False)
            info = step_result.get("info", {})

            rewards.append(reward)
            steps_taken = step
            action_str = json.dumps(action)

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            history.append(
                f"Step {step}: {action.get('action_type')} "
                f"section={action.get('target_section')} "
                f"content={action.get('content')} "
                f"→ reward={reward:+.4f}"
            )

            if done:
                # Extract final score from info — clamp to open (0, 1) as
                # required by the evaluation platform validator.
                final_score = info.get("final_score")
                if final_score is not None:
                    score = max(0.001, min(0.999, float(final_score)))
                break

        # If not done, score by reward sum — clamp to open (0, 1)
        if score == 0.0 and rewards:
            max_r = TASK_MAX_REWARDS.get(task_id, 3.0)
            score = max(0.001, min(0.999, sum(rewards) / max_r))

        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error for {task_id}: {exc}", flush=True)
        score = 0.001  # clamp: platform rejects exact 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"[DEBUG] Starting inference: model={MODEL_NAME} env={ENV_BASE_URL}", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_scores = {}
    start_time = time.time()

    for task_id in TASKS:
        print(f"\n[DEBUG] ===== Running {task_id} =====", flush=True)
        score = run_episode(client, task_id)
        all_scores[task_id] = score
        print(f"[DEBUG] {task_id} score: {score:.4f}", flush=True)

        # Safety: stop if we're close to 20-min limit
        elapsed = time.time() - start_time
        if elapsed > 1100:
            print("[DEBUG] Approaching time limit — skipping remaining tasks.", flush=True)
            break

    mean_score = sum(all_scores.values()) / max(len(all_scores), 1)
    print(f"\n[DEBUG] All task scores: {all_scores}", flush=True)
    print(f"[DEBUG] Mean score: {mean_score:.4f}", flush=True)


if __name__ == "__main__":
    main()
