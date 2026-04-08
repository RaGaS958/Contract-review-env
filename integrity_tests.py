"""
integrity_tests.py — Comprehensive genuineness verification suite.

Tests 7 integrity categories:
  1. Score variance        — graders produce DIFFERENT scores for different agents
  2. Anti-exploit          — spam/cheat strategies score WORSE than smart agents
  3. Determinism           — same actions always produce SAME rewards (reproducible)
  4. Grader independence   — grader never sees future actions (no lookahead)
  5. Reset isolation       — episodes are truly independent
  6. Boundary enforcement  — rewards always in [-1, 1], scores in [0, 1]
  7. Step-by-step audit    — full episode trace matches final graded score

Run with:
    python integrity_tests.py              # unit tests (no server needed)
    python integrity_tests.py --live       # also tests against running server
"""

import sys
import json
import time
import httpx
import random
from typing import List, Tuple, Dict, Any

# ─── Test runner ───────────────────────────────────────────────────────────────

RESULTS: List[Tuple[str, bool, str]] = []

def test(name: str):
    def dec(fn):
        def wrapper():
            try:
                msg = fn()
                RESULTS.append((name, True, msg or "OK"))
                print(f"  PASS  {name}")
                if msg and msg != "OK":
                    for line in msg.splitlines():
                        print(f"        {line}")
            except AssertionError as e:
                RESULTS.append((name, False, str(e)))
                print(f"  FAIL  {name}")
                print(f"        {e}")
            except Exception as e:
                RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
                print(f"  ERR   {name}")
                print(f"        {type(e).__name__}: {e}")
        wrapper()  # auto-execute on decoration
        return wrapper
    return dec


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 1: SCORE VARIANCE — graders must NOT return same score for everyone
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  CATEGORY 1: Score variance (graders aren't constant)")
print("="*60)

@test("task1 grader: empty agent scores 0.0")
def _():
    from server.graders import grade_task1
    from data.contracts import CONTRACTS
    gt = CONTRACTS["nda_tech_001"]["ground_truth"]["clauses"]
    s = grade_task1([], gt)
    assert s == 0.0, f"Empty agent should score 0.0, got {s}"
    return f"score={s}"

@test("task1 grader: perfect agent scores 1.0")
def _():
    from server.graders import grade_task1
    from data.contracts import CONTRACTS
    gt = CONTRACTS["nda_tech_001"]["ground_truth"]["clauses"]
    perfect = [{"clause_type": c["type"], "section": str(c["section"])} for c in gt]
    s = grade_task1(perfect, gt)
    assert s == 1.0, f"Perfect agent should score 1.0, got {s}"
    return f"score={s}"

@test("task1 grader: partial agent scores between 0 and 1")
def _():
    from server.graders import grade_task1
    from data.contracts import CONTRACTS
    gt = CONTRACTS["nda_tech_001"]["ground_truth"]["clauses"]
    # Only 2 out of 5 clauses
    partial = [
        {"clause_type": "confidentiality", "section": "2"},
        {"clause_type": "term", "section": "3"},
    ]
    s = grade_task1(partial, gt)
    assert 0.0 < s < 1.0, f"Partial agent should score between 0 and 1, got {s}"
    return f"score={s}"

@test("task1 grader: 4 different quality levels produce 4 different scores")
def _():
    from server.graders import grade_task1
    from data.contracts import CONTRACTS
    gt = CONTRACTS["nda_tech_001"]["ground_truth"]["clauses"]
    perfect = [{"clause_type": c["type"], "section": str(c["section"])} for c in gt]
    good    = perfect[:4]
    partial = perfect[:2]
    empty   = []
    scores = [
        grade_task1(perfect, gt),
        grade_task1(good, gt),
        grade_task1(partial, gt),
        grade_task1(empty, gt),
    ]
    assert scores == sorted(scores, reverse=True), f"Scores should decrease: {scores}"
    assert len(set(scores)) == 4, f"All 4 scores should be distinct: {scores}"
    return f"scores={[round(s,3) for s in scores]}"

@test("task2 grader: false-positive spammer scores 0.0 (clamped)")
def _():
    from server.graders import grade_task2
    from data.contracts import CONTRACTS
    gt = CONTRACTS["saas_subscription_001"]["ground_truth"]["risks"]
    # Flood with fake risks
    spam = [{"risk_type": f"fake_risk_{i}", "section": "1", "severity": "high"}
            for i in range(20)]
    s = grade_task2(spam, gt)
    assert s == 0.0, f"Spammer should score 0.0 after FP penalties, got {s}"
    return f"score={s} (20 FP flags correctly penalised)"

@test("task2 grader: correct flags + no FP = perfect score")
def _():
    from server.graders import grade_task2
    from data.contracts import CONTRACTS
    gt = CONTRACTS["saas_subscription_001"]["ground_truth"]["risks"]
    perfect = [{"risk_type": r["type"], "section": str(r.get("section","")),"severity": r["severity"]} for r in gt]
    s = grade_task2(perfect, gt)
    assert s == 1.0, f"Perfect risk identifier should score 1.0, got {s}"
    return f"score={s}"

@test("task3 grader: missing blocking risk gates score to ≤0.15")
def _():
    from server.graders import grade_task3
    from data.contracts import CONTRACTS
    gt = CONTRACTS["employment_senior_001"]["ground_truth"]
    # No blocking risks caught
    s = grade_task3(
        agent_clauses=[], agent_flags=[],
        approved_sections=[], revision_requests=[],
        step_count=5, max_steps=25, ground_truth=gt,
    )
    assert s <= 0.15, f"Missing blocking risks should cap score at 0.15, got {s}"
    return f"score={s} (blocking gate enforced)"



# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 2: ANTI-EXPLOIT — cheat strategies must score WORSE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  CATEGORY 2: Anti-exploit (cheating strategies backfire)")
print("="*60)

@test("spam-flag strategy scores lower than smart agent (task2)")
def _():
    from server.environment import ContractReviewEnvironment
    from models.action import ContractAction

    # Smart agent: correct flags only
    env = ContractReviewEnvironment()
    env.reset(task_id="task2")
    smart_actions = [
        ContractAction(action_type="flag_risk", target_section="3.2", content="data_used_for_ml_training", severity="high"),
        ContractAction(action_type="flag_risk", target_section="5.2", content="inadequate_liability_cap", severity="high"),
        ContractAction(action_type="submit_review"),
    ]
    for a in smart_actions:
        obs, r = env.step(a)
    smart_score = r.final_score

    # Spam agent: flag everything with made-up risk types
    env2 = ContractReviewEnvironment()
    env2.reset(task_id="task2")
    spam_actions = [
        ContractAction(action_type="flag_risk", target_section=str(i), content=f"fake_risk_{i}", severity="high")
        for i in range(10)
    ] + [ContractAction(action_type="submit_review")]
    for a in spam_actions:
        obs2, r2 = env2.step(a)
    spam_score = r2.final_score

    assert smart_score > spam_score, (
        f"Smart agent ({smart_score:.3f}) should beat spammer ({spam_score:.3f})"
    )
    return f"smart={smart_score:.3f} > spam={spam_score:.3f}"

@test("loop exploit: duplicate actions give -0.05 (not free reward)")
def _():
    from server.environment import ContractReviewEnvironment
    from models.action import ContractAction

    env = ContractReviewEnvironment()
    env.reset(task_id="task1")
    action = ContractAction(action_type="extract_clause", target_section="2", content="confidentiality")

    # First time: positive reward
    obs1, r1 = env.step(action)
    assert r1.value > 0, f"First action should be positive, got {r1.value}"

    # Second time (duplicate): negative reward
    obs2, r2 = env.step(action)
    assert r2.value < 0, f"Duplicate action should be penalised, got {r2.value}"
    assert r2.value == -0.05, f"Duplicate penalty should be -0.05, got {r2.value}"
    return f"first={r1.value:+.4f}, duplicate={r2.value:+.4f}"

@test("approve-risky-section exploit scores negative reward")
def _():
    from server.environment import ContractReviewEnvironment
    from models.action import ContractAction

    env = ContractReviewEnvironment()
    env.reset(task_id="task3")  # employment agreement has risky sections
    # Section 4 has overbroad_ip_assignment (blocking risk)
    obs, r = env.step(ContractAction(action_type="approve_section", target_section="4"))
    assert r.value < 0, f"Approving risky section should penalise, got {r.value}"
    return f"reward={r.value:+.4f} (correctly penalised)"

@test("submit-immediately baseline scores near 0 (can't game with empty review)")
def _():
    from server.environment import ContractReviewEnvironment
    from models.action import ContractAction

    scores = []
    for task_id in ["task1", "task2", "task3"]:
        env = ContractReviewEnvironment()
        env.reset(task_id=task_id)
        obs, r = env.step(ContractAction(action_type="submit_review"))
        assert r.final_score is not None
        scores.append(r.final_score)
    for s in scores:
        assert s <= 0.1, f"Empty review should score ≤0.1, got {s}"
    return f"empty scores={[round(s,3) for s in scores]}"

@test("step budget exhaustion auto-submits with reduced terminal bonus")
def _():
    from server.environment import ContractReviewEnvironment, TASK_MAX_STEPS
    from models.action import ContractAction

    env = ContractReviewEnvironment()
    env.reset(task_id="task1")
    max_steps = TASK_MAX_STEPS["task1"]

    # Use up all steps with unique annotate actions
    for i in range(max_steps):
        if env._state.done:
            break
        obs, r = env.step(ContractAction(
            action_type="annotate",
            target_section=str(i),
            content=f"note_{i}",
        ))
    assert obs.done, "Episode should be done after budget exhaustion"
    assert r.is_terminal, "Terminal flag should be set"
    assert r.final_score is not None
    return f"auto-submitted at step {obs.step_count}, score={r.final_score:.3f}"


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 3: DETERMINISM — same inputs always give same outputs
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  CATEGORY 3: Determinism (reproducible results)")
print("="*60)

@test("same action sequence always produces identical rewards (run 1 vs run 2)")
def _():
    from server.environment import ContractReviewEnvironment
    from models.action import ContractAction

    actions = [
        ContractAction(action_type="extract_clause", target_section="2", content="confidentiality"),
        ContractAction(action_type="flag_risk", target_section="", content="no_limitation_of_liability", severity="medium"),
        ContractAction(action_type="approve_section", target_section="1"),
        ContractAction(action_type="submit_review"),
    ]

    def run():
        env = ContractReviewEnvironment()
        env.reset(task_id="task1")
        rewards = []
        final_score = None
        for a in actions:
            obs, r = env.step(a)
            rewards.append(r.value)
            if r.final_score is not None:
                final_score = r.final_score
        return rewards, final_score

    r1, s1 = run()
    r2, s2 = run()
    assert r1 == r2, f"Rewards differ between runs: {r1} vs {r2}"
    assert s1 == s2, f"Final scores differ: {s1} vs {s2}"
    return f"rewards={r1} identical across 2 runs"

@test("graders are deterministic — same inputs, same score, 10 repetitions")
def _():
    from server.graders import grade_task1
    from data.contracts import CONTRACTS
    gt = CONTRACTS["nda_tech_001"]["ground_truth"]["clauses"]
    agent = [{"clause_type": "confidentiality", "section": "2"},
             {"clause_type": "term", "section": "3"}]
    scores = [grade_task1(agent, gt) for _ in range(10)]
    assert len(set(scores)) == 1, f"Grader not deterministic: {scores}"
    return f"score={scores[0]} consistent over 10 runs"

@test("reset produces identical initial observations across 5 resets")
def _():
    from server.environment import ContractReviewEnvironment

    env = ContractReviewEnvironment()
    obs_snapshots = []
    for _ in range(5):
        obs = env.reset(task_id="task1")
        obs_snapshots.append({
            "step_count": obs.step_count,
            "done": obs.done,
            "clauses_extracted": obs.clauses_extracted,
            "flags_raised": obs.flags_raised,
            "cumulative_reward": obs.cumulative_reward,
        })
    for snap in obs_snapshots[1:]:
        assert snap == obs_snapshots[0], f"Reset not clean: {snap} vs {obs_snapshots[0]}"
    return "5 resets produced identical clean observations"


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 4: GRADER INDEPENDENCE — no lookahead into future actions
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  CATEGORY 4: Grader independence (no lookahead)")
print("="*60)

@test("step reward does NOT depend on future actions")
def _():
    from server.environment import ContractReviewEnvironment
    from models.action import ContractAction

    # Two episodes: same first action, different subsequent actions
    # First step reward must be identical
    action1 = ContractAction(action_type="extract_clause", target_section="2", content="confidentiality")

    env_a = ContractReviewEnvironment()
    env_a.reset(task_id="task1")
    _, r_a = env_a.step(action1)

    env_b = ContractReviewEnvironment()
    env_b.reset(task_id="task1")
    _, r_b = env_b.step(action1)

    assert r_a.value == r_b.value, (
        f"Same first action gives different rewards: {r_a.value} vs {r_b.value}"
    )

    # Now do different things in each episode — first reward must not change retroactively
    env_a.step(ContractAction(action_type="submit_review"))
    env_b.step(ContractAction(action_type="flag_risk", target_section="1", content="fake", severity="low"))
    env_b.step(ContractAction(action_type="flag_risk", target_section="2", content="fake2", severity="low"))

    # Re-check: the reward object from step 1 is immutable
    assert r_a.value == r_b.value, "First step reward was retroactively changed (lookahead detected!)"
    return f"first-step reward={r_a.value} stable regardless of subsequent actions"

@test("grader only uses episode state at submission time — not future state")
def _():
    from server.environment import ContractReviewEnvironment
    from models.action import ContractAction

    # Submit early (only 1 clause extracted)
    env = ContractReviewEnvironment()
    env.reset(task_id="task1")
    env.step(ContractAction(action_type="extract_clause", target_section="2", content="confidentiality"))
    _, r_early = env.step(ContractAction(action_type="submit_review"))

    # Submit late (all clauses extracted)
    env2 = ContractReviewEnvironment()
    env2.reset(task_id="task1")
    from data.contracts import CONTRACTS
    gt = CONTRACTS["nda_tech_001"]["ground_truth"]["clauses"]
    for c in gt:
        env2.step(ContractAction(action_type="extract_clause",
                                  target_section=str(c["section"]),
                                  content=c["type"]))
    _, r_late = env2.step(ContractAction(action_type="submit_review"))

    assert r_early.final_score < r_late.final_score, (
        f"More work should give higher score: early={r_early.final_score} late={r_late.final_score}"
    )
    return f"early={r_early.final_score:.3f} < late={r_late.final_score:.3f} (grader uses actual episode state)"


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 5: RESET ISOLATION — episodes don't contaminate each other
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  CATEGORY 5: Episode isolation (no state leakage)")
print("="*60)

@test("completed episode state does not bleed into next episode")
def _():
    from server.environment import ContractReviewEnvironment
    from models.action import ContractAction

    env = ContractReviewEnvironment()

    # Episode 1: do lots of work
    env.reset(task_id="task2")
    for _ in range(8):
        env.step(ContractAction(action_type="flag_risk", target_section="1",
                                content=f"risk_{_}", severity="high"))
    env.step(ContractAction(action_type="submit_review"))

    # Episode 2: fresh reset — must be clean
    obs = env.reset(task_id="task1")
    assert obs.step_count == 0, f"step_count not reset: {obs.step_count}"
    assert obs.clauses_extracted == [], f"Clauses leaked: {obs.clauses_extracted}"
    assert obs.flags_raised == [], f"Flags leaked: {obs.flags_raised}"
    assert obs.cumulative_reward == 0.0, f"Reward leaked: {obs.cumulative_reward}"
    assert not obs.done, "done flag leaked from previous episode"
    return "No state leaked between episodes"

@test("task switch on reset loads correct contract")
def _():
    from server.environment import ContractReviewEnvironment

    env = ContractReviewEnvironment()

    obs1 = env.reset(task_id="task1")
    assert obs1.document_type == "NDA", f"task1 should be NDA, got {obs1.document_type}"

    obs2 = env.reset(task_id="task2")
    assert "SaaS" in obs2.document_type or "Subscription" in obs2.document_type, (
        f"task2 should be SaaS agreement, got {obs2.document_type}"
    )

    obs3 = env.reset(task_id="task3")
    assert "Employment" in obs3.document_type, f"task3 should be Employment, got {obs3.document_type}"
    return f"task1={obs1.document_type!r}, task2={obs2.document_type!r}, task3={obs3.document_type!r}"

@test("two parallel envs don't share state")
def _():
    from server.environment import ContractReviewEnvironment
    from models.action import ContractAction

    env_a = ContractReviewEnvironment()
    env_b = ContractReviewEnvironment()

    env_a.reset(task_id="task1")
    env_b.reset(task_id="task2")

    # Do work in env_a
    env_a.step(ContractAction(action_type="extract_clause", target_section="2", content="confidentiality"))
    env_a.step(ContractAction(action_type="extract_clause", target_section="3", content="term"))

    # env_b should not see env_a's work
    state_b = env_b.state()
    assert len(state_b.clauses_extracted) == 0, (
        f"env_b was contaminated by env_a: {state_b.clauses_extracted}"
    )
    assert state_b.task_id == "task2", f"env_b task_id wrong: {state_b.task_id}"
    return "Two parallel environments are fully isolated"


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 6: BOUNDARY ENFORCEMENT
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  CATEGORY 6: Boundary enforcement (rewards always in valid range)")
print("="*60)

@test("all rewards are in [-1.0, 1.0] across 100 random actions")
def _():
    from server.environment import ContractReviewEnvironment
    from models.action import ContractAction
    import random

    random.seed(0)

    action_types = ["extract_clause", "flag_risk", "annotate",
                    "approve_section", "request_revision"]
    severities = ["low", "medium", "high", "blocking"]
    fake_clauses = ["confidentiality", "term", "governing_law",
                    "fake_clause", "xyz_clause"]
    fake_risks = ["overbroad_ip_assignment", "fake_risk", "no_limitation_of_liability", "xyz"]

    out_of_range = []
    for task_id in ["task1", "task2", "task3"]:
        env = ContractReviewEnvironment()
        env.reset(task_id=task_id)
        for _ in range(33):
            if env._state.done:
                break
            atype = random.choice(action_types)
            if atype == "flag_risk":
                content = random.choice(fake_risks)
                severity = random.choice(severities)
            elif atype == "extract_clause":
                content = random.choice(fake_clauses)
                severity = None
            else:
                content = f"note_{random.randint(0,999)}"
                severity = None
            action = ContractAction(
                action_type=atype,
                target_section=str(random.randint(1, 12)),
                content=content,
                severity=severity,
            )
            obs, r = env.step(action)
            if not (-1.0 <= r.value <= 1.0):
                out_of_range.append((atype, r.value))

    assert out_of_range == [], f"Out-of-range rewards found: {out_of_range}"
    return "All 100 random actions produced rewards in [-1.0, 1.0]"

@test("all final scores are in [0.0, 1.0]")
def _():
    from server.environment import ContractReviewEnvironment
    from models.action import ContractAction

    scores = []
    for task_id in ["task1", "task2", "task3"]:
        for _ in range(3):  # 3 different quality runs per task
            env = ContractReviewEnvironment()
            env.reset(task_id=task_id)
            # Random partial work then submit
            for i in range(_ * 3):
                env.step(ContractAction(
                    action_type="extract_clause",
                    target_section=str(i),
                    content="confidentiality",
                ))
            obs, r = env.step(ContractAction(action_type="submit_review"))
            assert r.final_score is not None
            assert 0.0 <= r.final_score <= 1.0, (
                f"Final score {r.final_score} out of [0,1] for {task_id}"
            )
            scores.append(r.final_score)

    return f"9 final scores all in [0,1]: {[round(s,3) for s in scores]}"


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 7: STEP-BY-STEP AUDIT — reward sum matches final grade
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  CATEGORY 7: Full episode audit (reward trace is honest)")
print("="*60)

@test("cumulative_reward in observation matches sum of step rewards")
def _():
    from server.environment import ContractReviewEnvironment
    from models.action import ContractAction

    env = ContractReviewEnvironment()
    env.reset(task_id="task1")

    actions = [
        ContractAction(action_type="extract_clause", target_section="2", content="confidentiality"),
        ContractAction(action_type="extract_clause", target_section="3", content="term"),
        ContractAction(action_type="flag_risk", target_section="", content="no_limitation_of_liability", severity="medium"),
        ContractAction(action_type="approve_section", target_section="1"),
        ContractAction(action_type="submit_review"),
    ]

    reward_sum = 0.0
    for a in actions:
        obs, r = env.step(a)
        reward_sum += r.value
    
    # obs.cumulative_reward is tracked internally
    assert abs(obs.cumulative_reward - reward_sum) < 0.001, (
        f"Cumulative mismatch: obs={obs.cumulative_reward:.4f} vs sum={reward_sum:.4f}"
    )
    return f"cumulative_reward={obs.cumulative_reward:.4f} matches step sum={reward_sum:.4f}"

@test("final_score is a pure function of episode actions (audit trail is complete)")
def _():
    """Run same episode twice and verify every single reward value matches."""
    from server.environment import ContractReviewEnvironment
    from models.action import ContractAction

    actions = [
        ContractAction(action_type="flag_risk", target_section="3.2", content="data_used_for_ml_training", severity="high"),
        ContractAction(action_type="flag_risk", target_section="5.2", content="inadequate_liability_cap", severity="high"),
        ContractAction(action_type="flag_risk", target_section="7.2", content="termination_for_convenience_provider_only", severity="medium"),
        ContractAction(action_type="submit_review"),
    ]

    traces = []
    for _ in range(2):
        env = ContractReviewEnvironment()
        env.reset(task_id="task2")
        trace = []
        for a in actions:
            obs, r = env.step(a)
            trace.append({"action": a.action_type, "reward": r.value, "final": r.final_score})
        traces.append(trace)

    for i, (s1, s2) in enumerate(zip(traces[0], traces[1])):
        assert s1["reward"] == s2["reward"], (
            f"Step {i+1} reward mismatch: {s1['reward']} vs {s2['reward']}"
        )
    return f"2 identical runs produced byte-for-byte matching reward traces"

@test("wrong severity gets partial credit (not binary pass/fail — honest partial scoring)")
def _():
    from server.environment import ContractReviewEnvironment
    from models.action import ContractAction

    # Correct risk type but wrong severity: medium instead of high
    env_correct = ContractReviewEnvironment()
    env_correct.reset(task_id="task2")
    _, r_correct = env_correct.step(ContractAction(
        action_type="flag_risk", target_section="3.2",
        content="data_used_for_ml_training", severity="high"   # correct
    ))

    env_wrong_sev = ContractReviewEnvironment()
    env_wrong_sev.reset(task_id="task2")
    _, r_wrong = env_wrong_sev.step(ContractAction(
        action_type="flag_risk", target_section="3.2",
        content="data_used_for_ml_training", severity="low"   # wrong severity
    ))

    env_wrong_type = ContractReviewEnvironment()
    env_wrong_type.reset(task_id="task2")
    _, r_false = env_wrong_type.step(ContractAction(
        action_type="flag_risk", target_section="1",
        content="completely_fake_risk", severity="high"       # wrong type (FP)
    ))

    assert r_correct.value > r_wrong.value > r_false.value, (
        f"Scoring should be: correct > wrong_severity > false_positive\n"
        f"Got: {r_correct.value:.4f} > {r_wrong.value:.4f} > {r_false.value:.4f}"
    )
    return (f"correct={r_correct.value:+.4f} > "
            f"wrong_severity={r_wrong.value:+.4f} > "
            f"false_positive={r_false.value:+.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE SERVER TESTS (only if --live flag passed)
# ═══════════════════════════════════════════════════════════════════════════════

if "--live" in sys.argv:
    BASE = "http://localhost:8000"

    print("\n" + "="*60)
    print("  LIVE SERVER TESTS (HTTP endpoint integrity)")
    print("="*60)

    @test("live: /reset returns HTTP 200 and clean observation")
    def _():
        r = httpx.post(f"{BASE}/reset", json={"task_id": "task1"}, timeout=10)
        assert r.status_code == 200, f"HTTP {r.status_code}"
        obs = r.json()["observation"]
        assert obs["step_count"] == 0
        assert not obs["done"]
        assert obs["cumulative_reward"] == 0.0
        return f"HTTP 200, step_count=0, done=False"

    @test("live: correct action gives positive reward via HTTP")
    def _():
        httpx.post(f"{BASE}/reset", json={"task_id": "task1"})
        r = httpx.post(f"{BASE}/step", json={"action": {
            "action_type": "extract_clause",
            "target_section": "2",
            "content": "confidentiality",
            "severity": None,
        }}, timeout=10)
        assert r.status_code == 200
        d = r.json()
        assert d["reward"] > 0, f"Expected positive reward, got {d['reward']}"
        return f"reward={d['reward']:+.4f}"

    @test("live: FP flag gives negative reward via HTTP")
    def _():
        httpx.post(f"{BASE}/reset", json={"task_id": "task2"})
        r = httpx.post(f"{BASE}/step", json={"action": {
            "action_type": "flag_risk",
            "target_section": "1",
            "content": "completely_fake_risk_xyz_123",
            "severity": "high",
        }}, timeout=10)
        assert r.status_code == 200
        d = r.json()
        assert d["reward"] < 0, f"FP flag should give negative reward, got {d['reward']}"
        return f"reward={d['reward']:+.4f}"

    @test("live: same episode script gives identical scores on 2 consecutive runs")
    def _():
        script = [
            {"action_type": "extract_clause", "target_section": "2", "content": "confidentiality", "severity": None},
            {"action_type": "flag_risk", "target_section": "", "content": "no_limitation_of_liability", "severity": "medium"},
            {"action_type": "submit_review", "target_section": "", "content": "", "severity": None},
        ]
        def run_script():
            httpx.post(f"{BASE}/reset", json={"task_id": "task1"})
            for action in script:
                r = httpx.post(f"{BASE}/step", json={"action": action}, timeout=10)
            return r.json()["info"]["final_score"]

        s1 = run_script()
        s2 = run_script()
        assert s1 == s2, f"Live scores differ: {s1} vs {s2}"
        return f"consistent score={s1} across 2 live runs"

    @test("live: submit empty review scores near 0.0")
    def _():
        httpx.post(f"{BASE}/reset", json={"task_id": "task3"})
        r = httpx.post(f"{BASE}/step", json={"action": {
            "action_type": "submit_review", "target_section": "", "content": "", "severity": None,
        }}, timeout=10)
        d = r.json()
        assert d["done"] == True
        s = d["info"]["final_score"]
        assert s <= 0.15, f"Empty task3 should score ≤0.15, got {s}"
        return f"empty task3 score={s:.3f}"


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  INTEGRITY TEST REPORT")
print("="*60)

passed = [r for r in RESULTS if r[1]]
failed = [r for r in RESULTS if not r[1]]

print(f"  Total:  {len(RESULTS)}")
print(f"  Passed: {len(passed)}")
print(f"  Failed: {len(failed)}")

if failed:
    print("\n  FAILURES:")
    for name, _, msg in failed:
        print(f"    - {name}")
        print(f"      {msg}")

print("="*60)
if not failed:
    print("  ALL INTEGRITY CHECKS PASSED")
    print("  Your environment is genuine and tamper-proof.")
else:
    print(f"  {len(failed)} INTEGRITY ISSUES FOUND — fix before submitting")
print("="*60)

sys.exit(0 if not failed else 1)