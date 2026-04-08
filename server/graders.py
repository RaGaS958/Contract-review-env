"""
Deterministic graders for all 3 tasks.
No LLM calls — pure structured comparison against ground truth.
All graders return float in [0.0, 1.0].
"""

from typing import List, Dict, Any


# ─── Shared helpers ────────────────────────────────────────────────────────────

SEVERITY_WEIGHTS: Dict[str, float] = {
    "blocking": 1.0,
    "high": 0.8,
    "medium": 0.5,
    "low": 0.25,
}


def _normalize_type(s: str) -> str:
    """Lowercase and strip for comparison."""
    return s.lower().strip().replace("-", "_").replace(" ", "_")


def _clause_match(agent_type: str, gt_type: str) -> bool:
    return _normalize_type(agent_type) == _normalize_type(gt_type)


def _risk_match(agent_type: str, gt_type: str) -> bool:
    return _normalize_type(agent_type) == _normalize_type(gt_type)


def _section_match(agent_section: str, gt_section: str) -> float:
    """Exact section match = 1.0, same top-level section = 0.4, no match = 0.0."""
    a = str(agent_section).strip()
    g = str(gt_section).strip()
    if a == g:
        return 1.0
    if a.split(".")[0] == g.split(".")[0]:
        return 0.4
    return 0.0


# ─── Task 1: Clause Extraction ─────────────────────────────────────────────────

def grade_task1(
    agent_clauses: List[Dict[str, Any]],
    ground_truth_clauses: List[Dict[str, Any]],
) -> float:
    """
    Score = mean over GT clauses of best-match score from agent clauses.

    Scoring per GT clause:
      - Clause type matched + exact section  → 1.0
      - Clause type matched + partial section → 0.6
      - Clause type matched, no section info  → 0.4
      - No type match                         → 0.0
    """
    if not ground_truth_clauses:
        return 1.0

    total = 0.0
    for gt in ground_truth_clauses:
        best = 0.0
        for agent in agent_clauses:
            if _clause_match(agent.get("clause_type", ""), gt["type"]):
                sec_score = _section_match(
                    agent.get("section", ""),
                    str(gt.get("section", "")),
                )
                match_score = 0.4 + 0.6 * sec_score
                best = max(best, match_score)
        total += best

    return round(min(total / len(ground_truth_clauses), 1.0), 4)


# ─── Task 2: Risk Identification ───────────────────────────────────────────────

def grade_task2(
    agent_flags: List[Dict[str, Any]],
    ground_truth_risks: List[Dict[str, Any]],
) -> float:
    """
    Precision-recall style scoring weighted by severity.

    True positive  : agent flag matches GT risk type + severity
    Partial match  : agent flag matches GT risk type, wrong severity → 0.5 × weight
    False positive : agent flag has no GT match → -0.15 penalty (per flag)

    Score is clamped to [0, 1].
    """
    if not ground_truth_risks:
        return 1.0

    max_possible = sum(SEVERITY_WEIGHTS[r["severity"]] for r in ground_truth_risks)
    earned = 0.0
    fp_penalty = 0.0

    matched_gt_indices = set()

    for agent_flag in agent_flags:
        a_type = _normalize_type(agent_flag.get("risk_type", ""))
        a_sev = agent_flag.get("severity", "")
        matched = False

        for idx, gt in enumerate(ground_truth_risks):
            if not _risk_match(a_type, gt["type"]):
                continue
            matched = True
            if idx not in matched_gt_indices:
                matched_gt_indices.add(idx)
                weight = SEVERITY_WEIGHTS[gt["severity"]]
                if a_sev == gt["severity"]:
                    earned += weight          # exact match
                else:
                    earned += weight * 0.5    # right risk, wrong severity
            break  # don't double-count

        if not matched:
            fp_penalty += 0.15  # false positive

    raw = (earned - fp_penalty) / max_possible
    return round(max(0.0, min(raw, 1.0)), 4)


# ─── Task 3: Full Contract Redline ─────────────────────────────────────────────

def grade_task3(
    agent_clauses: List[Dict[str, Any]],
    agent_flags: List[Dict[str, Any]],
    approved_sections: List[str],
    revision_requests: List[str],
    step_count: int,
    max_steps: int,
    ground_truth: Dict[str, Any],
) -> float:
    """
    Multi-component score for the hard task.

    Components:
    1. Blocking risk gate   : if any blocking risk is MISSED → cap score at 0.15
    2. Clause coverage      : grade_task1 result × 0.30
    3. Risk identification  : grade_task2 result × 0.45
    4. Section disposition  : correct approve/revise of sections × 0.15
    5. Efficiency bonus     : steps saved × 0.10

    Total: up to 1.0
    """
    gt_risks = ground_truth.get("risks", [])
    gt_clauses = ground_truth.get("clauses", [])
    gt_approved = set(ground_truth.get("approved_sections", []))
    gt_risky = {r["section"] for r in gt_risks if r.get("section")}

    # ── Gate: blocking risks ───────────────────────────────────────────────────
    blocking_gt = [r for r in gt_risks if r["severity"] == "blocking"]
    blocking_caught = [
        f for f in agent_flags
        if _risk_match(f.get("risk_type", ""), "")
        or any(
            _risk_match(f.get("risk_type", ""), b["type"])
            for b in blocking_gt
        )
    ]

    # More precise blocking check
    blocking_caught_types = set()
    for f in agent_flags:
        for b in blocking_gt:
            if _risk_match(f.get("risk_type", ""), b["type"]):
                blocking_caught_types.add(b["type"])

    gate_passed = len(blocking_caught_types) >= len(blocking_gt)

    if not gate_passed:
        # Partial credit proportional to how many blocking issues were caught
        partial_blocking = len(blocking_caught_types) / max(len(blocking_gt), 1)
        return round(0.05 + 0.10 * partial_blocking, 4)

    # ── Component scores ───────────────────────────────────────────────────────
    clause_score = grade_task1(agent_clauses, gt_clauses)
    risk_score = grade_task2(agent_flags, gt_risks)

    # Section disposition score
    disposition_score = 0.0
    total_sections = len(gt_approved) + len(gt_risky - gt_approved)
    if total_sections > 0:
        correct = 0
        for sec in approved_sections:
            if sec in gt_approved:
                correct += 1
            elif sec in gt_risky:
                correct -= 0.5   # approved a risky section
        for sec in revision_requests:
            if sec in gt_risky:
                correct += 1
            elif sec in gt_approved:
                correct -= 0.25  # flagged a clean section
        disposition_score = max(0.0, min(correct / total_sections, 1.0))

    # Efficiency: reward finishing faster
    steps_used_ratio = step_count / max(max_steps, 1)
    efficiency_bonus = max(0.0, 1.0 - steps_used_ratio) * 0.5 + 0.5  # 0.5–1.0

    final = (
        clause_score      * 0.30 +
        risk_score        * 0.45 +
        disposition_score * 0.15 +
        efficiency_bonus  * 0.10
    )

    return round(min(final, 1.0), 4)


# ─── Unified grader dispatcher ─────────────────────────────────────────────────

def grade_episode(task_id: str, state: Any, ground_truth: Dict[str, Any]) -> float:
    """
    Dispatch to the correct grader based on task_id.

    Individual graders return true [0.0, 1.0] scores so integrity tests pass.
    The dispatcher clamps the final submitted score to the open interval (0, 1)
    because the evaluation platform rejects exact 0.0 and 1.0 as out-of-range.
    """
    agent_clauses = [
        {"clause_type": c.clause_type, "section": c.section}
        for c in state.clauses_extracted
    ]
    agent_flags = [
        {"risk_type": f.risk_type, "section": f.section, "severity": f.severity}
        for f in state.flags_raised
    ]

    if task_id == "task1":
        raw = grade_task1(agent_clauses, ground_truth.get("clauses", []))
    elif task_id == "task2":
        raw = grade_task2(agent_flags, ground_truth.get("risks", []))
    elif task_id == "task3":
        raw = grade_task3(
            agent_clauses=agent_clauses,
            agent_flags=agent_flags,
            approved_sections=state.approved_sections,
            revision_requests=state.revision_requests,
            step_count=state.step_count,
            max_steps=state.max_steps,
            ground_truth=ground_truth,
        )
    else:
        raise ValueError(f"Unknown task_id: {task_id}")

    # Clamp to open interval (0, 1) — platform rejects exact 0.0 and 1.0.
    # Individual graders remain pure so integrity tests continue to pass.
    return round(max(0.001, min(0.999, raw)), 4)
