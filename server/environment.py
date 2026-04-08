"""
ContractReviewEnvironment — core RL environment logic.
Implements reset(), step(), and state() per OpenEnv spec.
"""

import uuid
import hashlib
from typing import Any, Dict, Optional, Tuple

from models.action import ContractAction
from models.observation import (
    ContractObservation,
    ContractState,
    ContractReward,
    ExtractedClause,
    RaisedFlag,
)
from data.contracts import CONTRACTS, TASK_CONTRACT_MAP
from server.graders import grade_episode

# Max steps per task
TASK_MAX_STEPS: Dict[str, int] = {
    "task1": 15,
    "task2": 20,
    "task3": 25,
}

# Per-step cost to encourage efficiency
STEP_COST = 0.008


class ContractReviewEnvironment:
    """
    A contract document review environment for RL agent training.

    Episode lifecycle
    -----------------
    1. reset(task_id, contract_id) → ContractObservation (clean state)
    2. step(action) → (ContractObservation, ContractReward)  [repeat]
    3. Episode ends when agent calls submit_review or max_steps is reached.
    """

    def __init__(self) -> None:
        self._state: Optional[ContractState] = None
        self._contract_data: Optional[Dict[str, Any]] = None

    # ─── Public API ────────────────────────────────────────────────────────────

    def reset(
        self,
        task_id: str = "task1",
        contract_id: Optional[str] = None,
    ) -> ContractObservation:
        """
        Start a fresh episode.

        Parameters
        ----------
        task_id    : "task1", "task2", or "task3"
        contract_id: specific contract to use, or None to use task default
        """
        if task_id not in TASK_MAX_STEPS:
            raise ValueError(f"Unknown task_id '{task_id}'. Must be task1/task2/task3.")

        # Select contract
        if contract_id is None:
            options = TASK_CONTRACT_MAP.get(task_id, [])
            if not options:
                raise ValueError(f"No contracts configured for task '{task_id}'.")
            contract_id = options[0]

        if contract_id not in CONTRACTS:
            raise ValueError(f"Unknown contract_id '{contract_id}'.")

        self._contract_data = CONTRACTS[contract_id]

        self._state = ContractState(
            episode_id=str(uuid.uuid4()),
            task_id=task_id,
            contract_id=contract_id,
            max_steps=TASK_MAX_STEPS[task_id],
        )

        return self._build_observation(message="Episode started. Review the contract below.")

    def step(self, action: ContractAction) -> Tuple[ContractObservation, ContractReward]:
        """
        Apply an action and return the new observation and reward.
        """
        if self._state is None or self._contract_data is None:
            raise RuntimeError("Call reset() before step().")

        if self._state.done:
            obs = self._build_observation(message="Episode is already done.")
            return obs, ContractReward(value=0.0, reason="Episode already done.", is_terminal=True)

        self._state.step_count += 1

        # Dedup check — same action twice gives 0 reward
        action_hash = self._hash_action(action)
        if action_hash in self._state.seen_action_hashes:
            reward = ContractReward(
                value=-0.05,
                reason="Duplicate action — no new information extracted.",
            )
            self._state.cumulative_reward += reward.value
            self._state.last_action_reward = reward.value
            obs = self._build_observation(
                message="You already performed this action. Try something different."
            )
            return obs, reward

        self._state.seen_action_hashes.append(action_hash)

        # Dispatch
        if action.action_type == "submit_review":
            return self._handle_submit()

        reward = self._compute_step_reward(action)
        self._apply_action_to_state(action)

        # Step budget exhaustion
        if self._state.step_count >= self._state.max_steps:
            return self._handle_budget_exhausted()

        self._state.cumulative_reward += reward.value
        self._state.last_action_reward = reward.value

        obs = self._build_observation(message=reward.reason)
        return obs, reward

    def state(self) -> ContractState:
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._state

    # ─── Reward computation ────────────────────────────────────────────────────

    def _compute_step_reward(self, action: ContractAction) -> ContractReward:
        gt = self._contract_data["ground_truth"]
        a_type = action.action_type

        # Base step cost
        base = -STEP_COST

        if a_type == "extract_clause":
            return self._reward_extract_clause(action, gt, base)

        if a_type == "flag_risk":
            return self._reward_flag_risk(action, gt, base)

        if a_type == "approve_section":
            return self._reward_approve_section(action, gt, base)

        if a_type == "request_revision":
            return self._reward_request_revision(action, gt, base)

        if a_type == "annotate":
            # Annotations are free — no reward, no penalty (beyond step cost)
            return ContractReward(
                value=round(base, 4),
                reason="Annotation recorded.",
            )

        return ContractReward(value=round(base, 4), reason="Action recorded.")

    def _reward_extract_clause(
        self, action: ContractAction, gt: Dict, base: float
    ) -> ContractReward:
        from server.graders import _clause_match, _section_match

        agent_type = action.content.strip()
        agent_sec = action.target_section.strip()

        # Check against ground truth clauses
        for gt_clause in gt.get("clauses", []):
            if _clause_match(agent_type, gt_clause["type"]):
                sec_s = _section_match(agent_sec, str(gt_clause.get("section", "")))
                if sec_s == 1.0:
                    r = base + 0.18
                    return ContractReward(
                        value=round(r, 4),
                        reason=f"Correct clause '{agent_type}' extracted from correct section {agent_sec}.",
                    )
                elif sec_s > 0:
                    r = base + 0.08
                    return ContractReward(
                        value=round(r, 4),
                        reason=f"Clause '{agent_type}' found but section reference is approximate.",
                    )
                else:
                    r = base + 0.04
                    return ContractReward(
                        value=round(r, 4),
                        reason=f"Clause type '{agent_type}' is correct but section '{agent_sec}' is wrong.",
                    )

        # False positive — clause not in ground truth
        r = base - 0.04
        return ContractReward(
            value=round(r, 4),
            reason=f"No matching clause of type '{agent_type}' found in document.",
        )

    def _reward_flag_risk(
        self, action: ContractAction, gt: Dict, base: float
    ) -> ContractReward:
        from server.graders import _risk_match, SEVERITY_WEIGHTS

        if not action.severity:
            return ContractReward(
                value=round(base - 0.05, 4),
                reason="flag_risk requires a severity level.",
            )

        agent_type = action.content.strip()
        agent_sev = action.severity

        for gt_risk in gt.get("risks", []):
            if _risk_match(agent_type, gt_risk["type"]):
                weight = SEVERITY_WEIGHTS[gt_risk["severity"]]
                if agent_sev == gt_risk["severity"]:
                    r = base + weight * 0.35
                    return ContractReward(
                        value=round(r, 4),
                        reason=f"Risk '{agent_type}' correctly identified as {agent_sev}.",
                    )
                else:
                    r = base + weight * 0.15
                    return ContractReward(
                        value=round(r, 4),
                        reason=f"Risk '{agent_type}' found but severity should be "
                               f"'{gt_risk['severity']}' not '{agent_sev}'.",
                    )

        # False positive
        r = base - 0.12
        return ContractReward(
            value=round(r, 4),
            reason=f"'{agent_type}' is not a valid risk in this document — false positive.",
        )

    def _reward_approve_section(
        self, action: ContractAction, gt: Dict, base: float
    ) -> ContractReward:
        sec = action.target_section.strip()
        gt_approved = set(gt.get("approved_sections", []))
        # A section is risky if any risk lives in it or its subsections
        # e.g. approving "4" should penalise if "4.1" or "4.3" has risks
        all_risky_sections = {r["section"] for r in gt.get("risks", []) if r.get("section")}

        def section_has_risk(s: str) -> bool:
            """True if section s or any of its subsections has a risk."""
            return any(
                rs == s or rs.startswith(s + ".")
                for rs in all_risky_sections
            )

        if sec in gt_approved:
            return ContractReward(
                value=round(base + 0.08, 4),
                reason=f"Section {sec} is clean — correct to approve.",
            )
        if section_has_risk(sec):
            return ContractReward(
                value=round(base - 0.12, 4),
                reason=f"Section {sec} contains risks — should not be approved without revision.",
            )
        # Neutral section
        return ContractReward(
            value=round(base + 0.02, 4),
            reason=f"Section {sec} approved.",
        )

    def _reward_request_revision(
        self, action: ContractAction, gt: Dict, base: float
    ) -> ContractReward:
        sec = action.target_section.strip()
        all_risky_sections = {r["section"] for r in gt.get("risks", []) if r.get("section")}
        gt_approved = set(gt.get("approved_sections", []))

        def section_has_risk(s: str) -> bool:
            return any(rs == s or rs.startswith(s + ".") for rs in all_risky_sections)

        if section_has_risk(sec):
            return ContractReward(
                value=round(base + 0.10, 4),
                reason=f"Section {sec} does have issues — revision correctly requested.",
            )
        if sec in gt_approved:
            return ContractReward(
                value=round(base - 0.06, 4),
                reason=f"Section {sec} is clean — unnecessary revision request.",
            )
        return ContractReward(
            value=round(base + 0.02, 4),
            reason=f"Revision request for section {sec} recorded.",
        )

    # ─── State mutations ───────────────────────────────────────────────────────

    def _apply_action_to_state(self, action: ContractAction) -> None:
        s = self._state

        if action.action_type == "extract_clause":
            s.clauses_extracted.append(
                ExtractedClause(
                    clause_type=action.content,
                    section=action.target_section,
                    step_extracted=s.step_count,
                )
            )

        elif action.action_type == "flag_risk":
            s.flags_raised.append(
                RaisedFlag(
                    risk_type=action.content,
                    section=action.target_section,
                    severity=action.severity or "low",
                    step_raised=s.step_count,
                )
            )

        elif action.action_type == "approve_section":
            if action.target_section not in s.approved_sections:
                s.approved_sections.append(action.target_section)

        elif action.action_type == "request_revision":
            if action.target_section not in s.revision_requests:
                s.revision_requests.append(action.target_section)

    # ─── Terminal transitions ──────────────────────────────────────────────────

    def _handle_submit(self) -> Tuple[ContractObservation, ContractReward]:
        self._state.done = True
        final_score = grade_episode(
            self._state.task_id,
            self._state,
            self._contract_data["ground_truth"],
        )
        self._state.final_score = final_score

        # Terminal reward: big bonus proportional to final score
        terminal_r = round(final_score * 0.6, 4)
        self._state.cumulative_reward += terminal_r
        self._state.last_action_reward = terminal_r

        obs = self._build_observation(
            message=f"Review submitted. Final score: {final_score:.3f}"
        )
        reward = ContractReward(
            value=terminal_r,
            reason=f"Episode complete. Final graded score: {final_score:.3f}",
            is_terminal=True,
            final_score=final_score,
        )
        return obs, reward

    def _handle_budget_exhausted(self) -> Tuple[ContractObservation, ContractReward]:
        """Auto-submit when steps run out."""
        self._state.done = True
        final_score = grade_episode(
            self._state.task_id,
            self._state,
            self._contract_data["ground_truth"],
        )
        self._state.final_score = final_score
        terminal_r = round(final_score * 0.4, 4)  # smaller bonus for running out
        self._state.cumulative_reward += terminal_r
        self._state.last_action_reward = terminal_r

        obs = self._build_observation(
            message=f"Step budget exhausted. Auto-submitted. Final score: {final_score:.3f}"
        )
        reward = ContractReward(
            value=terminal_r,
            reason=f"Budget exhausted. Auto-graded score: {final_score:.3f}",
            is_terminal=True,
            final_score=final_score,
        )
        return obs, reward

    # ─── Helpers ───────────────────────────────────────────────────────────────

    def _build_observation(self, message: str = "") -> ContractObservation:
        s = self._state
        c = self._contract_data
        return ContractObservation(
            document_text=c["text"],
            document_type=c["type"],
            document_title=c["title"],
            step_count=s.step_count,
            max_steps=s.max_steps,
            steps_remaining=max(0, s.max_steps - s.step_count),
            clauses_extracted=list(s.clauses_extracted),
            flags_raised=list(s.flags_raised),
            approved_sections=list(s.approved_sections),
            revision_requests=list(s.revision_requests),
            last_action_reward=s.last_action_reward,
            cumulative_reward=round(s.cumulative_reward, 4),
            done=s.done,
            task_id=s.task_id,
            contract_id=s.contract_id,
            message=message,
        )

    @staticmethod
    def _hash_action(action: ContractAction) -> str:
        key = f"{action.action_type}|{action.target_section}|{action.content}|{action.severity}"
        return hashlib.md5(key.encode()).hexdigest()