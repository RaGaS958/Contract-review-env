"""Observation and State models for the contract review environment."""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class ExtractedClause(BaseModel):
    clause_type: str
    section: str
    step_extracted: int


class RaisedFlag(BaseModel):
    risk_type: str
    section: str
    severity: Literal["low", "medium", "high", "blocking"]
    step_raised: int


class ContractObservation(BaseModel):
    """
    Everything the agent sees after each step (or after reset).
    """

    # Document content
    document_text: str = Field(..., description="Full text of the contract document.")
    document_type: str = Field(..., description="E.g. 'NDA', 'SaaS Subscription Agreement'.")
    document_title: str = Field(..., description="Title of the contract document.")

    # Progress tracking
    step_count: int = Field(..., description="Current step number (1-indexed after first step).")
    max_steps: int = Field(..., description="Maximum steps allowed in this episode.")
    steps_remaining: int = Field(..., description="How many steps the agent has left.")

    # Agent's work so far
    clauses_extracted: List[ExtractedClause] = Field(
        default_factory=list,
        description="All clauses the agent has extracted so far.",
    )
    flags_raised: List[RaisedFlag] = Field(
        default_factory=list,
        description="All risk flags the agent has raised so far.",
    )
    approved_sections: List[str] = Field(
        default_factory=list,
        description="Sections the agent has marked as approved.",
    )
    revision_requests: List[str] = Field(
        default_factory=list,
        description="Sections the agent has flagged for revision.",
    )

    # Reward feedback
    last_action_reward: float = Field(
        default=0.0,
        description="Reward received for the last action (useful for in-context RL).",
    )
    cumulative_reward: float = Field(
        default=0.0,
        description="Total reward accumulated so far in this episode.",
    )

    # Episode status
    done: bool = Field(default=False, description="True when the episode has ended.")
    task_id: str = Field(..., description="Which task this episode is for.")
    contract_id: str = Field(..., description="Internal contract document ID.")

    # Hint for the agent
    available_actions: List[str] = Field(
        default_factory=lambda: [
            "extract_clause",
            "flag_risk",
            "annotate",
            "approve_section",
            "request_revision",
            "submit_review",
        ],
        description="Actions available to the agent.",
    )

    message: str = Field(
        default="",
        description="Optional feedback message to the agent about the last action.",
    )


class ContractState(BaseModel):
    """Internal server-side state — not sent to agent in full."""

    episode_id: str
    task_id: str
    contract_id: str

    # Episode progress
    step_count: int = 0
    max_steps: int = 20
    done: bool = False

    # Agent actions log
    clauses_extracted: List[ExtractedClause] = Field(default_factory=list)
    flags_raised: List[RaisedFlag] = Field(default_factory=list)
    approved_sections: List[str] = Field(default_factory=list)
    revision_requests: List[str] = Field(default_factory=list)

    # Dedup tracking
    seen_action_hashes: List[str] = Field(default_factory=list)

    # Reward tracking
    cumulative_reward: float = 0.0
    last_action_reward: float = 0.0

    # Final score (set when done)
    final_score: Optional[float] = None


class ContractReward(BaseModel):
    """Reward returned with each step result."""

    value: float = Field(..., description="Reward for this step, in [-1.0, 1.0].")
    reason: str = Field(..., description="Human-readable reason for this reward.")
    is_terminal: bool = Field(default=False, description="True if this ends the episode.")
    final_score: Optional[float] = Field(
        default=None,
        description="Final graded score [0, 1] — only set when is_terminal=True.",
    )
