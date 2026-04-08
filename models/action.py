"""ContractAction — typed action model for the contract review environment."""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class ContractAction(BaseModel):
    """
    An action the agent can take during a contract review episode.

    action_type options
    -------------------
    extract_clause      : Identify and record a specific clause type from the document.
    flag_risk           : Raise a risk flag on a section with an assigned severity.
    annotate            : Add a free-text annotation to a section (informational).
    approve_section     : Mark a section as reviewed and acceptable.
    request_revision    : Mark a section as requiring redline / revision.
    submit_review       : End the episode and submit the completed review.
    """

    action_type: Literal[
        "extract_clause",
        "flag_risk",
        "annotate",
        "approve_section",
        "request_revision",
        "submit_review",
    ] = Field(..., description="The type of review action to perform.")

    target_section: Optional[str] = Field(
        default="",
        description="The section number or identifier being acted on (e.g. '3.2', '5').",
    )

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    def __init__(self, **data):
        # Coerce None → "" for target_section and content
        if data.get("target_section") is None:
            data["target_section"] = ""
        if data.get("content") is None:
            data["content"] = ""
        super().__init__(**data)

    content: Optional[str] = Field(
        default="",
        description=(
            "For extract_clause: the clause type (e.g. 'confidentiality', 'term', "
            "'governing_law', 'dispute_resolution', 'ip_assignment', 'non_compete'). "
            "For flag_risk: the risk type identifier. "
            "For annotate / request_revision: free-text description. "
            "For approve_section / submit_review: can be empty."
        ),
    )

    severity: Optional[Literal["low", "medium", "high", "blocking"]] = Field(
        default=None,
        description="Required when action_type is 'flag_risk'. Severity of the identified risk.",
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "action_type": "extract_clause",
                    "target_section": "2",
                    "content": "confidentiality",
                    "severity": None,
                },
                {
                    "action_type": "flag_risk",
                    "target_section": "5.1",
                    "content": "worldwide_noncompete",
                    "severity": "blocking",
                },
                {
                    "action_type": "submit_review",
                    "target_section": "",
                    "content": "",
                    "severity": None,
                },
            ]
        }
