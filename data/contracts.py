"""
Synthetic contract dataset with embedded ground-truth annotations.
All documents are procedurally generated — no copyright issues.
"""

from typing import Dict, Any, List

CONTRACTS: Dict[str, Dict[str, Any]] = {

    # ─── TASK 1 DOCUMENTS: NDA / Simple (Easy) ────────────────────────────────
    "nda_tech_001": {
        "type": "NDA",
        "title": "Mutual Non-Disclosure Agreement — TechCorp & AlphaCo",
        "difficulty": "easy",
        "task_ids": ["task1"],
        "text": """
MUTUAL NON-DISCLOSURE AGREEMENT

This Mutual Non-Disclosure Agreement ("Agreement") is entered into as of January 1, 2024,
between TechCorp Inc., a Delaware corporation ("Party A"), and AlphaCo Ltd., a UK company ("Party B").

1. PURPOSE
The parties wish to explore a potential business relationship (the "Purpose") and may disclose
certain confidential information to each other in connection with the Purpose.

2. CONFIDENTIAL INFORMATION
2.1 "Confidential Information" means any non-public information disclosed by one party
("Disclosing Party") to the other party ("Receiving Party"), whether orally or in writing,
that is designated as confidential or that reasonably should be understood to be confidential.

2.2 Confidential Information does not include information that: (a) is or becomes publicly known
through no breach of this Agreement; (b) was rightfully known before disclosure; (c) is
independently developed without use of Confidential Information; or (d) is required to be
disclosed by law or court order.

3. TERM
3.1 This Agreement shall commence on the date first written above and shall continue for a
period of two (2) years, unless earlier terminated by either party with thirty (30) days'
written notice.

4. OBLIGATIONS
4.1 Each party agrees to: (a) hold the other party's Confidential Information in strict
confidence; (b) not disclose such information to third parties without prior written consent;
(c) use the Confidential Information solely for the Purpose; and (d) protect the Confidential
Information using the same degree of care it uses for its own confidential information, but in
no event less than reasonable care.

5. GOVERNING LAW
5.1 This Agreement shall be governed by and construed in accordance with the laws of the
State of Delaware, without regard to its conflict of laws provisions.

6. DISPUTE RESOLUTION
6.1 Any dispute arising out of or relating to this Agreement shall be resolved by binding
arbitration under the rules of the American Arbitration Association. The arbitration shall
take place in Wilmington, Delaware.

7. ENTIRE AGREEMENT
7.1 This Agreement constitutes the entire agreement between the parties with respect to the
subject matter hereof and supersedes all prior negotiations, representations, or agreements.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first written above.

TECHCORP INC.                          ALPHACO LTD.
By: ___________________                By: ___________________
Name: John Smith                       Name: Jane Doe
Title: CEO                             Title: Managing Director
""",
        "ground_truth": {
            "clauses": [
                {"type": "confidentiality", "section": "2", "present": True,
                 "key_terms": ["non-public", "designated as confidential"]},
                {"type": "term", "section": "3", "present": True,
                 "duration": "2 years", "notice_period": "30 days"},
                {"type": "governing_law", "section": "5", "present": True,
                 "jurisdiction": "Delaware"},
                {"type": "dispute_resolution", "section": "6", "present": True,
                 "mechanism": "arbitration", "venue": "Wilmington, Delaware"},
                {"type": "entire_agreement", "section": "7", "present": True},
            ],
            "risks": [
                {
                    "type": "no_limitation_of_liability",
                    "section": None,
                    "severity": "medium",
                    "description": "Agreement lacks a limitation of liability clause — "
                                   "exposure is uncapped if confidential info is misused."
                },
            ],
            "approved_sections": ["1", "4", "7"],
        }
    },

    "nda_saas_002": {
        "type": "NDA",
        "title": "One-Way NDA — SaaSVendor & ClientCo",
        "difficulty": "easy",
        "task_ids": ["task1"],
        "text": """
UNILATERAL NON-DISCLOSURE AGREEMENT

Date: March 15, 2024
Disclosing Party: SaaSVendor Inc. ("Company")
Receiving Party: ClientCo LLC ("Recipient")

1. DEFINITION OF CONFIDENTIAL INFORMATION
1.1 For purposes of this Agreement, "Confidential Information" means all technical,
business, financial, customer, or product information disclosed by Company to Recipient,
whether in oral, written, electronic, or any other form, including but not limited to:
source code, algorithms, product roadmaps, pricing strategies, and customer lists.

2. TERM OF AGREEMENT
2.1 This Agreement shall be effective from the date first written above and shall
remain in effect for three (3) years, unless terminated earlier by Company upon
fourteen (14) days written notice.

3. RECIPIENT'S OBLIGATIONS
3.1 Recipient agrees to: (a) maintain the confidentiality of all Confidential Information;
(b) use Confidential Information only for evaluating a potential business engagement with
Company; (c) restrict disclosure to employees with a need-to-know who are bound by
confidentiality obligations no less restrictive than those in this Agreement.

4. INTELLECTUAL PROPERTY
4.1 All Confidential Information disclosed under this Agreement remains the exclusive
property of Company. Nothing in this Agreement grants Recipient any license or right
to use Company's intellectual property except as expressly set forth herein.

5. GOVERNING LAW AND JURISDICTION
5.1 This Agreement is governed by the laws of California. Any legal action shall be
brought exclusively in the state or federal courts of Santa Clara County, California.

6. REMEDIES
6.1 Recipient acknowledges that breach of this Agreement may cause irreparable harm
and that monetary damages may be inadequate. Company shall be entitled to seek
equitable relief including injunction without posting bond.

AGREED AND ACCEPTED:
SaaSVendor Inc.                        ClientCo LLC
Signature: ____________                Signature: ____________
""",
        "ground_truth": {
            "clauses": [
                {"type": "confidentiality", "section": "1", "present": True,
                 "key_terms": ["technical", "business", "financial", "source code"]},
                {"type": "term", "section": "2", "present": True,
                 "duration": "3 years", "notice_period": "14 days"},
                {"type": "governing_law", "section": "5", "present": True,
                 "jurisdiction": "California"},
                {"type": "dispute_resolution", "section": "5", "present": True,
                 "mechanism": "litigation", "venue": "Santa Clara County"},
                {"type": "ip_ownership", "section": "4", "present": True},
                {"type": "remedies", "section": "6", "present": True},
            ],
            "risks": [
                {
                    "type": "no_limitation_of_liability",
                    "section": None,
                    "severity": "medium",
                    "description": "No limitation of liability clause present."
                },
                {
                    "type": "unilateral_agreement",
                    "section": "1",
                    "severity": "low",
                    "description": "Agreement is one-way — Recipient receives no "
                                   "confidentiality protection for its own disclosures."
                },
            ],
            "approved_sections": ["3", "6"],
        }
    },

    # ─── TASK 2 DOCUMENTS: SaaS Agreement (Medium) ────────────────────────────
    "saas_subscription_001": {
        "type": "SaaS Subscription Agreement",
        "title": "SaaS Master Subscription Agreement — CloudBase & Acme Corp",
        "difficulty": "medium",
        "task_ids": ["task2"],
        "text": """
MASTER SUBSCRIPTION AGREEMENT

This Master Subscription Agreement ("Agreement") is made as of February 1, 2024,
between CloudBase Technologies Inc. ("Provider") and Acme Corporation ("Customer").

1. SUBSCRIPTION SERVICES
1.1 Provider grants Customer a non-exclusive, non-transferable right to access and
use the Provider's cloud-based software platform ("Services") during the Subscription Term.

1.2 Customer may not sublicense, sell, resell, transfer, assign, or otherwise commercially
exploit or make available to any third party the Services.

2. FEES AND PAYMENT
2.1 Customer shall pay all fees as specified in the applicable Order Form. Fees are
non-refundable except as expressly set forth herein.

2.2 Provider reserves the right to modify fees upon ninety (90) days' notice. If Customer
does not agree to the modified fees, Customer may terminate the Agreement upon written
notice within thirty (30) days of receiving the fee modification notice.

2.3 All amounts past due shall accrue interest at the rate of 3% per month, or the
maximum rate permitted by law, whichever is greater.

3. DATA PROCESSING AND PRIVACY
3.1 Provider will process Customer Data only as necessary to provide the Services and
as instructed by Customer. Provider shall implement appropriate technical and
organizational measures to protect Customer Data.

3.2 Provider may use Customer Data for its own product improvement, analytics, and
machine learning model training without restriction or compensation to Customer.

3.3 In the event of a data breach, Provider shall notify Customer within seventy-two (72)
hours of becoming aware of the breach.

4. INTELLECTUAL PROPERTY
4.1 As between the parties, Customer retains all right, title, and interest in Customer Data.

4.2 Customer hereby grants Provider a perpetual, irrevocable, worldwide, royalty-free license
to use, copy, modify, create derivative works from, and distribute any feedback, suggestions,
or recommendations ("Feedback") provided by Customer.

4.3 Provider retains all rights in the Services, including all improvements made based on
Customer's use patterns, even where such improvements were directly caused by Customer's
unique use case.

5. LIMITATION OF LIABILITY
5.1 IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR INDIRECT, INCIDENTAL, SPECIAL, OR
CONSEQUENTIAL DAMAGES.

5.2 PROVIDER'S TOTAL CUMULATIVE LIABILITY SHALL NOT EXCEED THE LESSER OF (A) THE AMOUNTS
PAID BY CUSTOMER IN THE THREE (3) MONTHS PRECEDING THE CLAIM, OR (B) ONE THOUSAND DOLLARS ($1,000).

6. WARRANTIES AND DISCLAIMER
6.1 Provider warrants that the Services will perform materially in accordance with the
applicable documentation.

6.2 EXCEPT AS SET FORTH IN SECTION 6.1, THE SERVICES ARE PROVIDED "AS IS" WITHOUT WARRANTY
OF ANY KIND.

7. TERM AND TERMINATION
7.1 This Agreement commences on the Effective Date and continues for one (1) year,
automatically renewing for successive one (1) year terms unless terminated.

7.2 Provider may terminate or suspend Services immediately and without notice for any
breach or for any reason at Provider's sole discretion.

8. GOVERNING LAW
8.1 This Agreement is governed by the laws of the State of New York.

9. INDEMNIFICATION
9.1 Customer shall indemnify and hold harmless Provider from any claims arising from
Customer's use of the Services, breach of this Agreement, or violation of any law.
Provider shall have no indemnification obligations to Customer.
""",
        "ground_truth": {
            "clauses": [
                {"type": "subscription_rights", "section": "1", "present": True},
                {"type": "payment_terms", "section": "2", "present": True},
                {"type": "data_processing", "section": "3", "present": True},
                {"type": "ip_ownership", "section": "4", "present": True},
                {"type": "limitation_of_liability", "section": "5", "present": True},
                {"type": "warranty", "section": "6", "present": True},
                {"type": "term_termination", "section": "7", "present": True},
                {"type": "governing_law", "section": "8", "present": True},
                {"type": "indemnification", "section": "9", "present": True},
            ],
            "risks": [
                {
                    "type": "data_used_for_ml_training",
                    "section": "3.2",
                    "severity": "high",
                    "description": "Section 3.2 allows Provider to use Customer Data for "
                                   "ML model training without restriction or compensation — "
                                   "significant IP and confidentiality risk."
                },
                {
                    "type": "inadequate_liability_cap",
                    "section": "5.2",
                    "severity": "high",
                    "description": "Liability cap is lesser of 3-month fees or $1,000 — "
                                   "absurdly low and heavily favors Provider."
                },
                {
                    "type": "termination_for_convenience_provider_only",
                    "section": "7.2",
                    "severity": "medium",
                    "description": "Provider can terminate immediately without notice for "
                                   "any reason — Customer has no equivalent right."
                },
                {
                    "type": "unilateral_indemnification",
                    "section": "9.1",
                    "severity": "medium",
                    "description": "Indemnification is entirely one-sided — Customer must "
                                   "indemnify Provider but receives no indemnification."
                },
                {
                    "type": "perpetual_feedback_license",
                    "section": "4.2",
                    "severity": "low",
                    "description": "Feedback license is perpetual, irrevocable, and royalty-free — "
                                   "Provider benefits from Customer input indefinitely."
                },
            ],
            "approved_sections": ["1", "6", "8"],
        }
    },

    # ─── TASK 3 DOCUMENTS: Employment Agreement (Hard) ────────────────────────
    "employment_senior_001": {
        "type": "Employment Agreement",
        "title": "Senior Engineer Employment Agreement — MegaCorp & Candidate",
        "difficulty": "hard",
        "task_ids": ["task3"],
        "text": """
SENIOR ENGINEER EMPLOYMENT AGREEMENT

This Employment Agreement ("Agreement") is entered into as of April 1, 2024, between
MegaCorp Inc., a Delaware corporation ("Company"), and the individual identified below ("Employee").

1. POSITION AND DUTIES
1.1 Employee is hired as Senior Software Engineer reporting to the VP of Engineering.
1.2 Employee agrees to devote substantially all of their working time and attention to
the Company's business and shall not engage in any other employment or business activity
without prior written consent of the Company.

2. COMPENSATION
2.1 Base Salary: Employee shall receive an annual base salary of $180,000, paid in
accordance with Company's standard payroll schedule.
2.2 Performance Bonus: Employee is eligible for an annual discretionary bonus of up
to 20% of base salary, payable at Company's sole discretion.
2.3 Equity: Employee shall receive 10,000 restricted stock units ("RSUs"), vesting
over four (4) years with a one-year cliff.

3. BENEFITS
3.1 Employee is entitled to standard Company benefits including health insurance,
401(k) plan, and paid time off per Company policy.

4. INTELLECTUAL PROPERTY ASSIGNMENT
4.1 Employee agrees that all work product, inventions, discoveries, improvements,
and innovations ("Work Product") made or conceived by Employee during employment,
whether or not during working hours, whether or not using Company resources, and
whether or not related to Company's business, shall be the exclusive property of Company.

4.2 Employee hereby assigns and agrees to assign to Company all right, title, and
interest in all Work Product, including all patents, copyrights, and trade secrets.

4.3 This assignment applies to Employee's prior inventions to the extent they are
incorporated into Company's products or services.

5. NON-COMPETE COVENANT
5.1 During employment and for a period of two (2) years following termination for
any reason, Employee agrees not to, directly or indirectly: (a) engage in, own, manage,
operate, control, be employed by, consult for, or participate in any business that
competes with the Company anywhere in the world; (b) solicit any customer, client, or
business partner of the Company.

5.2 Employee acknowledges that the scope of this restriction is reasonable and
necessary to protect Company's legitimate business interests.

6. NON-SOLICITATION
6.1 For two (2) years post-termination, Employee shall not recruit, solicit, or
encourage any Company employee or contractor to leave the Company's employ.

7. CONFIDENTIALITY
7.1 Employee agrees to maintain in strict confidence all Confidential Information
and not to disclose or use any Confidential Information except as required by
Employee's duties at the Company.

8. TERMINATION
8.1 At-Will Employment: Employment is at-will. Either party may terminate employment
at any time, with or without cause.
8.2 Severance: No severance shall be paid upon termination for any reason.
8.3 Garden Leave: Upon resignation, Employee shall serve a mandatory ninety (90) day
garden leave period during which Employee remains employed but may be excluded from
all systems and communications.

9. DISPUTE RESOLUTION
9.1 Any dispute shall be resolved exclusively through binding arbitration under JAMS
rules. Employee waives any right to a jury trial.
9.2 Claims must be brought within six (6) months of the event giving rise to the claim.

10. NON-DISPARAGEMENT
10.1 Employee agrees never to make any negative, critical, or disparaging remarks
about the Company, its officers, directors, employees, products, or services, through
any medium, including social media, forever.

11. GOVERNING LAW
11.1 This Agreement is governed by the laws of Delaware.

12. ENTIRE AGREEMENT
12.1 This Agreement supersedes all prior agreements, representations, and understandings
related to Employee's employment.
""",
        "ground_truth": {
            "clauses": [
                {"type": "position_duties", "section": "1", "present": True},
                {"type": "compensation", "section": "2", "present": True},
                {"type": "benefits", "section": "3", "present": True},
                {"type": "ip_assignment", "section": "4", "present": True},
                {"type": "non_compete", "section": "5", "present": True},
                {"type": "non_solicitation", "section": "6", "present": True},
                {"type": "confidentiality", "section": "7", "present": True},
                {"type": "termination", "section": "8", "present": True},
                {"type": "dispute_resolution", "section": "9", "present": True},
                {"type": "non_disparagement", "section": "10", "present": True},
                {"type": "governing_law", "section": "11", "present": True},
                {"type": "entire_agreement", "section": "12", "present": True},
            ],
            "risks": [
                {
                    "type": "overbroad_ip_assignment",
                    "section": "4.1",
                    "severity": "blocking",
                    "description": "Section 4.1 assigns ALL inventions regardless of whether "
                                   "made during work hours or using company resources — violates "
                                   "employee rights in CA, WA, IL, MN, NC, and DE. Must be "
                                   "narrowed to work-related inventions using company resources."
                },
                {
                    "type": "worldwide_noncompete",
                    "section": "5.1",
                    "severity": "blocking",
                    "description": "Two-year worldwide non-compete is unenforceable in most "
                                   "US states including CA, ND, OK, and MN, and at risk in "
                                   "many others. Geographic scope must be dramatically reduced "
                                   "or clause removed entirely."
                },
                {
                    "type": "shortened_statute_of_limitations",
                    "section": "9.2",
                    "severity": "high",
                    "description": "6-month claim limitation is below statutory minimums for "
                                   "employment claims in most jurisdictions — likely unenforceable "
                                   "and exposes Company to challenge."
                },
                {
                    "type": "perpetual_non_disparagement",
                    "section": "10.1",
                    "severity": "high",
                    "description": "Perpetual non-disparagement with no carve-outs for protected "
                                   "activity (NLRA rights, whistleblowing, government reporting) "
                                   "is unenforceable and exposes Company to NLRB complaints."
                },
                {
                    "type": "no_severance",
                    "section": "8.2",
                    "severity": "medium",
                    "description": "Zero severance with 90-day mandatory garden leave creates "
                                   "risk of unjust enrichment claims — employee works garden "
                                   "leave without compensation path post-termination."
                },
                {
                    "type": "prior_inventions_sweep",
                    "section": "4.3",
                    "severity": "medium",
                    "description": "Prior inventions assignment is overbroad — should require "
                                   "Employee to schedule prior inventions they wish to retain."
                },
            ],
            "approved_sections": ["1", "2", "3", "6", "7", "11", "12"],
        }
    },
}


def get_contract(contract_id: str) -> Dict[str, Any]:
    if contract_id not in CONTRACTS:
        raise ValueError(f"Unknown contract: {contract_id}")
    return CONTRACTS[contract_id]


def get_contracts_for_task(task_id: str) -> List[str]:
    return [k for k, v in CONTRACTS.items() if task_id in v["task_ids"]]


TASK_CONTRACT_MAP = {
    "task1": ["nda_tech_001", "nda_saas_002"],
    "task2": ["saas_subscription_001"],
    "task3": ["employment_senior_001"],
}
