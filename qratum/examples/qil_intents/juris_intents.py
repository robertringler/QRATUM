"""
QIL Intent Examples - JURIS (Legal AI)

Example intents for legal contract analysis, reasoning, and compliance.
"""

from qratum.platform import PlatformIntent

# Example 1: Contract Risk Analysis
contract_analysis_intent = PlatformIntent(
    vertical="JURIS",
    task="analyze_contract",
    parameters={
        "contract_text": """
        SERVICE AGREEMENT
        
        This Agreement is entered into between Company A ("Provider") and Company B ("Client").
        
        1. SERVICES: Provider shall deliver software development services as specified in Exhibit A.
        
        2. PAYMENT: Client shall pay $100,000 upon completion, with 50% due within 30 days.
        
        3. INDEMNIFICATION: Provider shall indemnify Client against all claims arising from 
           intellectual property infringement.
        
        4. LIABILITY: Provider's total liability shall not exceed the fees paid under this Agreement.
        
        5. TERMINATION: Either party may terminate with 30 days notice. Upon breach, immediate 
           termination is permitted with penalty equal to 50% of remaining contract value.
        
        6. CONFIDENTIALITY: Both parties must maintain confidentiality of proprietary information 
           for 5 years after termination.
        """
    },
    requester_id="legal_team_001",
)


# Example 2: Legal Reasoning (IRAC Framework)
legal_reasoning_intent = PlatformIntent(
    vertical="JURIS",
    task="legal_reasoning",
    parameters={
        "issue": "Whether the defendant breached the non-compete agreement",
        "facts": """
        Employee signed non-compete preventing work in same industry for 2 years within 50 miles.
        Employee left company and joined competitor 6 months later, located 30 miles away.
        Employee argues non-compete is overly restrictive and unenforceable.
        """,
        "applicable_law": "State contract law generally enforces reasonable non-compete agreements",
    },
    requester_id="legal_team_001",
)


# Example 3: Litigation Outcome Prediction
litigation_prediction_intent = PlatformIntent(
    vertical="JURIS",
    task="predict_litigation",
    parameters={
        "case_facts": """
        Patent infringement case. Plaintiff holds patent on mobile payment technology.
        Defendant launched similar product 2 years after patent grant.
        Defendant argues patent is overly broad and invalid.
        Discovery shows defendant engineers reviewed patent documentation.
        """,
        "jurisdiction": "US District Court, Northern California",
    },
    requester_id="legal_team_001",
)


# Example 4: Regulatory Compliance Check
compliance_check_intent = PlatformIntent(
    vertical="JURIS",
    task="check_compliance",
    parameters={
        "document": """
        PRIVACY POLICY
        
        We collect user data including name, email, and usage patterns.
        Data is stored on secure servers with encryption.
        Data may be shared with third-party analytics providers.
        Users can request data deletion via email.
        """,
        "regulations": ["GDPR", "CCPA", "HIPAA"],
    },
    requester_id="compliance_team",
)


# Example 5: Complex M&A Contract Analysis
ma_contract_intent = PlatformIntent(
    vertical="JURIS",
    task="analyze_contract",
    parameters={
        "contract_text": """
        MERGER AGREEMENT
        
        Acquirer shall purchase 100% of Target's outstanding shares for $500M.
        
        Representations and Warranties:
        - Target has no undisclosed liabilities exceeding $10M
        - All intellectual property is owned free and clear
        - No pending litigation
        
        Material Adverse Change (MAC) Clause: Acquirer may terminate if Target experiences
        adverse change affecting business value by more than 15%.
        
        Indemnification: Seller indemnifies Acquirer for breach of representations up to $100M,
        with 18-month survival period.
        
        Conditions Precedent:
        - Regulatory approval from DOJ and FTC
        - Shareholder approval (66% threshold)
        - No material adverse change
        """
    },
    requester_id="ma_team_corp_dev",
)


def print_intent_examples():
    """Print all example intents"""

    print("JURIS (Legal AI) - QIL Intent Examples")
    print("=" * 80)
    print()

    examples = [
        ("Contract Risk Analysis", contract_analysis_intent),
        ("Legal Reasoning (IRAC)", legal_reasoning_intent),
        ("Litigation Prediction", litigation_prediction_intent),
        ("Compliance Check", compliance_check_intent),
        ("M&A Contract Analysis", ma_contract_intent),
    ]

    for name, intent in examples:
        print(f"\n{name}")
        print("-" * 80)
        print(f"Vertical: {intent.vertical}")
        print(f"Task: {intent.task}")
        print(f"Requester: {intent.requester_id}")
        print(f"Intent ID: {intent.intent_id}")
        print()


if __name__ == "__main__":
    print_intent_examples()
