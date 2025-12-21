# QRATUM-OMNILEX v1.0

## Sovereign Deterministic Legal Analysis Engine

QRATUM-OMNILEX is a comprehensive legal analysis engine that runs natively as a QRATUM workload. All legal reasoning executes as immutable, hash-chained contracts dispatched to the Frankenstein Cluster, guaranteeing deterministic replay, auditability, and sovereignty.

### ⚠️ IMPORTANT LEGAL DISCLAIMER

**THIS SYSTEM PROVIDES INFORMATION ONLY, NOT LEGAL ADVICE**

- No attorney-client relationship is created by using this system
- All analysis requires review and approval by a licensed attorney
- This system cannot replace professional legal judgment
- Consult with a qualified attorney licensed in your jurisdiction for specific legal advice

---

## Features

### Core Legal Analysis Capabilities

1. **IRAC Legal Reasoning** - Issue, Rule, Application, Conclusion framework
2. **Adversarial Simulation** - Models courtroom debate between opposing positions
3. **Conflict of Laws Resolution** - Resolves multi-jurisdictional conflicts
4. **Litigation Prediction** - Monte Carlo simulation for outcome forecasting
5. **Contract Analysis** - Automated contract review with red flag detection

### QRATUM Integration

OMNILEX enforces all 8 QRATUM fatal invariants:

1. ✅ Contract immutability (frozen dataclasses)
2. ✅ Zero policy logic in adapters
3. ✅ Mandatory event emission
4. ✅ Hash-chain integrity
5. ✅ Causal traceability
6. ✅ Authorized execution only
7. ✅ Deterministic serialization
8. ✅ Temporal constraint enforcement

---

## Quick Start

### Installation

```bash
pip install -e .
```

### Basic Usage

```python
from omnilex import QRATUMOmniLexEngine, LegalQILIntent
from omnilex.qil_legal import generate_intent_id
import time

# Initialize the engine
engine = QRATUMOmniLexEngine()

# Create a legal analysis intent
intent = LegalQILIntent(
    intent_id=generate_intent_id("irac_analysis", "US", time.time()),
    compute_task="irac_analysis",
    jurisdiction_primary="US",
    jurisdictions_secondary=(),
    legal_domain="contract",
    reasoning_framework="irac",
    attorney_supervised=True,
    raw_facts="""
    AlphaCorp and BetaServices entered into a service agreement requiring
    4-hour response time. BetaServices responded after 9 hours, causing
    $50,000 in damages.
    """,
    legal_question="Did BetaServices breach the contract?"
)

# Submit for analysis
response = engine.submit_legal_intent(intent)

# Access results
print(response["disclaimer"])  # UPL disclaimer
print(response["result"]["issue"])
print(response["result"]["rule"])
print(response["result"]["conclusion"])
```

---

## Analysis Types

### 1. IRAC Analysis

Performs structured legal analysis using Issue-Rule-Application-Conclusion framework.

```python
intent = LegalQILIntent(
    intent_id=generate_intent_id("irac_analysis", "US", time.time()),
    compute_task="irac_analysis",
    jurisdiction_primary="US",
    jurisdictions_secondary=(),
    legal_domain="contract",
    reasoning_framework="irac",
    attorney_supervised=True,
    raw_facts="[Your facts here]",
    legal_question="[Your question here]"
)
```

### 2. Adversarial Simulation

Simulates legal debate between opposing positions.

```python
intent = LegalQILIntent(
    intent_id=generate_intent_id("adversarial", "US", time.time()),
    compute_task="adversarial_simulation",
    jurisdiction_primary="US",
    jurisdictions_secondary=(),
    legal_domain="contract",
    reasoning_framework="irac",
    attorney_supervised=True,
    raw_facts="[Your facts here]",
    legal_question="[Your question here]"
)
```

### 3. Conflict of Laws Resolution

Determines which jurisdiction's law applies in multi-jurisdictional disputes.

```python
intent = LegalQILIntent(
    intent_id=generate_intent_id("conflict", "US-CA", time.time()),
    compute_task="conflict_of_laws",
    jurisdiction_primary="US-CA",
    jurisdictions_secondary=("US-NY", "US-TX"),
    legal_domain="contract",
    reasoning_framework="irac",
    attorney_supervised=True,
    raw_facts="[Your facts here]",
    legal_question="Which law applies?"
)
```

### 4. Litigation Prediction

Predicts case outcomes using Monte Carlo simulation.

```python
intent = LegalQILIntent(
    intent_id=generate_intent_id("prediction", "US", time.time()),
    compute_task="litigation_prediction",
    jurisdiction_primary="US",
    jurisdictions_secondary=(),
    legal_domain="contract",
    reasoning_framework="irac",
    attorney_supervised=True,
    raw_facts="[Your facts here]",
    legal_question="What is the likely outcome?"
)
```

### 5. Contract Review

Analyzes contracts for risks, red flags, and missing provisions.

```python
intent = LegalQILIntent(
    intent_id=generate_intent_id("review", "US", time.time()),
    compute_task="contract_review",
    jurisdiction_primary="US",
    jurisdictions_secondary=(),
    legal_domain="service_agreement",
    reasoning_framework="irac",
    attorney_supervised=True,
    raw_facts="[Full contract text here]",
    legal_question="Review this contract."
)
```

---

## QIL Examples

See `omnilex/examples/` for complete QIL-formatted legal analysis examples:

- `contract_breach.qil` - Breach of contract analysis
- `litigation_prediction.qil` - Outcome forecasting
- `multi_jurisdiction.qil` - Cross-border compliance
- `contract_review.qil` - Red flag detection

---

## Deterministic Replay & Audit

OMNILEX supports deterministic replay and audit of all legal analyses:

```python
# Replay a previous analysis
replayed = engine.replay_analysis("OMNILEX-abc123...")

# Audit an analysis for integrity
audit_report = engine.audit_analysis("OMNILEX-abc123...")
print(audit_report["hash_integrity"]["valid"])  # True if integrity maintained
print(audit_report["invariants_satisfied"])      # QRATUM invariant compliance
```

---

## Legal Domains Supported

- Constitutional Law
- Criminal Law
- Contract Law
- Tort Law
- Property Law
- Corporate Law
- Securities Law
- Bankruptcy Law
- Tax Law
- Intellectual Property
- Labor & Employment Law
- Environmental Law
- Family Law
- Immigration Law
- Human Rights Law
- Space Law
- Cyber Law
- AI Regulation

---

## Legal Traditions Supported

- Common Law
- Civil Law
- Religious Law (Islamic, Jewish, Canon)
- Socialist Law
- Customary Law
- Mixed Systems
- International Law
- Supranational Law

---

## Jurisdictions

Pre-configured jurisdictions include:

- United States (Federal and State)
- United Kingdom
- Germany
- (Extensible to additional jurisdictions)

---

## Testing

Run the comprehensive test suite:

```bash
pytest omnilex/tests/ -v
```

All 60 tests validate:
- IRAC reasoning engine
- Adversarial simulation
- Conflict resolution
- Litigation prediction
- Contract analysis
- Main engine orchestration
- QRATUM invariant enforcement
- Hash-chain integrity

---

## Architecture

```
omnilex/
├── __init__.py              # Package exports
├── ontology.py              # Legal domain ontology
├── qil_legal.py             # Extended QIL for legal
├── knowledge.py             # Legal knowledge base
├── reasoning.py             # IRAC reasoning engine
├── adversarial.py           # Adversarial simulation
├── conflicts.py             # Conflict of laws resolver
├── prediction.py            # Litigation prediction
├── contracts.py             # Contract analysis
├── engine.py                # Main OMNILEX engine
├── examples/                # QIL examples
│   ├── contract_breach.qil
│   ├── litigation_prediction.qil
│   ├── multi_jurisdiction.qil
│   └── contract_review.qil
└── tests/                   # Comprehensive tests
    ├── test_reasoning.py
    ├── test_prediction.py
    ├── test_contracts.py
    └── test_engine.py
```

---

## Version

**QRATUM-OMNILEX v1.0.0**

Status: Production
License: Apache 2.0

---

## Security

- ✅ No security vulnerabilities (CodeQL clean)
- ✅ All tests passing (60/60)
- ✅ Linting clean (ruff)
- ✅ Immutable contracts (frozen dataclasses)
- ✅ Deterministic execution
- ✅ Hash-chain integrity

---

## Contributing

OMNILEX is part of the QRATUM project. See `CONTRIBUTING.md` in the repository root.

---

## Support

For questions or issues, please consult with a qualified attorney licensed in your jurisdiction. This system provides informational tools only and does not constitute legal advice.

---

**Remember: Always have a licensed attorney review any legal analysis before making decisions or taking action based on the results.**
