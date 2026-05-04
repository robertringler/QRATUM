# QRATUM Human Factors Engineering Design

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0 |
| Last Updated | 2026-01-05 |
| Classification | Internal |
| Status | Design Document |
| Domain | Human Factors Engineering (CRITICAL - Previously Missing) |

---

## Table of Contents

1. [Misinterpretation of Deterministic Outputs](#1-misinterpretation-of-deterministic-outputs)
2. [Over-Trust Prevention Interfaces](#2-over-trust-prevention-interfaces)
3. [Operator Error Under Time Pressure](#3-operator-error-under-time-pressure)
4. [Explainability for Non-Technical Decision-Makers](#4-explainability-for-non-technical-decision-makers)
5. [Cognitive Overload Failure Modes](#5-cognitive-overload-failure-modes)
6. [Training Wheels for First-Time Operators](#6-training-wheels-for-first-time-operators)
7. [UX Constraints for Misuse Prevention](#7-ux-constraints-for-misuse-prevention)
8. [Courtroom Output Evaluation](#8-courtroom-output-evaluation)
9. [Unignorable Alert Design](#9-unignorable-alert-design)
10. [Confused User Red Team](#10-confused-user-red-team)

---

## 1. Misinterpretation of Deterministic Outputs

### 1.1 Design Objective

Analyze how humans misinterpret deterministic system outputs.

### 1.2 Misinterpretation Taxonomy

| Misinterpretation Type | Description | Consequence | Frequency |
|------------------------|-------------|-------------|-----------|
| **Certainty Inflation** | "Deterministic" = "definitely correct" | Over-reliance | High |
| **Confidence Confusion** | Mixing determinism with accuracy | Wrong trust level | High |
| **Completeness Assumption** | "System gave answer" = "all factors considered" | Incomplete analysis | Medium |
| **Neutrality Bias** | "Computer said it" = "unbiased" | Hidden bias acceptance | Medium |
| **Precision Illusion** | Many decimal places = high accuracy | False precision trust | High |

### 1.3 Detailed Analysis

```
┌─────────────────────────────────────────────────────────────────────────┐
│              Human Misinterpretation Patterns                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  PATTERN 1: DETERMINISM = CORRECTNESS FALLACY                           │
│                                                                          │
│    Human Mental Model:                                                  │
│      "This system always gives the same answer"                        │
│      → "Therefore the answer must be right"                            │
│                                                                          │
│    Reality:                                                             │
│      Determinism only means: same input → same output                  │
│      Says nothing about whether output is correct                      │
│      Example: A consistently wrong calculation is still deterministic  │
│                                                                          │
│    Mitigation:                                                          │
│      • Explicitly separate "consistent" from "correct" in UI           │
│      • Show validation status alongside output                         │
│      • Training: "Deterministic means reproducible, not infallible"   │
│                                                                          │
│  PATTERN 2: AUTOMATION COMPLACENCY                                      │
│                                                                          │
│    Human Mental Model:                                                  │
│      "The system is very sophisticated"                                │
│      → "I don't need to check its work"                                │
│                                                                          │
│    Reality:                                                             │
│      Sophisticated systems can still have systematic errors            │
│      Edge cases may not be handled correctly                           │
│      Input quality directly affects output quality                     │
│                                                                          │
│    Mitigation:                                                          │
│      • Mandatory human checkpoints for critical decisions              │
│      • Periodic "trust calibration" exercises                          │
│      • Show system limitations prominently                             │
│                                                                          │
│  PATTERN 3: BLACK BOX ACCEPTANCE                                        │
│                                                                          │
│    Human Mental Model:                                                  │
│      "I don't understand how it works"                                 │
│      → "But it's been approved, so I'll accept it"                     │
│                                                                          │
│    Reality:                                                             │
│      Humans tend to defer to authority (system/institution)            │
│      Lack of understanding reduces critical evaluation                 │
│                                                                          │
│    Mitigation:                                                          │
│      • Provide multiple explanation levels (expert → novice)          │
│      • Require acknowledgment of key decision factors                  │
│      • Enable easy "why?" queries at any point                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.4 Interface Corrections

```python
class OutputInterpretationGuard:
    """
    Guard against common misinterpretations
    """
    
    def format_deterministic_output(
        self,
        value: Any,
        context: OutputContext
    ) -> FormattedOutput:
        """
        Format output with interpretation guards
        """
        return FormattedOutput(
            # Primary result
            value=value,
            
            # Anti-misinterpretation labels
            determinism_label="REPRODUCIBLE: Same inputs will always produce this output",
            correctness_label=self._get_correctness_label(context),
            completeness_label=self._get_completeness_label(context),
            
            # Mandatory acknowledgment (for critical outputs)
            requires_acknowledgment=context.is_critical,
            acknowledgment_text=(
                "I understand this output is reproducible but may not reflect "
                "all relevant factors. I will apply appropriate judgment."
            )
        )
    
    def _get_correctness_label(self, context: OutputContext) -> str:
        if context.validation_status == "VERIFIED":
            return "VALIDATED: Output has been verified against known correct cases"
        elif context.validation_status == "PARTIAL":
            return "PARTIALLY VALIDATED: Some aspects verified, others not"
        else:
            return "UNVALIDATED: Reproducible but correctness not independently verified"
```

---

## 2. Over-Trust Prevention Interfaces

### 2.1 Design Objective

Design interfaces that prevent over-trust in QRATUM.

### 2.2 Over-Trust Indicators

| Signal | Meaning | Response |
|--------|---------|----------|
| User never overrides | May be over-trusting | Prompt for review |
| User accepts instantly | Not reading details | Require interaction |
| User skips explanations | Not understanding | Highlight key factors |
| Error rate too low | Rubber-stamping | Inject test cases |

### 2.3 Trust Calibration Interface

```
┌─────────────────────────────────────────────────────────────────────────┐
│              Over-Trust Prevention Interface Design                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  PRINCIPLE 1: FRICTION FOR CRITICAL DECISIONS                           │
│                                                                          │
│    Instead of: [Accept] [Reject]                                       │
│    Use:        [Review Details Before Deciding]                         │
│                └── Then: [I've reviewed, Accept] [I've reviewed, Reject]│
│                                                                          │
│  PRINCIPLE 2: MANDATORY ENGAGEMENT                                      │
│                                                                          │
│    Before allowing acceptance:                                          │
│      • Show key factors (user must scroll/expand)                      │
│      • Highlight uncertainties and limitations                         │
│      • Display confidence bounds, not just point estimates             │
│      • Require clicking on each critical section                       │
│                                                                          │
│  PRINCIPLE 3: PERIODIC TRUST CHECKS                                     │
│                                                                          │
│    Inject known-incorrect cases periodically:                          │
│      • If user accepts: Warning + remedial training                    │
│      • Track trust calibration over time                               │
│      • Adjust required friction based on calibration                   │
│                                                                          │
│  PRINCIPLE 4: VISIBLE UNCERTAINTY                                       │
│                                                                          │
│    Always show:                                                         │
│      • Confidence interval (not just point estimate)                   │
│      • What factors were NOT considered                                │
│      • When data was last updated                                      │
│      • Known limitations for this case type                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.4 Implementation

```python
class TrustCalibratedInterface:
    """
    Interface that prevents over-trust through calibration
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.trust_metrics = self.load_trust_metrics(user_id)
    
    def present_decision(self, decision: Decision) -> DecisionPresentation:
        """
        Present decision with appropriate trust calibration
        """
        # Determine required engagement based on user's trust calibration
        required_engagement = self._calculate_required_engagement(decision)
        
        presentation = DecisionPresentation(
            decision=decision,
            
            # Mandatory sections that must be viewed
            mandatory_sections=[
                Section("Key Factors", decision.key_factors, must_expand=True),
                Section("Limitations", decision.limitations, must_expand=True),
                Section("Confidence", decision.confidence_info, must_expand=True),
            ],
            
            # Engagement requirements
            min_review_time_seconds=required_engagement.min_time,
            must_expand_all=required_engagement.must_expand,
            quiz_before_accept=required_engagement.needs_quiz,
            
            # Trust calibration display
            user_trust_status=self._format_trust_status()
        )
        
        return presentation
    
    def inject_calibration_case(self) -> Optional[CalibrationCase]:
        """
        Periodically inject known cases to calibrate trust
        """
        if random.random() < 0.05:  # 5% of decisions
            return CalibrationCase(
                decision=self._generate_test_decision(),
                expected_response="REJECT",  # Known bad case
                learning_message=(
                    "This was a calibration case with known issues. "
                    "Review the following problems that should have been caught..."
                )
            )
        return None
    
    def record_decision(self, decision_id: str, user_action: str, time_spent: float):
        """
        Record decision for trust calibration
        """
        self.trust_metrics.record(
            decision_id=decision_id,
            action=user_action,
            time_spent=time_spent,
            expanded_sections=self.current_session.expanded_sections
        )
        
        # Check for over-trust signals
        if self.trust_metrics.detect_over_trust():
            self.schedule_calibration_intervention()
```

---

## 3. Operator Error Under Time Pressure

### 3.1 Design Objective

Model operator error under time pressure.

### 3.2 Time Pressure Error Taxonomy

| Error Type | Cause | Consequence | Frequency Under Pressure |
|------------|-------|-------------|-------------------------|
| **Rushed Reading** | Time constraints | Miss critical info | 3x higher |
| **Default Acceptance** | Cognitive shortcut | Accept errors | 4x higher |
| **Confirmation Bias** | Seek quick resolution | Ignore warnings | 2x higher |
| **Tunnel Vision** | Focus narrowing | Miss context | 3x higher |
| **Sequence Errors** | Process shortcuts | Wrong order | 5x higher |

### 3.3 Time Pressure Mitigation Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│              Time Pressure Error Mitigation                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  DESIGN PRINCIPLE: Time pressure should never compromise safety         │
│                                                                          │
│  MITIGATION 1: MINIMUM REVIEW TIME ENFORCEMENT                          │
│                                                                          │
│    For critical decisions:                                              │
│      • Accept button disabled until minimum time elapsed                │
│      • Progress indicator shows required review time                    │
│      • Cannot be overridden by user                                     │
│                                                                          │
│    Example:                                                             │
│      ┌────────────────────────────────────────┐                        │
│      │ Review Progress: ████████░░ 80%        │                        │
│      │ Accept available in 12 seconds         │                        │
│      │                                         │                        │
│      │ [Accept - Wait 12s] [Reject Available] │                        │
│      └────────────────────────────────────────┘                        │
│                                                                          │
│  MITIGATION 2: ESCALATION UNDER DEADLINE                               │
│                                                                          │
│    If operator attempts rapid decisions:                                │
│      • Automatically escalate to supervisor review                      │
│      • Log pattern for later analysis                                   │
│      • Offer assistance ("Need help with workload?")                   │
│                                                                          │
│  MITIGATION 3: BATCH PREVENTION                                         │
│                                                                          │
│    Prevent processing many decisions without breaks:                    │
│      • Maximum consecutive decisions without pause                      │
│      • Mandatory micro-breaks after N decisions                         │
│      • Fatigue warning after extended sessions                         │
│                                                                          │
│  MITIGATION 4: CRITICAL PATH HIGHLIGHTING                              │
│                                                                          │
│    Under time pressure, show only essential information:                │
│      • "Quick Review" mode shows critical factors only                 │
│      • Red/yellow/green summary for rapid assessment                   │
│      • "Full Review" still required for acceptance                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Time Pressure Detection

```python
class TimePressureDetector:
    """
    Detect when operators are under excessive time pressure
    """
    
    THRESHOLDS = {
        "rapid_decisions": 3,           # Decisions per minute
        "short_review_time": 5,         # Seconds
        "high_queue_size": 50,          # Pending items
        "approaching_deadline": 0.1,    # Fraction of deadline remaining
    }
    
    def detect_time_pressure(self, operator: Operator) -> TimePressureAssessment:
        metrics = self.collect_metrics(operator)
        
        pressure_factors = []
        
        if metrics.decisions_per_minute > self.THRESHOLDS["rapid_decisions"]:
            pressure_factors.append(PressureFactor(
                type="RAPID_DECISIONS",
                severity="HIGH",
                value=metrics.decisions_per_minute
            ))
        
        if metrics.avg_review_time < self.THRESHOLDS["short_review_time"]:
            pressure_factors.append(PressureFactor(
                type="SHORT_REVIEW",
                severity="MEDIUM",
                value=metrics.avg_review_time
            ))
        
        if metrics.queue_size > self.THRESHOLDS["high_queue_size"]:
            pressure_factors.append(PressureFactor(
                type="HIGH_WORKLOAD",
                severity="HIGH",
                value=metrics.queue_size
            ))
        
        overall_pressure = self._calculate_overall_pressure(pressure_factors)
        
        return TimePressureAssessment(
            pressure_level=overall_pressure,
            factors=pressure_factors,
            recommendations=self._get_recommendations(pressure_factors),
            escalation_required=(overall_pressure > 0.7)
        )
```

---

## 4. Explainability for Non-Technical Decision-Makers

### 4.1 Design Objective

Define explainability for non-technical decision-makers.

### 4.2 Audience-Specific Requirements

| Audience | Technical Level | Needs | Format |
|----------|-----------------|-------|--------|
| Executive | Low | Bottom line, risks, recommendations | Summary + bullet points |
| Manager | Medium | Process, alternatives, justification | Structured report |
| Legal | Medium | Evidence, compliance, defensibility | Formal documentation |
| Board | Low | Strategic impact, risk assessment | Visual + narrative |

### 4.3 Multi-Level Explanation Framework

```
┌─────────────────────────────────────────────────────────────────────────┐
│              Multi-Level Explanation Framework                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  LEVEL 1: EXECUTIVE SUMMARY (30 seconds)                                │
│    "The system recommends X because of Y. Key risk is Z."              │
│                                                                          │
│  LEVEL 2: BUSINESS EXPLANATION (5 minutes)                              │
│    • What was decided                                                   │
│    • Why (in business terms)                                            │
│    • What alternatives were considered                                  │
│    • Key factors that drove the decision                                │
│    • Risks and mitigations                                              │
│                                                                          │
│  LEVEL 3: PROCESS EXPLANATION (15 minutes)                              │
│    • Data sources used                                                  │
│    • Rules and policies applied                                         │
│    • How factors were weighted                                          │
│    • Where human judgment was involved                                  │
│    • Audit trail references                                             │
│                                                                          │
│  LEVEL 4: TECHNICAL EXPLANATION (1 hour)                                │
│    • Complete derivation chain                                          │
│    • Mathematical/logical basis                                         │
│    • Edge case handling                                                 │
│    • Sensitivity analysis                                               │
│    • Validation evidence                                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.4 Plain Language Generator

```python
class PlainLanguageExplainer:
    """
    Generate explanations for non-technical audiences
    """
    
    JARGON_TRANSLATIONS = {
        "deterministic": "always gives the same result for the same input",
        "Merkle tree": "a tamper-evident record",
        "Byzantine fault": "malicious actor",
        "consensus": "agreement among multiple systems",
        "invariant": "rule that must always be true"
    }
    
    def explain_for_executive(self, decision: Decision) -> ExecutiveExplanation:
        return ExecutiveExplanation(
            one_line=self._generate_one_liner(decision),
            recommendation=self._format_recommendation(decision),
            key_risks=self._extract_key_risks(decision, max_items=3),
            confidence=self._format_confidence_plain(decision.confidence),
            action_required=self._determine_action(decision)
        )
    
    def explain_for_legal(self, decision: Decision) -> LegalExplanation:
        return LegalExplanation(
            summary=self._generate_legal_summary(decision),
            evidence_chain=self._format_evidence_chain(decision),
            compliance_status=self._check_compliance(decision),
            defensibility_notes=self._assess_defensibility(decision),
            audit_references=self._get_audit_references(decision)
        )
    
    def _translate_jargon(self, text: str) -> str:
        """Replace technical terms with plain language"""
        result = text
        for jargon, plain in self.JARGON_TRANSLATIONS.items():
            result = result.replace(jargon, f"{plain} (technical term: {jargon})")
        return result
```

---

## 5. Cognitive Overload Failure Modes

### 5.1 Design Objective

Identify cognitive overload failure modes.

### 5.2 Overload Sources

| Source | Manifestation | System Design Issue |
|--------|---------------|-------------------|
| **Information Density** | Too much data on screen | Poor information hierarchy |
| **Decision Fatigue** | Quality degrades over time | No break enforcement |
| **Complexity Creep** | Increasingly complex interfaces | Feature accumulation |
| **Alert Fatigue** | Ignoring all alerts | Too many non-critical alerts |
| **Context Switching** | Errors when switching tasks | Poor task separation |

### 5.3 Overload Prevention Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│              Cognitive Overload Prevention                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  PRINCIPLE 1: PROGRESSIVE DISCLOSURE                                    │
│                                                                          │
│    Default view: Essential information only                             │
│    Expand on demand: Additional details                                 │
│    Deep dive available: Full technical view                            │
│                                                                          │
│  PRINCIPLE 2: INFORMATION CHUNKING                                      │
│                                                                          │
│    Maximum items per view: 7 ± 2                                       │
│    Group related information visually                                   │
│    Use whitespace to separate concepts                                  │
│                                                                          │
│  PRINCIPLE 3: DECISION BATCHING                                         │
│                                                                          │
│    Similar decisions grouped together                                   │
│    Context established once, applied to batch                          │
│    Clear separation between batches                                    │
│                                                                          │
│  PRINCIPLE 4: LOAD MONITORING                                           │
│                                                                          │
│    Track cognitive load indicators:                                     │
│      • Time per decision trending up                                   │
│      • Error rate trending up                                          │
│      • Review time decreasing                                          │
│    Auto-suggest breaks when load detected                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Training Wheels for First-Time Operators

### 6.1 Design Objective

Design training wheels for first-time operators.

### 6.2 Progressive Onboarding

```
┌─────────────────────────────────────────────────────────────────────────┐
│              First-Time Operator Training Wheels                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  STAGE 1: GUIDED MODE (First 50 decisions)                              │
│                                                                          │
│    • Step-by-step walkthrough for each decision type                   │
│    • Tooltips explain every field                                      │
│    • "Did you know?" tips at relevant moments                          │
│    • All decisions reviewed by experienced operator                    │
│    • Immediate feedback on every decision                              │
│                                                                          │
│  STAGE 2: ASSISTED MODE (Decisions 51-200)                              │
│                                                                          │
│    • Tooltips available on hover (not automatic)                       │
│    • Sample decisions reviewed by supervisor                           │
│    • Access to quick reference guide                                   │
│    • Escalation path prominently displayed                            │
│                                                                          │
│  STAGE 3: SUPERVISED MODE (Decisions 201-500)                           │
│                                                                          │
│    • Independent operation                                              │
│    • Periodic spot-checks by supervisor                                │
│    • Trust calibration exercises continue                              │
│    • Performance metrics tracked                                       │
│                                                                          │
│  STAGE 4: INDEPENDENT MODE (After 500 decisions)                        │
│                                                                          │
│    • Full operational authority                                        │
│    • Mentor available for complex cases                               │
│    • Ongoing calibration (reduced frequency)                          │
│    • May mentor new operators                                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. UX Constraints for Misuse Prevention

### 7.1 Design Objective

Propose UX constraints that prevent misuse.

### 7.2 Misuse Prevention Patterns

| Misuse Type | UX Constraint | Implementation |
|-------------|---------------|----------------|
| **Rubber-stamping** | Minimum engagement | Time + interaction requirements |
| **Cherry-picking** | Audit visibility | Show query patterns to supervisors |
| **Override abuse** | Justification required | Structured reason capture |
| **Data manipulation** | Input locking | Immutable after submission |
| **Explanation shopping** | Single canonical version | One explanation per decision |

### 7.3 Constraint Implementation

```python
class MisusePreventionConstraints:
    """
    UX constraints that prevent misuse by design
    """
    
    def enforce_engagement(self, decision_screen: Screen) -> None:
        """Prevent rubber-stamping through engagement requirements"""
        
        # Require expanding all critical sections
        for section in decision_screen.critical_sections:
            section.must_expand = True
            section.expansion_logged = True
        
        # Minimum time before accept enabled
        decision_screen.accept_button.min_delay_seconds = 15
        
        # Scroll-to-bottom required
        decision_screen.scroll_to_bottom_required = True
    
    def lock_inputs_after_start(self, decision_id: str) -> None:
        """Prevent data manipulation by locking inputs"""
        
        # Once decision process starts, inputs cannot change
        decision = self.get_decision(decision_id)
        decision.input_data.lock()
        
        # Hash inputs for audit
        decision.input_hash = sha3_256(decision.input_data.serialize())
        
        # Any attempt to modify raises alert
        decision.input_data.on_modification_attempt = self.raise_manipulation_alert
    
    def enforce_single_explanation(self, decision_id: str) -> None:
        """Prevent explanation shopping"""
        
        decision = self.get_decision(decision_id)
        
        if decision.explanation_generated:
            # Return cached explanation, don't regenerate
            return decision.canonical_explanation
        
        # Generate once and lock
        explanation = self.generate_explanation(decision)
        decision.canonical_explanation = explanation
        decision.explanation_generated = True
        decision.explanation_locked = True
```

---

## 8. Courtroom Output Evaluation

### 8.1 Design Objective

Evaluate QRATUM outputs in courtroom scenarios.

### 8.2 Courtroom Requirements

| Requirement | Standard | QRATUM Capability |
|-------------|----------|------------------|
| **Authentication** | FRE 901 | Cryptographic signatures |
| **Chain of Custody** | Evidence handling | Merkle chain audit |
| **Expert Testimony** | Daubert standard | Explainability framework |
| **Cross-examination** | Challenge opportunity | Full derivation trace |

### 8.3 Courtroom-Ready Output Package

```python
@dataclass
class CourtroomOutputPackage:
    """
    Output formatted for legal proceedings
    """
    
    # Decision summary
    decision_summary: DecisionSummary
    
    # Evidence authentication
    authentication: Authentication
    chain_of_custody: ChainOfCustody
    
    # Expert explanation
    technical_explanation: TechnicalExplanation
    plain_language_explanation: PlainLanguageExplanation
    
    # Challenge support
    input_data_certified: CertifiedData
    derivation_trace: DerivationTrace
    alternative_analysis: AlternativeAnalysis
    
    # Certification
    certifications: List[Certification]

class CourtroomFormatter:
    """
    Format QRATUM outputs for courtroom use
    """
    
    def prepare_for_court(self, decision: Decision) -> CourtroomOutputPackage:
        return CourtroomOutputPackage(
            decision_summary=self._create_summary(decision),
            authentication=self._create_authentication(decision),
            chain_of_custody=self._trace_custody(decision),
            technical_explanation=self._technical_explain(decision),
            plain_language_explanation=self._plain_explain(decision),
            input_data_certified=self._certify_inputs(decision),
            derivation_trace=decision.full_trace,
            alternative_analysis=self._analyze_alternatives(decision),
            certifications=self._gather_certifications(decision)
        )
```

---

## 9. Unignorable Alert Design

### 9.1 Design Objective

Design alerts that humans will not ignore.

### 9.2 Alert Effectiveness Hierarchy

| Alert Level | Characteristics | Use Case | Ignore Rate |
|-------------|-----------------|----------|-------------|
| **CRITICAL** | Modal, requires action | Safety/legal issues | <1% |
| **HIGH** | Persistent, prominent | Important warnings | ~5% |
| **MEDIUM** | Visible, dismissible | Guidance | ~20% |
| **LOW** | Subtle, background | Information | ~50% |

### 9.3 Unignorable Alert Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│              Unignorable Alert Design Principles                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  PRINCIPLE 1: ALERT SCARCITY                                            │
│                                                                          │
│    • Critical alerts must be rare (max 1-2 per day)                    │
│    • If over-used, humans habituate and ignore                         │
│    • Reserve critical level for genuine emergencies                    │
│                                                                          │
│  PRINCIPLE 2: ESCALATING INTRUSIVENESS                                  │
│                                                                          │
│    T+0: Visual alert appears                                           │
│    T+30s: Audio cue added                                              │
│    T+60s: Requires acknowledgment to continue                          │
│    T+120s: Escalates to supervisor                                     │
│                                                                          │
│  PRINCIPLE 3: MEANINGFUL DIFFERENTIATION                                │
│                                                                          │
│    Critical: Red, modal, stops workflow                                │
│    High: Orange, persistent banner                                     │
│    Medium: Yellow, dismissible                                         │
│    Low: Blue, informational                                            │
│                                                                          │
│  PRINCIPLE 4: ACTIONABLE CONTENT                                        │
│                                                                          │
│    Bad: "Error occurred"                                               │
│    Good: "Cannot proceed: Missing authorization. [Request Auth]"       │
│                                                                          │
│  PRINCIPLE 5: ACKNOWLEDGMENT VERIFICATION                               │
│                                                                          │
│    For critical alerts:                                                │
│      • Type specific phrase to acknowledge                             │
│      • Or: Quiz on alert content                                       │
│      • Prevents mechanical dismissal                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Confused User Red Team

### 10.1 Design Objective

Red-team QRATUM assuming confused but well-intentioned users.

### 10.2 Confusion Scenarios

| Scenario | User Mental State | Potential Harm | Frequency |
|----------|------------------|----------------|-----------|
| **Wrong Screen** | Thinks they're somewhere else | Wrong action | Common |
| **Misread Label** | Confuses similar fields | Wrong data | Common |
| **Stale Context** | Acting on old information | Outdated decision | Medium |
| **Partial Understanding** | Knows enough to be dangerous | Overconfident error | Medium |
| **Wrong Mental Model** | Fundamentally misunderstands | Systematic error | Rare |

### 10.3 Confusion Mitigation

```python
class ConfusedUserDefense:
    """
    Defenses against well-intentioned confused users
    """
    
    def verify_user_context(self, user: User, action: Action) -> ContextCheck:
        """
        Verify user understands their current context
        """
        checks = []
        
        # Check 1: Is user on the screen they think they're on?
        if action.expected_screen != user.current_screen:
            checks.append(ContextIssue(
                type="WRONG_SCREEN",
                message=f"You're on {user.current_screen}, not {action.expected_screen}"
            ))
        
        # Check 2: Is user's data context current?
        if user.data_context.age > timedelta(minutes=5):
            checks.append(ContextIssue(
                type="STALE_CONTEXT",
                message="Your data may be outdated. Refresh before proceeding."
            ))
        
        # Check 3: Is action consistent with user's recent actions?
        if self._is_action_inconsistent(user, action):
            checks.append(ContextIssue(
                type="INCONSISTENT_ACTION",
                message="This action seems inconsistent with your previous actions."
            ))
        
        return ContextCheck(issues=checks, can_proceed=len(checks) == 0)
    
    def provide_orientation_cues(self, screen: Screen) -> None:
        """
        Always show orientation cues to prevent confusion
        """
        screen.header.add_permanent_elements([
            OrientationElement("Current Task", screen.task_name),
            OrientationElement("Context", screen.context_summary),
            OrientationElement("Last Refresh", screen.data_timestamp),
            OrientationElement("Your Role", screen.user_role)
        ])
    
    def implement_confirmation_pattern(
        self,
        action: Action
    ) -> ConfirmationPattern:
        """
        Implement appropriate confirmation for action severity
        """
        if action.severity == "IRREVERSIBLE":
            return ConfirmationPattern(
                type="TYPE_TO_CONFIRM",
                prompt=f"Type '{action.confirmation_phrase}' to confirm",
                additional_warnings=action.warnings
            )
        elif action.severity == "SIGNIFICANT":
            return ConfirmationPattern(
                type="TWO_BUTTON",
                prompt="Are you sure?",
                details=action.impact_summary
            )
        else:
            return ConfirmationPattern(type="NONE")
```

---

## Appendix: Human Factors Checklist

| Design Element | Requirement | Verification |
|----------------|-------------|--------------|
| Information hierarchy | Critical info prominent | User testing |
| Alert levels | Properly differentiated | Usage analysis |
| Confirmation patterns | Appropriate for severity | Error rate tracking |
| Training progression | Staged complexity | Competency assessment |
| Cognitive load | Within limits | Task completion time |
| Time pressure handling | Enforced minimums | Audit log review |

## References

1. Norman, D. (2013). The Design of Everyday Things
2. Wickens, C. (2013). Engineering Psychology and Human Performance
3. Reason, J. (1990). Human Error
4. FAA Human Factors Design Guide
5. MIL-STD-1472: Human Engineering Design Criteria
