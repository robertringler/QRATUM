# JURIS-PEERAGE Implementation Summary

## Executive Summary

JURIS-PEERAGE is a deterministic legal-historical reasoning engine for English/UK peerage law, title succession, and heirship determination. It extends the QRATUM platform's legal AI capabilities (JURIS vertical) to handle the specialized domain of nobility law and peerage rights.

**Date**: 2026-01-19  
**Module**: `qratum/verticals/juris_peerage.py`  
**Tests**: `tests/test_juris_peerage.py` (11/11 passing)  
**Integration**: QRATUM verticals framework

---

## Critical Architectural Invariant

### DESCENT ≠ ENTITLEMENT

This principle is **MANDATORY** and **NON-NEGOTIABLE**:

- **GENEALOGY** module proves **DESCENT** (family relationships, documented lineage)
- **JURIS-PEERAGE** module determines **LEGAL RIGHTS** (title succession, heirship claims)

These are **SEPARATE** and **DISTINCT** determinations:
- Having genealogical descent from nobility **DOES NOT** equal legal entitlement to titles
- Legal analysis requires proven descent **PLUS** application of peerage law doctrines

### Module Separation

| Module | Responsibility | Output |
|--------|---------------|---------|
| **Genealogy** | Prove family descent via GPS standards | Verified lineage, primary sources, gaps identified |
| **JURIS-PEERAGE** | Determine legal rights under peerage law | Legal determination, applicable law, fatal defects |

**Correct Workflow:**
1. Genealogy module → GPS-certified proof of descent
2. JURIS-PEERAGE module → Legal analysis of proven descent
3. Clear handoff with explicit boundaries

---

## Module Capabilities

JURIS-PEERAGE provides five core capabilities:

### 1. Heirship Determination (`determine_heirship`)

Analyzes whether a subject has legal claim to any title under English/UK law.

**Applies systematic doctrine checks:**
- Title existence verification
- Creation date constraints
- Attainder blood corruption
- Cadet branch exclusion
- Genealogical proof completeness
- Modern justiciability

**Default bias**: `NO LEGAL CLAIM EXISTS` (prove otherwise)

### 2. Title Succession Analysis (`analyze_title_succession`)

Models succession mechanics for specific titles:
- Succession mode (heir male, heir general, cognatic)
- Current title state (extant, extinct, abeyant)
- Eligible heirs under succession rules
- Legal constraints
- Succession order

### 3. Attainder Effect Evaluation (`evaluate_attainder_effects`)

Determines impact of treason conviction on title transmission:
- Blood corruption (prevents transmission)
- Honor restoration (reputation only, NOT legal rights)
- Blood restoration (full restoration by Act of Parliament)
- Title transmission status

### 4. Cadet Branch Assessment (`assess_cadet_branch_rights`)

Applies cadet branch exclusion doctrine:
- Under primogeniture, only senior (eldest) line inherits
- Cadet branches (younger sons, junior lines) excluded
- NO claim regardless of descent proximity

### 5. Justiciability Determination (`determine_justiciability`)

Assesses if peerage claim can be heard in courts:
- Crown prerogative (royal prerogative of honor)
- Parliamentary supremacy (1689+)
- Non-justiciability doctrine
- Proper forum: Crown petition or Act of Parliament

---

## Formalized Legal Doctrines

### Title State Machine

```
CREATED → EXTANT → (EXTINCT | ABEYANT | DORMANT | MERGED)
          ↓
      ATTAINTED
```

**Rules:**
- No title exists without formal creation by Crown/Parliament
- No revival without Crown/Parliament action
- Extinction is permanent (absent new creation)
- Abeyance requires Crown termination

### Succession Modes

1. **HEIR_MALE**: Male-line descendants only (females excluded)
2. **HEIR_GENERAL**: Male-preference primogeniture (males inherit before females)
3. **MALE_PREFERENCE_PRIMOGENITURE**: Traditional English (eldest male, then eldest female)
4. **COGNATIC**: Equal male/female inheritance (rare in England)

### Attainder Doctrine

**Treason conviction effects:**
- **Blood Corrupted**: Title cannot pass through attainted line
- **Heirs Excluded**: Descendants of attainted person have NO claim

**Restoration types:**
- **Honor Only**: Reputation restored, **NOT** legal transmission rights
- **Blood Restoration**: Full legal restoration by Act of Parliament only

### Core Doctrines Implemented

| Doctrine | Rule | Effect |
|----------|------|--------|
| **Cadet Branch Exclusion** | Only senior line inherits | Junior lines have NO claim |
| **Creation Date Constraint** | Cannot inherit title created AFTER ancestor died | Fatal defect |
| **Attainder Blood Corruption** | Treason corrupts blood | No transmission |
| **Male-Preference Primogeniture** | Males before females; eldest among equals | Succession order |
| **Crown Prerogative** | Crown creates peerages | Courts cannot override |
| **Non-Justiciability** | Peerage claims not justiciable | Crown/Parliament authority |

---

## Default Bias: DENY UNLESS PROVEN

JURIS-PEERAGE applies **systematic denial** logic:

### Automatic Denials

1. ❌ **NO TITLE**: If no evidence of formal creation
2. ❌ **CREATION AFTER DEATH**: If title created after ancestor died
3. ❌ **ATTAINDER**: If ancestor attainted (unless blood restored)
4. ❌ **CADET BRANCH**: If subject from non-inheriting branch
5. ❌ **EXTINCT**: If title legally extinct
6. ❌ **UNDECIDABLE**: If genealogical proof has gaps

### Modern Reality

**All peerage claims**: `NON-JUSTICIABLE`
- Modern courts decline jurisdiction
- Proper remedy: Crown petition or Act of Parliament
- No enforceable rights without Crown/Parliamentary action

---

## Input/Output Contract

### Input Format (from Genealogy Module)

```python
{
    "genealogical_proof": {
        "subject_name": "Robert Ringler",
        "proven_ancestors": [...],           # GPS-verified only
        "documented_relationships": [...],   # With confidence levels
        "confidence_levels": {...},          # Per relationship
        "gaps_identified": [...],            # Explicit unproven segments
        "source_citations": [...],           # Primary sources
    },
    "title_context": {
        "title_created": True,
        "creation_date": "1605-01-01",
        "ancestor_death_date": "1552-01-01",
        "attainder_status": "ATTAINTED",
        "honor_restored_only": True,
        "cadet_branch": True,
        "current_state": "EXTINCT",
    }
}
```

### Output Format (Legal Determination)

```python
{
    "subject": "Robert Ringler",
    "determination": "NO LEGAL CLAIM EXISTS",
    "reasoning": [
        "DOCTRINE APPLICATION (systematic):",
        "→ ATTAINDER BLOOD CORRUPTION: Attainder prevents transmission (doctrine)",
        "→ Honor restoration ≠ restoration in blood (requires Act of Parliament)",
    ],
    "applicable_law": [
        "English common law - attainder",
        "Acts of Attainder (1500s)",
        "Parliamentary supremacy (1689+)",
    ],
    "fatal_defects": [
        "FATAL: Ancestor attainted for treason; blood corrupted; title untransmittable"
    ],
    "requirements_if_claimable": [],
    "modern_enforceability": "NON-JUSTICIABLE",
    "confidence": "DEFINITIVE",
    "disclaimer": "⚖️ This is legal analysis, not legal advice. DESCENT ≠ ENTITLEMENT..."
}
```

---

## Example Use Cases

### Case 1: Cadet Branch Denial

**Scenario**: Subject descends from younger son of peer

**Input**:
```python
title_context = {
    "cadet_branch": True,  # Junior line
    "title_created": True,
    "current_state": "EXTANT",
}
```

**Output**:
```python
{
    "determination": "NO LEGAL CLAIM EXISTS",
    "fatal_defects": [
        "FATAL: Subject descends from cadet (non-inheriting) branch"
    ],
    "reasoning": [
        "→ CADET BRANCH EXCLUSION: Only senior line inherits under primogeniture (doctrine)"
    ],
    "confidence": "DEFINITIVE"
}
```

**Rationale**: Primogeniture excludes junior lines absolutely. No claim exists regardless of genealogical proximity.

### Case 2: Attainder with Honor-Only Restoration

**Scenario**: Ancestor executed for treason, reputation later restored

**Input**:
```python
title_context = {
    "attainder_status": "ATTAINTED",
    "honor_restored_only": True,  # Reputation, not legal rights
}
```

**Output**:
```python
{
    "determination": "NO LEGAL CLAIM EXISTS",
    "fatal_defects": [
        "FATAL: Ancestor attainted for treason; blood corrupted; title untransmittable"
    ],
    "reasoning": [
        "→ ATTAINDER BLOOD CORRUPTION: Attainder prevents transmission (doctrine)",
        "→ Honor restoration ≠ restoration in blood (requires Act of Parliament)"
    ],
    "confidence": "DEFINITIVE"
}
```

**Rationale**: Honor restoration restores reputation only. Legal transmission rights require Act of Parliament (blood restoration).

### Case 3: Title Created After Ancestor Died

**Scenario**: Title created 1605, ancestor died 1552

**Input**:
```python
title_context = {
    "creation_date": "1605-01-01",
    "ancestor_death_date": "1552-01-01",
}
```

**Output**:
```python
{
    "determination": "NO LEGAL CLAIM EXISTS",
    "fatal_defects": [
        "FATAL: Title created 1605-01-01 AFTER ancestor died 1552-01-01"
    ],
    "reasoning": [
        "→ CREATION DATE CONSTRAINT: Cannot inherit title that didn't exist (doctrine)"
    ],
    "confidence": "DEFINITIVE"
}
```

**Rationale**: Cannot inherit title that didn't exist when ancestor was alive. Title must exist to be inherited.

### Case 4: Genealogical Proof Gaps

**Scenario**: Proven descent has missing generational links

**Input**:
```python
genealogical_proof = {
    "gaps_identified": [
        "Generation 5-6: No primary source proof",
        "Generation 8-9: Conflicting records",
    ]
}
```

**Output**:
```python
{
    "determination": "UNDECIDABLE (genealogical gaps)",
    "reasoning": [
        "→ GENEALOGICAL PROOF INCOMPLETE: 2 gap(s) in proven descent",
        "  • Generation 5-6: No primary source proof",
        "  • Generation 8-9: Conflicting records"
    ],
    "requirements_if_claimable": [
        "GPS-certified proof of all generational links required"
    ],
    "confidence": "UNDECIDABLE"
}
```

**Rationale**: Legal determination requires complete proven descent. Gaps make claim undecidable until resolved.

---

## Integration with Genealogy System

### Workflow: Descent → Rights Analysis

```
┌─────────────────────────────────────────────────────────────┐
│                   GENEALOGY MODULE                          │
│  - GPS-compliant proof of descent                          │
│  - Primary source citations                                │
│  - Explicit gap identification                             │
│  - Multi-generation verification                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
            GPS-VERIFIED OUTPUT
           (proven descent only)
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 JURIS-PEERAGE MODULE                        │
│  - Accept proven descent as input                          │
│  - Apply peerage law doctrines                             │
│  - Systematic denial logic                                 │
│  - Legal determination output                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
            LEGAL DETERMINATION
          (rights analysis output)
```

### Example: Robert Ringler Case

**Genealogy Module Output:**
```python
{
    "subject_name": "Robert Ringler",
    "proven_ancestors": [
        {"name": "Sir Thomas Arundell of Wardour", "dates": "c.1502-1552"},
        # ... other proven ancestors
    ],
    "gaps_identified": [
        "Connection from Ringler to colonial American families requires primary sources"
    ],
    "key_findings": {
        "proven": "Descent from English nobility (95%+ confidence)",
        "unproven": "Modern → colonial bridge requires GPS verification"
    }
}
```

**JURIS-PEERAGE Module Output:**
```python
{
    "subject": "Robert Ringler",
    "determination": "NO LEGAL CLAIM EXISTS",
    "reasoning": [
        "→ Sir Thomas Arundell attainted 1552; blood corrupted",
        "→ Honor restored 1570s, but NOT in blood",
        "→ Barony of Arundell of Wardour created 1605 (after Thomas died 1552)",
        "→ Robert Ringler descends from cadet branch (non-inheriting)",
        "→ Genealogical gaps remain in modern → colonial bridge"
    ],
    "fatal_defects": [
        "FATAL: Attainder prevents transmission (no blood restoration)",
        "FATAL: Title created AFTER ancestor died",
        "FATAL: Cadet branch exclusion applies"
    ],
    "modern_enforceability": "NON-JUSTICIABLE (Crown/Parliament authority)",
    "confidence": "DEFINITIVE"
}
```

**Key Points:**
- Genealogy proves descent ✅ (Edward I → Arundell line documented)
- Legal analysis applies doctrines ⚖️ (attainder, creation date, cadet branch)
- Result: Proven descent ≠ legal entitlement ❌

---

## Safety & Compliance

### Prohibited Uses

**JURIS-PEERAGE explicitly prohibits:**
1. Claiming titles without Crown/Parliamentary authorization
2. Conflating genealogical descent with legal entitlement
3. Bypassing constitutional procedures for title creation/restoration
4. Using romantic or identity-inflating language
5. Asserting rights in non-justiciable matters

### Required Compliance

**Module enforces:**
1. Constitutional law compliance
2. Crown prerogative respect
3. Parliamentary supremacy recognition (1689+)
4. Historical accuracy in legal doctrine
5. Clear separation: descent vs. entitlement

### Safety Disclaimer (Always Present)

Every determination includes:

> ⚖️ **LEGAL DISCLAIMER**: This analysis concerns historical peerage law. It does not constitute legal advice. Modern peerage claims are largely non-justiciable and require Crown or Parliamentary action. **DESCENT ≠ ENTITLEMENT**. Consult legal counsel for any actual claim.

---

## Testing & Validation

### Test Coverage: 11/11 Tests Passing ✅

**Tests validate:**
1. **Module Initialization**: Correct setup, disclaimers present
2. **Supported Tasks**: All five capabilities accessible
3. **Cadet Branch Denial**: Doctrine correctly excludes junior lines
4. **Attainder Denial**: Blood corruption prevents transmission
5. **Creation Date Constraint**: Title after death = no claim
6. **Genealogical Gaps**: Incomplete proof = undecidable
7. **Non-Justiciability**: Courts lack jurisdiction
8. **Cadet Branch Assessment**: Rights assessment correct
9. **Attainder Effects (Honor Only)**: Restoration types distinguished
10. **Attainder Effects (Blood Restored)**: Full restoration recognized
11. **Disclaimer Presence**: Legal disclaimer always included

### Test Examples

```python
def test_cadet_branch_denial(self):
    """Test cadet branch exclusion (fundamental doctrine)"""
    result = module.execute_task(
        task="determine_heirship",
        parameters={
            "genealogical_proof": {...},
            "title_context": {"cadet_branch": True},
        },
    )
    
    assert "NO LEGAL CLAIM" in result["determination"]
    assert "FATAL" in result["fatal_defects"][0]
    assert "cadet" in result["fatal_defects"][0].lower()
    assert result["confidence"] == "DEFINITIVE"
```

---

## Technical Implementation

### Module Architecture

**File**: `qratum/verticals/juris_peerage.py` (575 lines)

**Key Classes:**
- `JurisPeerageModule(VerticalModuleBase)`: Main module
- `TitleState(Enum)`: Title state machine
- `SuccessionMode(Enum)`: Succession types
- `AttainderStatus(Enum)`: Attainder effects
- `PeerageTitle`: Title representation
- `GenealogicalInput`: Input contract
- `LegalDetermination`: Output contract

**Key Methods:**
- `_determine_heirship()`: Heirship analysis
- `_analyze_title_succession()`: Succession mechanics
- `_evaluate_attainder_effects()`: Attainder impact
- `_assess_cadet_branch_rights()`: Cadet branch analysis
- `_determine_justiciability()`: Court jurisdiction

### Integration Points

**QRATUM Platform Integration:**
- Extends `VerticalModuleBase`
- Uses `PlatformContract` for execution authority
- Emits events to `MerkleEventChain`
- Maintains cryptographic auditability
- Enforces 8 fatal invariants

**Coexistence with JURIS:**
- **JURIS**: Contract analysis, litigation prediction, compliance (`qratum/verticals/juris.py`)
- **JURIS-PEERAGE**: Peerage law, title succession, heirship (`qratum/verticals/juris_peerage.py`)

Both modules independent, operate under QRATUM governance.

---

## Future Enhancements

### Potential Extensions

1. **Doctrine Knowledge Base Expansion**
   - Add Scottish peerage law doctrines
   - Add Irish peerage law doctrines
   - Model special remainders in patents
   - Handle courtesy titles

2. **Enhanced Title State Tracking**
   - Full title lifecycle modeling
   - Abeyance termination mechanics
   - Dormancy claim procedures
   - Royal warrant requirements

3. **Pedigree Validation**
   - Direct integration with genealogy module
   - Automated GPS → legal handoff
   - Batch heirship determination
   - Confidence scoring refinement

4. **Historical Precedent Database**
   - Case law integration
   - Committee for Privileges decisions
   - House of Lords rulings
   - Peerage claims precedents

5. **Crown Petition Preparation**
   - Generate petition documents
   - Required evidence checklists
   - Procedural guidance
   - Historical success rates

---

## Conclusion

JURIS-PEERAGE successfully extends QRATUM's legal AI capabilities to handle peerage law and nobility rights determination. The module maintains strict architectural separation (DESCENT ≠ ENTITLEMENT), applies formalized legal doctrines systematically, and provides definitive determinations with appropriate conservative bias (deny unless proven).

**Key Achievements:**
- ✅ Formalized 6 core peerage law doctrines
- ✅ Implemented 5 legal analysis capabilities
- ✅ Maintained strict genealogy/legal separation
- ✅ Default deny-unless-proven bias
- ✅ Non-justiciability recognition
- ✅ Comprehensive safety disclaimers
- ✅ 11/11 tests passing
- ✅ Integration with QRATUM platform
- ✅ Zero external dependencies

**Ready for Production**: Module can accept GPS-verified genealogical proof and provide legally sound determinations of peerage rights under English/UK law.

---

**Documentation Version**: 1.0  
**Last Updated**: 2026-01-19  
**Author**: GitHub Copilot (QRATUM Agent)  
**Status**: IMPLEMENTATION COMPLETE ✅
