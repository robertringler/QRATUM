"""
JURIS-PEERAGE - Peerage and Nobility Law Module

Deterministic legal-historical reasoning engine for English/UK peerage law,
title succession, and heirship determination.

HARD INVARIANTS (NON-NEGOTIABLE):
- DESCENT ≠ ENTITLEMENT
- GENEALOGY ≠ LAW
- NO TITLE EXISTS WITHOUT FORMAL CREATION
- NO REVIVAL WITHOUT CROWN OR PARLIAMENT
- DEFAULT BIAS = DENY UNLESS LEGALLY PROVEN
- NO ROMANTIC, NARRATIVE, OR IDENTITY LANGUAGE

This module accepts PROVEN genealogical output and determines legal rights
under English common law, UK peerage law, and constitutional precedent.
"""

from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from .base import VerticalModuleBase
from ..platform.core import PlatformContract, EventType
from ..platform.event_chain import MerkleEventChain


class TitleState(Enum):
    """Title existence states (state machine)"""
    CREATED = "created"           # Title formally created by Crown/Parliament
    EXTANT = "extant"             # Title currently exists with living holder
    EXTINCT = "extinct"           # Title permanently ceased (no eligible heirs)
    ABEYANT = "abeyant"           # Title suspended (multiple co-heirs)
    DORMANT = "dormant"           # Title unclaimed but potentially claimable
    ATTAINTED = "attainted"       # Title affected by Act of Attainder
    MERGED = "merged"             # Title merged into higher title


class SuccessionMode(Enum):
    """Legal succession modes"""
    HEIR_MALE = "heir_male"                    # Male-line only
    HEIR_GENERAL = "heir_general"              # Male-preference primogeniture
    COGNATIC = "cognatic"                      # Equal male/female inheritance
    MALE_PREFERENCE_PRIMOGENITURE = "male_preference_primogeniture"  # Traditional English
    SALIC = "salic"                            # Absolute male-only (rare in England)


class AttainderStatus(Enum):
    """Attainder effects"""
    NONE = "none"                              # No attainder
    ATTAINTED = "attainted"                    # Blood corrupted, no transmission
    HONOR_RESTORED = "honor_restored"          # Reputation restored (not in blood)
    BLOOD_RESTORED = "blood_restored"          # Full restoration by Parliament


@dataclass
class PeerageTitle:
    """Formal peerage title representation"""
    title_name: str                            # e.g., "Baron Arundell of Wardour"
    creation_date: Optional[str]               # ISO date or "unknown"
    patent_authority: str                      # "Crown" or "Parliament"
    succession_mode: SuccessionMode
    current_state: TitleState
    holder: Optional[str]                      # Current holder name (if extant)
    creation_lineage: str                      # Who received original creation
    attainder_status: AttainderStatus
    restoration_date: Optional[str]            # If restored
    extinction_date: Optional[str]             # If extinct
    legal_notes: List[str]                     # Key legal constraints


@dataclass
class GenealogicalInput:
    """Contract: Input from genealogy system (PROVEN ONLY)"""
    subject_name: str
    proven_ancestors: List[Dict[str, Any]]     # GPS-verified only
    documented_relationships: List[Dict[str, Any]]
    confidence_levels: Dict[str, float]         # Per relationship
    gaps_identified: List[str]                  # Explicit unproven segments
    source_citations: List[str]                 # Primary sources only


@dataclass
class LegalDetermination:
    """Output: Legal determination of rights"""
    subject_name: str
    determination: str                          # "NO LEGAL CLAIM" | "POTENTIALLY CLAIMABLE" | "CLAIM REQUIRES [X]"
    reasoning: List[str]                        # Step-by-step legal analysis
    applicable_law: List[str]                   # Statutes, precedents, doctrines
    fatal_defects: List[str]                    # Why claim fails (if denied)
    requirements_if_claimable: List[str]        # What would be needed (if potential)
    title_analysis: Optional[Dict[str, Any]]    # Title-specific analysis
    modern_enforceability: str                  # "ENFORCEABLE" | "NON-JUSTICIABLE" | "REQUIRES CROWN/PARLIAMENT"
    confidence: str                             # "DEFINITIVE" | "PROBABLE" | "UNDECIDABLE"


class JurisPeerageModule(VerticalModuleBase):
    """
    JURIS-PEERAGE: Peerage and Nobility Law Analysis Module
    
    Capabilities:
    - Title succession analysis
    - Heirship determination
    - Attainder effect modeling
    - Peerage law doctrine application
    - Crown prerogative and Parliamentary supremacy
    - Non-justiciability determination
    
    Does NOT:
    - Prove genealogical descent (handled by genealogy module)
    - Create titles or rights
    - Conflate descent with entitlement
    - Use romantic or identity-inflating language
    """
    
    def __init__(self):
        super().__init__(
            vertical_name="JURIS-PEERAGE",
            description="Peerage and nobility law analysis for title succession and heirship",
            safety_disclaimer=(
                "⚖️ LEGAL DISCLAIMER: This analysis concerns historical peerage law. "
                "It does not constitute legal advice. Modern peerage claims are largely "
                "non-justiciable and require Crown or Parliamentary action. "
                "DESCENT ≠ ENTITLEMENT. Consult legal counsel for any actual claim."
            ),
            prohibited_uses=[
                "Claiming titles without Crown/Parliamentary authorization",
                "Conflating genealogical descent with legal entitlement",
                "Bypassing constitutional procedures for title creation/restoration",
                "Using romantic or identity-inflating language",
                "Asserting rights in non-justiciable matters",
            ],
            required_compliance=[
                "Constitutional law compliance",
                "Crown prerogative respect",
                "Parliamentary supremacy recognition",
                "Historical accuracy in legal doctrine",
                "Clear separation: descent vs. entitlement",
            ],
        )
        
        # Initialize doctrine knowledge base
        self._load_peerage_doctrines()
    
    def _load_peerage_doctrines(self):
        """Load formalized peerage law doctrines"""
        self.doctrines = {
            "male_preference_primogeniture": {
                "rule": "Males inherit before females; eldest among equals",
                "applies_to": ["Most English peerages pre-1958"],
                "exceptions": ["Baronies by writ", "Special remainders in patent"],
            },
            "heir_male": {
                "rule": "Succession limited to male-line descendants only",
                "applies_to": ["Many baronies", "Dukedoms"],
                "extinction_if": "No male-line heir survives",
            },
            "cadet_branch_exclusion": {
                "rule": "Cadet (non-inheriting) branches have NO claim",
                "applies_to": "All primogeniture systems",
                "reason": "Title passes to senior line; junior lines excluded",
            },
            "attainder_blood_corruption": {
                "rule": "Attainder corrupts blood; title untransmittable",
                "restoration_types": {
                    "honor": "Reputation restored, NOT legal rights",
                    "blood": "Full restoration by Act of Parliament only",
                },
                "applies_to": "Treason convictions",
            },
            "creation_date_constraint": {
                "rule": "No claim to title created AFTER ancestor's death",
                "rationale": "Title did not exist to be inherited",
                "applies_to": "All titles",
            },
            "crown_prerogative": {
                "rule": "Crown creates peerages; Courts cannot override",
                "modern_status": "Parliamentary supremacy since 1689",
                "justiciability": "Peerage claims largely non-justiciable",
            },
        }
    
    def get_supported_tasks(self) -> List[str]:
        """Return supported peerage law tasks"""
        return [
            "determine_heirship",
            "analyze_title_succession",
            "evaluate_attainder_effects",
            "assess_cadet_branch_rights",
            "determine_justiciability",
        ]
    
    def execute_task(
        self,
        task: str,
        parameters: Dict[str, Any],
        contract: PlatformContract,
        event_chain: MerkleEventChain,
    ) -> Dict[str, Any]:
        """Execute peerage law analysis task"""
        
        # Validate task
        if task not in self.get_supported_tasks():
            raise ValueError(f"Unknown task: {task}. Supported: {self.get_supported_tasks()}")
        
        # Validate safety
        if not self.validate_safety(parameters):
            raise ValueError("Safety validation failed - prohibited use detected")
        
        # Emit task started event
        self.emit_task_event(
            event_type=EventType.TASK_STARTED,
            contract_id=contract.contract_id,
            task=task,
            data={"parameters": parameters},
            event_chain=event_chain,
        )
        
        # Route to appropriate handler
        if task == "determine_heirship":
            result = self._determine_heirship(parameters)
        elif task == "analyze_title_succession":
            result = self._analyze_title_succession(parameters)
        elif task == "evaluate_attainder_effects":
            result = self._evaluate_attainder_effects(parameters)
        elif task == "assess_cadet_branch_rights":
            result = self._assess_cadet_branch_rights(parameters)
        elif task == "determine_justiciability":
            result = self._determine_justiciability(parameters)
        else:
            raise ValueError(f"Task handler not implemented: {task}")
        
        # Emit task completed event
        self.emit_task_event(
            event_type=EventType.TASK_COMPLETED,
            contract_id=contract.contract_id,
            task=task,
            data={"result_type": type(result).__name__},
            event_chain=event_chain,
        )
        
        return result
    
    def _determine_heirship(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine if subject has legal heirship claim to any title.
        
        DEFAULT BIAS: NO LEGAL CLAIM EXISTS (prove otherwise)
        """
        genealogical_input = parameters.get("genealogical_proof")
        if not genealogical_input:
            raise ValueError("genealogical_proof required (from genealogy module)")
        
        subject = genealogical_input.get("subject_name", "Unknown")
        title_context = parameters.get("title_context", {})
        
        # Initialize determination
        determination = LegalDetermination(
            subject_name=subject,
            determination="NO LEGAL CLAIM EXISTS",
            reasoning=[],
            applicable_law=[],
            fatal_defects=[],
            requirements_if_claimable=[],
            title_analysis=None,
            modern_enforceability="NON-JUSTICIABLE",
            confidence="DEFINITIVE",
        )
        
        # Apply legal doctrines systematically
        determination.reasoning.append("DOCTRINE APPLICATION (systematic):")
        
        # 1. Check if title ever existed
        if not title_context.get("title_created"):
            determination.fatal_defects.append(
                "FATAL: No evidence of formal title creation by Crown/Parliament"
            )
            determination.reasoning.append(
                "→ NO TITLE EXISTS WITHOUT FORMAL CREATION (doctrine)"
            )
            determination.confidence = "DEFINITIVE"
            return self._format_determination(determination)
        
        # 2. Check creation date vs. ancestor date
        creation_date = title_context.get("creation_date")
        ancestor_death = title_context.get("ancestor_death_date")
        if creation_date and ancestor_death:
            if self._date_after(creation_date, ancestor_death):
                determination.fatal_defects.append(
                    f"FATAL: Title created {creation_date} AFTER ancestor died {ancestor_death}"
                )
                determination.reasoning.append(
                    "→ CREATION DATE CONSTRAINT: Cannot inherit title that didn't exist (doctrine)"
                )
                determination.confidence = "DEFINITIVE"
                return self._format_determination(determination)
        
        # 3. Check for attainder
        attainder_status = title_context.get("attainder_status", "NONE")
        if attainder_status == "ATTAINTED":
            determination.fatal_defects.append(
                "FATAL: Ancestor attainted for treason; blood corrupted; title untransmittable"
            )
            determination.reasoning.append(
                "→ ATTAINDER BLOOD CORRUPTION: Attainder prevents transmission (doctrine)"
            )
            if title_context.get("honor_restored_only"):
                determination.reasoning.append(
                    "→ Honor restoration ≠ restoration in blood (requires Act of Parliament)"
                )
            determination.confidence = "DEFINITIVE"
            return self._format_determination(determination)
        
        # 4. Check cadet branch status
        if title_context.get("cadet_branch", False):
            determination.fatal_defects.append(
                "FATAL: Subject descends from cadet (non-inheriting) branch"
            )
            determination.reasoning.append(
                "→ CADET BRANCH EXCLUSION: Only senior line inherits under primogeniture (doctrine)"
            )
            determination.confidence = "DEFINITIVE"
            return self._format_determination(determination)
        
        # 5. Check title current state
        title_state = title_context.get("current_state", "UNKNOWN")
        if title_state == "EXTINCT":
            determination.fatal_defects.append(
                "FATAL: Title legally extinct (no eligible heirs at extinction)"
            )
            determination.reasoning.append(
                "→ EXTINCTION: Title ceased; no revival without Crown/Parliament action"
            )
            determination.modern_enforceability = "REQUIRES CROWN/PARLIAMENT"
            determination.confidence = "DEFINITIVE"
            return self._format_determination(determination)
        
        # 6. Check genealogical proof gaps
        gaps = genealogical_input.get("gaps_identified", [])
        if gaps:
            determination.determination = "UNDECIDABLE (genealogical gaps)"
            determination.reasoning.append(
                f"→ GENEALOGICAL PROOF INCOMPLETE: {len(gaps)} gap(s) in proven descent"
            )
            for gap in gaps[:3]:  # Show first 3 gaps
                determination.reasoning.append(f"  • {gap}")
            determination.requirements_if_claimable.append(
                "GPS-certified proof of all generational links required"
            )
            determination.confidence = "UNDECIDABLE"
            return self._format_determination(determination)
        
        # 7. Modern justiciability
        determination.reasoning.append(
            "→ CROWN PREROGATIVE & PARLIAMENTARY SUPREMACY:"
        )
        determination.reasoning.append(
            "  Modern peerage claims are NON-JUSTICIABLE in courts"
        )
        determination.reasoning.append(
            "  Any claim requires Crown petition or Act of Parliament"
        )
        determination.modern_enforceability = "NON-JUSTICIABLE"
        
        # Default: no claim unless proven otherwise above
        determination.determination = "NO LEGAL CLAIM EXISTS"
        determination.reasoning.append(
            "→ CONCLUSION: No viable legal claim identified under English/UK peerage law"
        )
        determination.confidence = "DEFINITIVE"
        
        return self._format_determination(determination)
    
    def _analyze_title_succession(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze title succession mechanics for specific title"""
        title_data = parameters.get("title")
        if not title_data:
            raise ValueError("title data required")
        
        analysis = {
            "title": title_data.get("name"),
            "succession_mode": self._determine_succession_mode(title_data),
            "current_state": self._determine_title_state(title_data),
            "eligible_heirs": self._identify_eligible_heirs(title_data, parameters.get("proven_lineage")),
            "succession_order": [],
            "legal_constraints": [],
        }
        
        # Apply succession rules
        succession_mode = analysis["succession_mode"]
        if succession_mode == "HEIR_MALE":
            analysis["legal_constraints"].append(
                "Male-line only: females and female-line descendants excluded"
            )
        elif succession_mode == "MALE_PREFERENCE_PRIMOGENITURE":
            analysis["legal_constraints"].append(
                "Males inherit before females; eldest among equals"
            )
        
        # State-specific constraints
        if analysis["current_state"] == TitleState.EXTINCT.value:
            analysis["legal_constraints"].append(
                "Title EXTINCT: No revival without Crown/Parliament action"
            )
        elif analysis["current_state"] == TitleState.ABEYANT.value:
            analysis["legal_constraints"].append(
                "Title ABEYANT: Multiple co-heirs; Crown termination required"
            )
        
        return {
            "analysis": analysis,
            "determination": "See succession_order for eligible heirs (if any)",
            "modern_enforceability": "NON-JUSTICIABLE (Crown prerogative applies)",
        }
    
    def _evaluate_attainder_effects(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate effects of attainder on title transmission"""
        attainder_data = parameters.get("attainder")
        if not attainder_data:
            raise ValueError("attainder data required")
        
        effects = {
            "attainted_person": attainder_data.get("person"),
            "attainder_date": attainder_data.get("date"),
            "offense": attainder_data.get("offense", "Treason"),
            "blood_corruption": True,
            "title_transmission": "PREVENTED",
            "restoration_status": "NONE",
            "legal_effects": [],
        }
        
        # Attainder corrupts blood
        effects["legal_effects"].append(
            "Blood corrupted: title cannot pass through attainted line"
        )
        effects["legal_effects"].append(
            "Heirs of attainted person have NO claim"
        )
        
        # Check for restoration
        restoration = attainder_data.get("restoration")
        if restoration:
            if restoration.get("type") == "honor_only":
                effects["restoration_status"] = "HONOR RESTORED (not in blood)"
                effects["legal_effects"].append(
                    "Honor restoration ≠ legal restoration: No transmission rights"
                )
                effects["title_transmission"] = "STILL PREVENTED"
            elif restoration.get("type") == "blood":
                effects["restoration_status"] = "BLOOD RESTORED (Act of Parliament)"
                effects["legal_effects"].append(
                    "Full restoration by Parliament: Transmission rights restored"
                )
                effects["title_transmission"] = "PERMITTED (post-restoration)"
                effects["blood_corruption"] = False
        
        return {
            "effects": effects,
            "determination": f"Title transmission: {effects['title_transmission']}",
        }
    
    def _assess_cadet_branch_rights(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Assess rights of cadet (non-inheriting) branch members"""
        lineage = parameters.get("lineage")
        if not lineage:
            raise ValueError("lineage data required")
        
        assessment = {
            "branch_type": "CADET (non-inheriting)",
            "rights_to_title": "NONE",
            "reasoning": [],
            "doctrine_applied": "CADET BRANCH EXCLUSION",
        }
        
        assessment["reasoning"].append(
            "Under primogeniture, title passes to senior (eldest) line"
        )
        assessment["reasoning"].append(
            "Cadet branches (younger sons, junior lines) are excluded"
        )
        assessment["reasoning"].append(
            "No claim exists regardless of descent proximity"
        )
        assessment["reasoning"].append(
            "This is fundamental to English succession law"
        )
        
        return {
            "assessment": assessment,
            "determination": "NO LEGAL CLAIM (cadet branch exclusion applies)",
        }
    
    def _determine_justiciability(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Determine if peerage claim is justiciable in courts"""
        claim_type = parameters.get("claim_type", "peerage")
        
        justiciability = {
            "claim_type": claim_type,
            "justiciable_in_courts": False,
            "reasoning": [],
            "proper_forum": "Crown petition or Act of Parliament",
            "constitutional_basis": [],
        }
        
        justiciability["reasoning"].append(
            "Peerage matters are Crown prerogative since medieval times"
        )
        justiciability["reasoning"].append(
            "Parliamentary supremacy established 1689 (Bill of Rights)"
        )
        justiciability["reasoning"].append(
            "Modern courts generally refuse jurisdiction over peerage claims"
        )
        justiciability["reasoning"].append(
            "Proper remedy: Petition Crown or seek Act of Parliament"
        )
        
        justiciability["constitutional_basis"].append(
            "Crown prerogative (royal prerogative of honor)"
        )
        justiciability["constitutional_basis"].append(
            "Parliamentary supremacy (1689+)"
        )
        justiciability["constitutional_basis"].append(
            "Non-justiciability doctrine (courts decline jurisdiction)"
        )
        
        return {
            "justiciability": justiciability,
            "determination": "NON-JUSTICIABLE (Crown/Parliament authority)",
        }
    
    # Helper methods
    
    def _determine_succession_mode(self, title_data: Dict[str, Any]) -> str:
        """Determine succession mode for title"""
        mode = title_data.get("succession_mode")
        if mode:
            return mode
        # Default for English peerages
        return SuccessionMode.MALE_PREFERENCE_PRIMOGENITURE.value
    
    def _determine_title_state(self, title_data: Dict[str, Any]) -> str:
        """Determine current state of title"""
        state = title_data.get("current_state")
        if state:
            return state
        # If unknown, assume extinct (conservative)
        return TitleState.EXTINCT.value
    
    def _identify_eligible_heirs(
        self,
        title_data: Dict[str, Any],
        proven_lineage: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Identify eligible heirs under succession rules"""
        if not proven_lineage:
            return []
        
        # This would apply succession mode rules to proven lineage
        # For now, return empty (no heirs identified without full proof)
        return []
    
    def _date_after(self, date1: str, date2: str) -> bool:
        """Check if date1 is after date2 (simple comparison)"""
        try:
            return date1 > date2  # ISO date string comparison
        except:
            return False
    
    def _format_determination(self, determination: LegalDetermination) -> Dict[str, Any]:
        """Format determination as dictionary"""
        return {
            "subject": determination.subject_name,
            "determination": determination.determination,
            "reasoning": determination.reasoning,
            "applicable_law": determination.applicable_law,
            "fatal_defects": determination.fatal_defects,
            "requirements_if_claimable": determination.requirements_if_claimable,
            "modern_enforceability": determination.modern_enforceability,
            "confidence": determination.confidence,
            "disclaimer": (
                "⚖️ This is legal analysis, not legal advice. "
                "DESCENT ≠ ENTITLEMENT. Peerage claims are non-justiciable. "
                "Consult legal counsel. Crown/Parliament action required for any claim."
            ),
        }
