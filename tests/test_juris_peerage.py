"""
Tests for JURIS-PEERAGE Module

Validates peerage law analysis, title succession determination,
and heirship legal reasoning.
"""

import unittest
from qratum.verticals.juris_peerage import (
    JurisPeerageModule,
    TitleState,
    SuccessionMode,
    AttainderStatus,
)
from qratum.platform.core import PlatformContract, EventType
from qratum.platform.event_chain import MerkleEventChain


class TestJurisPeerageModule(unittest.TestCase):
    """Test JURIS-PEERAGE module functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        from qratum.platform.core import PlatformIntent, ContractStatus
        from datetime import datetime, timezone
        
        self.module = JurisPeerageModule()
        
        # Create a proper PlatformIntent
        intent = PlatformIntent(
            vertical="JURIS-PEERAGE",
            task="determine_heirship",
            parameters={},
            requester_id="test-client",
        )
        
        # Create a proper PlatformContract
        now = datetime.now(timezone.utc).isoformat()
        self.contract = PlatformContract(
            intent=intent,
            contract_id="test-peerage-001",
            authorized_by="qcore-test",
            status=ContractStatus.AUTHORIZED,
            created_at=now,
            authorized_at=now,
        )
        self.event_chain = MerkleEventChain()
    
    def test_module_initialization(self):
        """Test module initializes correctly"""
        self.assertEqual(self.module.vertical_name, "JURIS-PEERAGE")
        self.assertIn("LEGAL DISCLAIMER", self.module.safety_disclaimer)
        self.assertIn("DESCENT ≠ ENTITLEMENT", self.module.safety_disclaimer)
    
    def test_supported_tasks(self):
        """Test supported tasks list"""
        tasks = self.module.get_supported_tasks()
        self.assertIn("determine_heirship", tasks)
        self.assertIn("analyze_title_succession", tasks)
        self.assertIn("evaluate_attainder_effects", tasks)
        self.assertIn("assess_cadet_branch_rights", tasks)
        self.assertIn("determine_justiciability", tasks)
    
    def test_cadet_branch_denial(self):
        """Test cadet branch exclusion (fundamental doctrine)"""
        parameters = {
            "genealogical_proof": {
                "subject_name": "John Smith",
                "proven_ancestors": [],
                "documented_relationships": [],
                "confidence_levels": {},
                "gaps_identified": [],
                "source_citations": [],
            },
            "title_context": {
                "title_created": True,
                "creation_date": "1552-01-01",  # BEFORE ancestor died (so passes that check)
                "ancestor_death_date": "1600-01-01",
                "attainder_status": "NONE",
                "cadet_branch": True,  # KEY: Subject from cadet branch
                "current_state": "EXTANT",
            },
        }
        
        result = self.module.execute_task(
            task="determine_heirship",
            parameters=parameters,
            contract=self.contract,
            event_chain=self.event_chain,
        )
        
        self.assertIn("NO LEGAL CLAIM", result["determination"])
        self.assertIn("FATAL", result["fatal_defects"][0])
        self.assertIn("cadet", result["fatal_defects"][0].lower())
        self.assertEqual(result["confidence"], "DEFINITIVE")
    
    def test_attainder_denial(self):
        """Test attainder blood corruption prevents transmission"""
        parameters = {
            "genealogical_proof": {
                "subject_name": "Jane Doe",
                "proven_ancestors": [],
                "documented_relationships": [],
                "confidence_levels": {},
                "gaps_identified": [],
                "source_citations": [],
            },
            "title_context": {
                "title_created": True,
                "creation_date": "1500-01-01",
                "ancestor_death_date": "1552-01-01",
                "attainder_status": "ATTAINTED",  # KEY: Attainder blocks transmission
                "honor_restored_only": True,
                "cadet_branch": False,
                "current_state": "EXTINCT",
            },
        }
        
        result = self.module.execute_task(
            task="determine_heirship",
            parameters=parameters,
            contract=self.contract,
            event_chain=self.event_chain,
        )
        
        self.assertIn("NO LEGAL CLAIM", result["determination"])
        fatal_defect = result["fatal_defects"][0]
        self.assertIn("attainted", fatal_defect.lower())
        self.assertIn("blood corrupted", fatal_defect.lower())
        self.assertIn("Honor restoration ≠ restoration in blood", result["reasoning"][2])
    
    def test_creation_date_constraint(self):
        """Test title created AFTER ancestor died = no claim"""
        parameters = {
            "genealogical_proof": {
                "subject_name": "Robert Ringler",
                "proven_ancestors": [],
                "documented_relationships": [],
                "confidence_levels": {},
                "gaps_identified": [],
                "source_citations": [],
            },
            "title_context": {
                "title_created": True,
                "creation_date": "1605-01-01",  # Created 1605
                "ancestor_death_date": "1552-01-01",  # Died 1552 (before title existed)
                "attainder_status": "NONE",
                "cadet_branch": False,
                "current_state": "EXTANT",
            },
        }
        
        result = self.module.execute_task(
            task="determine_heirship",
            parameters=parameters,
            contract=self.contract,
            event_chain=self.event_chain,
        )
        
        self.assertIn("NO LEGAL CLAIM", result["determination"])
        self.assertIn("AFTER ancestor died", result["fatal_defects"][0])
        self.assertIn("CREATION DATE CONSTRAINT", result["reasoning"][1])
    
    def test_genealogical_gaps_undecidable(self):
        """Test genealogical proof gaps = undecidable"""
        parameters = {
            "genealogical_proof": {
                "subject_name": "Alice Johnson",
                "proven_ancestors": [],
                "documented_relationships": [],
                "confidence_levels": {},
                "gaps_identified": [
                    "Generation 5-6: No primary source proof",
                    "Generation 8-9: Conflicting records",
                ],
                "source_citations": [],
            },
            "title_context": {
                "title_created": True,
                "creation_date": "1500-01-01",
                "ancestor_death_date": "1600-01-01",
                "attainder_status": "NONE",
                "cadet_branch": False,
                "current_state": "EXTANT",
            },
        }
        
        result = self.module.execute_task(
            task="determine_heirship",
            parameters=parameters,
            contract=self.contract,
            event_chain=self.event_chain,
        )
        
        self.assertIn("UNDECIDABLE", result["determination"])
        self.assertIn("genealogical gaps", result["determination"].lower())
        self.assertEqual(result["confidence"], "UNDECIDABLE")
        self.assertIn("GPS-certified proof", result["requirements_if_claimable"][0])
    
    def test_non_justiciability(self):
        """Test peerage claims are non-justiciable"""
        parameters = {
            "claim_type": "peerage",
        }
        
        result = self.module.execute_task(
            task="determine_justiciability",
            parameters=parameters,
            contract=self.contract,
            event_chain=self.event_chain,
        )
        
        justiciability = result["justiciability"]
        self.assertFalse(justiciability["justiciable_in_courts"])
        self.assertIn("Crown petition or Act of Parliament", justiciability["proper_forum"])
        self.assertIn("Crown prerogative", justiciability["constitutional_basis"][0])
        self.assertIn("Parliamentary supremacy", justiciability["constitutional_basis"][1])
    
    def test_cadet_branch_assessment(self):
        """Test cadet branch rights assessment"""
        parameters = {
            "lineage": {
                "branch_type": "cadet",
                "senior_line_exists": True,
            },
        }
        
        result = self.module.execute_task(
            task="assess_cadet_branch_rights",
            parameters=parameters,
            contract=self.contract,
            event_chain=self.event_chain,
        )
        
        assessment = result["assessment"]
        self.assertEqual(assessment["rights_to_title"], "NONE")
        self.assertEqual(assessment["doctrine_applied"], "CADET BRANCH EXCLUSION")
        self.assertIn("NO LEGAL CLAIM", result["determination"])
    
    def test_attainder_effects_honor_only(self):
        """Test attainder with honor-only restoration"""
        parameters = {
            "attainder": {
                "person": "Sir Thomas Arundell",
                "date": "1552",
                "offense": "Treason",
                "restoration": {
                    "type": "honor_only",
                    "date": "1570s",
                },
            },
        }
        
        result = self.module.execute_task(
            task="evaluate_attainder_effects",
            parameters=parameters,
            contract=self.contract,
            event_chain=self.event_chain,
        )
        
        effects = result["effects"]
        self.assertTrue(effects["blood_corruption"])
        self.assertEqual(effects["title_transmission"], "STILL PREVENTED")
        self.assertEqual(effects["restoration_status"], "HONOR RESTORED (not in blood)")
        self.assertIn("Honor restoration ≠ legal restoration", effects["legal_effects"][2])
    
    def test_attainder_effects_blood_restored(self):
        """Test attainder with full blood restoration"""
        parameters = {
            "attainder": {
                "person": "Earl of Example",
                "date": "1715",
                "offense": "Treason (Jacobite Rising)",
                "restoration": {
                    "type": "blood",
                    "date": "1824",
                    "authority": "Act of Parliament",
                },
            },
        }
        
        result = self.module.execute_task(
            task="evaluate_attainder_effects",
            parameters=parameters,
            contract=self.contract,
            event_chain=self.event_chain,
        )
        
        effects = result["effects"]
        self.assertFalse(effects["blood_corruption"])
        self.assertEqual(effects["title_transmission"], "PERMITTED (post-restoration)")
        self.assertEqual(effects["restoration_status"], "BLOOD RESTORED (Act of Parliament)")
    
    def test_disclaimer_always_present(self):
        """Test disclaimer present in all determinations"""
        parameters = {
            "genealogical_proof": {
                "subject_name": "Test Subject",
                "proven_ancestors": [],
                "documented_relationships": [],
                "confidence_levels": {},
                "gaps_identified": [],
                "source_citations": [],
            },
            "title_context": {
                "title_created": False,
            },
        }
        
        result = self.module.execute_task(
            task="determine_heirship",
            parameters=parameters,
            contract=self.contract,
            event_chain=self.event_chain,
        )
        
        self.assertIn("disclaimer", result)
        self.assertIn("DESCENT ≠ ENTITLEMENT", result["disclaimer"])
        self.assertIn("non-justiciable", result["disclaimer"].lower())
        self.assertIn("Crown/Parliament", result["disclaimer"])


if __name__ == "__main__":
    unittest.main()
