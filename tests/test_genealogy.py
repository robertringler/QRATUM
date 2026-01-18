"""
Tests for genealogical research orchestration system.
"""

import unittest
from datetime import date

from genealogy.models.person import Person, Gender, Relationship, RelationshipType
from genealogy.models.record import (
    GenealogicalRecord,
    Source,
    Citation,
    RecordType,
    SourceType,
    EvidenceQuality,
)
from genealogy.agents import (
    ModernRecordsAgent,
    ColonialAmericanAgent,
    EnglishNobilityAgent,
    MedievalRoyalAgent,
    ProofAnalysisAgent,
    HeraldicAgent,
)
from genealogy.orchestrator import ChiefResearchOrchestrator


class TestPerson(unittest.TestCase):
    """Test Person model."""
    
    def test_person_creation(self):
        """Test creating a person."""
        person = Person(
            person_id="TEST_001",
            full_name="John Doe",
            gender=Gender.MALE,
        )
        self.assertEqual(person.person_id, "TEST_001")
        self.assertEqual(person.full_name, "John Doe")
        self.assertEqual(person.gender, Gender.MALE)
    
    def test_person_str(self):
        """Test person string representation."""
        person = Person(
            person_id="TEST_001",
            full_name="John Doe",
            gender=Gender.MALE,
            birth_date=date(1800, 1, 1),
            death_date=date(1880, 12, 31),
        )
        result = str(person)
        self.assertIn("John Doe", result)
        self.assertIn("1800", result)


class TestRelationship(unittest.TestCase):
    """Test Relationship model."""
    
    def test_relationship_creation(self):
        """Test creating a relationship."""
        rel = Relationship(
            relationship_id="REL_001",
            person1_id="PARENT_001",
            person2_id="CHILD_001",
            relationship_type=RelationshipType.PARENT_CHILD,
        )
        self.assertEqual(rel.relationship_id, "REL_001")
        self.assertEqual(rel.relationship_type, RelationshipType.PARENT_CHILD)


class TestSource(unittest.TestCase):
    """Test Source model."""
    
    def test_source_creation(self):
        """Test creating a source."""
        source = Source(
            source_id="SRC_001",
            title="Test Source",
            source_type=SourceType.ORIGINAL_DOCUMENT,
            reliability=EvidenceQuality.PRIMARY,
        )
        self.assertEqual(source.source_id, "SRC_001")
        self.assertEqual(source.reliability, EvidenceQuality.PRIMARY)
    
    def test_formatted_citation(self):
        """Test formatted citation generation."""
        source = Source(
            source_id="SRC_001",
            title="Birth Register",
            source_type=SourceType.ORIGINAL_DOCUMENT,
            repository="County Archives",
            reliability=EvidenceQuality.PRIMARY,
        )
        citation = source.formatted_citation()
        self.assertIn("Birth Register", citation)
        self.assertIn("County Archives", citation)


class TestModernRecordsAgent(unittest.TestCase):
    """Test ModernRecordsAgent."""
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = ModernRecordsAgent()
        self.assertEqual(agent.agent_id, "agent_1_modern_records")
        self.assertEqual(agent.status, "initialized")
    
    def test_agent_execution(self):
        """Test agent execution."""
        agent = ModernRecordsAgent()
        context = {"subject_name": "Test Subject"}
        result = agent.execute_research(context)
        
        self.assertTrue(result["success"])
        self.assertEqual(agent.status, "completed")
        self.assertGreater(len(agent.findings), 0)


class TestColonialAmericanAgent(unittest.TestCase):
    """Test ColonialAmericanAgent."""
    
    def test_agent_execution(self):
        """Test agent execution."""
        agent = ColonialAmericanAgent()
        context = {"subject_name": "Test Subject"}
        result = agent.execute_research(context)
        
        self.assertTrue(result["success"])
        self.assertEqual(agent.status, "completed")


class TestEnglishNobilityAgent(unittest.TestCase):
    """Test EnglishNobilityAgent."""
    
    def test_agent_execution(self):
        """Test agent execution."""
        agent = EnglishNobilityAgent()
        context = {"subject_name": "Test Subject"}
        result = agent.execute_research(context)
        
        self.assertTrue(result["success"])
        self.assertEqual(agent.status, "completed")


class TestMedievalRoyalAgent(unittest.TestCase):
    """Test MedievalRoyalAgent."""
    
    def test_agent_execution(self):
        """Test agent execution."""
        agent = MedievalRoyalAgent()
        context = {"subject_name": "Test Subject"}
        result = agent.execute_research(context)
        
        self.assertTrue(result["success"])
        self.assertEqual(agent.status, "completed")
        
        # Check that Edward I is documented
        self.assertIn("EDWARD_I", agent.persons)


class TestProofAnalysisAgent(unittest.TestCase):
    """Test ProofAnalysisAgent."""
    
    def test_agent_execution(self):
        """Test agent execution with mock data."""
        agent = ProofAnalysisAgent()
        context = {
            "subject_name": "Test Subject",
            "all_agents_results": [],
        }
        result = agent.execute_research(context)
        
        self.assertTrue(result["success"])
        self.assertIn("gps_compliance", result)
        self.assertIn("proof_determination", result)


class TestHeraldicAgent(unittest.TestCase):
    """Test HeraldicAgent."""
    
    def test_agent_execution(self):
        """Test agent execution."""
        agent = HeraldicAgent()
        context = {"subject_name": "Test Subject"}
        result = agent.execute_research(context)
        
        self.assertTrue(result["success"])
        self.assertEqual(agent.status, "completed")


class TestChiefOrchestrator(unittest.TestCase):
    """Test ChiefResearchOrchestrator."""
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        orchestrator = ChiefResearchOrchestrator(subject_name="Test Subject")
        self.assertEqual(orchestrator.subject_name, "Test Subject")
        self.assertEqual(len(orchestrator.agents), 6)
    
    def test_orchestrator_execution_sequential(self):
        """Test orchestrator execution in sequential mode."""
        orchestrator = ChiefResearchOrchestrator(subject_name="Robert Ringler Jr.")
        
        # Execute in sequential mode for testing
        results = orchestrator.execute_research(parallel=False)
        
        self.assertIn("subject", results)
        self.assertEqual(results["subject"], "Robert Ringler Jr.")
        self.assertIn("agents_results", results)
        self.assertIn("synthesis", results)
        
        # Check that all agents executed
        agents_results = results["agents_results"]
        self.assertIn("agent_1_modern", agents_results)
        self.assertIn("agent_2_colonial", agents_results)
        self.assertIn("agent_3_nobility", agents_results)
        self.assertIn("agent_4_medieval", agents_results)
        self.assertIn("agent_5_proof", agents_results)
        self.assertIn("agent_6_heraldic", agents_results)
        
        # Check synthesis components
        synthesis = results["synthesis"]
        self.assertIn("executive_summary", synthesis)
        self.assertIn("generation_table", synthesis)
        self.assertIn("descent_chart", synthesis)
        self.assertIn("proof_argument", synthesis)
        self.assertIn("limitations", synthesis)
        self.assertIn("recommendations", synthesis)


if __name__ == "__main__":
    unittest.main()
