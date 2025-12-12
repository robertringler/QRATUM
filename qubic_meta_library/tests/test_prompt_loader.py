"""Tests for PromptLoader service."""

import csv

import pytest

from qubic_meta_library.services import PromptLoader


class TestPromptLoader:
    """Test PromptLoader service."""

    @pytest.fixture
    def loader(self, tmp_path):
        """Create loader with test data."""
        # Create config directory
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        # Create domains config
        domains_content = """
domains:
  - id: D1
    name: Test Domain 1
    tier: 1
    id_range: [1, 10]
    primary_platform: QuASIM
    commercial_sector: Test
    keystones: [Key1]
  - id: D2
    name: Test Domain 2
    tier: 1
    id_range: [11, 20]
    primary_platform: QStack
    commercial_sector: Test
    keystones: [Key2]
"""
        (config_dir / "domains.yaml").write_text(domains_content)

        # Create data directory with prompts
        data_dir = tmp_path / "data"
        prompts_dir = data_dir / "prompts"
        prompts_dir.mkdir(parents=True)

        # Create test prompts CSV
        csv_file = prompts_dir / "test_prompts.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "id",
                    "category",
                    "description",
                    "domain",
                    "patentability_score",
                    "commercial_potential",
                    "keystone_nodes",
                    "synergy_connections",
                    "execution_layers",
                    "phase_deployment",
                    "output_type",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "id": 1,
                    "category": "Test",
                    "description": "Test prompt 1",
                    "domain": "D1",
                    "patentability_score": 0.85,
                    "commercial_potential": 0.90,
                    "keystone_nodes": "Node1;Node2",
                    "synergy_connections": "D2",
                    "execution_layers": "QuASIM",
                    "phase_deployment": 1,
                    "output_type": "simulation",
                }
            )
            writer.writerow(
                {
                    "id": 2,
                    "category": "Test",
                    "description": "Test prompt 2",
                    "domain": "D1",
                    "patentability_score": 0.92,
                    "commercial_potential": 0.88,
                    "keystone_nodes": "",
                    "synergy_connections": "",
                    "execution_layers": "QStack",
                    "phase_deployment": 2,
                    "output_type": "model",
                }
            )

        return PromptLoader(config_dir=config_dir, data_dir=data_dir)

    def test_load_domains(self, loader):
        """Test loading domains from config."""
        domains = loader.load_domains()

        assert len(domains) == 2
        assert "D1" in domains
        assert "D2" in domains
        assert domains["D1"].name == "Test Domain 1"

    def test_load_prompts_from_csv(self, loader, tmp_path):
        """Test loading prompts from CSV file."""
        csv_file = tmp_path / "data" / "prompts" / "test_prompts.csv"
        prompts = loader.load_prompts_from_csv(csv_file)

        assert len(prompts) == 2
        assert prompts[0].id == 1
        assert prompts[0].category == "Test"
        assert prompts[0].patentability_score == 0.85
        assert len(prompts[0].keystone_nodes) == 2

    def test_load_all_prompts(self, loader):
        """Test loading all prompts from data directory."""
        prompts = loader.load_all_prompts()

        assert len(prompts) == 2
        assert 1 in prompts
        assert 2 in prompts

    def test_get_prompts_by_domain(self, loader):
        """Test getting prompts by domain."""
        loader.load_all_prompts()

        d1_prompts = loader.get_prompts_by_domain("D1")
        assert len(d1_prompts) == 2

        d2_prompts = loader.get_prompts_by_domain("D2")
        assert len(d2_prompts) == 0

    def test_get_prompts_by_phase(self, loader):
        """Test getting prompts by phase."""
        loader.load_all_prompts()

        phase1_prompts = loader.get_prompts_by_phase(1)
        assert len(phase1_prompts) == 1
        assert phase1_prompts[0].id == 1

        phase2_prompts = loader.get_prompts_by_phase(2)
        assert len(phase2_prompts) == 1
        assert phase2_prompts[0].id == 2

    def test_get_high_value_prompts(self, loader):
        """Test getting high-value prompts."""
        loader.load_all_prompts()

        high_value = loader.get_high_value_prompts(threshold=0.8)
        assert len(high_value) == 2

        # No prompts meet 0.91 threshold (prompt 1: P=0.85 C=0.90, prompt 2: P=0.92 C=0.88)
        premium = loader.get_high_value_prompts(threshold=0.91)
        assert len(premium) == 0

    def test_parse_list(self, loader):
        """Test parsing semicolon-separated lists."""
        assert loader._parse_list("A;B;C") == ["A", "B", "C"]
        assert loader._parse_list("") == []
        assert loader._parse_list("  ") == []
        assert loader._parse_list("A ; B ; C") == ["A", "B", "C"]
