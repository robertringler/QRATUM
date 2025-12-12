"""Prompt loader service for Qubic Meta Library."""

import csv
from pathlib import Path

import yaml

from qubic_meta_library.models import Domain, Prompt


class PromptLoader:
    """Service for loading prompts and domains from configuration files."""

    def __init__(self, config_dir: Path | None = None, data_dir: Path | None = None):
        """
        Initialize prompt loader.

        Args:
            config_dir: Directory containing configuration files
            data_dir: Directory containing prompt data files
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "config"
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"

        self.config_dir = Path(config_dir)
        self.data_dir = Path(data_dir)
        self.domains: dict[str, Domain] = {}
        self.prompts: dict[int, Prompt] = {}

    def load_domains(self) -> dict[str, Domain]:
        """
        Load domain configurations.

        Returns:
            Dictionary mapping domain IDs to Domain objects
        """
        domains_file = self.config_dir / "domains.yaml"
        if not domains_file.exists():
            raise FileNotFoundError(f"Domains configuration not found: {domains_file}")

        with open(domains_file) as f:
            data = yaml.safe_load(f)

        self.domains = {}
        for domain_data in data.get("domains", []):
            domain = Domain.from_dict(domain_data)
            self.domains[domain.id] = domain

        return self.domains

    def load_prompts_from_csv(self, csv_file: Path) -> list[Prompt]:
        """
        Load prompts from CSV file.

        Args:
            csv_file: Path to CSV file

        Returns:
            List of Prompt objects
        """
        prompts = []
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = self._parse_prompt_row(row)
                prompts.append(prompt)
                self.prompts[prompt.id] = prompt

        return prompts

    def load_all_prompts(self) -> dict[int, Prompt]:
        """
        Load all prompts from data directory.

        Returns:
            Dictionary mapping prompt IDs to Prompt objects
        """
        prompts_dir = self.data_dir / "prompts"
        if not prompts_dir.exists():
            raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

        self.prompts = {}
        for csv_file in prompts_dir.glob("*.csv"):
            self.load_prompts_from_csv(csv_file)

        return self.prompts

    def get_prompts_by_domain(self, domain_id: str) -> list[Prompt]:
        """
        Get all prompts for a specific domain.

        Args:
            domain_id: Domain identifier (e.g., 'D1')

        Returns:
            List of prompts in the domain
        """
        return [p for p in self.prompts.values() if p.domain == domain_id]

    def get_prompts_by_phase(self, phase: int) -> list[Prompt]:
        """
        Get all prompts for a specific deployment phase.

        Args:
            phase: Phase number (1-4)

        Returns:
            List of prompts in the phase
        """
        return [p for p in self.prompts.values() if p.phase_deployment == phase]

    def get_high_value_prompts(self, threshold: float = 0.8) -> list[Prompt]:
        """
        Get high-value prompts based on patentability and commercial scores.

        Args:
            threshold: Minimum score threshold

        Returns:
            List of high-value prompts
        """
        return [p for p in self.prompts.values() if p.is_high_value(threshold)]

    def _parse_prompt_row(self, row: dict[str, str]) -> Prompt:
        """Parse CSV row into Prompt object."""
        return Prompt(
            id=int(row["id"]),
            category=row["category"],
            description=row["description"],
            domain=row["domain"],
            patentability_score=float(row["patentability_score"]),
            commercial_potential=float(row["commercial_potential"]),
            keystone_nodes=self._parse_list(row.get("keystone_nodes", "")),
            synergy_connections=self._parse_list(row.get("synergy_connections", "")),
            execution_layers=self._parse_list(row.get("execution_layers", "")),
            phase_deployment=int(row.get("phase_deployment", 1)),
            output_type=row.get("output_type", "simulation"),
        )

    def _parse_list(self, value: str) -> list[str]:
        """Parse semicolon-separated string into list."""
        if not value or value.strip() == "":
            return []
        return [item.strip() for item in value.split(";") if item.strip()]
