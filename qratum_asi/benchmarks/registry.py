"""Benchmark Registry for QRATUM SI Transition.

Maintains a registry of benchmark tasks across all cognitive domains
with human baselines for measuring progress toward superintelligence.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from qratum_asi.benchmarks.types import (
    BenchmarkCategory,
    BenchmarkTask,
    DifficultyLevel,
    HumanBaseline,
)


@dataclass
class TaskDefinition:
    """Definition for creating a benchmark task.

    Attributes:
        name: Task name
        category: Category
        description: Description
        difficulty: Difficulty level
        evaluation_criteria: How to evaluate
        input_template: Template for input
        expected_output_type: Type of expected output
    """

    name: str
    category: BenchmarkCategory
    description: str
    difficulty: DifficultyLevel
    evaluation_criteria: list[str]
    input_template: str
    expected_output_type: str


class BenchmarkRegistry:
    """Registry of benchmark tasks for SI evaluation.

    Maintains comprehensive benchmarks across all cognitive domains
    to measure progress toward universal superhuman performance.
    """

    def __init__(self):
        """Initialize the benchmark registry."""
        self.tasks: dict[str, BenchmarkTask] = {}
        self.baselines: dict[str, HumanBaseline] = {}
        self.categories_index: dict[BenchmarkCategory, list[str]] = {}

        self._task_counter = 0
        self._baseline_counter = 0

        # Initialize default benchmarks
        self._initialize_default_benchmarks()

    def _initialize_default_benchmarks(self) -> None:
        """Initialize default benchmark tasks."""
        # Theorem proving benchmarks
        self._add_default_task(
            TaskDefinition(
                name="First-Order Logic Theorem",
                category=BenchmarkCategory.THEOREM_PROVING,
                description="Prove theorems in first-order predicate logic",
                difficulty=DifficultyLevel.EXPERT,
                evaluation_criteria=["logical_validity", "completeness", "efficiency"],
                input_template="Given axioms: {axioms}. Prove: {goal}",
                expected_output_type="formal_proof",
            ),
            human_average=45.0,
            human_expert=85.0,
            world_record=98.0,
        )

        # Mathematical problem solving
        self._add_default_task(
            TaskDefinition(
                name="Mathematical Olympiad Problem",
                category=BenchmarkCategory.MATHEMATICAL_PROBLEM_SOLVING,
                description="Solve competition-level mathematics problems",
                difficulty=DifficultyLevel.FRONTIER,
                evaluation_criteria=["correctness", "elegance", "generalizability"],
                input_template="Problem: {problem_statement}",
                expected_output_type="solution_with_proof",
            ),
            human_average=20.0,
            human_expert=70.0,
            world_record=95.0,
        )

        # Scientific hypothesis generation
        self._add_default_task(
            TaskDefinition(
                name="Novel Hypothesis Generation",
                category=BenchmarkCategory.SCIENTIFIC_HYPOTHESIS,
                description="Generate testable scientific hypotheses from data",
                difficulty=DifficultyLevel.ADVANCED,
                evaluation_criteria=["novelty", "testability", "plausibility", "explanatory_power"],
                input_template="Domain: {domain}. Observations: {observations}",
                expected_output_type="hypothesis_with_predictions",
            ),
            human_average=40.0,
            human_expert=75.0,
            world_record=90.0,
        )

        # Strategic planning
        self._add_default_task(
            TaskDefinition(
                name="Multi-Agent Strategic Planning",
                category=BenchmarkCategory.STRATEGIC_PLANNING,
                description="Develop optimal strategies in complex multi-agent scenarios",
                difficulty=DifficultyLevel.EXPERT,
                evaluation_criteria=["optimality", "robustness", "adaptability"],
                input_template="Scenario: {scenario}. Objectives: {objectives}",
                expected_output_type="strategic_plan",
            ),
            human_average=35.0,
            human_expert=70.0,
            world_record=88.0,
        )

        # Game playing (Chess-level)
        self._add_default_task(
            TaskDefinition(
                name="Chess Position Evaluation",
                category=BenchmarkCategory.GAME_PLAYING,
                description="Evaluate chess positions and find optimal moves",
                difficulty=DifficultyLevel.EXPERT,
                evaluation_criteria=["accuracy", "depth", "speed"],
                input_template="Position: {fen_position}",
                expected_output_type="evaluation_and_best_move",
            ),
            human_average=50.0,
            human_expert=85.0,
            world_record=99.0,  # GM level
        )

        # Creative writing
        self._add_default_task(
            TaskDefinition(
                name="Short Story Composition",
                category=BenchmarkCategory.CREATIVE_WRITING,
                description="Compose original short stories with literary merit",
                difficulty=DifficultyLevel.ADVANCED,
                evaluation_criteria=["originality", "coherence", "style", "emotional_impact"],
                input_template="Theme: {theme}. Constraints: {constraints}",
                expected_output_type="narrative_text",
            ),
            human_average=50.0,
            human_expert=80.0,
            world_record=95.0,
        )

        # Poetry
        self._add_default_task(
            TaskDefinition(
                name="Formal Poetry Composition",
                category=BenchmarkCategory.POETRY_COMPOSITION,
                description="Compose poetry in specified forms (sonnet, haiku, etc.)",
                difficulty=DifficultyLevel.ADVANCED,
                evaluation_criteria=["form_adherence", "imagery", "meaning", "aesthetic_quality"],
                input_template="Form: {form}. Subject: {subject}",
                expected_output_type="poem",
            ),
            human_average=45.0,
            human_expert=75.0,
            world_record=92.0,
        )

        # Cross-domain reasoning
        self._add_default_task(
            TaskDefinition(
                name="Cross-Domain Synthesis",
                category=BenchmarkCategory.CROSS_DOMAIN_REASONING,
                description="Synthesize insights across disparate domains",
                difficulty=DifficultyLevel.FRONTIER,
                evaluation_criteria=["validity", "novelty", "usefulness", "connection_depth"],
                input_template="Domains: {domains}. Question: {question}",
                expected_output_type="synthesis_with_reasoning",
            ),
            human_average=30.0,
            human_expert=65.0,
            world_record=85.0,
        )

        # Negotiation
        self._add_default_task(
            TaskDefinition(
                name="Multi-Party Negotiation",
                category=BenchmarkCategory.NEGOTIATION,
                description="Navigate complex multi-party negotiations",
                difficulty=DifficultyLevel.EXPERT,
                evaluation_criteria=["outcome_quality", "fairness", "relationship_preservation"],
                input_template="Parties: {parties}. Interests: {interests}. BATNA: {batna}",
                expected_output_type="negotiation_strategy",
            ),
            human_average=45.0,
            human_expert=75.0,
            world_record=90.0,
        )

        # Paradigm generation (hardest)
        self._add_default_task(
            TaskDefinition(
                name="Novel Paradigm Invention",
                category=BenchmarkCategory.PARADIGM_GENERATION,
                description="Invent genuinely novel theoretical paradigms",
                difficulty=DifficultyLevel.FRONTIER,
                evaluation_criteria=["novelty", "coherence", "explanatory_power", "falsifiability"],
                input_template="Field: {field}. Current paradigm limitations: {limitations}",
                expected_output_type="paradigm_proposal",
            ),
            human_average=15.0,  # Very few can do this
            human_expert=50.0,
            world_record=75.0,  # Even geniuses struggle
        )

    def _add_default_task(
        self,
        definition: TaskDefinition,
        human_average: float,
        human_expert: float,
        world_record: float,
    ) -> BenchmarkTask:
        """Add a default task with baseline."""
        self._task_counter += 1
        task_id = f"task_{self._task_counter:04d}"

        self._baseline_counter += 1
        baseline_id = f"baseline_{self._baseline_counter:04d}"

        baseline = HumanBaseline(
            baseline_id=baseline_id,
            category=definition.category,
            human_average_score=human_average,
            human_expert_score=human_expert,
            world_record_score=world_record,
            source="QRATUM SI Benchmark Suite v1.0",
            sample_size=1000,  # Placeholder
            measurement_date=datetime.now(timezone.utc).isoformat(),
        )

        task = BenchmarkTask(
            task_id=task_id,
            category=definition.category,
            name=definition.name,
            description=definition.description,
            difficulty=definition.difficulty,
            input_format=definition.input_template,
            output_format=definition.expected_output_type,
            evaluation_criteria=definition.evaluation_criteria,
            human_baseline=baseline,
        )

        self.tasks[task_id] = task
        self.baselines[baseline_id] = baseline

        if definition.category not in self.categories_index:
            self.categories_index[definition.category] = []
        self.categories_index[definition.category].append(task_id)

        return task

    def get_task(self, task_id: str) -> BenchmarkTask | None:
        """Get a task by ID."""
        return self.tasks.get(task_id)

    def get_tasks_by_category(
        self, category: BenchmarkCategory
    ) -> list[BenchmarkTask]:
        """Get all tasks in a category."""
        task_ids = self.categories_index.get(category, [])
        return [self.tasks[tid] for tid in task_ids]

    def get_all_tasks(self) -> list[BenchmarkTask]:
        """Get all registered tasks."""
        return list(self.tasks.values())

    def get_all_categories(self) -> list[BenchmarkCategory]:
        """Get all categories with tasks."""
        return list(self.categories_index.keys())

    def add_custom_task(
        self,
        name: str,
        category: BenchmarkCategory,
        description: str,
        difficulty: DifficultyLevel,
        evaluation_criteria: list[str],
        input_format: str,
        output_format: str,
        human_baseline: HumanBaseline | None = None,
    ) -> BenchmarkTask:
        """Add a custom benchmark task."""
        self._task_counter += 1
        task_id = f"task_{self._task_counter:04d}"

        task = BenchmarkTask(
            task_id=task_id,
            category=category,
            name=name,
            description=description,
            difficulty=difficulty,
            input_format=input_format,
            output_format=output_format,
            evaluation_criteria=evaluation_criteria,
            human_baseline=human_baseline,
        )

        self.tasks[task_id] = task

        if category not in self.categories_index:
            self.categories_index[category] = []
        self.categories_index[category].append(task_id)

        return task

    def get_registry_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_tasks": len(self.tasks),
            "total_categories": len(self.categories_index),
            "tasks_per_category": {
                cat.value: len(tasks)
                for cat, tasks in self.categories_index.items()
            },
            "total_baselines": len(self.baselines),
        }
