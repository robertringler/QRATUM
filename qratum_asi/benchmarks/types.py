"""Type definitions for QRATUM Benchmark Suite."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class BenchmarkCategory(Enum):
    """Categories of benchmark tasks."""

    # Formal reasoning
    THEOREM_PROVING = "theorem_proving"
    LOGICAL_DEDUCTION = "logical_deduction"
    MATHEMATICAL_PROBLEM_SOLVING = "mathematical_problem_solving"

    # Scientific reasoning
    SCIENTIFIC_HYPOTHESIS = "scientific_hypothesis"
    EXPERIMENT_DESIGN = "experiment_design"
    DATA_INTERPRETATION = "data_interpretation"

    # Strategic and game-theoretic
    STRATEGIC_PLANNING = "strategic_planning"
    GAME_PLAYING = "game_playing"
    NEGOTIATION = "negotiation"

    # Creative
    CREATIVE_WRITING = "creative_writing"
    POETRY_COMPOSITION = "poetry_composition"
    ARTISTIC_DESIGN = "artistic_design"

    # Language and knowledge
    READING_COMPREHENSION = "reading_comprehension"
    KNOWLEDGE_SYNTHESIS = "knowledge_synthesis"
    EXPLANATION_GENERATION = "explanation_generation"

    # Cross-domain synthesis
    CROSS_DOMAIN_REASONING = "cross_domain_reasoning"
    PARADIGM_GENERATION = "paradigm_generation"
    NOVEL_INVENTION = "novel_invention"


class PerformanceLevel(Enum):
    """Performance level classification."""

    BELOW_HUMAN = "below_human"  # Below average human
    HUMAN_AVERAGE = "human_average"  # At average human level
    HUMAN_EXPERT = "human_expert"  # At expert human level
    SUPERHUMAN = "superhuman"  # Beyond best human performance


class DifficultyLevel(Enum):
    """Difficulty level of tasks."""

    ELEMENTARY = "elementary"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    FRONTIER = "frontier"  # At the edge of human capability


@dataclass
class HumanBaseline:
    """Human performance baseline for a benchmark.

    Attributes:
        baseline_id: Unique identifier
        category: Benchmark category
        human_average_score: Average human score
        human_expert_score: Expert human score
        world_record_score: Best known human score
        source: Source of baseline data
        sample_size: How many humans measured
        measurement_date: When measured
    """

    baseline_id: str
    category: BenchmarkCategory
    human_average_score: float
    human_expert_score: float
    world_record_score: float
    source: str
    sample_size: int
    measurement_date: str


@dataclass
class BenchmarkTask:
    """A specific benchmark task.

    Attributes:
        task_id: Unique identifier
        category: Category of task
        name: Task name
        description: What the task measures
        difficulty: Difficulty level
        input_format: Format of task input
        output_format: Expected output format
        evaluation_criteria: How to evaluate
        max_score: Maximum possible score
        human_baseline: Human baseline if available
    """

    task_id: str
    category: BenchmarkCategory
    name: str
    description: str
    difficulty: DifficultyLevel
    input_format: str
    output_format: str
    evaluation_criteria: list[str]
    max_score: float = 100.0
    human_baseline: HumanBaseline | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of a benchmark evaluation.

    Attributes:
        result_id: Unique identifier
        task_id: Task that was evaluated
        score: Achieved score
        max_score: Maximum possible score
        normalized_score: Score as fraction of max
        performance_level: Classification vs human baseline
        reasoning_chain: Reasoning used (if applicable)
        time_taken_seconds: Time to complete
        provenance_hash: Hash for provenance
        timestamp: Evaluation timestamp
    """

    result_id: str
    task_id: str
    score: float
    max_score: float
    normalized_score: float
    performance_level: PerformanceLevel
    reasoning_chain: list[str]
    time_taken_seconds: float
    provenance_hash: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_superhuman(self) -> bool:
        """Check if performance exceeds human expert level."""
        return self.performance_level == PerformanceLevel.SUPERHUMAN


@dataclass
class EvaluationSummary:
    """Summary of evaluation across multiple benchmarks.

    Attributes:
        summary_id: Unique identifier
        total_tasks: Total tasks evaluated
        tasks_by_category: Count per category
        average_normalized_score: Overall average
        superhuman_count: Tasks with superhuman performance
        expert_count: Tasks at expert level
        performance_distribution: Distribution of performance levels
        timestamp: Summary timestamp
    """

    summary_id: str
    total_tasks: int
    tasks_by_category: dict[str, int]
    average_normalized_score: float
    superhuman_count: int
    expert_count: int
    performance_distribution: dict[str, int]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
