"""PHASE IV: Abstraction Compression Engine

Intelligence must simplify itself over time.

Key capabilities:
- Detect repeated patterns across algorithms, graphs, workloads
- Replace special-case logic with generalized primitives
- Measure intelligence growth as compression ratio
- Fewer concepts explaining more behavior with equal/greater performance
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from datetime import datetime
from enum import Enum
from collections import defaultdict
import hashlib
import json


class PatternType(Enum):
    """Types of patterns that can be detected."""
    ALGORITHM = "algorithm"  # Repeated algorithmic pattern
    DATA_STRUCTURE = "data_structure"  # Repeated data structure usage
    CONTROL_FLOW = "control_flow"  # Repeated control flow pattern
    TRANSFORMATION = "transformation"  # Repeated data transformation
    INTERFACE = "interface"  # Repeated interface pattern


@dataclass
class Pattern:
    """A detected pattern."""
    pattern_id: str
    pattern_type: PatternType
    description: str
    occurrences: List[str]  # Where this pattern occurs (component IDs)
    complexity_score: float  # How complex is this pattern?
    frequency: int  # How often does it occur?
    discovered_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def get_compression_potential(self) -> float:
        """Estimate compression potential (higher = more worth abstracting)."""
        # More occurrences = more compression potential
        # Higher complexity = more compression potential
        return self.frequency * self.complexity_score


@dataclass
class AbstractionPrimitive:
    """A generalized primitive that replaces multiple special cases."""
    primitive_id: str
    name: str
    description: str
    replaces_patterns: List[str]  # Pattern IDs this primitive replaces
    replaces_count: int  # How many special cases does this replace?
    complexity: float  # Complexity of the primitive itself
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    usage_count: int = 0
    
    def get_compression_ratio(self, replaced_complexity: float) -> float:
        """Compute compression ratio.
        
        compression_ratio = (old_complexity) / (new_complexity)
        Higher is better (more compression achieved).
        """
        return replaced_complexity / max(self.complexity, 0.1)


@dataclass
class CompressionMetrics:
    """Metrics for measuring compression progress."""
    total_concepts: int  # How many distinct concepts/patterns?
    total_complexity: float  # Sum of all complexities
    compression_ratio: float  # Current compression ratio
    intelligence_score: float  # Intelligence = compression with performance
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class AbstractionCompressionEngine:
    """Engine for compressing system complexity through abstraction.
    
    Intelligence growth is measured as:
    - Fewer concepts (abstractions) 
    - Explaining more behavior (pattern coverage)
    - With equal or greater performance
    
    This is the key to recursive self-improvement: each iteration should
    make the system conceptually simpler while maintaining/improving capability.
    """
    
    def __init__(self):
        """Initialize compression engine."""
        self.patterns: Dict[str, Pattern] = {}
        self.primitives: Dict[str, AbstractionPrimitive] = {}
        self.metrics_history: List[CompressionMetrics] = []
        
        # Track code/concept instances
        self.concept_instances: Dict[str, List[str]] = defaultdict(list)
        
        # Performance baselines
        self.performance_baselines: Dict[str, float] = {}
    
    def detect_patterns(
        self,
        codebase_analysis: Dict[str, Any]
    ) -> List[Pattern]:
        """Detect repeated patterns across the codebase."""
        detected = []
        
        # Analyze algorithms
        algorithms = codebase_analysis.get("algorithms", {})
        algorithm_patterns = self._detect_algorithm_patterns(algorithms)
        detected.extend(algorithm_patterns)
        
        # Analyze data structures
        data_structures = codebase_analysis.get("data_structures", {})
        ds_patterns = self._detect_data_structure_patterns(data_structures)
        detected.extend(ds_patterns)
        
        # Analyze control flow
        control_flows = codebase_analysis.get("control_flows", {})
        cf_patterns = self._detect_control_flow_patterns(control_flows)
        detected.extend(cf_patterns)
        
        # Store detected patterns
        for pattern in detected:
            self.patterns[pattern.pattern_id] = pattern
        
        return detected
    
    def _detect_algorithm_patterns(
        self,
        algorithms: Dict[str, Any]
    ) -> List[Pattern]:
        """Detect repeated algorithmic patterns."""
        patterns = []
        
        # Group algorithms by similarity
        algorithm_signatures = {}
        for algo_id, algo_data in algorithms.items():
            # Compute signature (simplified: based on operations)
            operations = tuple(sorted(algo_data.get("operations", [])))
            if operations not in algorithm_signatures:
                algorithm_signatures[operations] = []
            algorithm_signatures[operations].append(algo_id)
        
        # Create patterns for repeated signatures
        for idx, (signature, occurrences) in enumerate(algorithm_signatures.items()):
            if len(occurrences) >= 2:  # At least 2 occurrences to be a pattern
                pattern = Pattern(
                    pattern_id=f"algo_pattern_{idx}",
                    pattern_type=PatternType.ALGORITHM,
                    description=f"Repeated algorithm with operations: {signature[:3]}...",
                    occurrences=occurrences,
                    complexity_score=len(signature) * 1.5,
                    frequency=len(occurrences)
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_data_structure_patterns(
        self,
        data_structures: Dict[str, Any]
    ) -> List[Pattern]:
        """Detect repeated data structure patterns."""
        patterns = []
        
        # Group by structure type and usage
        structure_groups = defaultdict(list)
        for ds_id, ds_data in data_structures.items():
            ds_type = ds_data.get("type", "unknown")
            operations = tuple(sorted(ds_data.get("operations", [])))
            key = (ds_type, operations)
            structure_groups[key].append(ds_id)
        
        # Create patterns
        for idx, (key, occurrences) in enumerate(structure_groups.items()):
            if len(occurrences) >= 2:
                ds_type, operations = key
                pattern = Pattern(
                    pattern_id=f"ds_pattern_{idx}",
                    pattern_type=PatternType.DATA_STRUCTURE,
                    description=f"Repeated {ds_type} with operations: {operations[:3]}...",
                    occurrences=occurrences,
                    complexity_score=len(operations),
                    frequency=len(occurrences)
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_control_flow_patterns(
        self,
        control_flows: Dict[str, Any]
    ) -> List[Pattern]:
        """Detect repeated control flow patterns."""
        patterns = []
        
        # Group by control flow structure
        flow_groups = defaultdict(list)
        for flow_id, flow_data in control_flows.items():
            # Simplified: group by structure type
            structure = flow_data.get("structure", "sequential")
            flow_groups[structure].append(flow_id)
        
        # Create patterns for common structures
        for structure, occurrences in flow_groups.items():
            if len(occurrences) >= 3:  # Higher threshold for control flow
                pattern = Pattern(
                    pattern_id=f"cf_pattern_{structure}",
                    pattern_type=PatternType.CONTROL_FLOW,
                    description=f"Repeated {structure} control flow",
                    occurrences=occurrences,
                    complexity_score=2.0,  # Control flow complexity
                    frequency=len(occurrences)
                )
                patterns.append(pattern)
        
        return patterns
    
    def propose_abstraction(
        self,
        pattern_ids: List[str],
        primitive_name: str,
        primitive_description: str,
        primitive_complexity: float
    ) -> AbstractionPrimitive:
        """Propose a primitive abstraction that replaces multiple patterns."""
        # Validate patterns exist
        patterns = []
        for pid in pattern_ids:
            if pid not in self.patterns:
                raise ValueError(f"Pattern not found: {pid}")
            patterns.append(self.patterns[pid])
        
        # Calculate what this primitive replaces
        replaces_count = sum(p.frequency for p in patterns)
        
        # Create primitive
        primitive = AbstractionPrimitive(
            primitive_id=f"primitive_{datetime.utcnow().timestamp()}",
            name=primitive_name,
            description=primitive_description,
            replaces_patterns=pattern_ids,
            replaces_count=replaces_count,
            complexity=primitive_complexity
        )
        
        self.primitives[primitive.primitive_id] = primitive
        return primitive
    
    def evaluate_abstraction(
        self,
        primitive: AbstractionPrimitive,
        performance_impact: float  # Multiplier: 1.0 = same, >1.0 = better, <1.0 = worse
    ) -> Dict[str, Any]:
        """Evaluate if an abstraction is worthwhile.
        
        Good abstraction:
        - Reduces complexity (compression)
        - Maintains or improves performance
        - Makes system conceptually simpler
        """
        # Calculate replaced complexity
        replaced_complexity = 0.0
        for pattern_id in primitive.replaces_patterns:
            if pattern_id in self.patterns:
                pattern = self.patterns[pattern_id]
                replaced_complexity += pattern.complexity_score * pattern.frequency
        
        # Calculate compression ratio
        compression_ratio = primitive.get_compression_ratio(replaced_complexity)
        
        # Calculate net benefit (compression * performance)
        net_benefit = compression_ratio * performance_impact
        
        # Worth adopting if:
        # 1. Compression ratio > 2.0 (at least 2x reduction)
        # 2. Performance not degraded (>=0.95)
        # 3. Net benefit > 1.5
        worthwhile = (
            compression_ratio >= 2.0 and
            performance_impact >= 0.95 and
            net_benefit >= 1.5
        )
        
        return {
            "primitive_id": primitive.primitive_id,
            "worthwhile": worthwhile,
            "compression_ratio": compression_ratio,
            "performance_impact": performance_impact,
            "net_benefit": net_benefit,
            "replaced_complexity": replaced_complexity,
            "new_complexity": primitive.complexity
        }
    
    def apply_abstraction(
        self,
        primitive_id: str,
        performance_multiplier: float = 1.0
    ) -> bool:
        """Apply an abstraction to the system.
        
        This represents actually replacing the pattern instances with
        the primitive abstraction.
        """
        if primitive_id not in self.primitives:
            return False
        
        primitive = self.primitives[primitive_id]
        
        # Evaluate if worthwhile
        evaluation = self.evaluate_abstraction(primitive, performance_multiplier)
        
        if not evaluation["worthwhile"]:
            return False
        
        # Mark primitive as used
        primitive.usage_count += 1
        
        # In real implementation, would actually refactor the code
        # For now, just record that patterns are replaced
        for pattern_id in primitive.replaces_patterns:
            if pattern_id in self.patterns:
                # Mark pattern as abstracted
                self.patterns[pattern_id].frequency = 0  # No longer repeated
        
        return True
    
    def compute_metrics(
        self,
        system_behavior_coverage: float = 1.0,
        system_performance: float = 1.0
    ) -> CompressionMetrics:
        """Compute current compression metrics.
        
        Args:
            system_behavior_coverage: How much of system behavior is explained (0-1)
            system_performance: Current system performance relative to baseline (0-inf)
        """
        # Count active concepts
        active_patterns = sum(1 for p in self.patterns.values() if p.frequency > 0)
        active_primitives = len(self.primitives)
        total_concepts = active_patterns + active_primitives
        
        # Calculate total complexity
        pattern_complexity = sum(
            p.complexity_score * p.frequency
            for p in self.patterns.values()
            if p.frequency > 0
        )
        primitive_complexity = sum(
            p.complexity * p.usage_count
            for p in self.primitives.values()
        )
        total_complexity = pattern_complexity + primitive_complexity
        
        # Compression ratio = behavior coverage / complexity
        # Higher = more behavior explained with less complexity
        compression_ratio = system_behavior_coverage / max(total_complexity, 1.0)
        
        # Intelligence score = compression * performance
        # System is more intelligent if it's both simpler AND more capable
        intelligence_score = compression_ratio * system_performance
        
        metrics = CompressionMetrics(
            total_concepts=total_concepts,
            total_complexity=total_complexity,
            compression_ratio=compression_ratio,
            intelligence_score=intelligence_score
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def measure_intelligence_growth(self) -> Dict[str, Any]:
        """Measure how intelligence has grown over time.
        
        Intelligence growth = system becoming simpler while maintaining/improving capability
        """
        if len(self.metrics_history) < 2:
            return {
                "intelligence_growth": 0.0,
                "compression_improvement": 0.0,
                "message": "Insufficient history"
            }
        
        # Compare first and last metrics
        first = self.metrics_history[0]
        last = self.metrics_history[-1]
        
        # Intelligence growth
        intelligence_growth = (
            (last.intelligence_score - first.intelligence_score) /
            max(first.intelligence_score, 0.01)
        )
        
        # Compression improvement
        compression_improvement = (
            (last.compression_ratio - first.compression_ratio) /
            max(first.compression_ratio, 0.01)
        )
        
        # Concept reduction
        concept_reduction = (
            (first.total_concepts - last.total_concepts) /
            max(first.total_concepts, 1)
        )
        
        return {
            "intelligence_growth": intelligence_growth,
            "compression_improvement": compression_improvement,
            "concept_reduction": concept_reduction,
            "metrics_count": len(self.metrics_history),
            "current_intelligence": last.intelligence_score,
            "current_concepts": last.total_concepts
        }
    
    def get_top_compression_opportunities(
        self,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """Get top opportunities for compression."""
        # Rank patterns by compression potential
        opportunities = []
        
        for pattern in self.patterns.values():
            if pattern.frequency == 0:
                continue  # Already abstracted
            
            potential = pattern.get_compression_potential()
            opportunities.append({
                "pattern_id": pattern.pattern_id,
                "pattern_type": pattern.pattern_type.value,
                "description": pattern.description,
                "compression_potential": potential,
                "frequency": pattern.frequency,
                "complexity": pattern.complexity_score
            })
        
        # Sort by potential
        opportunities.sort(key=lambda x: x["compression_potential"], reverse=True)
        
        return opportunities[:top_n]
    
    def get_abstraction_report(self) -> Dict[str, Any]:
        """Get comprehensive abstraction report."""
        # Current state
        current_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        # Primitives stats
        primitive_stats = {
            "total_primitives": len(self.primitives),
            "active_primitives": sum(1 for p in self.primitives.values() if p.usage_count > 0),
            "total_replacements": sum(p.replaces_count for p in self.primitives.values())
        }
        
        # Pattern stats
        pattern_stats = {
            "total_patterns_detected": len(self.patterns),
            "active_patterns": sum(1 for p in self.patterns.values() if p.frequency > 0),
            "abstracted_patterns": sum(1 for p in self.patterns.values() if p.frequency == 0)
        }
        
        # Growth
        growth = self.measure_intelligence_growth()
        
        return {
            "current_metrics": {
                "total_concepts": current_metrics.total_concepts if current_metrics else 0,
                "total_complexity": current_metrics.total_complexity if current_metrics else 0,
                "compression_ratio": current_metrics.compression_ratio if current_metrics else 0,
                "intelligence_score": current_metrics.intelligence_score if current_metrics else 0
            } if current_metrics else None,
            "primitive_stats": primitive_stats,
            "pattern_stats": pattern_stats,
            "intelligence_growth": growth,
            "top_opportunities": self.get_top_compression_opportunities()
        }
