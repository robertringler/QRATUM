"""Telemetry Dashboard Panel: Performance metrics and system monitoring.

Features:
- Nodes/sec, threads active, hash hit rate, recovery metrics
- Real-time performance graphs
- Memory usage and latency tracking
- Compliance/audit log status with ISO/NIST/EU AI alignment indicators
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import time

import numpy as np


class ComplianceStatus(Enum):
    """Compliance status levels."""
    COMPLIANT = "compliant"
    WARNING = "warning"
    NON_COMPLIANT = "non_compliant"
    UNKNOWN = "unknown"


@dataclass
class PerformanceMetrics:
    """Core performance metrics."""
    nodes_per_second: float = 0.0
    evaluations_per_second: float = 0.0
    moves_per_second: float = 0.0
    
    # Targets from Stage IV benchmarking
    target_nps_single: float = 70_000_000  # 70M single core
    target_nps_multi: float = 1_900_000_000  # 1.9B 32 threads
    target_mcts_rollouts: float = 500_000  # 500k/sec
    target_nn_latency_ms: float = 0.15  # ≤0.15ms


@dataclass
class ThreadMetrics:
    """Thread and parallelization metrics."""
    active_threads: int = 1
    max_threads: int = 1
    thread_utilization: float = 0.0  # 0-1
    load_balance_score: float = 1.0  # How well balanced


@dataclass
class CacheMetrics:
    """Hash table and cache metrics."""
    hash_hit_rate: float = 0.0  # Target: ≥93%
    hash_table_size_mb: int = 256
    hash_table_fill: float = 0.0
    
    # Cache performance
    l3_miss_rate: float = 0.0  # Target: ≤8%
    branch_misprediction: float = 0.0  # Target: ≤4%


@dataclass
class LatencyMetrics:
    """Latency measurements."""
    nn_eval_latency_ms: float = 0.0
    move_gen_latency_us: float = 0.0
    search_overhead_ms: float = 0.0
    
    # History
    latency_history: list[float] = field(default_factory=list)


@dataclass
class MemoryMetrics:
    """Memory usage metrics."""
    heap_used_mb: float = 0.0
    heap_total_mb: float = 0.0
    search_tree_mb: float = 0.0
    nn_weights_mb: float = 0.0


@dataclass
class RecoveryMetrics:
    """Fault recovery metrics."""
    recovery_count: int = 0
    last_recovery_time_ms: float = 0.0
    mean_recovery_time_ms: float = 0.0
    max_recovery_time_ms: float = 0.0  # Target: ≤500ms


@dataclass
class ComplianceMetrics:
    """Compliance and audit metrics."""
    iso_status: ComplianceStatus = ComplianceStatus.UNKNOWN
    nist_status: ComplianceStatus = ComplianceStatus.UNKNOWN
    eu_ai_status: ComplianceStatus = ComplianceStatus.UNKNOWN
    
    # Audit
    audit_log_enabled: bool = False
    audit_entries_count: int = 0
    last_audit_timestamp: float = 0.0


@dataclass
class TelemetryState:
    """Complete state of the telemetry panel."""
    # Metrics
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    threads: ThreadMetrics = field(default_factory=ThreadMetrics)
    cache: CacheMetrics = field(default_factory=CacheMetrics)
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    memory: MemoryMetrics = field(default_factory=MemoryMetrics)
    recovery: RecoveryMetrics = field(default_factory=RecoveryMetrics)
    compliance: ComplianceMetrics = field(default_factory=ComplianceMetrics)
    
    # History (for graphs)
    nps_history: list[float] = field(default_factory=list)
    hash_hit_history: list[float] = field(default_factory=list)
    memory_history: list[float] = field(default_factory=list)
    
    # Session info
    session_start_time: float = 0.0
    total_nodes_searched: int = 0
    total_positions_evaluated: int = 0
    
    # Display options
    show_performance: bool = True
    show_threads: bool = True
    show_cache: bool = True
    show_latency: bool = True
    show_memory: bool = True
    show_recovery: bool = False
    show_compliance: bool = True
    graph_time_window: float = 60.0  # seconds


class TelemetryPanel:
    """Telemetry Dashboard Panel for performance monitoring.
    
    This panel provides:
    - Real-time nodes/sec and evaluation speed metrics
    - Thread utilization and parallelization monitoring
    - Hash table hit rates and cache performance
    - Memory usage tracking
    - Latency measurements
    - Compliance status indicators
    """
    
    # Color coding for metrics
    COLORS = {
        'good': (0.2, 0.8, 0.3),  # Green
        'warning': (1.0, 0.8, 0.0),  # Yellow
        'bad': (1.0, 0.3, 0.2),  # Red
        'neutral': (0.5, 0.7, 0.9),  # Blue
    }
    
    # Thresholds for status coloring
    THRESHOLDS = {
        'hash_hit_good': 0.93,
        'hash_hit_warn': 0.85,
        'l3_miss_good': 0.08,
        'l3_miss_warn': 0.12,
        'recovery_good': 500,
        'recovery_warn': 1000,
    }
    
    def __init__(self, width: int = 400, height: int = 500) -> None:
        """Initialize telemetry panel.
        
        Args:
            width: Panel width in pixels
            height: Panel height in pixels
        """
        self.width = width
        self.height = height
        self.state = TelemetryState()
        self.state.session_start_time = time.time()
    
    def reset_session(self) -> None:
        """Reset session statistics."""
        self.state.session_start_time = time.time()
        self.state.total_nodes_searched = 0
        self.state.total_positions_evaluated = 0
        self.state.nps_history.clear()
        self.state.hash_hit_history.clear()
        self.state.memory_history.clear()
    
    def update_performance(
        self,
        nodes_searched: int,
        evaluations: int,
        elapsed_time: float,
    ) -> None:
        """Update performance metrics.
        
        Args:
            nodes_searched: Number of nodes searched
            evaluations: Number of evaluations performed
            elapsed_time: Time elapsed in seconds
        """
        if elapsed_time > 0:
            nps = nodes_searched / elapsed_time
            eps = evaluations / elapsed_time
            
            self.state.performance.nodes_per_second = nps
            self.state.performance.evaluations_per_second = eps
            
            # Update history
            self.state.nps_history.append(nps)
            if len(self.state.nps_history) > 1000:
                self.state.nps_history = self.state.nps_history[-1000:]
            
            # Update totals
            self.state.total_nodes_searched += nodes_searched
            self.state.total_positions_evaluated += evaluations
    
    def update_threads(
        self,
        active: int,
        max_threads: int,
        utilization: float,
    ) -> None:
        """Update thread metrics.
        
        Args:
            active: Number of active threads
            max_threads: Maximum available threads
            utilization: CPU utilization (0-1)
        """
        self.state.threads.active_threads = active
        self.state.threads.max_threads = max_threads
        self.state.threads.thread_utilization = utilization
        
        # Calculate load balance (simplified)
        if max_threads > 0:
            self.state.threads.load_balance_score = active / max_threads
    
    def update_cache(
        self,
        hash_hits: int,
        hash_probes: int,
        table_size_mb: int,
        entries_used: int,
        total_entries: int,
    ) -> None:
        """Update cache metrics.
        
        Args:
            hash_hits: Number of hash table hits
            hash_probes: Number of hash table probes
            table_size_mb: Hash table size in MB
            entries_used: Number of entries used
            total_entries: Total entries capacity
        """
        if hash_probes > 0:
            hit_rate = hash_hits / hash_probes
            self.state.cache.hash_hit_rate = hit_rate
            
            self.state.hash_hit_history.append(hit_rate)
            if len(self.state.hash_hit_history) > 1000:
                self.state.hash_hit_history = self.state.hash_hit_history[-1000:]
        
        self.state.cache.hash_table_size_mb = table_size_mb
        if total_entries > 0:
            self.state.cache.hash_table_fill = entries_used / total_entries
    
    def update_latency(
        self,
        nn_eval_ms: float,
        move_gen_us: float,
        search_overhead_ms: float,
    ) -> None:
        """Update latency metrics.
        
        Args:
            nn_eval_ms: Neural network evaluation latency in milliseconds
            move_gen_us: Move generation latency in microseconds
            search_overhead_ms: Search overhead in milliseconds
        """
        self.state.latency.nn_eval_latency_ms = nn_eval_ms
        self.state.latency.move_gen_latency_us = move_gen_us
        self.state.latency.search_overhead_ms = search_overhead_ms
        
        self.state.latency.latency_history.append(nn_eval_ms)
        if len(self.state.latency.latency_history) > 1000:
            self.state.latency.latency_history = self.state.latency.latency_history[-1000:]
    
    def update_memory(
        self,
        heap_used_mb: float,
        heap_total_mb: float,
        tree_mb: float = 0.0,
        nn_mb: float = 0.0,
    ) -> None:
        """Update memory metrics.
        
        Args:
            heap_used_mb: Heap memory used in MB
            heap_total_mb: Total heap memory in MB
            tree_mb: Search tree memory in MB
            nn_mb: Neural network weights memory in MB
        """
        self.state.memory.heap_used_mb = heap_used_mb
        self.state.memory.heap_total_mb = heap_total_mb
        self.state.memory.search_tree_mb = tree_mb
        self.state.memory.nn_weights_mb = nn_mb
        
        self.state.memory_history.append(heap_used_mb)
        if len(self.state.memory_history) > 1000:
            self.state.memory_history = self.state.memory_history[-1000:]
    
    def record_recovery(self, recovery_time_ms: float) -> None:
        """Record a fault recovery event.
        
        Args:
            recovery_time_ms: Recovery time in milliseconds
        """
        self.state.recovery.recovery_count += 1
        self.state.recovery.last_recovery_time_ms = recovery_time_ms
        self.state.recovery.max_recovery_time_ms = max(
            self.state.recovery.max_recovery_time_ms,
            recovery_time_ms
        )
        
        # Update mean
        n = self.state.recovery.recovery_count
        prev_mean = self.state.recovery.mean_recovery_time_ms
        self.state.recovery.mean_recovery_time_ms = prev_mean + (recovery_time_ms - prev_mean) / n
    
    def update_compliance(
        self,
        iso_compliant: bool,
        nist_compliant: bool,
        eu_ai_compliant: bool,
    ) -> None:
        """Update compliance status.
        
        Args:
            iso_compliant: ISO compliance status
            nist_compliant: NIST compliance status
            eu_ai_compliant: EU AI Act compliance status
        """
        self.state.compliance.iso_status = ComplianceStatus.COMPLIANT if iso_compliant else ComplianceStatus.NON_COMPLIANT
        self.state.compliance.nist_status = ComplianceStatus.COMPLIANT if nist_compliant else ComplianceStatus.NON_COMPLIANT
        self.state.compliance.eu_ai_status = ComplianceStatus.COMPLIANT if eu_ai_compliant else ComplianceStatus.NON_COMPLIANT
    
    def enable_audit_log(self, enabled: bool = True) -> None:
        """Enable or disable audit logging.
        
        Args:
            enabled: Whether audit logging is enabled
        """
        self.state.compliance.audit_log_enabled = enabled
        self.state.compliance.last_audit_timestamp = time.time()
    
    def _get_metric_color(self, metric: str, value: float) -> tuple[float, float, float]:
        """Get color for a metric based on value.
        
        Args:
            metric: Metric name
            value: Metric value
            
        Returns:
            RGB color tuple
        """
        if metric == 'hash_hit':
            if value >= self.THRESHOLDS['hash_hit_good']:
                return self.COLORS['good']
            elif value >= self.THRESHOLDS['hash_hit_warn']:
                return self.COLORS['warning']
            else:
                return self.COLORS['bad']
        elif metric == 'l3_miss':
            if value <= self.THRESHOLDS['l3_miss_good']:
                return self.COLORS['good']
            elif value <= self.THRESHOLDS['l3_miss_warn']:
                return self.COLORS['warning']
            else:
                return self.COLORS['bad']
        elif metric == 'recovery':
            if value <= self.THRESHOLDS['recovery_good']:
                return self.COLORS['good']
            elif value <= self.THRESHOLDS['recovery_warn']:
                return self.COLORS['warning']
            else:
                return self.COLORS['bad']
        
        return self.COLORS['neutral']
    
    def get_session_duration(self) -> float:
        """Get session duration in seconds."""
        return time.time() - self.state.session_start_time
    
    def get_render_data(self) -> dict[str, Any]:
        """Get all data needed for rendering.
        
        Returns:
            Dictionary with panel state for visualization
        """
        session_duration = self.get_session_duration()
        
        # Performance data
        perf = self.state.performance
        perf_data = {
            'nps': perf.nodes_per_second,
            'eps': perf.evaluations_per_second,
            'mps': perf.moves_per_second,
            'nps_formatted': self._format_large_number(perf.nodes_per_second),
            'target_nps_single': perf.target_nps_single,
            'target_nps_multi': perf.target_nps_multi,
            'nps_percent_single': min(100, perf.nodes_per_second / perf.target_nps_single * 100) if perf.target_nps_single else 0,
            'history': self.state.nps_history[-100:],
        }
        
        # Thread data
        threads = self.state.threads
        thread_data = {
            'active': threads.active_threads,
            'max': threads.max_threads,
            'utilization': threads.thread_utilization,
            'load_balance': threads.load_balance_score,
        }
        
        # Cache data
        cache = self.state.cache
        cache_data = {
            'hit_rate': cache.hash_hit_rate,
            'hit_rate_percent': cache.hash_hit_rate * 100,
            'target_hit_rate': 0.93,
            'table_size_mb': cache.hash_table_size_mb,
            'fill_percent': cache.hash_table_fill * 100,
            'l3_miss_rate': cache.l3_miss_rate,
            'branch_misprediction': cache.branch_misprediction,
            'history': self.state.hash_hit_history[-100:],
            'color': self._get_metric_color('hash_hit', cache.hash_hit_rate),
        }
        
        # Latency data
        lat = self.state.latency
        latency_data = {
            'nn_eval_ms': lat.nn_eval_latency_ms,
            'move_gen_us': lat.move_gen_latency_us,
            'search_overhead_ms': lat.search_overhead_ms,
            'target_nn_ms': 0.15,
            'nn_meets_target': lat.nn_eval_latency_ms <= 0.15,
            'history': lat.latency_history[-100:],
        }
        
        # Memory data
        mem = self.state.memory
        memory_data = {
            'heap_used_mb': mem.heap_used_mb,
            'heap_total_mb': mem.heap_total_mb,
            'heap_percent': (mem.heap_used_mb / mem.heap_total_mb * 100) if mem.heap_total_mb else 0,
            'tree_mb': mem.search_tree_mb,
            'nn_mb': mem.nn_weights_mb,
            'history': self.state.memory_history[-100:],
        }
        
        # Recovery data
        rec = self.state.recovery
        recovery_data = {
            'count': rec.recovery_count,
            'last_ms': rec.last_recovery_time_ms,
            'mean_ms': rec.mean_recovery_time_ms,
            'max_ms': rec.max_recovery_time_ms,
            'target_ms': 500,
            'meets_target': rec.max_recovery_time_ms <= 500,
            'color': self._get_metric_color('recovery', rec.max_recovery_time_ms),
        }
        
        # Compliance data
        comp = self.state.compliance
        compliance_data = {
            'iso': comp.iso_status.value,
            'nist': comp.nist_status.value,
            'eu_ai': comp.eu_ai_status.value,
            'audit_enabled': comp.audit_log_enabled,
            'audit_entries': comp.audit_entries_count,
        }
        
        return {
            'width': self.width,
            'height': self.height,
            'session': {
                'duration': session_duration,
                'duration_formatted': self._format_duration(session_duration),
                'total_nodes': self.state.total_nodes_searched,
                'total_evals': self.state.total_positions_evaluated,
            },
            'performance': perf_data,
            'threads': thread_data,
            'cache': cache_data,
            'latency': latency_data,
            'memory': memory_data,
            'recovery': recovery_data,
            'compliance': compliance_data,
            'display_options': {
                'show_performance': self.state.show_performance,
                'show_threads': self.state.show_threads,
                'show_cache': self.state.show_cache,
                'show_latency': self.state.show_latency,
                'show_memory': self.state.show_memory,
                'show_recovery': self.state.show_recovery,
                'show_compliance': self.state.show_compliance,
                'graph_time_window': self.state.graph_time_window,
            },
            'colors': self.COLORS,
            'thresholds': self.THRESHOLDS,
        }
    
    def _format_large_number(self, n: float) -> str:
        """Format large number with suffix."""
        if n >= 1_000_000_000:
            return f"{n / 1_000_000_000:.2f}B"
        elif n >= 1_000_000:
            return f"{n / 1_000_000:.2f}M"
        elif n >= 1_000:
            return f"{n / 1_000:.2f}K"
        else:
            return f"{n:.0f}"
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable form."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"
    
    def to_json(self) -> str:
        """Serialize render data to JSON."""
        import json
        return json.dumps(self.get_render_data())
