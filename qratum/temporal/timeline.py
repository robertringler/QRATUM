"""
Timeline Management for QRATUM Temporal Computing

Manages the multiverse of computed timelines, including creation, branching,
merging, and navigation across the possibility space.
"""

from typing import List, Optional, Dict, Tuple, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from .state import TemporalState, StateChain, TemporalCoordinate
from .verification import TemporalVerifier


@dataclass
class TimelineBranch:
    """
    Represents a divergence point between timelines.
    
    Attributes:
        branch_point_depth: Depth at which branch occurred
        parent_timeline_id: ID of parent timeline
        child_timeline_ids: IDs of child timelines
        branch_reason: Why the branch was created
        branch_timestamp: When the branch was created
        metadata: Additional branch information
    """
    branch_point_depth: int
    parent_timeline_id: str
    child_timeline_ids: List[str]
    branch_reason: str
    branch_timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return (f"Branch at depth {self.branch_point_depth}: "
                f"{self.parent_timeline_id} -> {len(self.child_timeline_ids)} children")


@dataclass
class Timeline:
    """
    Single timeline with full state history and branch points.
    
    A timeline represents a single path through the possibility space,
    with a complete history of states and records of where it branched
    from other timelines.
    
    Attributes:
        timeline_id: Unique identifier
        chain: State chain for this timeline
        parent_branch: Branch point if this split from another timeline
        child_branches: Branch points where this timeline split
        creation_time: When this timeline was created
        is_active: Whether this timeline is still being computed
        metadata: Additional timeline information
    """
    timeline_id: str
    chain: StateChain
    parent_branch: Optional[TimelineBranch] = None
    child_branches: List[TimelineBranch] = field(default_factory=list)
    creation_time: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_state(self, state: TemporalState) -> None:
        """Add a state to this timeline"""
        if state.coordinate.timeline_id != self.timeline_id:
            raise ValueError(f"State timeline {state.coordinate.timeline_id} "
                           f"doesn't match timeline {self.timeline_id}")
        self.chain.add_state(state)
    
    def get_state_at_time(self, computational_t: float) -> Optional[TemporalState]:
        """Get state at or before given computational time"""
        return self.chain.get_state_at_time(computational_t)
    
    def get_state_at_depth(self, depth: int) -> Optional[TemporalState]:
        """Get state at given depth"""
        return self.chain.get_state_at_depth(depth)
    
    def get_latest_state(self) -> Optional[TemporalState]:
        """Get the most recent state"""
        if len(self.chain.states) > 0:
            return self.chain.states[-1]
        return None
    
    def length(self) -> int:
        """Get number of states in timeline"""
        return len(self.chain.states)
    
    def branch(
        self,
        branch_point_depth: int,
        child_timeline_ids: List[str],
        reason: str = "manual_branch",
    ) -> TimelineBranch:
        """
        Record a branch from this timeline.
        
        Args:
            branch_point_depth: Depth where branch occurs
            child_timeline_ids: IDs of child timelines
            reason: Reason for branching
            
        Returns:
            TimelineBranch object
        """
        branch = TimelineBranch(
            branch_point_depth=branch_point_depth,
            parent_timeline_id=self.timeline_id,
            child_timeline_ids=child_timeline_ids,
            branch_reason=reason,
        )
        self.child_branches.append(branch)
        return branch
    
    def get_ancestry(self) -> List[str]:
        """
        Get list of ancestor timeline IDs.
        
        Returns:
            List of timeline IDs from root to this timeline
        """
        ancestry = [self.timeline_id]
        if self.parent_branch:
            # Would need reference to parent timeline to recurse
            ancestry.insert(0, self.parent_branch.parent_timeline_id)
        return ancestry


class TimelineManager:
    """
    Multiverse manager for creating, branching, merging, and navigating timelines.
    
    The TimelineManager maintains the complete graph of all timelines and provides
    operations for exploring the possibility space.
    
    Example:
        >>> manager = TimelineManager()
        >>> timeline = manager.create_timeline("main")
        >>> branches = manager.branch_timeline(
        ...     parent_id="main",
        ...     branch_names=["option_a", "option_b"],
        ...     branch_point_depth=10,
        ... )
    """
    
    def __init__(self, enable_verification: bool = True):
        """
        Initialize timeline manager.
        
        Args:
            enable_verification: Enable Merkle verification of timelines
        """
        self.timelines: Dict[str, Timeline] = {}
        self.verifier = TemporalVerifier() if enable_verification else None
        self.creation_count = 0
        self.branch_count = 0
    
    def create_timeline(
        self,
        timeline_id: Optional[str] = None,
        parent_branch: Optional[TimelineBranch] = None,
    ) -> Timeline:
        """
        Create a new timeline.
        
        Args:
            timeline_id: Optional ID (auto-generated if None)
            parent_branch: Optional parent branch
            
        Returns:
            New Timeline object
        """
        if timeline_id is None:
            timeline_id = f"timeline_{uuid.uuid4().hex[:8]}"
        
        if timeline_id in self.timelines:
            raise ValueError(f"Timeline {timeline_id} already exists")
        
        timeline = Timeline(
            timeline_id=timeline_id,
            chain=StateChain(timeline_id=timeline_id),
            parent_branch=parent_branch,
        )
        
        self.timelines[timeline_id] = timeline
        self.creation_count += 1
        
        return timeline
    
    def get_timeline(self, timeline_id: str) -> Optional[Timeline]:
        """Get timeline by ID"""
        return self.timelines.get(timeline_id)
    
    def delete_timeline(self, timeline_id: str) -> bool:
        """
        Delete a timeline.
        
        Args:
            timeline_id: Timeline to delete
            
        Returns:
            True if deleted, False if not found
        """
        if timeline_id in self.timelines:
            del self.timelines[timeline_id]
            return True
        return False
    
    def branch_timeline(
        self,
        parent_id: str,
        branch_names: List[str],
        branch_point_depth: int,
        reason: str = "manual_branch",
    ) -> Dict[str, Timeline]:
        """
        Branch a timeline into multiple child timelines.
        
        Args:
            parent_id: Parent timeline ID
            branch_names: Names for child branches
            branch_point_depth: Depth to branch at
            reason: Reason for branching
            
        Returns:
            Dictionary mapping branch names to Timeline objects
        """
        parent = self.timelines.get(parent_id)
        if not parent:
            raise ValueError(f"Parent timeline {parent_id} not found")
        
        if branch_point_depth >= parent.length():
            raise ValueError(f"Branch point {branch_point_depth} beyond timeline length")
        
        # Create child timelines
        child_timelines: Dict[str, Timeline] = {}
        child_ids: List[str] = []
        
        for branch_name in branch_names:
            child_id = f"{parent_id}_branch_{branch_name}"
            
            # Create branch record
            branch = TimelineBranch(
                branch_point_depth=branch_point_depth,
                parent_timeline_id=parent_id,
                child_timeline_ids=[child_id],
                branch_reason=reason,
            )
            
            # Create child timeline
            child = self.create_timeline(
                timeline_id=child_id,
                parent_branch=branch,
            )
            
            # Copy states up to branch point
            for i in range(branch_point_depth + 1):
                parent_state = parent.chain.states[i]
                
                # Create new state with updated timeline ID
                new_coord = TemporalCoordinate(
                    state_hash=parent_state.coordinate.state_hash,
                    timeline_id=child_id,
                    computational_t=parent_state.coordinate.computational_t,
                    physical_t=parent_state.coordinate.physical_t,
                    depth=parent_state.coordinate.depth,
                )
                
                new_state = TemporalState(
                    coordinate=new_coord,
                    data=parent_state.data,
                    parent_hash=parent_state.parent_hash,
                    metadata=parent_state.metadata.copy(),
                )
                
                child.add_state(new_state)
            
            child_timelines[branch_name] = child
            child_ids.append(child_id)
        
        # Record branch in parent
        parent.branch(
            branch_point_depth=branch_point_depth,
            child_timeline_ids=child_ids,
            reason=reason,
        )
        
        self.branch_count += 1
        
        return child_timelines
    
    def merge_timelines(
        self,
        timeline_ids: List[str],
        merge_fn: Callable[[List[TemporalState]], Any],
        merged_timeline_id: Optional[str] = None,
    ) -> Timeline:
        """
        Merge multiple timelines into one.
        
        Args:
            timeline_ids: IDs of timelines to merge
            merge_fn: Function to merge final states
            merged_timeline_id: Optional ID for merged timeline
            
        Returns:
            Merged Timeline object
        """
        if len(timeline_ids) < 2:
            raise ValueError("Need at least 2 timelines to merge")
        
        # Get latest states from each timeline
        latest_states = []
        for tid in timeline_ids:
            timeline = self.timelines.get(tid)
            if not timeline:
                raise ValueError(f"Timeline {tid} not found")
            
            latest = timeline.get_latest_state()
            if latest:
                latest_states.append(latest)
        
        # Create merged timeline
        if merged_timeline_id is None:
            merged_timeline_id = f"merged_{uuid.uuid4().hex[:8]}"
        
        merged_timeline = self.create_timeline(merged_timeline_id)
        
        # Apply merge function
        merged_data = merge_fn(latest_states)
        
        # Create merged state
        coord = TemporalCoordinate(
            state_hash="",
            timeline_id=merged_timeline_id,
            computational_t=max(s.coordinate.computational_t for s in latest_states),
            physical_t=max(s.coordinate.physical_t for s in latest_states),
            depth=0,
        )
        
        merged_state = TemporalState(
            coordinate=coord,
            data=merged_data,
            parent_hash=None,
            metadata={
                'merged_from': timeline_ids,
                'merge_count': len(timeline_ids),
            }
        )
        merged_state.coordinate.state_hash = merged_state.compute_hash()
        
        merged_timeline.add_state(merged_state)
        
        return merged_timeline
    
    def find_optimal_timeline(
        self,
        objective_fn: Callable[[Timeline], float],
        timeline_ids: Optional[List[str]] = None,
    ) -> Tuple[Optional[Timeline], float]:
        """
        Find optimal timeline according to objective function.
        
        Args:
            objective_fn: Function that scores timelines (higher is better)
            timeline_ids: Optional list of timeline IDs to search (searches all if None)
            
        Returns:
            Tuple of (best_timeline, best_score)
        """
        if timeline_ids is None:
            timeline_ids = list(self.timelines.keys())
        
        best_timeline = None
        best_score = float('-inf')
        
        for tid in timeline_ids:
            timeline = self.timelines.get(tid)
            if timeline and timeline.is_active:
                score = objective_fn(timeline)
                if score > best_score:
                    best_score = score
                    best_timeline = timeline
        
        return best_timeline, best_score
    
    def get_timeline_graph(self) -> Dict[str, List[str]]:
        """
        Get graph of timeline relationships.
        
        Returns:
            Dictionary mapping timeline IDs to their child timeline IDs
        """
        graph: Dict[str, List[str]] = {}
        
        for tid, timeline in self.timelines.items():
            children = []
            for branch in timeline.child_branches:
                children.extend(branch.child_timeline_ids)
            graph[tid] = children
        
        return graph
    
    def get_all_leaves(self) -> List[Timeline]:
        """
        Get all leaf timelines (timelines with no children).
        
        Returns:
            List of leaf Timeline objects
        """
        graph = self.get_timeline_graph()
        all_children = set()
        for children in graph.values():
            all_children.update(children)
        
        leaves = []
        for tid, timeline in self.timelines.items():
            if tid not in all_children or len(timeline.child_branches) == 0:
                leaves.append(timeline)
        
        return leaves
    
    def verify_timeline(self, timeline_id: str) -> bool:
        """
        Verify integrity of a timeline.
        
        Args:
            timeline_id: Timeline to verify
            
        Returns:
            True if timeline is valid
        """
        timeline = self.timelines.get(timeline_id)
        if not timeline or not self.verifier:
            return False
        
        return self.verifier.verify_state_chain(timeline.chain)
    
    def verify_all_timelines(self) -> Dict[str, bool]:
        """
        Verify all timelines.
        
        Returns:
            Dictionary mapping timeline IDs to verification results
        """
        if not self.verifier:
            return {}
        
        return self.verifier.batch_verify([t.chain for t in self.timelines.values()])
    
    def prune_timelines(
        self,
        keep_fn: Callable[[Timeline], bool],
    ) -> int:
        """
        Prune timelines based on a predicate.
        
        Args:
            keep_fn: Function that returns True for timelines to keep
            
        Returns:
            Number of timelines pruned
        """
        to_delete = []
        for tid, timeline in self.timelines.items():
            if not keep_fn(timeline):
                to_delete.append(tid)
        
        for tid in to_delete:
            del self.timelines[tid]
        
        return len(to_delete)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get timeline manager statistics"""
        total_states = sum(t.length() for t in self.timelines.values())
        active_timelines = sum(1 for t in self.timelines.values() if t.is_active)
        
        return {
            'total_timelines': len(self.timelines),
            'active_timelines': active_timelines,
            'total_states': total_states,
            'total_created': self.creation_count,
            'total_branches': self.branch_count,
            'leaf_timelines': len(self.get_all_leaves()),
        }
