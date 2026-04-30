#!/usr/bin/env python3
"""
Production-Grade Workflow Checkpointing System

Provides robust checkpointing and restart capabilities for long-running
genomics pipelines. Enables fault tolerance and efficient resource utilization.

Features:
- Automatic checkpoint creation at pipeline stages
- State persistence to disk with compression
- Fast restart from last successful checkpoint
- Progress tracking and ETA estimation
- Integration with WGS pipeline

Author: QRATUM Team
License: See LICENSE file
"""

import gzip
import hashlib
import json
import logging
import pickle
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CheckpointStage(Enum):
    """Pipeline stages for checkpointing"""
    INITIALIZED = "initialized"
    ALIGNMENT_STARTED = "alignment_started"
    ALIGNMENT_COMPLETE = "alignment_complete"
    SORTING_COMPLETE = "sorting_complete"
    DEDUP_COMPLETE = "deduplication_complete"
    BQSR_COMPLETE = "base_quality_recalibration_complete"
    VARIANT_CALLING_STARTED = "variant_calling_started"
    VARIANT_CALLING_COMPLETE = "variant_calling_complete"
    ANNOTATION_COMPLETE = "annotation_complete"
    RARITY_ANALYSIS_COMPLETE = "rarity_analysis_complete"
    PIPELINE_COMPLETE = "pipeline_complete"


@dataclass
class Checkpoint:
    """Represents a pipeline checkpoint"""
    checkpoint_id: str
    stage: CheckpointStage
    timestamp: str
    pipeline_config: Dict[str, Any]
    stage_outputs: Dict[str, str]  # File paths
    stage_metrics: Dict[str, Any]
    previous_checkpoint_id: Optional[str] = None
    elapsed_time_seconds: float = 0.0
    estimated_remaining_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['stage'] = self.stage.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Checkpoint':
        """Create from dictionary"""
        data['stage'] = CheckpointStage(data['stage'])
        return cls(**data)


class CheckpointManager:
    """
    Manages pipeline checkpoints
    
    Provides checkpoint creation, loading, and management with
    automatic cleanup of old checkpoints.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints",
                 max_checkpoints: int = 10,
                 compress: bool = True):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.compress = compress

        # Database for checkpoint metadata
        self.db_path = self.checkpoint_dir / "checkpoints.db"
        self._initialize_database()

        logger.info(f"Checkpoint manager initialized: {self.checkpoint_dir}")

    def _initialize_database(self):
        """Initialize SQLite database for checkpoint metadata"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                stage TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                pipeline_config TEXT NOT NULL,
                stage_outputs TEXT NOT NULL,
                stage_metrics TEXT NOT NULL,
                previous_checkpoint_id TEXT,
                elapsed_time_seconds REAL,
                estimated_remaining_seconds REAL,
                file_path TEXT NOT NULL,
                file_size_bytes INTEGER,
                compressed BOOLEAN
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON checkpoints(timestamp DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_stage 
            ON checkpoints(stage)
        """)

        conn.commit()
        conn.close()

    def create_checkpoint(self, stage: CheckpointStage,
                          pipeline_config: Dict[str, Any],
                          stage_outputs: Dict[str, str],
                          stage_metrics: Dict[str, Any],
                          previous_checkpoint_id: Optional[str] = None,
                          elapsed_time: float = 0.0) -> Checkpoint:
        """
        Create and save a checkpoint
        
        Args:
            stage: Pipeline stage
            pipeline_config: Full pipeline configuration
            stage_outputs: File paths produced at this stage
            stage_metrics: Performance metrics for this stage
            previous_checkpoint_id: ID of previous checkpoint
            elapsed_time: Time elapsed since pipeline start
        
        Returns:
            Checkpoint object
        """
        # Generate checkpoint ID
        checkpoint_id = self._generate_checkpoint_id(stage, pipeline_config)

        # Estimate remaining time
        estimated_remaining = self._estimate_remaining_time(
            stage, elapsed_time, pipeline_config
        )

        # Create checkpoint object
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            stage=stage,
            timestamp=datetime.now().isoformat(),
            pipeline_config=pipeline_config,
            stage_outputs=stage_outputs,
            stage_metrics=stage_metrics,
            previous_checkpoint_id=previous_checkpoint_id,
            elapsed_time_seconds=elapsed_time,
            estimated_remaining_seconds=estimated_remaining
        )

        # Save checkpoint to disk
        file_path = self._save_checkpoint_to_disk(checkpoint)

        # Save metadata to database
        self._save_checkpoint_metadata(checkpoint, file_path)

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        logger.info(f"Checkpoint created: {checkpoint_id} at stage {stage.value}")
        logger.info(f"  Elapsed: {elapsed_time:.1f}s, Estimated remaining: {estimated_remaining:.1f}s" if estimated_remaining else "")

        return checkpoint

    def _generate_checkpoint_id(self, stage: CheckpointStage,
                                  config: Dict[str, Any]) -> str:
        """Generate unique checkpoint ID"""
        # Use hash of stage + timestamp + config subset
        timestamp = datetime.now().isoformat()
        config_subset = {
            'sample_id': config.get('sample_id', 'unknown'),
            'reference': config.get('reference', 'unknown')
        }

        content = f"{stage.value}:{timestamp}:{json.dumps(config_subset, sort_keys=True)}"
        hash_val = hashlib.sha256(content.encode()).hexdigest()[:16]

        return f"ckpt_{stage.value}_{hash_val}"

    def _save_checkpoint_to_disk(self, checkpoint: Checkpoint) -> Path:
        """Save checkpoint to disk with optional compression"""
        filename = f"{checkpoint.checkpoint_id}.pkl"
        if self.compress:
            filename += ".gz"

        file_path = self.checkpoint_dir / filename

        # Serialize checkpoint
        data = checkpoint.to_dict()

        if self.compress:
            with gzip.open(file_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        file_size = file_path.stat().st_size
        logger.debug(f"Checkpoint saved: {file_path} ({file_size} bytes)")

        return file_path

    def _save_checkpoint_metadata(self, checkpoint: Checkpoint, file_path: Path):
        """Save checkpoint metadata to database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO checkpoints
            (checkpoint_id, stage, timestamp, pipeline_config, stage_outputs,
             stage_metrics, previous_checkpoint_id, elapsed_time_seconds,
             estimated_remaining_seconds, file_path, file_size_bytes, compressed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            checkpoint.checkpoint_id,
            checkpoint.stage.value,
            checkpoint.timestamp,
            json.dumps(checkpoint.pipeline_config),
            json.dumps(checkpoint.stage_outputs),
            json.dumps(checkpoint.stage_metrics),
            checkpoint.previous_checkpoint_id,
            checkpoint.elapsed_time_seconds,
            checkpoint.estimated_remaining_seconds,
            str(file_path),
            file_path.stat().st_size,
            self.compress
        ))

        conn.commit()
        conn.close()

    def load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load checkpoint by ID"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT file_path, compressed FROM checkpoints
            WHERE checkpoint_id = ?
        """, (checkpoint_id,))

        row = cursor.fetchone()
        conn.close()

        if not row:
            logger.warning(f"Checkpoint not found: {checkpoint_id}")
            return None

        file_path, compressed = row

        # Load from disk
        try:
            if compressed:
                with gzip.open(file_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)

            checkpoint = Checkpoint.from_dict(data)
            logger.info(f"Checkpoint loaded: {checkpoint_id}")
            return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None

    def get_latest_checkpoint(self) -> Optional[Checkpoint]:
        """Get most recent checkpoint"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT checkpoint_id FROM checkpoints
            ORDER BY timestamp DESC
            LIMIT 1
        """)

        row = cursor.fetchone()
        conn.close()

        if row:
            return self.load_checkpoint(row[0])

        return None

    def get_checkpoint_by_stage(self, stage: CheckpointStage) -> Optional[Checkpoint]:
        """Get most recent checkpoint for a specific stage"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT checkpoint_id FROM checkpoints
            WHERE stage = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (stage.value,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return self.load_checkpoint(row[0])

        return None

    def list_checkpoints(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent checkpoints"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT checkpoint_id, stage, timestamp, elapsed_time_seconds,
                   estimated_remaining_seconds, file_size_bytes
            FROM checkpoints
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        checkpoints = []
        for row in cursor.fetchall():
            checkpoints.append({
                'checkpoint_id': row[0],
                'stage': row[1],
                'timestamp': row[2],
                'elapsed_time_seconds': row[3],
                'estimated_remaining_seconds': row[4],
                'file_size_bytes': row[5]
            })

        conn.close()
        return checkpoints

    def delete_checkpoint(self, checkpoint_id: str):
        """Delete a checkpoint"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Get file path
        cursor.execute("""
            SELECT file_path FROM checkpoints
            WHERE checkpoint_id = ?
        """, (checkpoint_id,))

        row = cursor.fetchone()
        if row:
            file_path = Path(row[0])
            if file_path.exists():
                file_path.unlink()

            # Delete from database
            cursor.execute("""
                DELETE FROM checkpoints
                WHERE checkpoint_id = ?
            """, (checkpoint_id,))

            conn.commit()
            logger.info(f"Checkpoint deleted: {checkpoint_id}")

        conn.close()

    def _cleanup_old_checkpoints(self):
        """Delete old checkpoints beyond max_checkpoints limit"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Get checkpoints beyond limit
        cursor.execute("""
            SELECT checkpoint_id FROM checkpoints
            ORDER BY timestamp DESC
            LIMIT -1 OFFSET ?
        """, (self.max_checkpoints,))

        old_checkpoints = [row[0] for row in cursor.fetchall()]
        conn.close()

        for checkpoint_id in old_checkpoints:
            self.delete_checkpoint(checkpoint_id)

        if old_checkpoints:
            logger.info(f"Cleaned up {len(old_checkpoints)} old checkpoints")

    def _estimate_remaining_time(self, stage: CheckpointStage,
                                   elapsed_time: float,
                                   config: Dict[str, Any]) -> Optional[float]:
        """Estimate remaining pipeline time"""
        # Define typical stage durations (as percentage of total)
        stage_percentages = {
            CheckpointStage.INITIALIZED: 0.0,
            CheckpointStage.ALIGNMENT_STARTED: 0.05,
            CheckpointStage.ALIGNMENT_COMPLETE: 0.40,
            CheckpointStage.SORTING_COMPLETE: 0.50,
            CheckpointStage.DEDUP_COMPLETE: 0.55,
            CheckpointStage.BQSR_COMPLETE: 0.60,
            CheckpointStage.VARIANT_CALLING_STARTED: 0.65,
            CheckpointStage.VARIANT_CALLING_COMPLETE: 0.85,
            CheckpointStage.ANNOTATION_COMPLETE: 0.90,
            CheckpointStage.RARITY_ANALYSIS_COMPLETE: 0.95,
            CheckpointStage.PIPELINE_COMPLETE: 1.0
        }

        current_progress = stage_percentages.get(stage, 0.5)

        if current_progress > 0 and current_progress < 1.0:
            total_estimated = elapsed_time / current_progress
            remaining = total_estimated - elapsed_time
            return max(0, remaining)

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get checkpoint statistics"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Total checkpoints
        cursor.execute("SELECT COUNT(*) FROM checkpoints")
        total = cursor.fetchone()[0]

        # Total disk usage
        cursor.execute("SELECT SUM(file_size_bytes) FROM checkpoints")
        total_size = cursor.fetchone()[0] or 0

        # By stage
        cursor.execute("""
            SELECT stage, COUNT(*) FROM checkpoints
            GROUP BY stage
        """)
        by_stage = {row[0]: row[1] for row in cursor.fetchall()}

        conn.close()

        return {
            "total_checkpoints": total,
            "total_size_mb": total_size / (1024 * 1024),
            "checkpoints_by_stage": by_stage,
            "checkpoint_dir": str(self.checkpoint_dir),
            "max_checkpoints": self.max_checkpoints,
            "compression_enabled": self.compress
        }


class CheckpointedPipeline:
    """
    Base class for checkpointed pipelines
    
    Provides checkpoint/restart functionality for long-running pipelines.
    """

    def __init__(self, pipeline_config: Dict[str, Any],
                 checkpoint_manager: Optional[CheckpointManager] = None):
        self.config = pipeline_config
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.start_time = time.time()
        self.current_checkpoint_id: Optional[str] = None

    def execute_with_checkpoints(self, resume_from: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute pipeline with automatic checkpointing
        
        Args:
            resume_from: Checkpoint ID to resume from (None for fresh start)
        
        Returns:
            Pipeline results
        """
        # Load checkpoint if resuming
        resume_checkpoint = None
        if resume_from:
            resume_checkpoint = self.checkpoint_manager.load_checkpoint(resume_from)
            if not resume_checkpoint:
                logger.warning(f"Could not load checkpoint {resume_from}, starting fresh")
        elif resume_from is None:
            # Check for latest checkpoint automatically
            resume_checkpoint = self.checkpoint_manager.get_latest_checkpoint()
            if resume_checkpoint:
                logger.info(f"Found existing checkpoint, resuming from {resume_checkpoint.stage.value}")

        # Determine starting stage
        if resume_checkpoint:
            start_stage = resume_checkpoint.stage
            self.start_time = time.time() - resume_checkpoint.elapsed_time_seconds
            self.current_checkpoint_id = resume_checkpoint.checkpoint_id
        else:
            start_stage = CheckpointStage.INITIALIZED
            self.current_checkpoint_id = None

        # Execute pipeline stages
        return self._execute_stages(start_stage, resume_checkpoint)

    def _execute_stages(self, start_stage: CheckpointStage,
                        resume_checkpoint: Optional[Checkpoint]) -> Dict[str, Any]:
        """Execute pipeline stages with checkpointing"""
        # This is a template - override in subclass
        raise NotImplementedError("Subclass must implement _execute_stages")

    def _create_checkpoint(self, stage: CheckpointStage,
                           outputs: Dict[str, str],
                           metrics: Dict[str, Any]) -> Checkpoint:
        """Create checkpoint at current stage"""
        elapsed = time.time() - self.start_time

        checkpoint = self.checkpoint_manager.create_checkpoint(
            stage=stage,
            pipeline_config=self.config,
            stage_outputs=outputs,
            stage_metrics=metrics,
            previous_checkpoint_id=self.current_checkpoint_id,
            elapsed_time=elapsed
        )

        self.current_checkpoint_id = checkpoint.checkpoint_id
        return checkpoint


def main():
    """Demo/test checkpointing system"""
    print("\n" + "="*80)
    print("PRODUCTION WORKFLOW CHECKPOINTING SYSTEM")
    print("="*80 + "\n")

    # Initialize checkpoint manager
    manager = CheckpointManager(checkpoint_dir="test_checkpoints")

    # Simulate pipeline stages
    pipeline_config = {
        "sample_id": "SAMPLE001",
        "reference": "hg38",
        "input_fastq": "reads.fq.gz"
    }

    print("Creating test checkpoints...\n")

    # Stage 1: Alignment complete
    checkpoint1 = manager.create_checkpoint(
        stage=CheckpointStage.ALIGNMENT_COMPLETE,
        pipeline_config=pipeline_config,
        stage_outputs={"bam": "aligned.bam"},
        stage_metrics={"reads_aligned": 100000000, "mapping_rate": 0.95},
        elapsed_time=3600.0  # 1 hour
    )

    print(f"✓ Created checkpoint: {checkpoint1.checkpoint_id}")
    print(f"  Stage: {checkpoint1.stage.value}")
    print(f"  Elapsed: {checkpoint1.elapsed_time_seconds}s")
    print(f"  Estimated remaining: {checkpoint1.estimated_remaining_seconds}s\n")

    # Stage 2: Variant calling complete
    checkpoint2 = manager.create_checkpoint(
        stage=CheckpointStage.VARIANT_CALLING_COMPLETE,
        pipeline_config=pipeline_config,
        stage_outputs={
            "bam": "aligned.bam",
            "vcf": "variants.vcf.gz"
        },
        stage_metrics={
            "total_variants": 5000000,
            "snps": 4500000,
            "indels": 500000
        },
        previous_checkpoint_id=checkpoint1.checkpoint_id,
        elapsed_time=7200.0  # 2 hours
    )

    print(f"✓ Created checkpoint: {checkpoint2.checkpoint_id}")
    print(f"  Stage: {checkpoint2.stage.value}")
    print(f"  Elapsed: {checkpoint2.elapsed_time_seconds}s")
    print(f"  Estimated remaining: {checkpoint2.estimated_remaining_seconds}s\n")

    # List checkpoints
    print("Recent checkpoints:")
    checkpoints = manager.list_checkpoints()
    for ckpt in checkpoints:
        print(f"  - {ckpt['checkpoint_id']}: {ckpt['stage']} ({ckpt['elapsed_time_seconds']:.0f}s)")

    print()

    # Statistics
    stats = manager.get_statistics()
    print("Checkpoint Statistics:")
    print(json.dumps(stats, indent=2))

    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
