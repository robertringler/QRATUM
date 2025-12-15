"""Persistent audit registry for constraint violations and execution traces.

Provides SQLite-backed storage for:
- Constraint violations
- Reasoning traces
- CI/CD test results
- Deterministic reproduction data

Mathematical Foundation:
    Each audit entry contains:
    - Timestamp (UTC)
    - Violation type and severity
    - Context (module, function, parameters)
    - Reproducibility data (seed, configuration hash)
    - Resolution status
"""

from __future__ import annotations

import json
import sqlite3
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class ViolationType(Enum):
    """Types of audit violations."""
    
    CONSERVATION_VIOLATION = "conservation_violation"
    EQUIVALENCE_VIOLATION = "equivalence_violation"
    NUMERICAL_INSTABILITY = "numerical_instability"
    CONSTRAINT_VIOLATION = "constraint_violation"
    DETERMINISM_VIOLATION = "determinism_violation"


@dataclass
class AuditEntry:
    """Single audit entry.
    
    Attributes:
        id: Unique entry ID (auto-generated)
        timestamp: Entry timestamp
        violation_type: Type of violation
        severity: Severity level (0-10)
        module: Module where violation occurred
        function: Function where violation occurred
        message: Human-readable description
        context: Additional context data
        seed: Seed used for reproducibility
        config_hash: Configuration hash
        resolved: Whether violation has been resolved
    """
    
    id: Optional[int] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    violation_type: ViolationType = ViolationType.CONSTRAINT_VIOLATION
    severity: int = 5
    module: str = ""
    function: str = ""
    message: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    config_hash: str = ""
    resolved: bool = False


class AuditRegistry:
    """Persistent audit registry using SQLite.
    
    Provides queryable storage for violations and execution traces.
    
    Attributes:
        db_path: Path to SQLite database
        conn: Database connection
    """
    
    def __init__(self, db_path: str = "xenon_audit.db"):
        """Initialize audit registry.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.conn = None
        self._initialize_db()
    
    def _initialize_db(self) -> None:
        """Initialize database schema."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                violation_type TEXT NOT NULL,
                severity INTEGER NOT NULL,
                module TEXT NOT NULL,
                function TEXT NOT NULL,
                message TEXT NOT NULL,
                context TEXT,
                seed INTEGER NOT NULL,
                config_hash TEXT,
                resolved INTEGER NOT NULL DEFAULT 0
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_log(timestamp)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_violation_type ON audit_log(violation_type)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_severity ON audit_log(severity)
        """)
        self.conn.commit()
    
    def log(self, entry: AuditEntry) -> int:
        """Log an audit entry.
        
        Args:
            entry: Audit entry to log
            
        Returns:
            Entry ID
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO audit_log (
                timestamp, violation_type, severity, module, function,
                message, context, seed, config_hash, resolved
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.timestamp,
            entry.violation_type.value,
            entry.severity,
            entry.module,
            entry.function,
            entry.message,
            json.dumps(entry.context),
            entry.seed,
            entry.config_hash,
            1 if entry.resolved else 0,
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def query(
        self,
        violation_type: Optional[ViolationType] = None,
        min_severity: int = 0,
        max_severity: int = 10,
        resolved: Optional[bool] = None,
        limit: int = 100,
    ) -> List[AuditEntry]:
        """Query audit entries.
        
        Args:
            violation_type: Filter by violation type
            min_severity: Minimum severity
            max_severity: Maximum severity
            resolved: Filter by resolution status
            limit: Maximum number of results
            
        Returns:
            List of audit entries
        """
        query = "SELECT * FROM audit_log WHERE severity >= ? AND severity <= ?"
        params = [min_severity, max_severity]
        
        if violation_type is not None:
            query += " AND violation_type = ?"
            params.append(violation_type.value)
        
        if resolved is not None:
            query += " AND resolved = ?"
            params.append(1 if resolved else 0)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor = self.conn.execute(query, params)
        
        entries = []
        for row in cursor.fetchall():
            entries.append(AuditEntry(
                id=row[0],
                timestamp=row[1],
                violation_type=ViolationType(row[2]),
                severity=row[3],
                module=row[4],
                function=row[5],
                message=row[6],
                context=json.loads(row[7]) if row[7] else {},
                seed=row[8],
                config_hash=row[9] or "",
                resolved=bool(row[10]),
            ))
        
        return entries
    
    def mark_resolved(self, entry_id: int) -> None:
        """Mark an entry as resolved.
        
        Args:
            entry_id: Entry ID to mark resolved
        """
        self.conn.execute(
            "UPDATE audit_log SET resolved = 1 WHERE id = ?",
            (entry_id,)
        )
        self.conn.commit()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit statistics.
        
        Returns:
            Dictionary of statistics
        """
        cursor = self.conn.execute("""
            SELECT 
                violation_type,
                COUNT(*) as count,
                AVG(severity) as avg_severity,
                SUM(CASE WHEN resolved = 1 THEN 1 ELSE 0 END) as resolved_count
            FROM audit_log
            GROUP BY violation_type
        """)
        
        stats = {}
        for row in cursor.fetchall():
            stats[row[0]] = {
                "count": row[1],
                "avg_severity": row[2],
                "resolved_count": row[3],
            }
        
        return stats
    
    def export_to_json(self, output_path: str) -> None:
        """Export audit log to JSON.
        
        Args:
            output_path: Output JSON file path
        """
        entries = self.query(limit=1000000)  # Export all
        
        data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "total_entries": len(entries),
            "entries": [asdict(entry) for entry in entries],
            "statistics": self.get_statistics(),
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
