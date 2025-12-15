"""Tests for audit registry."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from xenon.bioinformatics.audit import (
    AuditRegistry,
    AuditEntry,
    ViolationType,
)


class TestAuditRegistry:
    """Test suite for audit registry."""
    
    def test_initialization(self):
        """Test registry initialization."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        registry = AuditRegistry(db_path=db_path)
        assert registry.db_path == Path(db_path)
        assert registry.conn is not None
        registry.close()
        
        # Clean up
        Path(db_path).unlink()
    
    def test_log_entry(self):
        """Test logging an audit entry."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        registry = AuditRegistry(db_path=db_path)
        
        entry = AuditEntry(
            violation_type=ViolationType.CONSERVATION_VIOLATION,
            severity=5,
            module="test_module",
            function="test_function",
            message="Test violation",
            seed=42,
        )
        
        entry_id = registry.log(entry)
        assert entry_id > 0
        
        registry.close()
        Path(db_path).unlink()
    
    def test_query_entries(self):
        """Test querying audit entries."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        registry = AuditRegistry(db_path=db_path)
        
        # Log multiple entries
        for severity in [3, 5, 7, 9]:
            entry = AuditEntry(
                violation_type=ViolationType.CONSTRAINT_VIOLATION,
                severity=severity,
                module="test",
                function="test",
                message=f"Severity {severity}",
                seed=42,
            )
            registry.log(entry)
        
        # Query high severity
        high_severity = registry.query(min_severity=7)
        assert len(high_severity) == 2
        
        # Query all
        all_entries = registry.query(limit=100)
        assert len(all_entries) == 4
        
        registry.close()
        Path(db_path).unlink()
    
    def test_mark_resolved(self):
        """Test marking entry as resolved."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        registry = AuditRegistry(db_path=db_path)
        
        entry = AuditEntry(
            violation_type=ViolationType.DETERMINISM_VIOLATION,
            severity=5,
            module="test",
            function="test",
            message="Test",
            seed=42,
        )
        
        entry_id = registry.log(entry)
        
        # Mark resolved
        registry.mark_resolved(entry_id)
        
        # Query resolved
        resolved = registry.query(resolved=True)
        assert len(resolved) == 1
        
        unresolved = registry.query(resolved=False)
        assert len(unresolved) == 0
        
        registry.close()
        Path(db_path).unlink()
    
    def test_get_statistics(self):
        """Test getting statistics."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        registry = AuditRegistry(db_path=db_path)
        
        # Log entries with different types
        for vtype in [ViolationType.CONSERVATION_VIOLATION, ViolationType.EQUIVALENCE_VIOLATION]:
            for i in range(3):
                entry = AuditEntry(
                    violation_type=vtype,
                    severity=5,
                    module="test",
                    function="test",
                    message="Test",
                    seed=42,
                )
                entry_id = registry.log(entry)
                if i == 0:
                    registry.mark_resolved(entry_id)
        
        stats = registry.get_statistics()
        
        assert len(stats) == 2
        assert stats[ViolationType.CONSERVATION_VIOLATION.value]["count"] == 3
        assert stats[ViolationType.CONSERVATION_VIOLATION.value]["resolved_count"] == 1
        
        registry.close()
        Path(db_path).unlink()
