"""Tests for full genome sequencing pipeline.

Validates:
- Pipeline initialization
- Data generation
- Engine integration
- Output generation
- Audit logging
- Reproducibility
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

import numpy as np
import pytest

from xenon.bioinformatics.full_genome_sequencing import (
    FullGenomeSequencingPipeline,
    GenomeSequencingConfig,
)


class TestFullGenomeSequencingPipeline:
    """Test suite for full genome sequencing pipeline."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""

        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def pipeline_config(self, temp_output_dir):
        """Create test configuration."""

        return GenomeSequencingConfig(
            global_seed=42,
            max_threads=2,
            output_dir=temp_output_dir,
            prefer_gpu=False,
            prefer_qpu=False,
            audit_db_path=os.path.join(temp_output_dir, "test_audit.db"),
        )

    def test_pipeline_initialization(self, pipeline_config):
        """Test that pipeline initializes correctly."""

        pipeline = FullGenomeSequencingPipeline(config=pipeline_config)

        assert pipeline.config.global_seed == 42
        assert pipeline.config.max_threads == 2
        assert pipeline.alignment_engine is not None
        assert pipeline.fusion_engine is not None
        assert pipeline.te_engine is not None
        assert pipeline.inference_engine is not None
        assert pipeline.audit is not None

    def test_synthetic_sequence_generation(self, pipeline_config):
        """Test synthetic sequence generation."""

        pipeline = FullGenomeSequencingPipeline(config=pipeline_config)
        sequences = pipeline._generate_synthetic_sequences(n_sequences=5)

        assert len(sequences) == 5
        for seq_id, sequence in sequences:
            assert isinstance(seq_id, str)
            assert isinstance(sequence, str)
            assert len(sequence) >= 50
            assert len(sequence) <= 200

    def test_synthetic_omics_generation(self, pipeline_config):
        """Test synthetic omics data generation."""

        pipeline = FullGenomeSequencingPipeline(config=pipeline_config)
        omics_data = pipeline._generate_synthetic_omics_data()

        assert "genomics" in omics_data
        assert "transcriptomics" in omics_data
        assert "epigenomics" in omics_data

        for layer, data in omics_data.items():
            assert isinstance(data, np.ndarray)
            assert len(data) == 100

    def test_synthetic_timeseries_generation(self, pipeline_config):
        """Test synthetic time-series data generation."""

        pipeline = FullGenomeSequencingPipeline(config=pipeline_config)
        timeseries_data = pipeline._generate_synthetic_timeseries_data()

        assert len(timeseries_data) >= 3
        for var, data in timeseries_data.items():
            assert isinstance(data, np.ndarray)
            assert len(data) == 100

    def test_synthetic_variant_graph_generation(self, pipeline_config):
        """Test synthetic variant graph generation."""

        pipeline = FullGenomeSequencingPipeline(config=pipeline_config)
        variant_graph = pipeline._generate_synthetic_variant_graph()

        assert variant_graph["n_nodes"] == 20
        assert variant_graph["n_edges"] == 30
        assert variant_graph["node_features"].shape == (20, 10)
        assert variant_graph["edge_index"].shape == (2, 30)

    def test_alignment_execution(self, pipeline_config):
        """Test alignment phase execution."""

        pipeline = FullGenomeSequencingPipeline(config=pipeline_config)
        sequences = pipeline._generate_synthetic_sequences(n_sequences=3)

        results = pipeline.execute_alignment(sequences)

        assert "alignments" in results
        assert "summary" in results
        assert len(results["alignments"]) == 2  # n-1 pairs
        assert results["summary"]["total_alignments"] == 2

    def test_fusion_execution(self, pipeline_config):
        """Test fusion phase execution."""

        pipeline = FullGenomeSequencingPipeline(config=pipeline_config)
        omics_data = pipeline._generate_synthetic_omics_data()

        results = pipeline.execute_fusion(omics_data)

        assert "decompositions" in results
        assert "summary" in results
        assert len(results["decompositions"]) >= 1

    def test_transfer_entropy_execution(self, pipeline_config):
        """Test transfer entropy phase execution."""

        pipeline = FullGenomeSequencingPipeline(config=pipeline_config)
        timeseries_data = pipeline._generate_synthetic_timeseries_data()

        results = pipeline.execute_transfer_entropy(timeseries_data)

        assert "transfer_entropies" in results
        assert "summary" in results
        assert len(results["transfer_entropies"]) >= 1

    def test_inference_execution(self, pipeline_config):
        """Test inference phase execution."""

        pipeline = FullGenomeSequencingPipeline(config=pipeline_config)
        variant_graph = pipeline._generate_synthetic_variant_graph()

        results = pipeline.execute_inference(variant_graph)

        assert "predictions" in results
        assert "summary" in results
        assert len(results["predictions"]) == 20  # n_nodes

    def test_audit_summary_generation(self, pipeline_config):
        """Test audit summary generation."""

        pipeline = FullGenomeSequencingPipeline(config=pipeline_config)

        # Run a simple operation to generate some audit entries
        sequences = pipeline._generate_synthetic_sequences(n_sequences=2)
        pipeline.execute_alignment(sequences)

        summary = pipeline.generate_audit_summary()

        assert "total_entries" in summary
        assert "unresolved_critical" in summary
        assert "json_export" in summary
        assert os.path.exists(summary["json_export"])

    def test_reproducibility_report_generation(self, pipeline_config):
        """Test reproducibility report generation."""

        pipeline = FullGenomeSequencingPipeline(config=pipeline_config)

        report = pipeline.generate_reproducibility_report()

        assert report["seed_used"] == 42
        assert report["deterministic"] is True
        assert "engines" in report
        assert "validation" in report

    def test_full_pipeline_run(self, pipeline_config):
        """Test complete pipeline execution."""

        pipeline = FullGenomeSequencingPipeline(config=pipeline_config)

        deployment_report = pipeline.run()

        # Check deployment report structure
        assert deployment_report["status"] == "SUCCESS"
        assert "metrics" in deployment_report
        assert "audit_summary" in deployment_report
        assert "reproducibility" in deployment_report
        assert "outputs" in deployment_report

        # Check output files exist
        expected_files = [
            "alignment_result.json",
            "fusion_result.json",
            "transfer_entropy.json",
            "functional_predictions.json",
            "audit_summary.json",
            "deployment_report.json",
        ]

        for filename in expected_files:
            filepath = os.path.join(pipeline_config.output_dir, filename)
            assert os.path.exists(filepath), f"Missing output file: {filename}"

            # Verify JSON is valid
            with open(filepath) as f:
                data = json.load(f)
                assert isinstance(data, dict)

    def test_deterministic_execution(self, pipeline_config, temp_output_dir):
        """Test that pipeline produces deterministic results."""

        # Run 1
        pipeline1 = FullGenomeSequencingPipeline(config=pipeline_config)
        report1 = pipeline1.run()

        # Load alignment results from run 1
        alignment_path = os.path.join(temp_output_dir, "alignment_result.json")
        with open(alignment_path) as f:
            alignment_results1 = json.load(f)

        # Run 2 with same seed
        pipeline2 = FullGenomeSequencingPipeline(config=pipeline_config)
        report2 = pipeline2.run()

        # Load alignment results from run 2
        with open(alignment_path) as f:
            alignment_results2 = json.load(f)

        # Compare results - should be identical
        assert len(alignment_results1["alignments"]) == len(alignment_results2["alignments"])

        for i, (align1, align2) in enumerate(
            zip(alignment_results1["alignments"], alignment_results2["alignments"])
        ):
            assert align1["score"] == align2["score"], f"Alignment {i} score differs"
            assert (
                align1["circuit_depth"] == align2["circuit_depth"]
            ), f"Alignment {i} depth differs"

    def test_hardware_detection(self, pipeline_config):
        """Test hardware detection and backend selection."""

        pipeline = FullGenomeSequencingPipeline(config=pipeline_config)

        assert pipeline.hardware_detector is not None
        assert pipeline.available_hardware is not None
        assert pipeline.metrics.hardware_used in ["CPU", "GPU_NVIDIA", "GPU_AMD", "QPU"]
        assert pipeline.metrics.backend_used in [
            "classical",
            "gpu_accelerated",
            "quantum",
        ]

    def test_performance_instrumentation(self, pipeline_config):
        """Test that performance metrics are collected."""

        pipeline = FullGenomeSequencingPipeline(config=pipeline_config)

        deployment_report = pipeline.run()
        metrics = deployment_report["metrics"]

        assert metrics["total_duration_ms"] > 0
        assert metrics["alignment_duration_ms"] >= 0
        assert metrics["fusion_duration_ms"] >= 0
        assert metrics["transfer_entropy_duration_ms"] >= 0
        assert metrics["inference_duration_ms"] >= 0
        assert metrics["memory_peak_mb"] >= 0

    def test_thread_safety(self, pipeline_config):
        """Test thread-safe engine wrapper."""

        pipeline = FullGenomeSequencingPipeline(config=pipeline_config)

        assert pipeline.thread_safe_alignment is not None

        # Test that we can execute alignment through thread-safe wrapper
        sequences = pipeline._generate_synthetic_sequences(n_sequences=2)
        seq1 = sequences[0][1]
        seq2 = sequences[1][1]

        result = pipeline.thread_safe_alignment.execute("align", seq1[:50], seq2[:50])

        assert result is not None
        assert hasattr(result, "score")

    def test_output_structure(self, pipeline_config):
        """Test that output files have correct structure."""

        pipeline = FullGenomeSequencingPipeline(config=pipeline_config)
        pipeline.run()

        # Check alignment result structure
        alignment_path = os.path.join(pipeline_config.output_dir, "alignment_result.json")
        with open(alignment_path) as f:
            alignment_data = json.load(f)

        assert "alignments" in alignment_data
        assert "summary" in alignment_data
        for alignment in alignment_data["alignments"]:
            assert "seq1_id" in alignment
            assert "seq2_id" in alignment
            assert "score" in alignment
            assert "circuit_depth" in alignment

        # Check fusion result structure
        fusion_path = os.path.join(pipeline_config.output_dir, "fusion_result.json")
        with open(fusion_path) as f:
            fusion_data = json.load(f)

        assert "decompositions" in fusion_data
        assert "summary" in fusion_data
        for decomp in fusion_data["decompositions"]:
            assert "source1" in decomp
            assert "source2" in decomp
            assert "unique_s1" in decomp
            assert "redundant" in decomp

        # Check transfer entropy structure
        te_path = os.path.join(pipeline_config.output_dir, "transfer_entropy.json")
        with open(te_path) as f:
            te_data = json.load(f)

        assert "transfer_entropies" in te_data
        assert "summary" in te_data
        for te in te_data["transfer_entropies"]:
            assert "source" in te
            assert "target" in te
            assert "te_value" in te

        # Check inference result structure
        inference_path = os.path.join(pipeline_config.output_dir, "functional_predictions.json")
        with open(inference_path) as f:
            inference_data = json.load(f)

        assert "predictions" in inference_data
        assert "summary" in inference_data
