"""Main automation script for QRATUM-Chess benchmarking.

Orchestrates the complete Stage IV benchmark suite:
- Environment verification
- Benchmark execution
- Stage III certification
- Motif extraction
- Report generation

Usage:
    from qratum_chess.benchmarks.auto_benchmark import AutoBenchmark

    auto = AutoBenchmark()
    results = auto.run_full_suite()
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from qratum_chess.benchmarks.motif_extractor import MotifExtractor, MotifType
from qratum_chess.benchmarks.runner import BenchmarkConfig, BenchmarkRunner, BenchmarkSummary
from qratum_chess.search.aas import AsymmetricAdaptiveSearch

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class AutoBenchmarkConfig:
    """Configuration for automated benchmarking.

    Attributes:
        quick_mode: Run with reduced iterations for faster execution.
        certify: Run Stage III certification verification.
        extract_motifs: Enable motif extraction from telemetry.
        output_dir: Base output directory for results.
        gpu_enabled: Force GPU acceleration (if available).
        cpu_only: Disable GPU and run on CPU only.
        checkpoint_enabled: Enable checkpoint/resume capability.
        benchmark_config: Underlying benchmark configuration.
    """

    quick_mode: bool = False
    certify: bool = False
    extract_motifs: bool = True
    output_dir: str = "benchmarks/auto_run"
    gpu_enabled: bool | None = None
    cpu_only: bool = False
    checkpoint_enabled: bool = True
    benchmark_config: BenchmarkConfig = field(default_factory=BenchmarkConfig)


@dataclass
class EnvironmentInfo:
    """System environment information."""

    python_version: str
    platform: str
    cpu_count: int
    gpu_available: bool
    gpu_name: str
    dependencies_ok: bool
    missing_dependencies: list[str] = field(default_factory=list)


class AutoBenchmark:
    """Automated benchmarking orchestrator for QRATUM-Chess.

    Handles the complete benchmarking pipeline:
    1. Environment verification
    2. Engine initialization
    3. Benchmark execution
    4. Motif extraction
    5. Report generation
    """

    def __init__(self, config: AutoBenchmarkConfig | None = None):
        """Initialize automated benchmark system.

        Args:
            config: Automation configuration.
        """
        self.config = config or AutoBenchmarkConfig()
        self.env_info: EnvironmentInfo | None = None
        self.runner: BenchmarkRunner | None = None
        self.motif_extractor: MotifExtractor | None = None
        self.checkpoint_path: Path | None = None

        # Apply quick mode settings
        if self.config.quick_mode:
            self.config.benchmark_config.torture_depth = 8
            self.config.benchmark_config.resilience_iterations = 3

    def verify_environment(self) -> EnvironmentInfo:
        """Verify system environment and dependencies.

        Returns:
            Environment information.
        """
        logger.info("Verifying environment...")

        # Python version
        python_version = (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )
        logger.info(f"Python version: {python_version}")

        # Platform info
        platform_info = f"{platform.system()} {platform.release()}"
        logger.info(f"Platform: {platform_info}")

        # CPU count
        cpu_count = os.cpu_count() or 1
        logger.info(f"CPU cores: {cpu_count}")

        # GPU detection
        gpu_available = False
        gpu_name = "None"

        if not self.config.cpu_only:
            try:
                import torch

                if torch.cuda.is_available():
                    gpu_available = True
                    gpu_name = torch.cuda.get_device_name(0)
                    logger.info(f"GPU available: {gpu_name}")
                else:
                    logger.info("GPU not available, using CPU")
            except ImportError:
                logger.warning("PyTorch not installed, GPU detection unavailable")
        else:
            logger.info("CPU-only mode enabled")

        # Check dependencies
        missing_deps = []
        dependencies_ok = True

        required_modules = [
            "numpy",
            "json",
            "csv",
        ]

        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_deps.append(module)
                dependencies_ok = False
                logger.error(f"Missing required dependency: {module}")

        if dependencies_ok:
            logger.info("All dependencies satisfied")
        else:
            logger.error(f"Missing dependencies: {', '.join(missing_deps)}")

        self.env_info = EnvironmentInfo(
            python_version=python_version,
            platform=platform_info,
            cpu_count=cpu_count,
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            dependencies_ok=dependencies_ok,
            missing_dependencies=missing_deps,
        )

        return self.env_info

    def run_full_suite(self) -> dict[str, Any]:
        """Run the complete benchmarking suite.

        Returns:
            Complete results dictionary with all outputs.
        """
        start_time = time.perf_counter()

        logger.info("=" * 80)
        logger.info("QRATUM-Chess Automated Benchmarking Suite")
        logger.info("=" * 80)

        # 1. Verify environment
        env_info = self.verify_environment()
        if not env_info.dependencies_ok:
            logger.error("Environment verification failed. Cannot proceed.")
            return {
                "success": False,
                "error": "Missing dependencies",
                "environment": env_info.__dict__,
            }

        # 2. Initialize engine
        logger.info("\nInitializing AsymmetricAdaptiveSearch engine...")
        try:
            engine = AsymmetricAdaptiveSearch()
            logger.info("Engine initialized successfully")
        except Exception as e:
            logger.error(f"Engine initialization failed: {e}")
            return {
                "success": False,
                "error": f"Engine initialization failed: {e}",
                "environment": env_info.__dict__,
            }

        # 3. Initialize benchmark runner
        logger.info("\nInitializing benchmark runner...")
        self.runner = BenchmarkRunner(self.config.benchmark_config)
        logger.info("Benchmark runner initialized")

        # 4. Run benchmarks
        logger.info("\n" + "=" * 80)
        logger.info("EXECUTING BENCHMARK SUITE")
        logger.info("=" * 80 + "\n")

        try:
            summary = self.runner.run(engine)
            logger.info("\nBenchmark execution completed")
        except Exception as e:
            logger.error(f"Benchmark execution failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Benchmark execution failed: {e}",
                "environment": env_info.__dict__,
            }

        # 5. Run Stage III certification (if requested)
        if self.config.certify:
            logger.info("\nRunning Stage III certification verification...")
            try:
                summary.certification = self.runner.certify_against_stage_iii(summary)
                logger.info("Certification verification completed")
            except Exception as e:
                logger.error(f"Certification verification failed: {e}")

        # 6. Extract motifs (if requested)
        motifs = []
        if self.config.extract_motifs:
            logger.info("\nExtracting novel motifs from telemetry...")
            try:
                motifs = self._extract_motifs(self.runner.telemetry)
                logger.info(f"Extracted {len(motifs)} novel motifs")
            except Exception as e:
                logger.error(f"Motif extraction failed: {e}", exc_info=True)

        # 7. Save all results
        logger.info("\nSaving results...")
        try:
            output_path = self._save_all_results(summary, motifs)
            logger.info(f"Results saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}", exc_info=True)
            output_path = None

        # 8. Print summary
        if self.runner:
            self.runner.print_summary(summary)

        elapsed_time = time.perf_counter() - start_time
        logger.info(f"\nTotal execution time: {elapsed_time:.2f} seconds")

        # 9. Return complete results
        return {
            "success": True,
            "environment": env_info.__dict__,
            "summary": summary.to_dict(),
            "motifs": [m.to_dict() for m in motifs],
            "output_path": str(output_path) if output_path else None,
            "elapsed_time": elapsed_time,
        }

    def _extract_motifs(self, telemetry) -> list:
        """Extract motifs from telemetry data.

        Args:
            telemetry: TelemetryOutput instance.

        Returns:
            List of discovered motifs.
        """
        # Export telemetry to temporary location
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        telemetry.export_json(temp_path)

        # Load and parse
        with open(temp_path) as f:
            telemetry_data = json.load(f)

        # Clean up temp file
        os.unlink(temp_path)

        # Extract motifs
        self.motif_extractor = MotifExtractor(
            novelty_threshold=0.6,
            divergence_threshold=0.5,
            min_cortex_activation=0.3,
        )

        motifs = self.motif_extractor.extract_from_telemetry(telemetry_data)

        return motifs

    def _save_all_results(self, summary: BenchmarkSummary, motifs: list) -> Path:
        """Save all benchmark and motif results.

        Args:
            summary: Benchmark summary.
            motifs: List of discovered motifs.

        Returns:
            Path to output directory.
        """
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(self.config.output_dir) / timestamp
        output_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        motifs_dir = output_path / "motifs"
        telemetry_dir = output_path / "telemetry"
        logs_dir = output_path / "logs"

        motifs_dir.mkdir(exist_ok=True)
        telemetry_dir.mkdir(exist_ok=True)
        logs_dir.mkdir(exist_ok=True)

        # Save benchmark results (using existing runner methods)
        if self.runner:
            # Save main results
            json_path = output_path / "benchmark_results.json"
            with open(json_path, "w") as f:
                json.dump(summary.to_dict(), f, indent=2, default=str)

            # Save CSV summary
            csv_path = output_path / "benchmark_metrics.csv"
            self.runner._export_metrics_csv(csv_path, summary)

            # Save HTML report
            html_path = output_path / "benchmark_report.html"
            self.runner._generate_html_report(html_path, summary, timestamp)

            # Save telemetry
            self.runner.telemetry.export_json(str(telemetry_dir / "telemetry_data.json"))

            # Save certification status
            if summary.certification:
                cert_path = output_path / "certification_status.json"
                with open(cert_path, "w") as f:
                    json.dump(summary.certification.to_dict(), f, indent=2)

        # Save motif results (if any)
        if motifs and self.motif_extractor:
            # JSON catalog
            self.motif_extractor.export_catalog_json(motifs_dir / "motif_catalog.json")

            # CSV summary
            self.motif_extractor.export_summary_csv(motifs_dir / "motifs_summary.csv")

            # HTML report
            self.motif_extractor.generate_html_report(motifs_dir / "motifs_report.html")

            # PGN exports by type
            for motif_type in MotifType:
                pgn_path = motifs_dir / f"{motif_type.value}_motifs.pgn"
                self.motif_extractor.export_pgn(pgn_path, motif_type)

        # Save environment info
        if self.env_info:
            env_path = output_path / "environment_info.json"
            with open(env_path, "w") as f:
                json.dump(self.env_info.__dict__, f, indent=2)

        return output_path

    def load_checkpoint(self, checkpoint_path: Path) -> dict[str, Any]:
        """Load checkpoint from previous run.

        Args:
            checkpoint_path: Path to checkpoint file.

        Returns:
            Checkpoint data.
        """
        with open(checkpoint_path) as f:
            return json.load(f)

    def save_checkpoint(self, data: dict[str, Any]) -> None:
        """Save checkpoint for resume capability.

        Args:
            data: Data to checkpoint.
        """
        if not self.config.checkpoint_enabled:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_{timestamp}.json"

        with open(checkpoint_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        self.checkpoint_path = checkpoint_path
        logger.info(f"Checkpoint saved: {checkpoint_path}")
