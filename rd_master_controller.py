#!/usr/bin/env python3
"""
R&D Master Controller - Fully Automated Research & Development Pipeline
Handles complete automation of quantum simulation research pipeline from data generation to publication-ready outputs.
"""

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class RDStage:
    """Research & Development stage configuration"""

    name: str
    enabled: bool
    priority: int
    dependencies: List[str]
    timeout_hours: float
    retry_count: int
    quality_gates: Dict[str, Any]
    outputs: List[str]


@dataclass
class RDPipelineConfig:
    """Master R&D pipeline configuration"""

    project_name: str
    version: str
    stages: List[RDStage]
    global_timeout_hours: float
    parallel_workers: int
    archive_results: bool
    generate_reports: bool
    auto_publish: bool


class RDMasterController:
    """Master controller for R&D automation pipeline"""

    def __init__(self, config_file: str = "automation/rd_pipeline_config.yaml"):
        self.config_file = config_file
        self.config = self._load_config()
        self.workspace_root = Path(__file__).parent.parent
        self.run_id = self._generate_run_id()
        self.output_dir = self.workspace_root / "rd_runs" / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = self._setup_logging()

        # Stage status tracking
        self.stage_status = {}
        self.stage_outputs = {}
        self.stage_metrics = {}

        # Initialize stages
        self._initialize_stages()

    def _load_config(self) -> RDPipelineConfig:
        """Load R&D pipeline configuration"""
        config_path = self.workspace_root / self.config_file
        if not config_path.exists():
            self._create_default_config(config_path)

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        stages = [RDStage(**stage) for stage in config_data["stages"]]

        return RDPipelineConfig(
            project_name=config_data["project_name"],
            version=config_data["version"],
            stages=stages,
            global_timeout_hours=config_data["global_timeout_hours"],
            parallel_workers=config_data["parallel_workers"],
            archive_results=config_data["archive_results"],
            generate_reports=config_data["generate_reports"],
            auto_publish=config_data["auto_publish"],
        )

    def _create_default_config(self, config_path: Path):
        """Create default R&D pipeline configuration"""
        default_config = {
            "project_name": "Quantum Phase Transition Cosmogenesis",
            "version": "1.0.0",
            "global_timeout_hours": 48.0,
            "parallel_workers": 4,
            "archive_results": True,
            "generate_reports": True,
            "auto_publish": False,
            "stages": [
                {
                    "name": "data_generation",
                    "enabled": True,
                    "priority": 1,
                    "dependencies": [],
                    "timeout_hours": 12.0,
                    "retry_count": 2,
                    "quality_gates": {
                        "min_datasets": 3,
                        "max_trace_deviation": 1e-9,
                        "min_simulation_time": 5.0,
                    },
                    "outputs": ["simulation_data", "gamma_sweeps", "trajectory_data"],
                },
                {
                    "name": "analysis_processing",
                    "enabled": True,
                    "priority": 2,
                    "dependencies": ["data_generation"],
                    "timeout_hours": 8.0,
                    "retry_count": 2,
                    "quality_gates": {
                        "min_d2_estimate": 0.1,
                        "max_bootstrap_halfwidth": 0.05,
                        "min_logical_fidelity": 0.6,
                    },
                    "outputs": ["analysis_results", "convergence_plots", "phase_diagrams"],
                },
                {
                    "name": "finite_size_scaling",
                    "enabled": True,
                    "priority": 3,
                    "dependencies": ["analysis_processing"],
                    "timeout_hours": 6.0,
                    "retry_count": 2,
                    "quality_gates": {
                        "min_scaling_exponent": 0.5,
                        "max_fitting_error": 0.1,
                        "min_lattice_sizes": 4,
                    },
                    "outputs": ["fss_fits", "scaling_plots", "critical_exponents"],
                },
                {
                    "name": "advanced_analysis",
                    "enabled": True,
                    "priority": 4,
                    "dependencies": ["finite_size_scaling"],
                    "timeout_hours": 4.0,
                    "retry_count": 1,
                    "quality_gates": {"min_fisher_information": 10.0, "max_instanton_error": 0.05},
                    "outputs": ["fisher_analysis", "instanton_metrics", "info_geometry"],
                },
                {
                    "name": "validation_testing",
                    "enabled": True,
                    "priority": 5,
                    "dependencies": ["advanced_analysis"],
                    "timeout_hours": 2.0,
                    "retry_count": 1,
                    "quality_gates": {"min_test_coverage": 0.8, "max_regression_tests_failed": 0},
                    "outputs": ["test_results", "validation_reports", "benchmark_data"],
                },
                {
                    "name": "report_generation",
                    "enabled": True,
                    "priority": 6,
                    "dependencies": ["validation_testing"],
                    "timeout_hours": 2.0,
                    "retry_count": 1,
                    "quality_gates": {
                        "min_figures_generated": 10,
                        "latex_compilation_success": True,
                    },
                    "outputs": ["latex_report", "figures", "summary_html"],
                },
            ],
        }

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)

        self.logger.info(f"Created default R&D config at {config_path}")

    def _generate_run_id(self) -> str:
        """Generate unique run ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{timestamp}_{self.config.project_name}_{self.config.version}"
        run_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"rd_{timestamp}_{run_hash}"

    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger("rd_master")
        logger.setLevel(logging.INFO)

        # File handler
        log_file = self.output_dir / "rd_master.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def _initialize_stages(self):
        """Initialize stage tracking"""
        for stage in self.config.stages:
            self.stage_status[stage.name] = "pending"
            self.stage_outputs[stage.name] = []
            self.stage_metrics[stage.name] = {}

    def _check_dependencies(self, stage: RDStage) -> bool:
        """Check if stage dependencies are satisfied"""
        for dep in stage.dependencies:
            if self.stage_status.get(dep) != "completed":
                return False
        return True

    def _execute_stage(self, stage: RDStage) -> bool:
        """Execute a single R&D stage"""
        self.logger.info(f"Starting stage: {stage.name}")
        self.stage_status[stage.name] = "running"

        stage_start = time.time()
        success = False

        try:
            if stage.name == "data_generation":
                success = self._execute_data_generation(stage)
            elif stage.name == "analysis_processing":
                success = self._execute_analysis_processing(stage)
            elif stage.name == "finite_size_scaling":
                success = self._execute_finite_size_scaling(stage)
            elif stage.name == "advanced_analysis":
                success = self._execute_advanced_analysis(stage)
            elif stage.name == "validation_testing":
                success = self._execute_validation_testing(stage)
            elif stage.name == "report_generation":
                success = self._execute_report_generation(stage)
            else:
                self.logger.warning(f"Unknown stage: {stage.name}")
                success = False

        except Exception as e:
            self.logger.error(f"Stage {stage.name} failed with exception: {e}")
            success = False

        stage_duration = time.time() - stage_start

        if success:
            self.stage_status[stage.name] = "completed"
            self.logger.info(f"Stage {stage.name} completed in {stage_duration:.1f}s")
        else:
            self.stage_status[stage.name] = "failed"
            self.logger.error(f"Stage {stage.name} failed after {stage_duration:.1f}s")

        return success

    def _execute_data_generation(self, stage: RDStage) -> bool:
        """Execute data generation stage"""
        self.logger.info("Executing comprehensive data generation pipeline")

        configs = [
            "automation/pipeline_config_quick.yaml",
            "automation/pipeline_config_medium.yaml",
            "automation/pipeline_config_robust.yaml",
        ]

        for config in configs:
            config_path = self.workspace_root / config
            if not config_path.exists():
                self.logger.warning(f"Config {config} not found, skipping")
                continue

            cmd = [sys.executable, "-m", "automation.run_pipeline", "--config", str(config_path)]

            self.logger.info(f"Running pipeline with config: {config}")
            result = subprocess.run(
                cmd,
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                timeout=3600 * stage.timeout_hours,
            )

            if result.returncode != 0:
                self.logger.error(f"Pipeline failed for {config}: {result.stderr}")
                return False

            self.logger.info(f"Pipeline completed for {config}")

        # Collect generated data
        data_files = list(self.workspace_root.glob("*_metrics.json"))
        data_files.extend(list(self.workspace_root.glob("automation_runs/*/index.json")))

        self.stage_outputs[stage.name] = [str(f) for f in data_files]
        self.stage_metrics[stage.name] = {"datasets_generated": len(data_files)}

        return len(data_files) >= stage.quality_gates.get("min_datasets", 1)

    def _execute_analysis_processing(self, stage: RDStage) -> bool:
        """Execute analysis processing stage"""
        self.logger.info("Executing comprehensive analysis processing")

        # Run all analyses
        cmd = [
            sys.executable,
            "all_analyses.py",
            "--Lx",
            "16",
            "--Ly",
            "16",
            "--T",
            "5",
            "--dt",
            "0.02",
            "--disorder",
            "0.2",
            "--convergence",
            "--msd",
            "--otoc",
            "--logical",
            "--traj-variance",
            "--phase-diagram",
        ]

        result = subprocess.run(
            cmd,
            cwd=self.workspace_root,
            capture_output=True,
            text=True,
            timeout=3600 * stage.timeout_hours,
        )

        if result.returncode != 0:
            self.logger.error(f"Analysis processing failed: {result.stderr}")
            return False

        # Run finite size scaling
        cmd_fss = [sys.executable, "finite_size_scaling.py", "--quick"]
        result_fss = subprocess.run(
            cmd_fss, cwd=self.workspace_root, capture_output=True, text=True, timeout=1800
        )

        # Collect outputs
        analysis_files = list(self.workspace_root.glob("all_*.png"))
        analysis_files.extend(list(self.workspace_root.glob("*_fss_*.json")))

        self.stage_outputs[stage.name] = [str(f) for f in analysis_files]

        return len(analysis_files) > 0

    def _execute_finite_size_scaling(self, stage: RDStage) -> bool:
        """Execute finite size scaling analysis"""
        self.logger.info("Executing finite size scaling analysis")

        cmd = [sys.executable, "finite_size_scaling.py", "--full"]
        result = subprocess.run(
            cmd,
            cwd=self.workspace_root,
            capture_output=True,
            text=True,
            timeout=3600 * stage.timeout_hours,
        )

        if result.returncode != 0:
            self.logger.error(f"FSS analysis failed: {result.stderr}")
            return False

        # Collect FSS outputs
        fss_files = list(self.workspace_root.glob("*_fss_*.json"))
        fss_files.extend(list(self.workspace_root.glob("*_fss_*.png")))

        self.stage_outputs[stage.name] = [str(f) for f in fss_files]

        return len(fss_files) > 0

    def _execute_advanced_analysis(self, stage: RDStage) -> bool:
        """Execute advanced analysis stage"""
        self.logger.info("Executing advanced analysis")

        # Run information geometry analysis if available
        info_geom_script = self.workspace_root / "information_geometry.py"
        if info_geom_script.exists():
            cmd = [sys.executable, "information_geometry.py", "--full"]
            subprocess.run(cmd, cwd=self.workspace_root, timeout=1800)

        # Run PTQ instanton analysis if available
        ptq_script = self.workspace_root / "adaptive_ptq_controller.py"
        if ptq_script.exists():
            cmd = [sys.executable, "adaptive_ptq_controller.py", "--analyze"]
            subprocess.run(cmd, cwd=self.workspace_root, timeout=1800)

        # Collect advanced analysis outputs
        adv_files = list(self.workspace_root.glob("*_fisher_*.json"))
        adv_files.extend(list(self.workspace_root.glob("*_instanton_*.json")))
        adv_files.extend(list(self.workspace_root.glob("*_info_geom_*.png")))

        self.stage_outputs[stage.name] = [str(f) for f in adv_files]

        return True  # Advanced analysis is optional

    def _execute_validation_testing(self, stage: RDStage) -> bool:
        """Execute validation and testing stage"""
        self.logger.info("Executing validation and testing")

        # Run quick test to validate pipeline
        cmd = [sys.executable, "quick_test_run.py"]
        result = subprocess.run(
            cmd, cwd=self.workspace_root, capture_output=True, text=True, timeout=1800
        )

        # Run benchmark tests
        benchmark_script = self.workspace_root / "automation" / "logical_benchmark.py"
        if benchmark_script.exists():
            cmd_bench = [sys.executable, "-m", "automation.logical_benchmark", "--L", "8"]
            subprocess.run(cmd_bench, cwd=self.workspace_root, timeout=1800)

        # Collect validation outputs
        test_files = list(self.workspace_root.glob("*_test_*.json"))
        test_files.extend(list(self.workspace_root.glob("*_benchmark_*.json")))

        self.stage_outputs[stage.name] = [str(f) for f in test_files]

        return result.returncode == 0

    def _execute_report_generation(self, stage: RDStage) -> bool:
        """Execute report generation stage"""
        self.logger.info("Executing report generation")

        # Generate LaTeX report
        cmd_latex = ["make", "latex"]
        result_latex = subprocess.run(
            cmd_latex, cwd=self.workspace_root, capture_output=True, text=True, timeout=1800
        )

        # Generate circuit diagrams if available
        cmd_circuits = ["make", "circuits"]
        subprocess.run(cmd_circuits, cwd=self.workspace_root, timeout=900)

        # Generate HTML summary
        self._generate_html_summary()

        # Collect report outputs
        report_files = list(self.workspace_root.glob("*.pdf"))
        report_files.extend(list(self.workspace_root.glob("summary.html")))
        report_files.extend(list(self.workspace_root.glob("circuits/*.pdf")))

        self.stage_outputs[stage.name] = [str(f) for f in report_files]

        return len(report_files) > 0

    def _generate_html_summary(self):
        """Generate comprehensive HTML summary"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>R&D Pipeline Summary - {self.run_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .stage {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .completed {{ background-color: #d4edda; }}
                .failed {{ background-color: #f8d7da; }}
                .pending {{ background-color: #fff3cd; }}
                .metrics {{ font-family: monospace; background: #f8f9fa; padding: 10px; }}
            </style>
        </head>
        <body>
            <h1>R&D Pipeline Summary</h1>
            <p><strong>Run ID:</strong> {self.run_id}</p>
            <p><strong>Project:</strong> {self.config.project_name}</p>
            <p><strong>Version:</strong> {self.config.version}</p>
            <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <h2>Stage Status</h2>
        """

        for stage in self.config.stages:
            status = self.stage_status[stage.name]
            html_content += f"""
            <div class="stage {status}">
                <h3>{stage.name.replace('_', ' ').title()}</h3>
                <p><strong>Status:</strong> {status}</p>
                <p><strong>Outputs:</strong> {len(self.stage_outputs[stage.name])} files</p>
                <div class="metrics">
                    <strong>Metrics:</strong><br>
                    {json.dumps(self.stage_metrics[stage.name], indent=2)}
                </div>
            </div>
            """

        html_content += """
        </body>
        </html>
        """

        summary_file = self.output_dir / "summary.html"
        with open(summary_file, "w") as f:
            f.write(html_content)

    def run_pipeline(self) -> bool:
        """Run the complete R&D pipeline"""
        self.logger.info(f"Starting R&D Master Pipeline - Run ID: {self.run_id}")

        pipeline_start = time.time()

        # Sort stages by priority
        stages_by_priority = sorted(self.config.stages, key=lambda s: s.priority)

        # Execute stages
        for stage in stages_by_priority:
            if not stage.enabled:
                self.logger.info(f"Skipping disabled stage: {stage.name}")
                continue

            # Check dependencies
            if not self._check_dependencies(stage):
                self.logger.error(f"Dependencies not met for stage: {stage.name}")
                self.stage_status[stage.name] = "failed"
                continue

            # Execute stage with retries
            success = False
            for attempt in range(stage.retry_count + 1):
                if attempt > 0:
                    self.logger.info(f"Retrying stage {stage.name} (attempt {attempt + 1})")

                success = self._execute_stage(stage)
                if success:
                    break

                if attempt < stage.retry_count:
                    time.sleep(30)  # Wait before retry

            if not success:
                self.logger.error(
                    f"Stage {stage.name} failed after {stage.retry_count + 1} attempts"
                )
                # Continue with other stages unless this is critical

        pipeline_duration = time.time() - pipeline_start

        # Generate final summary
        self._generate_final_summary(pipeline_duration)

        # Archive results if requested
        if self.config.archive_results:
            self._archive_results()

        # Check overall success
        failed_stages = [name for name, status in self.stage_status.items() if status == "failed"]

        if failed_stages:
            self.logger.error(f"Pipeline completed with failures: {failed_stages}")
            return False
        else:
            self.logger.info(f"Pipeline completed successfully in {pipeline_duration:.1f}s")
            return True

    def _generate_final_summary(self, duration: float):
        """Generate final pipeline summary"""
        summary = {
            "run_id": self.run_id,
            "project": self.config.project_name,
            "version": self.config.version,
            "start_time": datetime.now().isoformat(),
            "duration_seconds": duration,
            "stages": {},
        }

        for stage in self.config.stages:
            summary["stages"][stage.name] = {
                "status": self.stage_status[stage.name],
                "outputs": self.stage_outputs[stage.name],
                "metrics": self.stage_metrics[stage.name],
            }

        summary_file = self.output_dir / "pipeline_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Final summary saved to {summary_file}")

    def _archive_results(self):
        """Archive pipeline results"""
        self.logger.info("Archiving pipeline results")

        archive_dir = self.output_dir / "archived_outputs"
        archive_dir.mkdir(exist_ok=True)

        # Copy key outputs to archive
        patterns = [
            "*.json",
            "*.png",
            "*.pdf",
            "*.csv",
            "*.log",
            "automation_runs/*/",
            "circuits/*.pdf",
        ]

        for pattern in patterns:
            for file_path in self.workspace_root.glob(pattern):
                if file_path.is_file():
                    shutil.copy2(file_path, archive_dir)
                elif file_path.is_dir() and "automation_runs" in str(file_path):
                    shutil.copytree(file_path, archive_dir / file_path.name, dirs_exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="R&D Master Controller")
    parser.add_argument(
        "--config",
        default="automation/rd_pipeline_config.yaml",
        help="R&D pipeline configuration file",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show pipeline plan without executing"
    )

    args = parser.parse_args()

    controller = RDMasterController(args.config)

    if args.dry_run:
        print(f"R&D Pipeline Plan - {controller.config.project_name}")
        print("Stages to execute:")
        for stage in sorted(controller.config.stages, key=lambda s: s.priority):
            if stage.enabled:
                print(f"  {stage.priority}. {stage.name} (deps: {stage.dependencies})")
        return 0

    success = controller.run_pipeline()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
