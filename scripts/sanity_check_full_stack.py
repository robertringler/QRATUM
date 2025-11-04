#!/usr/bin/env python3
"""
QuASIM Full Stack Sanity Check

This script performs a comprehensive sanity check on the QuASIM full stack build:
- Validates Docker Compose configuration
- Builds Docker images
- Starts services
- Tests health endpoints
- Validates API functionality
- Tests frontend-backend communication
- Cleans up resources

Usage:
    python3 scripts/sanity_check_full_stack.py [--skip-docker] [--keep-running]

Options:
    --skip-docker    Skip Docker build and startup (useful if services already running)
    --keep-running   Keep services running after tests complete
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import requests
except ImportError:
    print("ERROR: 'requests' module not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:8080"
MAX_STARTUP_WAIT = 120  # seconds
HEALTH_CHECK_INTERVAL = 5  # seconds

# Test configuration constants
TEST_SEED = 42
TEST_SCALE = 1.5
EXPECTED_METRIC_PREFIX = "autonomous_systems_requests_total"
EXPECTED_FRONTEND_CONTENT = ["Autonomous Systems", "Run Kernel"]


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_header(message: str) -> None:
    """Print a formatted header."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{message:^70}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'=' * 70}{Colors.RESET}\n")


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {message}{Colors.RESET}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"{Colors.RED}✗ {message}{Colors.RESET}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.RESET}")


def print_info(message: str) -> None:
    """Print an info message."""
    print(f"{Colors.BLUE}ℹ {message}{Colors.RESET}")


def check_docker_available() -> bool:
    """Check if Docker is available."""
    try:
        result = subprocess.run(
            ["docker", "version"],
            capture_output=True,
            check=True,
            timeout=10
        )
        print_success("Docker is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        print_error("Docker is not available or not running")
        return False


def validate_docker_compose_config() -> bool:
    """Validate docker-compose.yml configuration."""
    print_info("Validating docker-compose configuration...")
    try:
        result = subprocess.run(
            ["docker", "compose", "config"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
            timeout=30
        )
        print_success("docker-compose.yml is valid")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"docker-compose.yml validation failed: {e.stderr.decode()}")
        return False
    except subprocess.TimeoutExpired:
        print_error("docker-compose config validation timed out")
        return False


def build_docker_images() -> bool:
    """Build Docker images for the stack."""
    print_info("Building Docker images (this may take a few minutes)...")
    try:
        result = subprocess.run(
            ["docker", "compose", "build"],
            cwd=REPO_ROOT,
            timeout=600  # 10 minutes
        )
        if result.returncode == 0:
            print_success("Docker images built successfully")
            return True
        else:
            print_error(f"Docker build failed with exit code {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        print_error("Docker build timed out after 10 minutes")
        return False


def start_services() -> bool:
    """Start services using docker-compose."""
    print_info("Starting services with docker-compose...")
    try:
        # Start in detached mode
        result = subprocess.run(
            ["docker", "compose", "up", "-d"],
            cwd=REPO_ROOT,
            capture_output=True,
            timeout=120
        )
        if result.returncode == 0:
            print_success("Services started successfully")
            return True
        else:
            print_error(f"Failed to start services: {result.stderr.decode()}")
            return False
    except subprocess.TimeoutExpired:
        print_error("Service startup timed out")
        return False


def stop_services() -> None:
    """Stop services using docker-compose."""
    print_info("Stopping services...")
    try:
        subprocess.run(
            ["docker", "compose", "down"],
            cwd=REPO_ROOT,
            capture_output=True,
            timeout=60
        )
        print_success("Services stopped")
    except subprocess.TimeoutExpired:
        print_warning("Service shutdown timed out")


def wait_for_service(url: str, service_name: str, timeout: int = MAX_STARTUP_WAIT) -> bool:
    """Wait for a service to become available."""
    print_info(f"Waiting for {service_name} to become available at {url}...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                elapsed = time.time() - start_time
                print_success(f"{service_name} is available (took {elapsed:.1f}s)")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(HEALTH_CHECK_INTERVAL)
    
    print_error(f"{service_name} did not become available within {timeout}s")
    return False


def test_backend_health() -> bool:
    """Test backend health endpoint."""
    print_info("Testing backend health endpoint...")
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "healthy":
                print_success("Backend health check passed")
                return True
            else:
                print_error(f"Backend returned unhealthy status: {data}")
                return False
        else:
            print_error(f"Backend health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Backend health check failed: {e}")
        return False


def test_backend_kernel() -> bool:
    """Test backend kernel endpoint."""
    print_info("Testing backend kernel endpoint...")
    try:
        payload = {"seed": TEST_SEED, "scale": TEST_SCALE}
        response = requests.post(
            f"{BACKEND_URL}/kernel",
            json=payload,
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            if "result" in data:
                result = data["result"]
                # Validate result structure
                if all(key in result for key in ["state_vector", "energy", "convergence"]):
                    print_success("Backend kernel endpoint works correctly")
                    print_info(f"  Energy: {result['energy']}")
                    print_info(f"  Convergence: {result['convergence']}")
                    return True
                else:
                    print_error(f"Kernel result missing expected keys: {result}")
                    return False
            else:
                print_error(f"Kernel response missing 'result' key: {data}")
                return False
        else:
            print_error(f"Kernel endpoint failed with status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Backend kernel test failed: {e}")
        return False


def test_backend_metrics() -> bool:
    """Test backend metrics endpoint."""
    print_info("Testing backend metrics endpoint...")
    try:
        response = requests.get(f"{BACKEND_URL}/metrics", timeout=10)
        if response.status_code == 200:
            metrics_text = response.text
            # Check for expected metrics
            if EXPECTED_METRIC_PREFIX in metrics_text:
                print_success("Backend metrics endpoint works correctly")
                return True
            else:
                print_error("Metrics endpoint missing expected metrics")
                return False
        else:
            print_error(f"Metrics endpoint failed with status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Backend metrics test failed: {e}")
        return False


def test_frontend_accessible() -> bool:
    """Test that frontend is accessible."""
    print_info("Testing frontend accessibility...")
    try:
        response = requests.get(FRONTEND_URL, timeout=10)
        if response.status_code == 200:
            html = response.text
            # Check for expected content
            if all(content in html for content in EXPECTED_FRONTEND_CONTENT):
                print_success("Frontend is accessible and contains expected content")
                return True
            else:
                print_error("Frontend missing expected content")
                return False
        else:
            print_error(f"Frontend request failed with status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Frontend accessibility test failed: {e}")
        return False


def check_service_logs() -> None:
    """Display service logs if there are issues."""
    print_info("Checking service logs...")
    try:
        result = subprocess.run(
            ["docker", "compose", "logs", "--tail=20"],
            cwd=REPO_ROOT,
            capture_output=True,
            timeout=30
        )
        if result.stdout:
            print("\n--- Recent Service Logs ---")
            print(result.stdout.decode())
            print("--- End Logs ---\n")
    except Exception as e:
        print_warning(f"Could not retrieve service logs: {e}")


def run_sanity_check(skip_docker: bool = False, keep_running: bool = False) -> int:
    """Run the full sanity check suite."""
    print_header("QuASIM Full Stack Sanity Check")
    
    results: Dict[str, bool] = {}
    
    # Pre-checks
    if not skip_docker:
        if not check_docker_available():
            return 1
        
        results["docker_compose_config"] = validate_docker_compose_config()
        if not results["docker_compose_config"]:
            return 1
        
        results["docker_build"] = build_docker_images()
        if not results["docker_build"]:
            return 1
        
        results["service_startup"] = start_services()
        if not results["service_startup"]:
            return 1
    else:
        print_info("Skipping Docker operations (--skip-docker flag)")
    
    # Wait for services to be ready
    results["backend_ready"] = wait_for_service(f"{BACKEND_URL}/health", "Backend")
    if not results["backend_ready"]:
        check_service_logs()
        if not keep_running and not skip_docker:
            stop_services()
        return 1
    
    results["frontend_ready"] = wait_for_service(FRONTEND_URL, "Frontend", timeout=30)
    if not results["frontend_ready"]:
        check_service_logs()
        print_warning("Frontend not ready, but continuing with backend tests...")
    
    # Run tests
    results["backend_health"] = test_backend_health()
    results["backend_kernel"] = test_backend_kernel()
    results["backend_metrics"] = test_backend_metrics()
    
    if results["frontend_ready"]:
        results["frontend_accessible"] = test_frontend_accessible()
    
    # Cleanup
    if not keep_running and not skip_docker:
        stop_services()
    elif keep_running:
        print_info("Services left running (--keep-running flag)")
        print_info(f"  Backend: {BACKEND_URL}")
        print_info(f"  Frontend: {FRONTEND_URL}")
        print_info("  Run 'docker compose down' to stop services")
    
    # Summary
    print_header("Sanity Check Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check_name, passed_status in results.items():
        status_str = f"{Colors.GREEN}PASS{Colors.RESET}" if passed_status else f"{Colors.RED}FAIL{Colors.RESET}"
        print(f"  {check_name}: {status_str}")
    
    print(f"\n{Colors.BOLD}Results: {passed}/{total} checks passed{Colors.RESET}")
    
    if all(results.values()):
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ All sanity checks passed!{Colors.RESET}")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ Some sanity checks failed{Colors.RESET}")
        return 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run sanity checks on QuASIM full stack build"
    )
    parser.add_argument(
        "--skip-docker",
        action="store_true",
        help="Skip Docker build and startup (assumes services already running)"
    )
    parser.add_argument(
        "--keep-running",
        action="store_true",
        help="Keep services running after tests complete"
    )
    
    args = parser.parse_args()
    
    try:
        return run_sanity_check(
            skip_docker=args.skip_docker,
            keep_running=args.keep_running
        )
    except KeyboardInterrupt:
        print_warning("\nSanity check interrupted by user")
        if not args.skip_docker:
            stop_services()
        return 130


if __name__ == "__main__":
    sys.exit(main())
