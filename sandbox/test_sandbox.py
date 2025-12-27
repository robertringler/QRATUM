#!/usr/bin/env python3
"""
QRATUM Sandbox Test Script

Verifies all services are running and validates the sandbox environment.
"""

import sys
import time
from typing import Tuple

import requests

# ANSI color codes
GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
NC = "\033[0m"  # No Color


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{'=' * 70}")
    print(f"{CYAN}{text}{NC}")
    print("=" * 70)


def print_test(name: str, passed: bool, message: str = "") -> None:
    """Print test result."""
    status = f"{GREEN}✓ PASS{NC}" if passed else f"{RED}✗ FAIL{NC}"
    print(f"  [{status}] {name}")
    if message:
        print(f"         {message}")


def test_service_health(service_name: str, url: str) -> Tuple[bool, str]:
    """Test if a service is healthy."""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return True, f"Status: {response.status_code}"
        else:
            return False, f"Unexpected status: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Connection refused"
    except requests.exceptions.Timeout:
        return False, "Request timeout"
    except Exception as e:
        return False, f"Error: {str(e)}"


def test_qradle_chain() -> Tuple[bool, str]:
    """Test QRADLE Merkle chain initialization."""
    try:
        response = requests.get("http://localhost:8001/api/chain/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            chain_length = data.get("chain_length", 0)
            root_hash = data.get("root_hash", "")

            if chain_length > 0 and root_hash:
                return True, f"Chain length: {chain_length}, Root: {root_hash[:16]}..."
            else:
                return False, "Chain not properly initialized"
        return False, f"Status: {response.status_code}"
    except Exception as e:
        return False, f"Error: {str(e)}"


def test_qradle_engine() -> Tuple[bool, str]:
    """Test QRADLE deterministic engine."""
    try:
        response = requests.get("http://localhost:8001/api/engine/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            initialized = data.get("initialized", False)

            if initialized:
                return True, "Engine initialized and operational"
            else:
                return False, "Engine not initialized"
        return False, f"Status: {response.status_code}"
    except Exception as e:
        return False, f"Error: {str(e)}"


def test_platform_status() -> Tuple[bool, str]:
    """Test QRATUM Platform status."""
    try:
        response = requests.get("http://localhost:8002/api/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            status = data.get("status", "unknown")
            modules = data.get("modules_loaded", 0)
            return True, f"Status: {status}, Modules: {modules}"
        return False, f"Status: {response.status_code}"
    except Exception as e:
        return False, f"Error: {str(e)}"


def run_all_tests() -> bool:
    """Run all sandbox tests."""
    print_header("🧪 QRATUM Sandbox Test Suite")

    all_passed = True

    # Test 1: QRADLE Health
    print("\n1. Testing QRADLE Service...")
    passed, message = test_service_health("QRADLE", "http://localhost:8001/health")
    print_test("QRADLE Health Check", passed, message)
    all_passed = all_passed and passed

    # Test 2: QRADLE Chain
    passed, message = test_qradle_chain()
    print_test("QRADLE Merkle Chain", passed, message)
    all_passed = all_passed and passed

    # Test 3: QRADLE Engine
    passed, message = test_qradle_engine()
    print_test("QRADLE Deterministic Engine", passed, message)
    all_passed = all_passed and passed

    # Test 4: Platform Health
    print("\n2. Testing QRATUM Platform...")
    passed, message = test_service_health("QRATUM Platform", "http://localhost:8002/")
    print_test("QRATUM Platform Health", passed, message)
    all_passed = all_passed and passed

    # Test 5: Platform Status
    passed, message = test_platform_status()
    print_test("QRATUM Platform Status", passed, message)
    all_passed = all_passed and passed

    # Summary
    print_header("Test Summary")
    if all_passed:
        print(f"{GREEN}✓ All tests passed!{NC}")
        print("\nSandbox Status: HEALTHY")
        print("\nAvailable Services:")
        print("  🛡️  QRADLE:          http://localhost:8001")
        print("  🚀 QRATUM Platform: http://localhost:8002")
        return True
    else:
        print(f"{RED}✗ Some tests failed{NC}")
        print("\nPlease check the service logs for details:")
        print("  QRADLE:  /tmp/qradle.log")
        print("  Platform: /tmp/platform.log")
        return False


def wait_for_services(timeout: int = 30) -> bool:
    """Wait for services to be ready."""
    print(f"{CYAN}Waiting for services to start...{NC}")

    start_time = time.time()
    services = [("QRADLE", "http://localhost:8001/health"), ("Platform", "http://localhost:8002/")]

    while time.time() - start_time < timeout:
        all_ready = True
        for name, url in services:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code != 200:
                    all_ready = False
            except Exception:
                all_ready = False

        if all_ready:
            print(f"{GREEN}✓ All services are ready{NC}")
            return True

        print(".", end="", flush=True)
        time.sleep(1)

    print(f"\n{YELLOW}Warning: Timeout waiting for services{NC}")
    return False


def main():
    """Main entry point."""
    print(f"{CYAN}QRATUM Sandbox Test Script{NC}")

    # Wait for services to be ready
    wait_for_services()

    # Run tests
    success = run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
