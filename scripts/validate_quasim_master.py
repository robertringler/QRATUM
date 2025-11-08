#!/usr/bin/env python3
"""
Validation script for QuASIM Master integration
Verifies all components are properly integrated and functional
"""

import os
import subprocess
import sys
from pathlib import Path


def check_file_exists(path, description):
    """Check if a file exists and report status"""
    if Path(path).exists():
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ {description} MISSING: {path}")
        return False


def run_command(cmd, description):
    """Run a command and report status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"✅ {description}")
            return True
        else:
            print(f"❌ {description} FAILED")
            print(f"   Error: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"❌ {description} ERROR: {str(e)}")
        return False


def main():
    """Main validation routine"""
    print("=" * 70)
    print("QuASIM Master Integration Validation")
    print("=" * 70)
    print()

    repo_root = Path(__file__).parent.parent
    os.chdir(repo_root)

    checks = []

    # File existence checks
    print("## File Existence Checks")
    checks.append(check_file_exists("quasim_master_all.py", "Main source file"))
    checks.append(check_file_exists("QUASIM_MASTER_SUMMARY.md", "Summary document"))
    checks.append(
        check_file_exists(".github/workflows/quasim-master-build.yml", "Main CI workflow")
    )
    checks.append(check_file_exists(".github/workflows/cuda-build.yml", "CUDA CI workflow"))
    checks.append(check_file_exists("QuASIM/CMakeLists.txt", "CMake config"))
    checks.append(check_file_exists("QuASIM/demo.cpp", "Demo application"))
    checks.append(check_file_exists("QuASIM/src/quasim_tensor_solve.cpp", "Core solver"))
    checks.append(check_file_exists("QuASIM/include/quasim_core.h", "Core header"))
    checks.append(check_file_exists("QuASIM/Dockerfile.cuda", "CUDA Dockerfile"))
    checks.append(check_file_exists("QuASIM/onera/benchmarks.csv", "Benchmark data"))
    print()

    # Functional checks
    print("## Functional Checks")
    checks.append(run_command("python3 quasim_master_all.py", "Self-test execution"))
    checks.append(
        run_command(
            "python3 quasim_master_all.py --emit /tmp/quasim_val_test",
            "Scaffold emission",
        )
    )
    checks.append(run_command("python3 -m py_compile quasim_master_all.py", "Python syntax check"))
    print()

    # Summary
    print("=" * 70)
    passed = sum(checks)
    total = len(checks)
    print(f"Validation Results: {passed}/{total} checks passed")

    if passed == total:
        print("✅ ALL VALIDATIONS PASSED")
        print("=" * 70)
        return 0
    else:
        print(f"❌ {total - passed} VALIDATION(S) FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
