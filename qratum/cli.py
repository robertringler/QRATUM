"""QRATUM CLI - Unified Command Line Interface"""

import sys
import argparse


def main():
    """Main QRATUM CLI router"""
    parser = argparse.ArgumentParser(
        prog="qratum",
        description="QRATUM Ecosystem Unified CLI",
        epilog="Use 'qratum <subsystem> --help' for subsystem-specific options"
    )
    parser.add_argument(
        "subsystem",
        choices=["nova", "visor", "quantis", "xenon", "core-os", "cryptex", "status"],
        help="QRATUM subsystem to invoke"
    )
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Subsystem arguments")
    
    args = parser.parse_args()
    
    if args.subsystem == "status":
        from qratum import get_subsystem_status
        status = get_subsystem_status()
        print("╔══════════════════════════════════════════════╗")
        print("║       QRATUM Subsystem Status Report         ║")
        print("╚══════════════════════════════════════════════╝")
        for name, info in status.items():
            avail = "✓" if info["available"] else "✗"
            version = info["version"]
            print(f"  {avail} {name:12} v{version}")
        return 0
    
    # Route to appropriate CLI
    subsystem_map = {
        "nova": "quasim.cli:main",
        "visor": "qubic.visualization.cli:main",
        "quantis": "qunimbus.cli:main",
        "xenon": "xenon.cli:main",
        "core-os": "qnx.cli:main",
        "cryptex": "qstack.cli:main",
    }
    
    target = subsystem_map.get(args.subsystem)
    if not target:
        print(f"Subsystem '{args.subsystem}' not yet implemented")
        return 1
    
    # Update sys.argv for target CLI
    sys.argv = [f"qratum-{args.subsystem}"] + args.args
    
    # Import and execute
    module_path, func_name = target.split(":")
    try:
        module = __import__(module_path, fromlist=[func_name])
        func = getattr(module, func_name)
        return func()
    except ImportError as e:
        print(f"Error: Subsystem '{args.subsystem}' not installed")
        print(f"Details: {e}")
        return 1
    except Exception as e:
        print(f"Error invoking {args.subsystem}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
