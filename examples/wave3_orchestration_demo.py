#!/usr/bin/env python3
"""Example: QuNimbus Wave 3 Orchestration.

This example demonstrates the dual execution of:
1. Wave 3 Launch: 1,000 pilots/day
2. China Photonic Factory: Global scale integration

Usage:
    python examples/wave3_orchestration_demo.py
"""

from __future__ import annotations

import asyncio
import logging
import sys

from quasim.qunimbus.china_integration import ChinaPhotonicFactory
from quasim.qunimbus.orchestrator import (ComplianceFramework, ExecutionMode,
                                          OrchestrationConfig,
                                          QuNimbusOrchestrator)
from quasim.qunimbus.pilot_factory import PilotFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


async def demo_wave3_launch():
    """Demonstrate Wave 3 launch execution."""
    logger.info("=" * 70)
    logger.info("DEMO: Wave 3 Launch (1,000 pilots/day)")
    logger.info("=" * 70)
    logger.info("")

    # Create configuration
    config = OrchestrationConfig(
        parallel=False,  # Sequential for demo
        tasks=["wave3_launch"],
        mode=ExecutionMode.LIVE_ACCELERATED,
        pilot_target=1000,
        china_enabled=False,
    )

    # Create orchestrator
    orchestrator = QuNimbusOrchestrator(config)

    # Execute Wave 3 launch
    result = await orchestrator.execute_wave3_launch()

    logger.info("")
    logger.info("Wave 3 Launch Result:")
    logger.info(f"  Status: {result['status']}")
    logger.info(f"  Pilots Generated: {result['pilots_generated']}")
    logger.info(f"  Efficiency: {result['efficiency']}")
    logger.info(f"  MERA Compression: {result['mera_compression']}")
    logger.info(f"  RL Convergence: {result['rl_convergence']}")
    logger.info(f"  Veto Rate: {result['veto_rate']}")
    logger.info(f"  Fidelity: {result['fidelity']}")

    # Display pilot snapshot
    logger.info("")
    factory = PilotFactory()
    factory.display_wave3_snapshot()

    return result


async def demo_china_factory():
    """Demonstrate China Photonic Factory integration."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("DEMO: China Photonic Factory Integration")
    logger.info("=" * 70)
    logger.info("")

    # Create configuration
    config = OrchestrationConfig(
        parallel=False,
        tasks=["china_photonic_scale"],
        mode=ExecutionMode.LIVE_ACCELERATED,
        china_enabled=True,
    )

    # Create orchestrator
    orchestrator = QuNimbusOrchestrator(config)

    # Execute China factory integration
    result = await orchestrator.execute_china_photonic_scale()

    logger.info("")
    logger.info("China Factory Result:")
    logger.info(f"  Status: {result['status']}")
    logger.info(f"  Pilots Generated: {result['pilots_generated']}")
    logger.info(f"  Capacity: {result['capacity']}")
    logger.info(f"  QKD Latency: {result['qkd_latency_ms']} ms")
    logger.info(f"  Compliance: {result['compliance']}")
    logger.info(f"  Uptime: {result['uptime']}")

    return result


async def demo_parallel_execution():
    """Demonstrate parallel execution of both tasks."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("DEMO: Parallel Execution (Wave 3 + China Factory)")
    logger.info("=" * 70)
    logger.info("")

    # Create configuration for parallel execution
    config = OrchestrationConfig(
        parallel=True,
        tasks=["wave3_launch", "china_photonic_scale"],
        auth="cac://quantum.lead@akron.us",
        compliance=[
            ComplianceFramework.CMMC_L2,
            ComplianceFramework.DO_178C,
            ComplianceFramework.ISO_13485,
            ComplianceFramework.CHINA_MLPS,
        ],
        mode=ExecutionMode.LIVE_ACCELERATED,
        pilot_target=1000,
        china_enabled=True,
    )

    # Create orchestrator
    orchestrator = QuNimbusOrchestrator(config)

    # Execute parallel orchestration
    result = await orchestrator.orchestrate()

    logger.info("")
    logger.info("Parallel Execution Result:")
    logger.info(f"  Status: {result['status']}")
    logger.info(f"  Mode: {result['mode']}")
    logger.info(f"  Combined Pilots/Day: {result.get('combined_pilots_per_day', 'N/A')}")
    logger.info(f"  Total Value Unlocked: {result.get('total_value_unlocked', 'N/A')}")

    # Display metrics
    logger.info("")
    metrics = orchestrator.get_metrics()
    logger.info("Final Metrics:")
    logger.info(f"  Total Pilots: {metrics.pilots_generated}")
    logger.info(f"  Akron Pilots: {metrics.akron_pilots}")
    logger.info(f"  China Pilots: {metrics.china_pilots}")
    logger.info(f"  Efficiency: {metrics.efficiency_multiplier}×")
    logger.info(f"  MERA Compression: {metrics.mera_compression}×")
    logger.info(f"  RL Convergence: {metrics.rl_convergence}%")
    logger.info(f"  Veto Rate: {metrics.veto_rate * 100:.1f}%")
    logger.info(f"  QKD Latency: {metrics.qkd_latency_ms} ms")
    logger.info(f"  Avg Fidelity: {metrics.fidelity_avg}")

    return result


async def demo_pilot_generation():
    """Demonstrate pilot generation."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("DEMO: Pilot Generation")
    logger.info("=" * 70)
    logger.info("")

    factory = PilotFactory(target_per_day=1000, veto_rate=0.008)

    # Generate a batch of pilots
    logger.info("Generating 100 pilots...")
    pilots = factory.generate_batch(count=100)

    logger.info("")
    logger.info(f"Generated {len(pilots)} pilots")
    logger.info("")

    # Display statistics
    stats = factory.get_stats()
    logger.info("Factory Statistics:")
    logger.info(f"  Total Pilots: {stats['pilots_generated']}")
    logger.info(f"  Vetoes: {stats['vetoes']}")
    logger.info(f"  Veto Rate: {stats['veto_rate'] * 100:.2f}%")
    logger.info(f"  Target/Day: {stats['target_per_day']}")

    # Show some example pilots
    logger.info("")
    logger.info("Sample Pilots:")
    for i, pilot in enumerate(pilots[:5], 1):
        logger.info(
            f"  {i}. {pilot.vertical} - {pilot.workload} "
            f"({pilot.runtime_s}s, {pilot.fidelity} fidelity, {pilot.backend})"
        )

    return pilots


async def demo_china_factory_standalone():
    """Demonstrate China factory standalone operations."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("DEMO: China Factory Standalone")
    logger.info("=" * 70)
    logger.info("")

    factory = ChinaPhotonicFactory()

    # Connect to factory
    logger.info("Connecting to China Photonic Factory...")
    factory.connect()

    # Display dashboard
    logger.info("")
    factory.display_dashboard()

    # Generate pilots
    logger.info("")
    logger.info("Generating 50 pilots...")
    result = factory.generate_pilots(count=50)
    logger.info(f"Result: {result}")

    # Get compliance status
    logger.info("")
    logger.info("Compliance Status:")
    compliance = factory.get_compliance_status()
    logger.info(f"  China MLPS: {compliance['china_standards']['status']}")
    logger.info(f"  US CMMC L2: {compliance['us_bridge']['cmmc_l2']}")
    logger.info(f"  QKD Security: {compliance['qkd_security']['security_level']}")

    # Disconnect
    logger.info("")
    factory.disconnect()

    return result


async def main():
    """Run all demos."""
    logger.info("=" * 70)
    logger.info("QuNimbus Wave 3 Orchestration Demo")
    logger.info("=" * 70)
    logger.info("")

    # Run individual demos
    await demo_wave3_launch()
    await demo_china_factory()
    await demo_pilot_generation()
    await demo_china_factory_standalone()

    # Run parallel execution demo
    await demo_parallel_execution()

    logger.info("")
    logger.info("=" * 70)
    logger.info("Demo Complete!")
    logger.info("=" * 70)
    logger.info("")
    logger.info("To use the CLI:")
    logger.info("  qunimbus orchestrate --parallel \\")
    logger.info('    --task "wave3_launch" \\')
    logger.info('    --task "china_photonic_scale" \\')
    logger.info('    --auth "cac://quantum.lead@akron.us" \\')
    logger.info('    --compliance "CMMC-L2,DO-178C,ISO-13485,China-MLPS" \\')
    logger.info('    --mode "live_accelerated"')


if __name__ == "__main__":
    asyncio.run(main())
