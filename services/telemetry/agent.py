"""Telemetry agent emitting ORD metrics for QuASIM×QuNimbus Phase VI."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import random
import re
import time
from typing import Iterable, Iterator, Sequence

from prometheus_client import Gauge, start_http_server

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class RegionConfig:
    name: str
    racks: int
    base_energy_kwh: float
    qevf_target_ops_per_kwh: float
    entanglement_efficiency: float
    mera_compression: float

    @property
    def baseline_throughput(self) -> float:
        return self.qevf_target_ops_per_kwh * self.entanglement_efficiency


class TelemetryAgent:
    """Simulates ORD telemetry using deterministic seeds."""

    METRIC_NAMES = {
        "qevf_ops_per_kwh": "Φ_QEVF operations per kWh",
        "energy_kwh": "Energy consumed in kWh",
        "fidelity": "Quantum circuit fidelity",
        "throughput": "Processed operations per second",
        "entanglement_yield": "Effective entanglement yield",
    }

    def __init__(
        self,
        regions: Iterable[RegionConfig],
        seed: int,
        collection_interval_s: float = 15.0,
    ) -> None:
        self._regions: list[RegionConfig] = list(regions)
        self._seed = seed
        self._rng = random.Random(seed)
        self._interval = collection_interval_s
        self._gauges: dict[str, Gauge] = {
            metric: Gauge(f"ord_{metric}", description, labelnames=("region",))
            for metric, description in self.METRIC_NAMES.items()
        }
        LOGGER.debug("TelemetryAgent initialized for %d regions", len(self._regions))

    def _simulate_tick(self, region: RegionConfig, t: float) -> dict[str, float]:
        phase = math.sin((t / 900.0) + (hash(region.name) % 10))
        noise = (self._rng.random() - 0.5) * 0.02
        qevf = region.qevf_target_ops_per_kwh * (1.0 + 0.01 * phase + noise)
        energy = region.base_energy_kwh * (1.0 + 0.005 * phase)
        fidelity = max(0.0, min(1.0, 0.998 + 0.0005 * phase + noise))
        throughput = region.baseline_throughput * (1.0 + 0.02 * phase)
        entanglement_yield = region.entanglement_efficiency * (1.0 + 0.01 * phase)
        return {
            "qevf_ops_per_kwh": qevf,
            "energy_kwh": energy,
            "fidelity": fidelity,
            "throughput": throughput,
            "entanglement_yield": entanglement_yield,
        }

    def samples(self) -> Iterator[dict[str, float]]:
        start = time.time()
        while True:
            t = time.time() - start
            for region in self._regions:
                metrics = self._simulate_tick(region, t)
                LOGGER.debug("Metrics for region %s: %s", region.name, metrics)
                yield {f"{key}:{region.name}": value for key, value in metrics.items()}
            time.sleep(self._interval)

    def serve_forever(self, duration_s: float | None = None) -> None:
        start = time.time()
        for metrics in self.samples():
            for metric_key, value in metrics.items():
                metric_name, region = metric_key.split(":", maxsplit=1)
                self._gauges[metric_name].labels(region=region).set(value)
            if duration_s is not None and (time.time() - start) >= duration_s:
                LOGGER.info("Telemetry agent duration reached %.2fs", duration_s)
                break


def agent_from_inputs(
    ord_hours: int,
    racks: int,
    energy_price_usd_per_kwh: float,
    entanglement_efficiency: float,
    mera_compression: float,
    qevf_target_ops_per_kwh: float,
    regions: Sequence[str],
    seed: int = 424242,
    collection_interval_s: float = 15.0,
) -> TelemetryAgent:
    base_energy_per_rack = (ord_hours / 72) * 10.0
    region_configs = [
        RegionConfig(
            name=region,
            racks=racks // max(len(regions), 1),
            base_energy_kwh=base_energy_per_rack,
            qevf_target_ops_per_kwh=qevf_target_ops_per_kwh,
            entanglement_efficiency=entanglement_efficiency,
            mera_compression=mera_compression,
        )
        for region in regions
    ]
    LOGGER.info(
        "Creating telemetry agent for %d regions, %d racks, seed=%d",
        len(region_configs),
        racks,
        seed,
    )
    return TelemetryAgent(region_configs, seed=seed, collection_interval_s=collection_interval_s)


def parse_duration(duration: str) -> float:
    match = re.fullmatch(r"(?i)(\d+)([smhd])", duration.strip())
    if not match:
        raise ValueError(f"Unsupported duration format: {duration}")
    value = int(match.group(1))
    unit = match.group(2).lower()
    factors = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    return value * factors[unit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ORD telemetry agent")
    parser.add_argument("--port", type=int, default=9000, help="Prometheus metrics port")
    parser.add_argument("--collection-interval", type=float, default=15.0)
    parser.add_argument("--duration", type=str, default="72h")
    parser.add_argument("--seed", type=int, default=424242)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    agent = agent_from_inputs(
        ord_hours=72,
        racks=60,
        energy_price_usd_per_kwh=0.11,
        entanglement_efficiency=0.93,
        mera_compression=114.6,
        qevf_target_ops_per_kwh=1.0e17,
        regions=("us-east", "us-west", "eu-central", "ap-sg"),
        seed=args.seed,
        collection_interval_s=args.collection_interval,
    )
    duration_s = parse_duration(args.duration) if args.duration else None
    start_http_server(args.port)
    LOGGER.info("Starting telemetry agent event loop on port %d", args.port)
    try:
        agent.serve_forever(duration_s=duration_s)
    except KeyboardInterrupt:
        LOGGER.info("Telemetry agent shutdown requested")


if __name__ == "__main__":
    main()
