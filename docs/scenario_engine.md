# Scenario Engine

The scenario engine introduces deterministic Scenario and Timeline abstractions. A Scenario binds:

- **Configuration**: domains, parameters, policies
- **Timeline**: ordered logical ticks containing typed events
- **Drivers**: deterministic hooks into QuASIM simulations and QuNimbus governance

## Timeline and Events

Events are grouped by tick and streamed in sorted order. Supported event flavors include system, market, mission, and node events. There is no wall-clock time; only logical ticks advance.

## Execution Flow

1. Build a Scenario with a Timeline and drivers.
2. Policies are applied deterministically per event.
3. Drivers execute domain simulators and optionally governance and node hooks.
4. State metrics and incidents are accumulated.

## Outcomes and Reporting

Outcomes are evaluated via required metrics and incidents. The ScenarioReport captures configuration, event sequence, metrics, incidents, and a plain-language narrative for operators.
