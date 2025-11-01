# Power and Thermal Management

The GB10 platform applies adaptive DVFS across CPU, GPU, and NIC domains. Thermal telemetry flows through an I2C-attached sensor array aggregated by the firmware and exposed via the `gb10_cpu` driver.

* **P-states:** Eight voltage/frequency operating points for the CPU complex ranging 0.6 V @ 800 MHz to 1.1 V @ 3.2 GHz.
* **G-states:** GPU performance states aligned with tensor workloads, with FP8 boost bins and RT core prioritization.
* **Cooling:** A dual-stage vapor chamber heatsink with PID-controlled fans (up to 6000 RPM) maintaining ≤ 85 °C GPU temperature.

Power budgeting is orchestrated by firmware tables in `fw/uefi/power_table.c` and enforced at runtime by a Linux power governor module. Telemetry counters are exported via sysfs and ingested by the profiler in `sdk/profiler/gb10_profiler.py`.
