# Firmware Boot Flow

1. **Boot ROM** — Authenticity verification of the firmware capsule using RSA-4096 and SHA-384 digest.
2. **LPDDR5x Training** — Firmware calibrates DQ/DQS timing with on-chip BIST before enabling the unified memory controller.
3. **PCIe & NVLink Bring-up** — Enumerate the ConnectX-7 and NVMe controllers, initialize NVLink-C2C coherence, and expose BARs.
4. **Thermal & Power Init** — Load DVFS tables, calibrate thermal sensors, and start the fan controller PID loop.
5. **UEFI Services** — Publish a device tree and runtime services to the operating system loader. The firmware also exposes logging over USB-C with CDC ACM support.
6. **Kernel Handoff** — Transfer control to the DGX OS kernel with memory map descriptors for unified CPU/GPU access.

The boot firmware is implemented in portable C with a minimal hardware abstraction layer in `fw/include/`. Logging macros allow developers to trace boot stages when running under QEMU.
