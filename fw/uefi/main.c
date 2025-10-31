// SPDX-License-Identifier: Apache-2.0
#include "gb10_fw.h"
#include <stdio.h>

static void enumerate_devices(void) {
    gb10_log(GB10_LOG_INFO, "Firmware: Initializing NVLink and PCIe fabric\n");
    gb10_log(GB10_LOG_INFO, "Firmware: ConnectX-7 NIC detected\n");
    gb10_log(GB10_LOG_INFO, "Firmware: NVMe controller ready\n");
}

void firmware_main(void) {
    gb10_log(GB10_LOG_INFO, "Firmware: Starting platform initialization\n");
    gb10_load_dvfs_tables();
    enumerate_devices();
    gb10_log(GB10_LOG_INFO, "Firmware: Boot complete, handoff to OS\n");
}
