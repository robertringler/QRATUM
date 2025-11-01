// SPDX-License-Identifier: Apache-2.0
#include "gb10_fw.h"
#include <stdio.h>

typedef struct {
    uint32_t freq_mhz;
    uint32_t voltage_mv;
} dvfs_entry_t;

static dvfs_entry_t cpu_pstates[] = {
    {800, 600},
    {1200, 650},
    {1600, 700},
    {2000, 760},
    {2400, 820},
    {2800, 900},
    {3000, 980},
    {3200, 1100},
};

void gb10_load_dvfs_tables(void) {
    for (size_t i = 0; i < sizeof(cpu_pstates)/sizeof(cpu_pstates[0]); ++i) {
        gb10_log(GB10_LOG_INFO, "DVFS: CPU P%zu -> %u MHz @ %u mV\n", i, cpu_pstates[i].freq_mhz, cpu_pstates[i].voltage_mv);
    }
}
