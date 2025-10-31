// SPDX-License-Identifier: Apache-2.0
#ifndef GB10_FW_H
#define GB10_FW_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GB10_LOG_INFO,
    GB10_LOG_WARN,
    GB10_LOG_ERROR
} gb10_log_level_t;

void gb10_log(gb10_log_level_t level, const char *fmt, ...);
void gb10_init_uart(void);
void gb10_load_dvfs_tables(void);

#ifdef __cplusplus
}
#endif

#endif // GB10_FW_H
