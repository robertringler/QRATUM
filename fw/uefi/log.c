// SPDX-License-Identifier: Apache-2.0
#include "gb10_fw.h"
#include <stdarg.h>
#include <stdio.h>

static const char *level_to_string(gb10_log_level_t level) {
    switch (level) {
    case GB10_LOG_WARN:
        return "WARN";
    case GB10_LOG_ERROR:
        return "ERROR";
    case GB10_LOG_INFO:
    default:
        return "INFO";
    }
}

void gb10_init_uart(void) {
    // In the reference platform we simply ensure stdio is ready.
    setvbuf(stdout, NULL, _IONBF, 0);
}

void gb10_log(gb10_log_level_t level, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    printf("[%s] ", level_to_string(level));
    vprintf(fmt, ap);
    va_end(ap);
}
