/* SPDX-License-Identifier: Apache-2.0 */
#ifndef GB10_DRV_COMMON_H
#define GB10_DRV_COMMON_H

#include <linux/device.h>
#include <linux/module.h>

struct gb10_telemetry {
    u64 perf_counter;
    u64 temperature_mdeg;
};

int gb10_register_device(struct device *dev, const char *name);
void gb10_unregister_device(struct device *dev, const char *name);

#endif /* GB10_DRV_COMMON_H */
