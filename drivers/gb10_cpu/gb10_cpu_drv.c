// SPDX-License-Identifier: Apache-2.0
#include <linux/hwmon.h>
#include <linux/module.h>
#include "../include/gb10_drv_common.h"

static struct gb10_telemetry telemetry = {
    .perf_counter = 1000,
    .temperature_mdeg = 65000,
};

static ssize_t perf_show(struct device *dev, struct device_attribute *attr, char *buf)
{
    telemetry.perf_counter += 256;
    return sysfs_emit(buf, "%llu\n", telemetry.perf_counter);
}

static ssize_t temp_show(struct device *dev, struct device_attribute *attr, char *buf)
{
    telemetry.temperature_mdeg += 10;
    return sysfs_emit(buf, "%llu\n", telemetry.temperature_mdeg);
}

static DEVICE_ATTR_RO(perf);
static DEVICE_ATTR_RO(temp);

static struct attribute *gb10_cpu_attrs[] = {
    &dev_attr_perf.attr,
    &dev_attr_temp.attr,
    NULL,
};

ATTRIBUTE_GROUPS(gb10_cpu);

static int gb10_cpu_probe(struct platform_device *pdev)
{
    return gb10_register_device(&pdev->dev, "cpu");
}

static int gb10_cpu_remove(struct platform_device *pdev)
{
    gb10_unregister_device(&pdev->dev, "cpu");
    return 0;
}

static struct platform_driver gb10_cpu_driver = {
    .driver = {
        .name = "gb10_cpu",
        .dev_groups = gb10_cpu_groups,
    },
    .probe = gb10_cpu_probe,
    .remove = gb10_cpu_remove,
};

module_platform_driver(gb10_cpu_driver);

MODULE_LICENSE("Apache-2.0");
MODULE_AUTHOR("GB10 Project");
MODULE_DESCRIPTION("Synthetic GB10 CPU driver");
