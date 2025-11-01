// SPDX-License-Identifier: Apache-2.0
#include <linux/blkdev.h>
#include <linux/crypto.h>
#include <linux/module.h>
#include "../include/gb10_drv_common.h"

static int gb10_nvme_major;

static int gb10_nvme_open(struct block_device *bdev, fmode_t mode)
{
    return 0;
}

static void gb10_nvme_release(struct gendisk *disk, fmode_t mode)
{
}

static const struct block_device_operations gb10_nvme_fops = {
    .owner = THIS_MODULE,
    .open = gb10_nvme_open,
    .release = gb10_nvme_release,
};

static struct gendisk *gb10_disk;

static int __init gb10_nvme_init(void)
{
    gb10_nvme_major = register_blkdev(0, "gb10nvme");
    if (gb10_nvme_major <= 0)
        return gb10_nvme_major;

    gb10_disk = alloc_disk(1);
    if (!gb10_disk) {
        unregister_blkdev(gb10_nvme_major, "gb10nvme");
        return -ENOMEM;
    }

    gb10_disk->major = gb10_nvme_major;
    gb10_disk->first_minor = 0;
    gb10_disk->fops = &gb10_nvme_fops;
    snprintf(gb10_disk->disk_name, sizeof(gb10_disk->disk_name), "gb10nvme0");
    set_capacity(gb10_disk, 1024);
    add_disk(gb10_disk);
    pr_info("GB10 NVMe driver registered\n");
    return 0;
}

static void __exit gb10_nvme_exit(void)
{
    del_gendisk(gb10_disk);
    put_disk(gb10_disk);
    unregister_blkdev(gb10_nvme_major, "gb10nvme");
    pr_info("GB10 NVMe driver unloaded\n");
}

module_init(gb10_nvme_init);
module_exit(gb10_nvme_exit);

MODULE_LICENSE("Apache-2.0");
MODULE_AUTHOR("GB10 Project");
MODULE_DESCRIPTION("Synthetic GB10 NVMe driver");
