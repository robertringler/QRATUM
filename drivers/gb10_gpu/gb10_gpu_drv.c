// SPDX-License-Identifier: Apache-2.0
#include <linux/fs.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/uaccess.h>
#include "../include/gb10_drv_common.h"

#define DEVICE_NAME "gb10_gpu"

static DEFINE_MUTEX(gb10_gpu_lock);
static u64 tensor_counter;

static ssize_t gb10_gpu_read(struct file *file, char __user *buf, size_t len, loff_t *off)
{
    char buffer[64];
    int written;

    mutex_lock(&gb10_gpu_lock);
    tensor_counter += 128;
    written = scnprintf(buffer, sizeof(buffer), "%llu\n", tensor_counter);
    mutex_unlock(&gb10_gpu_lock);

    if (*off >= written)
        return 0;
    if (len < written)
        return -EINVAL;
    if (copy_to_user(buf, buffer, written))
        return -EFAULT;
    *off += written;
    return written;
}

static const struct file_operations gb10_gpu_fops = {
    .owner = THIS_MODULE,
    .read = gb10_gpu_read,
};

static int gb10_gpu_major;

static int __init gb10_gpu_init(void)
{
    gb10_gpu_major = register_chrdev(0, DEVICE_NAME, &gb10_gpu_fops);
    if (gb10_gpu_major < 0)
        return gb10_gpu_major;
    pr_info("GB10 GPU driver loaded on major %d\n", gb10_gpu_major);
    return 0;
}

static void __exit gb10_gpu_exit(void)
{
    unregister_chrdev(gb10_gpu_major, DEVICE_NAME);
    pr_info("GB10 GPU driver unloaded\n");
}

module_init(gb10_gpu_init);
module_exit(gb10_gpu_exit);

MODULE_LICENSE("Apache-2.0");
MODULE_AUTHOR("GB10 Project");
MODULE_DESCRIPTION("Synthetic GB10 GPU driver");
