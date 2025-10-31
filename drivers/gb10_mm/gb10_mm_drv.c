// SPDX-License-Identifier: Apache-2.0
#include <linux/module.h>
#include <linux/mm.h>
#include <linux/slab.h>
#include "../include/gb10_drv_common.h"

static struct kmem_cache *gb10_page_cache;

static int gb10_mm_init_pool(void)
{
    gb10_page_cache = kmem_cache_create("gb10_mm", PAGE_SIZE, 0, SLAB_HWCACHE_ALIGN, NULL);
    if (!gb10_page_cache)
        return -ENOMEM;
    return 0;
}

static void gb10_mm_destroy_pool(void)
{
    if (gb10_page_cache)
        kmem_cache_destroy(gb10_page_cache);
}

static int __init gb10_mm_init(void)
{
    int ret = gb10_mm_init_pool();
    if (ret)
        return ret;
    pr_info("GB10 MMU driver ready with pooled allocator\n");
    return 0;
}

static void __exit gb10_mm_exit(void)
{
    gb10_mm_destroy_pool();
    pr_info("GB10 MMU driver unloaded\n");
}

module_init(gb10_mm_init);
module_exit(gb10_mm_exit);

MODULE_LICENSE("Apache-2.0");
MODULE_AUTHOR("GB10 Project");
MODULE_DESCRIPTION("Synthetic GB10 unified memory driver");
