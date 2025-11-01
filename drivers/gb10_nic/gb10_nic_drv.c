// SPDX-License-Identifier: Apache-2.0
#include <linux/ethtool.h>
#include <linux/module.h>
#include <linux/netdevice.h>
#include "../include/gb10_drv_common.h"

static netdev_tx_t gb10_nic_start_xmit(struct sk_buff *skb, struct net_device *dev)
{
    dev->stats.tx_packets++;
    dev->stats.tx_bytes += skb->len;
    dev_kfree_skb(skb);
    return NETDEV_TX_OK;
}

static int gb10_nic_open(struct net_device *dev)
{
    netif_start_queue(dev);
    return 0;
}

static int gb10_nic_stop(struct net_device *dev)
{
    netif_stop_queue(dev);
    return 0;
}

static const struct net_device_ops gb10_nic_ops = {
    .ndo_open = gb10_nic_open,
    .ndo_stop = gb10_nic_stop,
    .ndo_start_xmit = gb10_nic_start_xmit,
};

static void gb10_setup(struct net_device *dev)
{
    ether_setup(dev);
    dev->netdev_ops = &gb10_nic_ops;
    dev->min_mtu = 1500;
    dev->max_mtu = 9000;
}

static struct net_device *gb10_netdev;

static int __init gb10_nic_init(void)
{
    int ret;

    gb10_netdev = alloc_netdev(0, "gb10nic%d", NET_NAME_UNKNOWN, gb10_setup);
    if (!gb10_netdev)
        return -ENOMEM;
    ret = register_netdev(gb10_netdev);
    if (ret)
        free_netdev(gb10_netdev);
    pr_info("GB10 NIC driver registered\n");
    return ret;
}

static void __exit gb10_nic_exit(void)
{
    unregister_netdev(gb10_netdev);
    free_netdev(gb10_netdev);
    pr_info("GB10 NIC driver unloaded\n");
}

module_init(gb10_nic_init);
module_exit(gb10_nic_exit);

MODULE_LICENSE("Apache-2.0");
MODULE_AUTHOR("GB10 Project");
MODULE_DESCRIPTION("Synthetic GB10 NIC driver");
