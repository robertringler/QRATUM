// SPDX-License-Identifier: Apache-2.0
#include "gb10_fw.h"
#include <stdio.h>
#include <string.h>

static const uint8_t firmware_signature[] = {0x42, 0x10, 0xDE, 0xAD};

static int verify_firmware(const uint8_t *image, size_t size) {
    if (size < sizeof(firmware_signature)) {
        return -1;
    }
    return memcmp(image, firmware_signature, sizeof(firmware_signature));
}

int bootrom_entry(const uint8_t *image, size_t size) {
    gb10_init_uart();
    gb10_log(GB10_LOG_INFO, "BootROM: Starting secure boot checks\n");
    if (verify_firmware(image, size) != 0) {
        gb10_log(GB10_LOG_ERROR, "BootROM: Firmware signature invalid\n");
        return -1;
    }
    gb10_log(GB10_LOG_INFO, "BootROM: Verification successful, jumping to firmware\n");
    return 0;
}
