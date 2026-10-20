// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "backend/device.h"
#include "backend/objects.h"
#include "momoten/version.h"

#include <stdio.h>
#include <string.h>

// Property queries only read the supplied physical-device description. Keep
// unrelated discovery entry points linkable without loading an OpenCL runtime.
const std::vector<VkExtensionProperties>& instance_extensions()
{
    static const std::vector<VkExtensionProperties> empty;
    return empty;
}

std::vector<VkPhysicalDevice> discover_opencl_devices(VkInstance)
{
    return std::vector<VkPhysicalDevice>();
}

std::vector<VkExtensionProperties> device_extensions(VkPhysicalDevice)
{
    return std::vector<VkExtensionProperties>();
}

namespace momoten_detail {

OpenCLApi g_opencl = {};

bool debug_enabled()
{
    return false;
}

} // namespace momoten_detail

int main()
{
    VkPhysicalDevice_T physical = {};
    physical.name = "device-independent version test";
    physical.driver = "test OpenCL backend";
    physical.address_bits = 64;

    const uint32_t release = MOMOTEN_RELEASE_VERSION;
    const uint32_t expected_version = VK_MAKE_VERSION(
        release / 10000u - 2000u, release / 100u % 100u, release % 100u);
    VkPhysicalDeviceProperties properties = {};
    momoten_detail::impl_get_physical_device_properties(&physical, &properties);
    if (properties.driverVersion != expected_version)
    {
        fprintf(stderr, "driver_version_test: driverVersion does not encode the release date\n");
        return 1;
    }

    VkPhysicalDeviceDriverProperties driver = {};
    driver.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    VkPhysicalDeviceProperties2 properties2 = {};
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties2.pNext = &driver;
    memset(driver.driverInfo, '!', sizeof(driver.driverInfo));
    momoten_detail::impl_get_physical_device_properties2(&physical, &properties2);
    if (properties2.properties.driverVersion != expected_version
        || properties2.properties.driverVersion != properties.driverVersion)
    {
        fprintf(stderr, "driver_version_test: Properties2 reports a different driverVersion\n");
        return 1;
    }
    if (driver.driverInfo[VK_MAX_DRIVER_INFO_SIZE - 1] != '\0'
        || !strstr(driver.driverInfo, MOMOTEN_RELEASE_VERSION_STRING)
        || !strstr(driver.driverInfo, physical.driver.c_str()))
    {
        fprintf(stderr, "driver_version_test: driverInfo is missing the release date or backend\n");
        return 1;
    }

    physical.driver.assign(VK_MAX_DRIVER_INFO_SIZE * 2, 'x');
    memset(driver.driverInfo, '!', sizeof(driver.driverInfo));
    momoten_detail::impl_get_physical_device_properties2(&physical, &properties2);
    if (driver.driverInfo[VK_MAX_DRIVER_INFO_SIZE - 1] != '\0'
        || strlen(driver.driverInfo) != VK_MAX_DRIVER_INFO_SIZE - 1
        || !strstr(driver.driverInfo, MOMOTEN_RELEASE_VERSION_STRING)
        || !strstr(driver.driverInfo, "OpenCL backend: "))
    {
        fprintf(stderr, "driver_version_test: truncated driverInfo loses its date or terminator\n");
        return 1;
    }

    return 0;
}
