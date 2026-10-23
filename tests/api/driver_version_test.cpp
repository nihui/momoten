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
    const std::string expected_device_name = "momoten: " + physical.name;
    if (strcmp(properties.deviceName, expected_device_name.c_str()) != 0)
    {
        fprintf(stderr, "driver_version_test: deviceName does not preserve the OpenCL device name\n");
        return 1;
    }
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
    if (driver.driverID != static_cast<VkDriverId>(34))
    {
        fprintf(stderr, "driver_version_test: driverID does not report VK_DRIVER_ID_MOMOTEN (34)\n");
        return 1;
    }
    if (properties2.properties.driverVersion != expected_version
        || properties2.properties.driverVersion != properties.driverVersion
        || strcmp(properties2.properties.deviceName, properties.deviceName) != 0)
    {
        fprintf(stderr, "driver_version_test: Properties2 reports a different driver or device identity\n");
        return 1;
    }
    const std::string expected_info = "momoten " MOMOTEN_RELEASE_VERSION_STRING "; OpenCL backend: " + physical.driver;
    if (strcmp(driver.driverName, "momoten") != 0
        || driver.driverInfo[VK_MAX_DRIVER_INFO_SIZE - 1] != '\0'
        || strcmp(driver.driverInfo, expected_info.c_str()) != 0
        || driver.conformanceVersion.major != 0
        || driver.conformanceVersion.minor != 0
        || driver.conformanceVersion.subminor != 0
        || driver.conformanceVersion.patch != 0)
    {
        fprintf(stderr, "driver_version_test: concise driver identity or zero-conformance status is inconsistent\n");
        return 1;
    }

    physical.name.assign(VK_MAX_PHYSICAL_DEVICE_NAME_SIZE * 2, 'x');
    physical.driver.assign(VK_MAX_DRIVER_INFO_SIZE * 2, 'x');
    memset(driver.driverInfo, '!', sizeof(driver.driverInfo));
    momoten_detail::impl_get_physical_device_properties2(&physical, &properties2);
    if (properties2.properties.deviceName[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE - 1] != '\0'
        || strlen(properties2.properties.deviceName) != VK_MAX_PHYSICAL_DEVICE_NAME_SIZE - 1
        || strncmp(properties2.properties.deviceName, "momoten: ", 9) != 0
        || driver.driverInfo[VK_MAX_DRIVER_INFO_SIZE - 1] != '\0'
        || strlen(driver.driverInfo) != VK_MAX_DRIVER_INFO_SIZE - 1
        || !strstr(driver.driverInfo, MOMOTEN_RELEASE_VERSION_STRING)
        || !strstr(driver.driverInfo, "OpenCL backend: "))
    {
        fprintf(stderr, "driver_version_test: truncated deviceName or driverInfo loses its identity or terminator\n");
        return 1;
    }

    return 0;
}
