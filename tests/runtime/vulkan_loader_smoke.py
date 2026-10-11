#!/usr/bin/env python3
# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

"""Validate the ICD ABI and exercise discovery using the system Vulkan Loader."""

import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: vulkan_loader_smoke.py LIBRARY MANIFEST")

    library = Path(sys.argv[1]).resolve()
    manifest = Path(sys.argv[2]).resolve()
    info = json.loads(manifest.read_text())
    assert info["file_format_version"] == "1.0.0"
    assert info["ICD"]["api_version"] == "1.1.0"
    assert (manifest.parent / info["ICD"]["library_path"]).resolve() == library

    driver = ctypes.CDLL(str(library))
    negotiate = driver.vk_icdNegotiateLoaderICDInterfaceVersion
    negotiate.restype = ctypes.c_int32
    negotiate.argtypes = [ctypes.POINTER(ctypes.c_uint32)]

    version = ctypes.c_uint32(7)
    assert negotiate(ctypes.byref(version)) == 0 and version.value == 5
    version.value = 5
    assert negotiate(ctypes.byref(version)) == 0 and version.value == 5
    version.value = 4
    assert negotiate(ctypes.byref(version)) != 0
    for name in ("vk_icdGetInstanceProcAddr", "vk_icdGetPhysicalDeviceProcAddr",
                 "vkGetInstanceProcAddr", "vkCreateInstance"):
        assert getattr(driver, name, None) is not None, name

    # Use the *system* loader, not a direct call to the ICD. A machine with
    # no eligible OpenCL device can legitimately reject instance creation.
    script = """
import ctypes
import faulthandler
faulthandler.enable()
print('loading libvulkan.so.1',flush=True)
v = ctypes.CDLL('libvulkan.so.1')
print('loaded system Vulkan loader',flush=True)
class App(ctypes.Structure):
    _fields_=[('sType',ctypes.c_int),('pNext',ctypes.c_void_p),
              ('pApplicationName',ctypes.c_char_p),('applicationVersion',ctypes.c_uint32),
              ('pEngineName',ctypes.c_char_p),('engineVersion',ctypes.c_uint32),
              ('apiVersion',ctypes.c_uint32)]
class Create(ctypes.Structure):
    _fields_=[('sType',ctypes.c_int),('pNext',ctypes.c_void_p),
              ('flags',ctypes.c_uint32),('pApplicationInfo',ctypes.POINTER(App)),
              ('enabledLayerCount',ctypes.c_uint32),('ppEnabledLayerNames',ctypes.c_void_p),
              ('enabledExtensionCount',ctypes.c_uint32),('ppEnabledExtensionNames',ctypes.c_void_p)]
app=App(0,None,b'momoten-loader-smoke',0,None,0,(1<<22)|(1<<12))
ci=Create(1,None,0,ctypes.pointer(app),0,None,0,None)
instance=ctypes.c_void_p()
fn=v.vkCreateInstance
fn.argtypes=[ctypes.POINTER(Create),ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p)]
fn.restype=ctypes.c_int32
print('calling vkCreateInstance',flush=True)
ret=fn(ctypes.byref(ci),None,ctypes.byref(instance))
print('Vulkan Loader vkCreateInstance returned',ret)
assert ret == 0, 'Loader failed to create a Vulkan instance'
print('created Vulkan instance; enumerating physical devices',flush=True)
count=ctypes.c_uint32()
enum=v.vkEnumeratePhysicalDevices
enum.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint32),ctypes.c_void_p]
enum.restype=ctypes.c_int32
assert enum(instance,ctypes.byref(count),None) == 0
print('Physical device count:',count.value,flush=True)
assert count.value > 0, 'PoCL device not exposed through Vulkan Loader'
# Exercise the system loader's vkCreateDevice chain, not just ICD discovery.
physical=(ctypes.c_void_p*count.value)()
assert enum(instance,ctypes.byref(count),ctypes.cast(physical,ctypes.c_void_p)) == 0
class QueueCreate(ctypes.Structure):
    _fields_=[('sType',ctypes.c_int),('pNext',ctypes.c_void_p),
              ('flags',ctypes.c_uint32),('queueFamilyIndex',ctypes.c_uint32),
              ('queueCount',ctypes.c_uint32),('pQueuePriorities',ctypes.POINTER(ctypes.c_float))]
class DeviceCreate(ctypes.Structure):
    _fields_=[('sType',ctypes.c_int),('pNext',ctypes.c_void_p),
              ('flags',ctypes.c_uint32),('queueCreateInfoCount',ctypes.c_uint32),
              ('pQueueCreateInfos',ctypes.POINTER(QueueCreate)),
              ('enabledLayerCount',ctypes.c_uint32),('ppEnabledLayerNames',ctypes.c_void_p),
              ('enabledExtensionCount',ctypes.c_uint32),('ppEnabledExtensionNames',ctypes.c_void_p),
              ('pEnabledFeatures',ctypes.c_void_p)]
priority=ctypes.c_float(1.0)
queue=QueueCreate(2,None,0,0,1,ctypes.pointer(priority))
device_info=DeviceCreate(3,None,0,1,ctypes.pointer(queue),0,None,0,None,None)
device=ctypes.c_void_p()
create_device=v.vkCreateDevice
create_device.argtypes=[ctypes.c_void_p,ctypes.POINTER(DeviceCreate),
                        ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p)]
create_device.restype=ctypes.c_int32
ret=create_device(physical[0],ctypes.byref(device_info),None,ctypes.byref(device))
print('Vulkan Loader vkCreateDevice returned',ret,flush=True)
assert ret == 0, 'Loader failed to create Vulkan device'
v.vkDestroyDevice.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
v.vkDestroyDevice(device,None)
print('destroyed Vulkan device',flush=True)
print('destroying Vulkan instance',flush=True)
v.vkDestroyInstance.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
v.vkDestroyInstance(instance,None)
print('destroyed Vulkan instance',flush=True)
"""
    env = os.environ.copy()
    env["VK_DRIVER_FILES"] = str(manifest)
    env["VK_ICD_FILENAMES"] = str(manifest)  # compatibility with older loaders
    env["VK_LOADER_DEBUG"] = "driver"
    result = subprocess.run([sys.executable, "-u", "-X", "faulthandler", "-c", script], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, timeout=30)
    print(result.stdout, flush=True)
    print(result.stderr, flush=True)
    assert result.returncode == 0, f"system Vulkan Loader process failed (exit {result.returncode})"
    assert manifest.name in result.stderr, "system loader did not discover momoten ICD"


if __name__ == "__main__":
    main()
