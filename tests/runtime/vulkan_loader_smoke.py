#!/usr/bin/env python3
# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

"""Validate the ICD ABI and exercise discovery using the system Vulkan Loader."""

import argparse
import ctypes
import json
import os
import platform
from pathlib import Path
import subprocess
import sys


class OpenCLEnumerationError(RuntimeError):
    def __init__(self, result):
        super().__init__(f"clGetDeviceIDs failed: {result}")
        self.result = result


def has_opencl_device(path):
    """Probe the actual backend, so missing hardware cannot hide an ICD failure."""
    cl = ctypes.CDLL(path)  # A missing framework/library is an error, not a skip.
    uint = ctypes.c_uint32
    handle = ctypes.c_void_p
    cl.clGetPlatformIDs.argtypes = [uint, ctypes.POINTER(handle), ctypes.POINTER(uint)]
    cl.clGetPlatformIDs.restype = ctypes.c_int32
    cl.clGetDeviceIDs.argtypes = [handle, ctypes.c_uint64, uint,
                                 ctypes.POINTER(handle), ctypes.POINTER(uint)]
    cl.clGetDeviceIDs.restype = ctypes.c_int32
    cl.clGetDeviceInfo.argtypes = [handle, uint, ctypes.c_size_t,
                                  ctypes.c_void_p, ctypes.c_void_p]
    cl.clGetDeviceInfo.restype = ctypes.c_int32
    count = uint()
    result = cl.clGetPlatformIDs(0, None, ctypes.byref(count))
    if result == -1001:  # CL_PLATFORM_NOT_FOUND_KHR
        return False
    assert result == 0, f"clGetPlatformIDs failed: {result}"
    if count.value == 0:
        return False
    platforms = (handle * count.value)()
    assert cl.clGetPlatformIDs(count, platforms, None) == 0
    compatible = 0
    for platform in platforms:
        count = uint()
        result = cl.clGetDeviceIDs(platform, 0xffffffff, 0, None, ctypes.byref(count))
        if result == -1:  # CL_DEVICE_NOT_FOUND
            continue
        if result != 0:
            raise OpenCLEnumerationError(result)
        if count.value == 0:
            continue
        devices = (handle * count.value)()
        assert cl.clGetDeviceIDs(platform, 0xffffffff, count, devices, None) == 0
        for device in devices:
            # Match momoten's minimum backend requirements, independently of its loader.
            values = []
            for parameter in (0x1027, 0x1028, 0x1026, 0x100d, 0x1003):
                value = uint()
                result = cl.clGetDeviceInfo(device, parameter, ctypes.sizeof(value),
                                            ctypes.byref(value), None)
                assert result == 0, f"clGetDeviceInfo({parameter:#x}) failed: {result}"
                values.append(value.value)
            available, compiler, little_endian, address_bits, dimensions = values
            compatible += bool(available and compiler and little_endian and
                               address_bits in (32, 64) and dimensions >= 3)
    print(f"Compatible OpenCL devices in {path}: {compatible}", flush=True)
    return compatible > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library")
    parser.add_argument("manifest")
    parser.add_argument("--loader", default="libvulkan.so.1",
                        help="system Vulkan Loader library (not the momoten ICD)")
    parser.add_argument("--probe-opencl", metavar="LIBRARY",
                        help="return 77 only if this backend has no compatible devices")
    parser.add_argument("--apple-opencl-unavailable-probe", metavar="EXECUTABLE",
                        help="on hosted macOS ARM64, confirm Apple's -30 error with a native SDK probe")
    args = parser.parse_args()
    if args.apple_opencl_unavailable_probe:
        assert (sys.platform == "darwin" and platform.machine() == "arm64" and
                os.environ.get("GITHUB_ACTIONS") == "true" and
                os.environ.get("RUNNER_ENVIRONMENT") == "github-hosted" and
                os.environ.get("RUNNER_OS") == "macOS" and
                os.environ.get("RUNNER_ARCH") == "ARM64" and
                args.probe_opencl == "/System/Library/Frameworks/OpenCL.framework/OpenCL"), \
            "The Apple OpenCL environment exception is restricted to hosted macOS ARM64"
    library = Path(args.library).resolve()
    manifest = Path(args.manifest).resolve()
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

    print("ICD manifest, dynamic loading and ABI exports passed:", library, flush=True)
    # Run this probe only when explicitly requested by hardware-limited CI.
    # The ABI checks above still run. API errors fail, apart from the narrowly
    # verified hosted Apple ARM64 environment limitation below.
    if args.probe_opencl:
        try:
            available = has_opencl_device(args.probe_opencl)
        except OpenCLEnumerationError as error:
            if error.result != -30 or not args.apple_opencl_unavailable_probe:
                raise
            # Do not infer absent hardware from CL_INVALID_VALUE. Reproduce the
            # failure independently, without ctypes or any momoten code.
            probe = subprocess.run([args.apple_opencl_unavailable_probe], check=False, timeout=30)
            if probe.returncode != 78:
                raise
            print("SKIP: Apple OpenCL enumeration unavailable (CL_INVALID_VALUE); "
                  "native SDK probe agrees; system Vulkan Loader runtime untested", flush=True)
            return 78
        if not available:
            print("SKIP: no compatible OpenCL device; system Vulkan Loader runtime untested",
                  flush=True)
            return 77

    # Use the *system* loader, not a direct call to the ICD.
    script = """
import ctypes
import faulthandler
import sys
faulthandler.enable()
print('loading',sys.argv[1],flush=True)
v = ctypes.CDLL(sys.argv[1])
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
assert count.value > 0, 'OpenCL device not exposed through Vulkan Loader'
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
# Exercise all five dispatchable handle types through the system loader.
queue_handle=ctypes.c_void_p()
v.vkGetDeviceQueue.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32,
                            ctypes.POINTER(ctypes.c_void_p)]
v.vkGetDeviceQueue.restype=None
v.vkGetDeviceQueue(device,0,0,ctypes.byref(queue_handle))
assert queue_handle.value, 'Loader failed to return a Vulkan queue'
class PoolCreate(ctypes.Structure):
    _fields_=[('sType',ctypes.c_int),('pNext',ctypes.c_void_p),
              ('flags',ctypes.c_uint32),('queueFamilyIndex',ctypes.c_uint32)]
class CommandAllocate(ctypes.Structure):
    _fields_=[('sType',ctypes.c_int),('pNext',ctypes.c_void_p),
              ('commandPool',ctypes.c_uint64),('level',ctypes.c_int),
              ('commandBufferCount',ctypes.c_uint32)]
class CommandBegin(ctypes.Structure):
    _fields_=[('sType',ctypes.c_int),('pNext',ctypes.c_void_p),
              ('flags',ctypes.c_uint32),('pInheritanceInfo',ctypes.c_void_p)]
class Submit(ctypes.Structure):
    _fields_=[('sType',ctypes.c_int),('pNext',ctypes.c_void_p),
              ('waitSemaphoreCount',ctypes.c_uint32),('pWaitSemaphores',ctypes.c_void_p),
              ('pWaitDstStageMask',ctypes.c_void_p),('commandBufferCount',ctypes.c_uint32),
              ('pCommandBuffers',ctypes.POINTER(ctypes.c_void_p)),
              ('signalSemaphoreCount',ctypes.c_uint32),('pSignalSemaphores',ctypes.c_void_p)]
pool=ctypes.c_uint64()
pool_info=PoolCreate(39,None,0,0)
v.vkCreateCommandPool.argtypes=[ctypes.c_void_p,ctypes.POINTER(PoolCreate),
                               ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint64)]
v.vkCreateCommandPool.restype=ctypes.c_int32
assert v.vkCreateCommandPool(device,ctypes.byref(pool_info),None,ctypes.byref(pool)) == 0
command=ctypes.c_void_p()
allocate=CommandAllocate(40,None,pool.value,0,1)
v.vkAllocateCommandBuffers.argtypes=[ctypes.c_void_p,ctypes.POINTER(CommandAllocate),
                                    ctypes.POINTER(ctypes.c_void_p)]
v.vkAllocateCommandBuffers.restype=ctypes.c_int32
assert v.vkAllocateCommandBuffers(device,ctypes.byref(allocate),ctypes.byref(command)) == 0
assert command.value, 'Loader failed to allocate a Vulkan command buffer'
begin=CommandBegin(42,None,1,None)
v.vkBeginCommandBuffer.argtypes=[ctypes.c_void_p,ctypes.POINTER(CommandBegin)]
v.vkBeginCommandBuffer.restype=ctypes.c_int32
assert v.vkBeginCommandBuffer(command,ctypes.byref(begin)) == 0
v.vkCmdPipelineBarrier.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32,
                                ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p,
                                ctypes.c_uint32,ctypes.c_void_p,ctypes.c_uint32,ctypes.c_void_p]
v.vkCmdPipelineBarrier.restype=None
v.vkCmdPipelineBarrier(command,0x800,0x800,0,0,None,0,None,0,None)
v.vkEndCommandBuffer.argtypes=[ctypes.c_void_p]
v.vkEndCommandBuffer.restype=ctypes.c_int32
assert v.vkEndCommandBuffer(command) == 0
submit=Submit(4,None,0,None,None,1,ctypes.pointer(command),0,None)
v.vkQueueSubmit.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.POINTER(Submit),ctypes.c_uint64]
v.vkQueueSubmit.restype=ctypes.c_int32
assert v.vkQueueSubmit(queue_handle,1,ctypes.byref(submit),0) == 0
v.vkQueueWaitIdle.argtypes=[ctypes.c_void_p]
v.vkQueueWaitIdle.restype=ctypes.c_int32
assert v.vkQueueWaitIdle(queue_handle) == 0
print('submitted command buffer and waited for Vulkan queue',flush=True)
v.vkFreeCommandBuffers.argtypes=[ctypes.c_void_p,ctypes.c_uint64,ctypes.c_uint32,
                                ctypes.POINTER(ctypes.c_void_p)]
v.vkFreeCommandBuffers.restype=None
v.vkFreeCommandBuffers(device,pool,1,ctypes.byref(command))
v.vkDestroyCommandPool.argtypes=[ctypes.c_void_p,ctypes.c_uint64,ctypes.c_void_p]
v.vkDestroyCommandPool.restype=None
v.vkDestroyCommandPool(device,pool,None)
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
    result = subprocess.run([sys.executable, "-u", "-X", "faulthandler", "-c", script, args.loader], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, timeout=30)
    print(result.stdout, flush=True)
    print(result.stderr, flush=True)
    assert result.returncode == 0, f"system Vulkan Loader process failed (exit {result.returncode})"
    assert manifest.name in result.stderr, "system loader did not discover momoten ICD"


if __name__ == "__main__":
    sys.exit(main())
