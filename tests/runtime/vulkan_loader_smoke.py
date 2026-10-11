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
v = ctypes.CDLL('libvulkan.so.1')
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
ret=fn(ctypes.byref(ci),None,ctypes.byref(instance))
print('Vulkan Loader vkCreateInstance returned',ret)
if ret == 0:
    v.vkDestroyInstance.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
    v.vkDestroyInstance(instance,None)
"""
    env = os.environ.copy()
    env["VK_DRIVER_FILES"] = str(manifest)
    env["VK_ICD_FILENAMES"] = str(manifest)  # compatibility with older loaders
    env["VK_LOADER_DEBUG"] = "driver"
    result = subprocess.run([sys.executable, "-c", script], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, timeout=30)
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0, "system Vulkan Loader process failed"
    assert manifest.name in result.stderr, "system loader did not discover momoten ICD"


if __name__ == "__main__":
    main()
