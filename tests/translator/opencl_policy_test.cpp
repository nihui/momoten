// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator/translator_internal.h"

#include <stdio.h>
#include <string.h>

int main()
{
    using namespace momoten;
    const struct AddressSpaceCase
    {
        StorageClass storage;
        OpenCLAddressSpace address_space;
        const char* qualifier;
    } address_spaces[] = {
        {StorageClassFunction, OpenCLAddressSpace::Private, "__private"},
        {StorageClassPrivate, OpenCLAddressSpace::Private, "__private"},
        {StorageClassPushConstant, OpenCLAddressSpace::Private, "__private"},
        {StorageClassWorkgroup, OpenCLAddressSpace::Local, "__local"},
        {StorageClassUniform, OpenCLAddressSpace::Global, "__global"},
        {StorageClassStorageBuffer, OpenCLAddressSpace::Global, "__global"},
        {StorageClassCrossWorkgroup, OpenCLAddressSpace::Global, "__global"},
        {StorageClassUniformConstant, OpenCLAddressSpace::Unsupported, 0},
        {StorageClassInput, OpenCLAddressSpace::Unsupported, 0}};
    for (size_t i = 0; i < sizeof(address_spaces) / sizeof(address_spaces[0]); i++)
    {
        const OpenCLAddressSpace address_space = opencl_address_space(address_spaces[i].storage);
        const char* qualifier = opencl_address_space_qualifier(address_space);
        if (address_space != address_spaces[i].address_space
            || (qualifier == 0) != (address_spaces[i].qualifier == 0)
            || (qualifier && strcmp(qualifier, address_spaces[i].qualifier) != 0))
        {
            fprintf(stderr, "opencl_policy_test: address space for storage class %u\n",
                    static_cast<unsigned>(address_spaces[i].storage));
            return 1;
        }
    }

    const struct ArrayCase
    {
        const char* name;
        SPIRType::BaseType basetype;
        uint32_t width;
        uint32_t vecsize;
        uint32_t columns;
        bool array;
        StorageClass storage;
        bool needs_volatile;
    } arrays[] = {
        {"signed vec2 array", SPIRType::Int, 32, 2, 1, true, StorageClassFunction, true},
        {"signed vec3 array", SPIRType::Int, 32, 3, 1, true, StorageClassFunction, true},
        {"signed vec4 array", SPIRType::Int, 32, 4, 1, true, StorageClassFunction, true},
        {"unsigned vector array", SPIRType::UInt, 32, 4, 1, true, StorageClassFunction, true},
        {"scalar array", SPIRType::Int, 32, 1, 1, true, StorageClassFunction, false},
        {"non-array vector", SPIRType::Int, 32, 4, 1, false, StorageClassFunction, false},
        {"matrix array", SPIRType::Int, 32, 4, 4, true, StorageClassFunction, false},
        {"16-bit array", SPIRType::Short, 16, 4, 1, true, StorageClassFunction, false},
        {"64-bit array", SPIRType::Int64, 64, 4, 1, true, StorageClassFunction, false},
        {"float array", SPIRType::Float, 32, 4, 1, true, StorageClassFunction, false},
        {"boolean array", SPIRType::Boolean, 32, 4, 1, true, StorageClassFunction, false},
        {"module private array", SPIRType::Int, 32, 4, 1, true, StorageClassPrivate, false},
        {"workgroup array", SPIRType::Int, 32, 4, 1, true, StorageClassWorkgroup, false},
        {"storage buffer array", SPIRType::Int, 32, 4, 1, true, StorageClassStorageBuffer, false}};
    for (size_t i = 0; i < sizeof(arrays) / sizeof(arrays[0]); i++)
    {
        SPIRType type(arrays[i].array ? OpTypeArray : OpTypeVector);
        type.basetype = arrays[i].basetype;
        type.width = arrays[i].width;
        type.vecsize = arrays[i].vecsize;
        type.columns = arrays[i].columns;
        if (arrays[i].array)
            type.array.push_back(5);
        if (needs_volatile_private_vector_array(type, arrays[i].storage) != arrays[i].needs_volatile)
        {
            fprintf(stderr, "opencl_policy_test: volatile policy for %s\n", arrays[i].name);
            return 1;
        }
    }
    return 0;
}
