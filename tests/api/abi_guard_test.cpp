// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "abi_guard.h"

#include <new>
#include <stdexcept>

namespace abi_guard_test {

VkResult throw_bad_alloc()
{
    throw std::bad_alloc();
}

VkResult throw_result_exception()
{
    throw std::runtime_error("result exception");
}

void throw_void_exception()
{
    throw std::runtime_error("void exception");
}

PFN_vkVoidFunction throw_proc_exception()
{
    throw std::runtime_error("proc exception");
}

} // namespace abi_guard_test

int main()
{
    if (momoten_detail::VulkanAbiGuard<
            decltype(&abi_guard_test::throw_bad_alloc),
            &abi_guard_test::throw_bad_alloc>::call()
        != VK_ERROR_OUT_OF_HOST_MEMORY)
        return 1;

    if (momoten_detail::VulkanAbiGuard<
            decltype(&abi_guard_test::throw_result_exception),
            &abi_guard_test::throw_result_exception>::call()
        != VK_ERROR_INITIALIZATION_FAILED)
        return 1;

    momoten_detail::VulkanAbiGuard<
        decltype(&abi_guard_test::throw_void_exception),
        &abi_guard_test::throw_void_exception>::call();

    if (momoten_detail::VulkanAbiGuard<
            decltype(&abi_guard_test::throw_proc_exception),
            &abi_guard_test::throw_proc_exception>::call()
        != 0)
        return 1;

    return 0;
}
