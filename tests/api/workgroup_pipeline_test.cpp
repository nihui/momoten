// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "backend/device.h"
#include "backend/objects.h"
#include "backend/pipeline_build.h"
#include "backend/replay.h"

#include <spirv-tools/libspirv.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

struct _cl_program
{
    std::string source;
    size_t attempt;
    unsigned int application_references;
    unsigned int live_kernels;
    bool destroyed;
};

struct _cl_kernel
{
    cl_program program;
    bool alive;
};

struct _cl_mem
{
    bool alive;
};

struct _cl_event
{
    unsigned int references;
};

struct TestCase
{
    std::string name;
    uint32_t logical_size[3];
    size_t device_limit;
    size_t profile_limit;
    size_t dimension_limits[3];
    cl_ulong local_memory_limit;
    bool barrier;
    bool native_subgroups;
    size_t kernel_limits[2];
    cl_ulong kernel_local_memory[2];
    size_t subgroup_sizes[2];
    cl_int program_creation_errors[2];
    cl_int build_errors[2];
    cl_int kernel_creation_errors[2];
    cl_int query_errors[2];
    cl_int subgroup_query_errors[2];
    bool succeeds;
    momoten::WorkgroupMode expected_mode;
    size_t expected_local_size[3];
    size_t expected_global_size[3];
    size_t expected_chunks;
    size_t expected_program_calls;
    size_t expected_build_calls;
    size_t expected_kernel_calls;
    size_t expected_subgroup_calls;

    explicit TestCase(const char* name_)
        : name(name_), device_limit(1024), profile_limit(256), local_memory_limit(65536), barrier(false), native_subgroups(false), succeeds(true), expected_mode(momoten::WorkgroupModeDirect), expected_chunks(1), expected_program_calls(1), expected_build_calls(1), expected_kernel_calls(1), expected_subgroup_calls(0)
    {
        logical_size[0] = expected_local_size[0] = 4;
        logical_size[1] = expected_local_size[1] = 3;
        logical_size[2] = expected_local_size[2] = 2;
        expected_global_size[0] = 12;
        expected_global_size[1] = 6;
        expected_global_size[2] = 8;
        for (size_t d = 0; d < 3; d++)
            dimension_limits[d] = 1024;
        for (size_t i = 0; i < 2; i++)
        {
            kernel_limits[i] = 1024;
            kernel_local_memory[i] = 0;
            subgroup_sizes[i] = 8;
            program_creation_errors[i] = build_errors[i] = kernel_creation_errors[i] = query_errors[i] = subgroup_query_errors[i] = CL_SUCCESS;
        }
    }

    void expect_virtual(size_t local_x, size_t chunks, size_t global_x, size_t attempts = 1)
    {
        expected_mode = momoten::WorkgroupModeVirtual;
        expected_local_size[0] = local_x;
        expected_local_size[1] = expected_local_size[2] = 1;
        expected_global_size[0] = global_x;
        expected_global_size[1] = 2;
        expected_global_size[2] = 4;
        expected_chunks = chunks;
        expected_program_calls = expected_build_calls = expected_kernel_calls = attempts;
    }

    void expect_failure(size_t programs, size_t builds, size_t kernels)
    {
        succeeds = false;
        expected_program_calls = programs;
        expected_build_calls = builds;
        expected_kernel_calls = kernels;
    }
};

struct Shape
{
    size_t values[3];
};

struct MockState
{
    const TestCase& test;
    std::vector<std::unique_ptr<_cl_program> > programs;
    std::vector<std::unique_ptr<_cl_kernel> > kernels;
    std::vector<std::unique_ptr<_cl_mem> > memories;
    std::vector<std::unique_ptr<_cl_event> > events;
    std::vector<Shape> subgroup_shapes;
    Shape global_size;
    Shape local_size;
    size_t program_calls;
    size_t build_calls;
    size_t kernel_calls;
    size_t argument_calls;
    size_t enqueue_calls;
    bool invalid_lifetime;

    explicit MockState(const TestCase& value)
        : test(value), program_calls(0), build_calls(0), kernel_calls(0), argument_calls(0), enqueue_calls(0), invalid_lifetime(false)
    {
        for (size_t d = 0; d < 3; d++)
            global_size.values[d] = local_size.values[d] = 0;
    }
};

static MockState* mock;

namespace momoten_detail {

OpenCLApi g_opencl = {};

bool debug_enabled()
{
    return false;
}

} // namespace momoten_detail

void log_error(const char*, cl_int)
{
}

std::string opencl_device_string(cl_device_id, cl_device_info)
{
    return "workgroup pipeline test device";
}

static bool program_alive(cl_program program)
{
    return program && !program->destroyed && (program->application_references || program->live_kernels);
}

static void destroy_unused_program(cl_program program)
{
    if (!program->application_references && !program->live_kernels)
        program->destroyed = true;
}

static cl_program CL_API_CALL create_program(cl_context, cl_uint count, const char** sources, const size_t* lengths, cl_int* error)
{
    const size_t attempt = mock->program_calls++;
    // A rejected candidate must be released before building the next source.
    for (size_t i = 0; i < mock->programs.size(); i++)
        if (!mock->programs[i]->destroyed)
            mock->invalid_lifetime = true;
    *error = attempt < 2 ? mock->test.program_creation_errors[attempt] : CL_OUT_OF_RESOURCES;
    if (*error != CL_SUCCESS)
        return 0;
    std::unique_ptr<_cl_program> program(new _cl_program);
    program->attempt = attempt;
    program->application_references = 1;
    program->live_kernels = 0;
    program->destroyed = false;
    for (cl_uint i = 0; i < count; i++)
        program->source.append(sources[i], lengths ? lengths[i] : strlen(sources[i]));
    const cl_program result = program.get();
    mock->programs.push_back(std::move(program));
    return result;
}

static cl_int CL_API_CALL build_program(cl_program program, cl_uint, const cl_device_id*, const char*, void(CL_CALLBACK*)(cl_program, void*), void*)
{
    mock->build_calls++;
    if (!program_alive(program))
    {
        mock->invalid_lifetime = true;
        return CL_INVALID_PROGRAM;
    }
    return mock->test.build_errors[program->attempt];
}

static cl_int CL_API_CALL retain_program(cl_program program)
{
    if (!program_alive(program))
    {
        mock->invalid_lifetime = true;
        return CL_INVALID_PROGRAM;
    }
    program->application_references++;
    return CL_SUCCESS;
}

static cl_int CL_API_CALL release_program(cl_program program)
{
    if (!program || program->destroyed || !program->application_references)
    {
        mock->invalid_lifetime = true;
        return CL_INVALID_PROGRAM;
    }
    program->application_references--;
    destroy_unused_program(program);
    return CL_SUCCESS;
}

static cl_int CL_API_CALL program_build_info(cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t* size)
{
    if (size)
        *size = 0;
    return CL_SUCCESS;
}

static cl_kernel CL_API_CALL create_kernel(cl_program program, const char*, cl_int* error)
{
    mock->kernel_calls++;
    if (!program_alive(program))
    {
        mock->invalid_lifetime = true;
        *error = CL_INVALID_PROGRAM;
        return 0;
    }
    *error = mock->test.kernel_creation_errors[program->attempt];
    if (*error != CL_SUCCESS)
        return 0;
    std::unique_ptr<_cl_kernel> kernel(new _cl_kernel);
    kernel->program = program;
    kernel->alive = true;
    program->live_kernels++;
    const cl_kernel result = kernel.get();
    mock->kernels.push_back(std::move(kernel));
    return result;
}

static cl_int CL_API_CALL release_kernel(cl_kernel kernel)
{
    if (!kernel || !kernel->alive || !program_alive(kernel->program) || !kernel->program->live_kernels)
    {
        mock->invalid_lifetime = true;
        return CL_INVALID_KERNEL;
    }
    kernel->alive = false;
    kernel->program->live_kernels--;
    destroy_unused_program(kernel->program);
    return CL_SUCCESS;
}

template<typename T>
static cl_int write_info(T value, size_t capacity, void* output, size_t* size)
{
    if (size)
        *size = sizeof(value);
    if (output)
    {
        if (capacity < sizeof(value))
            return CL_INVALID_VALUE;
        memcpy(output, &value, sizeof(value));
    }
    return CL_SUCCESS;
}

static cl_int CL_API_CALL kernel_info(cl_kernel kernel, cl_device_id, cl_kernel_work_group_info parameter, size_t capacity, void* output, size_t* size)
{
    if (!kernel || !kernel->alive || !program_alive(kernel->program))
    {
        mock->invalid_lifetime = true;
        return CL_INVALID_KERNEL;
    }
    const size_t attempt = kernel->program->attempt;
    if (mock->test.query_errors[attempt] != CL_SUCCESS)
        return mock->test.query_errors[attempt];
    if (parameter == CL_KERNEL_WORK_GROUP_SIZE)
        return write_info(mock->test.kernel_limits[attempt], capacity, output, size);
    if (parameter == CL_KERNEL_LOCAL_MEM_SIZE)
        return write_info(mock->test.kernel_local_memory[attempt], capacity, output, size);
    return CL_INVALID_VALUE;
}

static cl_int CL_API_CALL subgroup_info(cl_kernel kernel, cl_device_id, cl_kernel_sub_group_info parameter, size_t input_size, const void* input, size_t capacity, void* output, size_t* size)
{
    if (!kernel || !kernel->alive || !program_alive(kernel->program))
    {
        mock->invalid_lifetime = true;
        return CL_INVALID_KERNEL;
    }
    if (parameter != CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE_KHR || input_size != sizeof(Shape) || !input)
        return CL_INVALID_VALUE;
    Shape shape;
    memcpy(shape.values, input, sizeof(shape.values));
    mock->subgroup_shapes.push_back(shape);
    const size_t attempt = kernel->program->attempt;
    if (mock->test.subgroup_query_errors[attempt] != CL_SUCCESS)
        return mock->test.subgroup_query_errors[attempt];
    return write_info(mock->test.subgroup_sizes[attempt], capacity, output, size);
}

static cl_mem CL_API_CALL create_buffer(cl_context, cl_mem_flags, size_t, void*, cl_int* error)
{
    std::unique_ptr<_cl_mem> memory(new _cl_mem);
    memory->alive = true;
    const cl_mem result = memory.get();
    mock->memories.push_back(std::move(memory));
    *error = CL_SUCCESS;
    return result;
}

static cl_int CL_API_CALL release_memory(cl_mem memory)
{
    if (!memory || !memory->alive)
    {
        mock->invalid_lifetime = true;
        return CL_INVALID_MEM_OBJECT;
    }
    memory->alive = false;
    return CL_SUCCESS;
}

static cl_int CL_API_CALL set_argument(cl_kernel kernel, cl_uint, size_t size, const void* value)
{
    if (!kernel || !kernel->alive || !value || size == 0)
    {
        mock->invalid_lifetime = true;
        return CL_INVALID_ARG_VALUE;
    }
    mock->argument_calls++;
    return CL_SUCCESS;
}

static cl_int CL_API_CALL enqueue_kernel(cl_command_queue, cl_kernel kernel, cl_uint dimensions, const size_t* offset, const size_t* global, const size_t* local, cl_uint wait_count, const cl_event*, cl_event* event)
{
    if (!kernel || !kernel->alive || dimensions != 3 || offset || !global || !local || wait_count != 0 || !event)
    {
        mock->invalid_lifetime = true;
        return CL_INVALID_VALUE;
    }
    memcpy(mock->global_size.values, global, sizeof(mock->global_size.values));
    memcpy(mock->local_size.values, local, sizeof(mock->local_size.values));
    mock->enqueue_calls++;
    std::unique_ptr<_cl_event> result(new _cl_event);
    result->references = 1;
    *event = result.get();
    mock->events.push_back(std::move(result));
    return CL_SUCCESS;
}

static cl_int CL_API_CALL retain_event(cl_event event)
{
    if (!event || !event->references)
    {
        mock->invalid_lifetime = true;
        return CL_INVALID_EVENT;
    }
    event->references++;
    return CL_SUCCESS;
}

static cl_int CL_API_CALL release_event(cl_event event)
{
    if (!event || !event->references)
    {
        mock->invalid_lifetime = true;
        return CL_INVALID_EVENT;
    }
    event->references--;
    return CL_SUCCESS;
}

static void install_mock()
{
    momoten_detail::g_opencl = momoten_detail::OpenCLApi();
    momoten_detail::g_opencl.p_clCreateProgramWithSource = create_program;
    momoten_detail::g_opencl.p_clBuildProgram = build_program;
    momoten_detail::g_opencl.p_clRetainProgram = retain_program;
    momoten_detail::g_opencl.p_clReleaseProgram = release_program;
    momoten_detail::g_opencl.p_clGetProgramBuildInfo = program_build_info;
    momoten_detail::g_opencl.p_clCreateKernel = create_kernel;
    momoten_detail::g_opencl.p_clReleaseKernel = release_kernel;
    momoten_detail::g_opencl.p_clGetKernelWorkGroupInfo = kernel_info;
    momoten_detail::g_opencl.p_clGetKernelSubGroupInfoKHR = subgroup_info;
    momoten_detail::g_opencl.p_clCreateBuffer = create_buffer;
    momoten_detail::g_opencl.p_clReleaseMemObject = release_memory;
    momoten_detail::g_opencl.p_clSetKernelArg = set_argument;
    momoten_detail::g_opencl.p_clEnqueueNDRangeKernel = enqueue_kernel;
    momoten_detail::g_opencl.p_clRetainEvent = retain_event;
    momoten_detail::g_opencl.p_clReleaseEvent = release_event;
}

static bool assemble(const TestCase& test, std::vector<uint32_t>& words)
{
    std::ostringstream source;
    source << "OpCapability Shader\n";
    if (test.native_subgroups)
        source << "OpCapability GroupNonUniform\n";
    source << "OpMemoryModel Logical GLSL450\n"
              "OpEntryPoint GLCompute %main \"main\" %local_id %local_index";
    if (test.native_subgroups)
        source << " %subgroup_size";
    source << "\nOpExecutionMode %main LocalSize " << test.logical_size[0] << ' ' << test.logical_size[1] << ' ' << test.logical_size[2] << "\n"
           << "OpDecorate %local_id BuiltIn LocalInvocationId\n"
              "OpDecorate %local_index BuiltIn LocalInvocationIndex\n"
              "OpDecorate %array ArrayStride 4\n"
              "OpMemberDecorate %block 0 Offset 0\n"
              "OpDecorate %block Block\n"
              "OpDecorate %data DescriptorSet 0\n"
              "OpDecorate %data Binding 0\n"
              "OpDecorate %data NonReadable\n";
    if (test.native_subgroups)
        source << "OpDecorate %subgroup_size BuiltIn SubgroupSize\n";
    source << "%void = OpTypeVoid\n"
              "%uint = OpTypeInt 32 0\n"
              "%uint3 = OpTypeVector %uint 3\n"
              "%function = OpTypeFunction %void\n"
              "%zero = OpConstant %uint 0\n"
              "%scope = OpConstant %uint 2\n"
              "%semantics = OpConstant %uint 264\n"
              "%array = OpTypeRuntimeArray %uint\n"
              "%block = OpTypeStruct %array\n"
              "%input_uint = OpTypePointer Input %uint\n"
              "%input_uint3 = OpTypePointer Input %uint3\n"
              "%storage_block = OpTypePointer StorageBuffer %block\n"
              "%storage_uint = OpTypePointer StorageBuffer %uint\n"
              "%local_id = OpVariable %input_uint3 Input\n"
              "%local_index = OpVariable %input_uint Input\n"
              "%data = OpVariable %storage_block StorageBuffer\n";
    if (test.native_subgroups)
        source << "%subgroup_size = OpVariable %input_uint Input\n";
    source << "%main = OpFunction %void None %function\n"
              "%entry = OpLabel\n"
              "%local = OpLoad %uint3 %local_id\n"
              "%index = OpLoad %uint %local_index\n"
              "%x = OpCompositeExtract %uint %local 0\n"
              "%y = OpCompositeExtract %uint %local 1\n"
              "%z = OpCompositeExtract %uint %local 2\n"
              "%xy = OpIAdd %uint %x %y\n"
              "%xyz = OpIAdd %uint %xy %z\n";
    if (test.native_subgroups)
        source << "%subgroup = OpLoad %uint %subgroup_size\n"
                  "%result = OpIAdd %uint %xyz %subgroup\n";
    else
        source << "%result = OpCopyObject %uint %xyz\n";
    if (test.barrier)
        source << "OpControlBarrier %scope %scope %semantics\n";
    source << "%pointer = OpAccessChain %storage_uint %data %zero %index\n"
              "OpStore %pointer %result\n"
              "OpReturn\n"
              "OpFunctionEnd\n";

    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_1);
    tools.SetMessageConsumer([&test](spv_message_level_t, const char*, const spv_position_t& position, const char* message) {
        fprintf(stderr, "workgroup_pipeline_test: %s SPIR-V %zu:%zu: %s\n", test.name.c_str(), position.line, position.column, message);
    });
    return tools.Assemble(source.str(), &words) && tools.Validate(words);
}

static int require(const TestCase& test, bool condition, const char* message)
{
    if (condition)
        return 0;
    fprintf(stderr, "workgroup_pipeline_test: %s: %s\n", test.name.c_str(), message);
    return 1;
}

static int check_workgroup_validation()
{
    const TestCase test("shared workgroup validation");
    VkPhysicalDevice_T physical = {};
    physical.address_bits = 64;
    physical.max_parameter_size = 1024;
    physical.max_workgroup_size = 1024;
    physical.max_compute_workgroup_invocations = 24;
    for (size_t d = 0; d < 3; d++)
        physical.max_work_item_sizes[d] = 1024;
    VkDevice_T device = {};
    device.physical_device = &physical;
    momoten::KernelABI abi;
    abi.address_bits = 64;
    for (size_t d = 0; d < 3; d++)
        abi.local_size[d] = test.logical_size[d];

    std::string diagnostic;
    int status = require(test, momoten_detail::validate_pipeline_abi(&device, abi, diagnostic), "direct workgroup at the profile limit was rejected");
    physical.max_compute_workgroup_invocations = 23;
    status |= require(test, !momoten_detail::validate_pipeline_abi(&device, abi, diagnostic) && diagnostic.find("profile") != std::string::npos, "direct workgroup exceeding the profile limit was accepted");
    abi.workgroup_mode = momoten::WorkgroupModeVirtual;
    abi.workgroup_splittable = true;
    status |= require(test, momoten_detail::validate_pipeline_abi(&device, abi, diagnostic), "virtual logical workgroup exceeding the profile limit was rejected");
    abi.local_size[0] = 0;
    status |= require(test, !momoten_detail::validate_pipeline_abi(&device, abi, diagnostic) && diagnostic.find("zero") != std::string::npos, "virtual mode accepted a zero logical dimension");
    for (size_t d = 0; d < 3; d++)
        abi.local_size[d] = 0xffffffffu;
    status |= require(test, !momoten_detail::validate_pipeline_abi(&device, abi, diagnostic) && diagnostic.find("overflows") != std::string::npos, "virtual mode accepted an overflowing logical invocation count");
    return status;
}

static int check_mock_lifetime(bool program_first)
{
    const TestCase test(program_first ? "application program reference released before kernel" : "kernel released before application program reference");
    MockState state(test);
    mock = &state;
    install_mock();
    const char* source = "__kernel void lifecycle_test() {}";
    cl_int error = CL_SUCCESS;
    cl_program program = create_program(0, 1, &source, 0, &error);
    cl_kernel kernel = create_kernel(program, "lifecycle_test", &error);
    if (!program || !kernel || error != CL_SUCCESS)
        return require(test, false, "could not create mock program and kernel");
    int status = require(test, program->application_references == 1 && program->live_kernels == 1 && program_alive(program), "kernel did not keep its program alive");
    if (program_first)
    {
        status |= require(test, release_program(program) == CL_SUCCESS && !state.invalid_lifetime, "releasing the application reference while a kernel is live was rejected");
        status |= require(test, program->application_references == 0 && program->live_kernels == 1 && program_alive(program), "program was destroyed before its live kernel was released");
        status |= require(test, release_program(program) == CL_INVALID_PROGRAM && state.invalid_lifetime && program_alive(program), "duplicate application-reference release was not detected");
        state.invalid_lifetime = false;
    }
    size_t kernel_limit = 0;
    status |= require(test, kernel_info(kernel, 0, CL_KERNEL_WORK_GROUP_SIZE, sizeof(kernel_limit), &kernel_limit, 0) == CL_SUCCESS && kernel_limit == test.kernel_limits[0], "kernel query failed while its program was retained by the kernel");
    const Shape shape = {{4, 3, 2}};
    size_t subgroup_size = 0;
    status |= require(test, subgroup_info(kernel, 0, CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE_KHR, sizeof(shape), &shape, sizeof(subgroup_size), &subgroup_size, 0) == CL_SUCCESS && subgroup_size == test.subgroup_sizes[0], "subgroup query failed while its program was retained by the kernel");
    status |= require(test, release_kernel(kernel) == CL_SUCCESS && program->live_kernels == 0, "kernel release did not drop the internal program reference");
    if (!program_first)
    {
        status |= require(test, program_alive(program) && program->application_references == 1, "kernel release destroyed an application-retained program");
        status |= require(test, release_program(program) == CL_SUCCESS, "final application program release failed");
    }
    status |= require(test, program->destroyed && !program_alive(program) && program->application_references == 0 && !state.invalid_lifetime, "program was not destroyed exactly after its final reference disappeared");
    return status;
}

static int run_case(const TestCase& test)
{
    MockState state(test);
    mock = &state;
    install_mock();
    VkPhysicalDevice_T physical = {};
    physical.address_bits = 64;
    physical.max_parameter_size = 1024;
    physical.max_workgroup_size = test.device_limit;
    physical.max_compute_workgroup_invocations = test.profile_limit;
    physical.local_memory = test.local_memory_limit;
    for (size_t d = 0; d < 3; d++)
        physical.max_work_item_sizes[d] = test.dimension_limits[d];
    if (test.native_subgroups)
    {
        physical.extensions = "cl_khr_subgroups cl_khr_subgroup_non_uniform_vote";
        physical.shader_profile.subgroup_mode = momoten::SubgroupModeNative;
        physical.shader_profile.subgroup_size = 8;
    }
    VkDevice_T device = {};
    device.physical_device = &physical;

    std::shared_ptr<ShaderModule> module(new ShaderModule());
    module->device = &device;
    if (!assemble(test, module->words))
        return 1;
    std::shared_ptr<PipelineLayout> layout(new PipelineLayout());
    layout->device = &device;
    layout->descriptor_layout.reset(new DescriptorSetLayout());
    layout->descriptor_layout->device = &device;
    VkDescriptorSetLayoutBinding binding = {};
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    layout->descriptor_layout->bindings.push_back(binding);
    const VkShaderModule module_handle = g_shader_modules.make_handle(module);
    const VkPipelineLayout layout_handle = g_pipeline_layouts.make_handle(layout);
    VkComputePipelineCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = module_handle;
    info.stage.pName = "main";
    info.layout = layout_handle;
    VkPipeline handle = VK_NULL_HANDLE;
    const VkResult result = momoten_detail::build_compute_pipelines(&device, VK_NULL_HANDLE, 1, &info, 0, &handle);
    int status = require(test, (result == VK_SUCCESS) == test.succeeds, "unexpected pipeline creation result");
    status |= require(test, (handle != VK_NULL_HANDLE) == test.succeeds, "failed candidate was published or successful candidate was not published");
    status |= require(test, state.program_calls == test.expected_program_calls && state.build_calls == test.expected_build_calls && state.kernel_calls == test.expected_kernel_calls, "unexpected build or retry count");
    if (state.programs.size() == 2)
        status |= require(test, state.programs[0]->source != state.programs[1]->source, "retry reused the direct source for the virtual program");
    status |= require(test, state.subgroup_shapes.size() == test.expected_subgroup_calls, "unexpected native subgroup query count");
    for (size_t i = 0; i < state.subgroup_shapes.size(); i++)
    {
        for (size_t d = 0; d < 3; d++)
        {
            const size_t expected = i == 0 ? test.logical_size[d] : test.expected_local_size[d];
            status |= require(test, state.subgroup_shapes[i].values[d] == expected, "subgroup query did not use the original 3D shape or virtual 1D shape");
        }
    }

    std::shared_ptr<Pipeline> pipeline = g_pipelines.get_handle(handle);
    if (pipeline)
    {
        status |= require(test, pipeline->abi.workgroup_mode == test.expected_mode, "pipeline recorded the wrong workgroup mode");
        status |= require(test, pipeline->abi.workgroup_splittable == !test.barrier, "shader synchronization capability was lost");
        status |= require(test, pipeline->workgroup_chunk_count == test.expected_chunks, "unexpected virtual chunk count");
        for (size_t d = 0; d < 3; d++)
        {
            status |= require(test, pipeline->abi.local_size[d] == test.logical_size[d], "logical workgroup size changed during fallback");
            status |= require(test, pipeline->opencl_local_size[d] == test.expected_local_size[d], "incorrect physical local size");
        }

        std::shared_ptr<DeviceMemory> memory(new DeviceMemory());
        memory->device = &device;
        memory->size = 65536;
        memory->range_map.reset(memory->size);
        cl_int error = CL_SUCCESS;
        memory->memory = create_buffer(0, CL_MEM_READ_WRITE, static_cast<size_t>(memory->size), 0, &error);
        std::shared_ptr<Buffer> buffer(new Buffer());
        buffer->device = &device;
        buffer->size = memory->size;
        buffer->memory = memory;
        RecordedCommand command;
        command.type = RecordedCommand::Dispatch;
        command.pipeline = pipeline;
        command.descriptors[0].buffer = buffer;
        command.group_count[0] = 3;
        command.group_count[1] = 2;
        command.group_count[2] = 4;
        ReplayState replay(&device);
        const cl_int replay_result = replay_dispatch(replay, command);
        status |= require(test, replay_result == CL_SUCCESS && state.enqueue_calls == 1 && state.argument_calls == 3, "dispatch replay failed or did not enqueue exactly one kernel");
        for (size_t d = 0; d < 3; d++)
        {
            status |= require(test, state.local_size.values[d] == test.expected_local_size[d], "enqueue used an incorrect local size");
            status |= require(test, state.global_size.values[d] == test.expected_global_size[d], "enqueue used an incorrect global size");
        }
        abort_replay(replay);
    }

    g_pipelines.erase_handle(handle);
    pipeline.reset();
    g_shader_modules.erase_handle(module_handle);
    g_pipeline_layouts.erase_handle(layout_handle);
    for (size_t i = 0; i < state.programs.size(); i++)
        status |= require(test, state.programs[i]->destroyed && !state.programs[i]->application_references && !state.programs[i]->live_kernels, "program was leaked");
    for (size_t i = 0; i < state.kernels.size(); i++)
        status |= require(test, !state.kernels[i]->alive, "kernel was leaked");
    for (size_t i = 0; i < state.memories.size(); i++)
        status |= require(test, !state.memories[i]->alive, "dispatch memory was leaked");
    for (size_t i = 0; i < state.events.size(); i++)
        status |= require(test, state.events[i]->references == 0, "dispatch event was leaked");
    status |= require(test, !state.invalid_lifetime, "object was used after release, released twice, or retained across retry");
    return status;
}

static void disable_artifact_environment()
{
    const char* names[] = {"MOMOTEN_CACHE", "NCNN_MOMOTEN_CACHE", "MOMOTEN_DUMP_DIR", "NCNN_MOMOTEN_DUMP_DIR"};
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++)
    {
#if defined(_WIN32)
        _putenv_s(names[i], "");
#else
        unsetenv(names[i]);
#endif
    }
}

int main()
{
    disable_artifact_environment();
    int status = check_workgroup_validation();
    status |= check_mock_lifetime(true);
    status |= check_mock_lifetime(false);
    TestCase test("original 3D workgroup");
    status |= run_case(test);
    test = TestCase("barrier keeps a legal original 3D workgroup");
    test.barrier = true;
    status |= run_case(test);

    const char* dimension_names[] = {"x dimension requires virtual mode", "y dimension requires virtual mode", "z dimension requires virtual mode"};
    for (size_t d = 0; d < 3; d++)
    {
        test = TestCase(dimension_names[d]);
        test.dimension_limits[d] = test.logical_size[d] - 1;
        test.expect_virtual(d == 0 ? 3 : 24, d == 0 ? 8 : 1, 72);
        status |= run_case(test);
    }
    test = TestCase("device total workgroup limit requires virtual mode");
    test.device_limit = test.profile_limit = test.kernel_limits[0] = 16;
    test.expect_virtual(16, 2, 96);
    status |= run_case(test);
    test = TestCase("profile limit 256 splits a 384-invocation logical workgroup");
    test.logical_size[0] = 8;
    test.logical_size[1] = 16;
    test.logical_size[2] = 3;
    test.expect_virtual(256, 2, 1536);
    status |= run_case(test);

    test = TestCase("built kernel maximum requires one virtual retry");
    test.kernel_limits[0] = test.kernel_limits[1] = 8;
    test.expect_virtual(8, 3, 72, 2);
    status |= run_case(test);
    test = TestCase("built kernel local memory requires one virtual retry");
    test.kernel_local_memory[0] = test.local_memory_limit + 1;
    test.expect_virtual(24, 1, 72, 2);
    status |= run_case(test);
    test = TestCase("matching native subgroup uses original 3D query");
    test.native_subgroups = true;
    test.expected_subgroup_calls = 1;
    status |= run_case(test);
    test = TestCase("original 3D subgroup mismatch retries virtual 1D query");
    test.native_subgroups = true;
    test.subgroup_sizes[0] = 4;
    test.expected_subgroup_calls = 2;
    test.expect_virtual(24, 1, 72, 2);
    status |= run_case(test);
    test = TestCase("virtual subgroup mismatch stops after one retry");
    test.native_subgroups = true;
    test.subgroup_sizes[0] = test.subgroup_sizes[1] = 4;
    test.expected_subgroup_calls = 2;
    test.expect_virtual(24, 1, 72, 2);
    test.expect_failure(2, 2, 2);
    status |= run_case(test);

    test = TestCase("compiler workgroup failure retries virtual source");
    test.build_errors[0] = CL_BUILD_PROGRAM_FAILURE;
    test.expect_virtual(24, 1, 72, 2);
    test.expected_kernel_calls = 1;
    status |= run_case(test);
    test = TestCase("compiler failure in both modes stops after one retry");
    test.build_errors[0] = test.build_errors[1] = CL_BUILD_PROGRAM_FAILURE;
    test.expect_failure(2, 2, 0);
    status |= run_case(test);
    test = TestCase("compiler OOM does not retry");
    test.build_errors[0] = CL_OUT_OF_HOST_MEMORY;
    test.expect_failure(1, 1, 0);
    status |= run_case(test);
    test = TestCase("unrelated compiler error does not retry");
    test.build_errors[0] = CL_INVALID_BUILD_OPTIONS;
    test.expect_failure(1, 1, 0);
    status |= run_case(test);
    test = TestCase("program creation OOM does not retry");
    test.program_creation_errors[0] = CL_OUT_OF_HOST_MEMORY;
    test.expect_failure(1, 0, 0);
    status |= run_case(test);
    test = TestCase("kernel creation OOM does not retry");
    test.kernel_creation_errors[0] = CL_OUT_OF_HOST_MEMORY;
    test.expect_failure(1, 1, 1);
    status |= run_case(test);
    test = TestCase("kernel limit query error does not retry");
    test.query_errors[0] = CL_INVALID_VALUE;
    test.expect_failure(1, 1, 1);
    status |= run_case(test);
    test = TestCase("subgroup query error does not retry");
    test.native_subgroups = true;
    test.subgroup_query_errors[0] = CL_INVALID_VALUE;
    test.expected_subgroup_calls = 1;
    test.expect_failure(1, 1, 1);
    status |= run_case(test);

    test = TestCase("barrier cannot bypass a device dimension limit");
    test.barrier = true;
    test.dimension_limits[1] = 2;
    test.expect_failure(0, 0, 0);
    status |= run_case(test);
    test = TestCase("barrier cannot bypass a built kernel limit");
    test.barrier = true;
    test.kernel_limits[0] = 8;
    test.expect_failure(1, 1, 1);
    status |= run_case(test);
    test = TestCase("virtual kernel with zero capacity does not retry again");
    test.kernel_limits[0] = 8;
    test.kernel_limits[1] = 0;
    test.expect_failure(2, 2, 2);
    status |= run_case(test);
    return status;
}
