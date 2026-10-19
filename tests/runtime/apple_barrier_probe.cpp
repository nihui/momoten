// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#define CL_SILENCE_DEPRECATION
#include <OpenCL/opencl.h>
#include <cstdio>
#include <cstring>

int main()
{
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    if (clGetPlatformIDs(1, &platform, 0) != CL_SUCCESS ||
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, 0) != CL_SUCCESS)
        return 1;
    char name[256] = {};
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, 0);
    printf("device: %s\n", name);
    cl_int result = 0;
    cl_context context = clCreateContext(0, 1, &device, 0, 0, &result);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &result);
    cl_mem output = clCreateBuffer(context, CL_MEM_READ_WRITE, 4 * sizeof(cl_uint), 0, &result);
    const char* sources[] = {
        "__kernel void k(__global uint*p) {p[get_global_id(0)] = get_local_id(0);}",
        "__kernel __attribute__((reqd_work_group_size(4,1,1))) void k(__global uint*p) {p[get_global_id(0)] = get_local_id(0);}",
        "__kernel void k(__global uint*p) {__local uint a[4]; uint x=get_local_id(0); a[x]=x; barrier(CLK_LOCAL_MEM_FENCE); p[get_global_id(0)]=a[3-x];}",
        "__kernel __attribute__((reqd_work_group_size(4,1,1))) void k(__global uint*p) {__local uint a[4]; uint x=get_local_id(0); a[x]=x; barrier(CLK_LOCAL_MEM_FENCE); p[get_global_id(0)]=a[3-x];}",
        "__attribute__((reqd_work_group_size(4,1,1))) __kernel void k(__global uint*p) {__local uint a[4]; uint x=get_local_id(0); a[x]=x; barrier(CLK_LOCAL_MEM_FENCE); p[get_global_id(0)]=a[3-x];}",
        "__kernel __attribute__((reqd_work_group_size(4,1,1))) void k(__global uint*p) {uint x=get_local_id(0); mem_fence(CLK_GLOBAL_MEM_FENCE); barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE); p[get_global_id(0)]=x;}",
        "__kernel __attribute__((reqd_work_group_size(4,1,1))) void k(__global uint*p) {__local uint a[4]; uint x=get_local_id(0); if(get_global_id(0)<4) {a[x]=x; barrier(CLK_LOCAL_MEM_FENCE); p[get_global_id(0)]=a[3-x];}}",
        "typedef struct {float scale; int bias; uint count;} Push; __kernel __attribute__((reqd_work_group_size(4,1,1))) void k(__global uint*p, Push push) {__local uint a[4]; uint x=get_local_id(0); if(get_global_id(0)<push.count) {a[x]=x; barrier(CLK_LOCAL_MEM_FENCE); p[get_global_id(0)]=a[3-x];}}"
    };
    const char* options[] = {"", "-cl-opt-disable", "-cl-std=CL1.1", "-cl-std=CL1.2", "-cl-auto-vectorize-enable", "-cl-autovectorize-disable"};
    for (unsigned s = 0; s < sizeof(sources)/sizeof(sources[0]); ++s)
    for (unsigned o = 0; o < sizeof(options)/sizeof(options[0]); ++o)
    {
        cl_program program = clCreateProgramWithSource(context, 1, &sources[s], 0, &result);
        result = clBuildProgram(program, 1, &device, options[o], 0, 0);
        printf("source=%u options='%s' build=%d ", s, options[o], result);
        if (result != CL_SUCCESS) {char log[8192]={};clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,sizeof(log),log,0);printf("%s\n",log);clReleaseProgram(program);continue;}
        char build_log[8192] = {};
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log, 0);
        if (build_log[0]) printf("build_log=[%s] ", build_log);
        cl_kernel kernel = clCreateKernel(program, "k", &result);
        if (!kernel) {printf("create_kernel=%d\n", result); clReleaseProgram(program); continue;}
        size_t maximum=0, required[3]={};cl_ulong local=0;cl_uint nargs=0;
        clGetKernelWorkGroupInfo(kernel,device,CL_KERNEL_WORK_GROUP_SIZE,sizeof(maximum),&maximum,0);
        clGetKernelWorkGroupInfo(kernel,device,CL_KERNEL_COMPILE_WORK_GROUP_SIZE,sizeof(required),required,0);
        clGetKernelWorkGroupInfo(kernel,device,CL_KERNEL_LOCAL_MEM_SIZE,sizeof(local),&local,0);
        clGetKernelInfo(kernel,CL_KERNEL_NUM_ARGS,sizeof(nargs),&nargs,0);
        result=clSetKernelArg(kernel,0,sizeof(output),&output);
        printf("arg0(size=%zu)=%d ", sizeof(output), result);
        struct Push {cl_float scale; cl_int bias; cl_uint count;} push={2.5f,-3,4};
        if(s==7 && result==CL_SUCCESS) { result=clSetKernelArg(kernel,1,sizeof(push),&push); printf("arg1(size=%zu)=%d ", sizeof(push), result); }
        printf("args=%u max=%zu required=%zu,%zu,%zu local=%llu set=%d ",nargs,maximum,required[0],required[1],required[2],(unsigned long long)local,result);
        size_t global_size[3]={4,1,1},local_size[3]={4,1,1};
        if(result==CL_SUCCESS)result=clEnqueueNDRangeKernel(queue,kernel,3,0,global_size,local_size,0,0,0);
        printf("enqueue3D=%d ",result);
        if (result != CL_SUCCESS) {result=clEnqueueNDRangeKernel(queue,kernel,1,0,global_size,local_size,0,0,0); printf("enqueue1D=%d ",result);}
        cl_uint values[4]={};
        if(result==CL_SUCCESS)result=clEnqueueReadBuffer(queue,output,CL_TRUE,0,sizeof(values),values,0,0,0);
        printf("read=%d output=%u,%u,%u,%u\n",result,values[0],values[1],values[2],values[3]);
        fflush(stdout);
        clReleaseKernel(kernel);clReleaseProgram(program);
    }
    clReleaseMemObject(output);clReleaseCommandQueue(queue);clReleaseContext(context);
}
