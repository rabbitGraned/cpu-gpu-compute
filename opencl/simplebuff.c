/*
* CPU-GPU-compute examples
* License: GNU GPL v3
* **
* A simple application for reading and writing data on an OpenCL device: C variation
*
* ICX:    icx simplebuff.c -o simplebuff OpenCL.lib
*/

#include <stdio.h>
#include <stdlib.h>

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif

#include <CL/cl.h>

int main(void)
{
    cl_uint platformCount = 0;
    cl_int err = clGetPlatformIDs(0, NULL, &platformCount);
    if (err != CL_SUCCESS || platformCount == 0) {
        fprintf(stderr, "No OpenCL platform found.\n");
        return EXIT_FAILURE;
    }

    cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platformCount);
    if (!platforms) {
        fprintf(stderr, "Failed to allocate memory for platforms.\n");
        return EXIT_FAILURE;
    }

    err = clGetPlatformIDs(platformCount, platforms, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to retrieve OpenCL platforms.\n");
        free(platforms);
        return EXIT_FAILURE;
    }

    cl_device_id selectedDevice = NULL;
    cl_context context = NULL;
    cl_platform_id selectedPlatform = NULL;

    for (cl_uint i = 0; i < platformCount && !selectedDevice; ++i) {
        cl_uint deviceCount = 0;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &deviceCount);
        if (err != CL_SUCCESS || deviceCount == 0) continue;

        cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id) * deviceCount);
        if (!devices) continue;

        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, deviceCount, devices, NULL);
        if (err != CL_SUCCESS) {
            free(devices);
            continue;
        }

        // We choose ANY GPU, just to CL_DEVICE_MAX_COMPUTE_UNITS > 0
        for (cl_uint j = 0; j < deviceCount; ++j) {
            cl_uint maxComputeUnits = 0;
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);
            if (maxComputeUnits > 0) {
                selectedDevice = devices[j];
                selectedPlatform = platforms[i];
                break;
            }
        }

        free(devices);
    }

    free(platforms);

    if (!selectedDevice) {
        fprintf(stderr, "No suitable GPU device found.\n");
        return EXIT_FAILURE;
    }

    context = clCreateContext(NULL, 1, &selectedDevice, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create OpenCL context.\n");
        return EXIT_FAILURE;
    }

    char deviceName[256] = { 0 };
    clGetDeviceInfo(selectedDevice, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
    printf("Selected GPU: %s\n", deviceName);

    const char* kernelSource =
        "__kernel void copy_kernel(__global const float* input, __global float* output) {\n"
        "    int id = get_global_id(0);\n"
        "    output[id] = input[id];\n"
        "}\n";

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create OpenCL program.\n");
        clReleaseContext(context);
        return EXIT_FAILURE;
    }

    err = clBuildProgram(program, 1, &selectedDevice, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, selectedDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        char* buildLog = (char*)malloc(logSize + 1);
        clGetProgramBuildInfo(program, selectedDevice, CL_PROGRAM_BUILD_LOG, logSize, buildLog, NULL);
        buildLog[logSize] = '\0';
        fprintf(stderr, "Build error:\n%s\n", buildLog);
        free(buildLog);
        clReleaseProgram(program);
        clReleaseContext(context);
        return EXIT_FAILURE;
    }

    cl_kernel kernel = clCreateKernel(program, "copy_kernel", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create kernel.\n");
        clReleaseProgram(program);
        clReleaseContext(context);
        return EXIT_FAILURE;
    }

    const size_t bufferSize = 1024 * sizeof(float);
    float* hostBuffer = (float*)malloc(bufferSize);
    if (!hostBuffer) {
        fprintf(stderr, "Failed to allocate host buffer.\n");
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseContext(context);
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < 1024; ++i) {
        hostBuffer[i] = (float)(i * 2 + 1);
    }

    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        bufferSize, hostBuffer, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create input buffer.\n");
        free(hostBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseContext(context);
        return EXIT_FAILURE;
    }

    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bufferSize, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create output buffer.\n");
        clReleaseMemObject(inputBuffer);
        free(hostBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseContext(context);
        return EXIT_FAILURE;
    }

    cl_command_queue queue;
#if defined(CL_VERSION_2_0) && CL_TARGET_OPENCL_VERSION >= 200
    // OpenCL 2.0
    queue = clCreateCommandQueueWithProperties(context, selectedDevice, NULL, &err);
#else
    // OpenCL 1.0
    queue = clCreateCommandQueue(context, selectedDevice, 0, &err);
#endif
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create command queue.\n");
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
        free(hostBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseContext(context);
        return EXIT_FAILURE;
        return EXIT_FAILURE;
    }
/*
    cl_command_queue queue = clCreateCommandQueue(context, selectedDevice, 0, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create command queue.\n");
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
        free(hostBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseContext(context);
        return EXIT_FAILURE;
    }
*/

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);

    size_t globalWorkSize = 1024;
    printf("Buffer has been sent to the GPU.\n");
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to enqueue kernel.\n");
    }
    else {
        clFinish(queue);
        printf("done. The buffer has been on the GPU.\n");

        float* result = (float*)malloc(bufferSize);
        clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, bufferSize, result, 0, NULL, NULL);
        free(result);
    }

    clReleaseCommandQueue(queue);
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    free(hostBuffer);

    return EXIT_SUCCESS;
}