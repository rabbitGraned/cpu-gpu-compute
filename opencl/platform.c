/*
* CPU-GPU-compute examples
* License: GNU GPL v3
* **
* An app for displaying available platforms.
*
* ICX:    icx platform.c -o platform OpenCL.lib
*/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif

#include "CL/cl.h"

void handle_error(cl_int err, const char* file, int line) {
    if (err == CL_SUCCESS) return;
    const char* msg = "Unknown error";
    switch (err) {
    case CL_DEVICE_NOT_FOUND:        return;
    case CL_INVALID_VALUE:           msg = "Invalid value"; break;
    case CL_INVALID_DEVICE_TYPE:     msg = "Invalid device type"; break;
    case CL_INVALID_PLATFORM:        msg = "Invalid platform"; break;
    case CL_INVALID_DEVICE:          msg = "Invalid device"; break;
    case CL_OUT_OF_HOST_MEMORY:      msg = "Out of host memory"; break;
    case CL_OUT_OF_RESOURCES:        msg = "Out of resources"; break;
    }
    fprintf(stderr, "Error: %s (%d) at %s:%d\n", msg, err, file, line);
    abort();
}

#define CL_CHECK(err) handle_error((err), __FILE__, __LINE__)

int main(void)
{
    cl_uint nplatforms;
    CL_CHECK(clGetPlatformIDs(0, NULL, &nplatforms));
    if (nplatforms == 0) {
        fprintf(stderr, "No OpenCL platforms found.\n");
        return 1;
    }

    cl_platform_id* platforms = malloc(nplatforms * sizeof(cl_platform_id));
    assert(platforms);
    CL_CHECK(clGetPlatformIDs(nplatforms, platforms, NULL));

    char buf[2048];
    for (cl_uint p = 0; p < nplatforms; ++p) {
        cl_platform_id pid = platforms[p];

        CL_CHECK(clGetPlatformInfo(pid, CL_PLATFORM_NAME, sizeof(buf), buf, NULL));
        printf("Platform: %s\n", buf);
        CL_CHECK(clGetPlatformInfo(pid, CL_PLATFORM_VERSION, sizeof(buf), buf, NULL));
        printf("Version: %s\n", buf);
        CL_CHECK(clGetPlatformInfo(pid, CL_PLATFORM_VENDOR, sizeof(buf), buf, NULL));
        printf("Vendor: %s\n", buf);
        printf("\n");

        cl_uint ndevices;
        CL_CHECK(clGetDeviceIDs(pid, CL_DEVICE_TYPE_ALL, 0, NULL, &ndevices));
        if (ndevices == 0) continue;

        cl_device_id* devices = malloc(ndevices * sizeof(cl_device_id));
        assert(devices);
        CL_CHECK(clGetDeviceIDs(pid, CL_DEVICE_TYPE_ALL, ndevices, devices, NULL));

        for (cl_uint d = 0; d < ndevices; ++d) {
            cl_device_id did = devices[d];

            CL_CHECK(clGetDeviceInfo(did, CL_DEVICE_NAME, sizeof(buf), buf, NULL));
            printf("  Device: %s\n", buf);
            CL_CHECK(clGetDeviceInfo(did, CL_DEVICE_VERSION, sizeof(buf), buf, NULL));
            printf("  OpenCL version: %s\n", buf);

            cl_uint cu;
            CL_CHECK(clGetDeviceInfo(did, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL));
            printf("  Compute units: %u\n", cu);

            cl_uint dims;
            CL_CHECK(clGetDeviceInfo(did, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(dims), &dims, NULL));
            size_t* sizes = malloc(dims * sizeof(size_t));
            CL_CHECK(clGetDeviceInfo(did, CL_DEVICE_MAX_WORK_ITEM_SIZES, dims * sizeof(size_t), sizes, NULL));
            printf("  Max work item sizes: ");
            for (cl_uint i = 0; i < dims; ++i) printf("%zu ", sizes[i]);
            printf("\n");
            free(sizes);

            size_t wgsize;
            CL_CHECK(clGetDeviceInfo(did, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(wgsize), &wgsize, NULL));
            printf("  Max work group size: %zu\n", wgsize);

            cl_bool compiler, linker;
            CL_CHECK(clGetDeviceInfo(did, CL_DEVICE_COMPILER_AVAILABLE, sizeof(compiler), &compiler, NULL));
            CL_CHECK(clGetDeviceInfo(did, CL_DEVICE_LINKER_AVAILABLE, sizeof(linker), &linker, NULL));
            printf("  Compiler: %s\n", compiler ? "yes" : "no");
            printf("  Linker: %s\n", linker ? "yes" : "no");
            printf("\n");
        }

        free(devices);
    }

    free(platforms);
    return 0;
}
