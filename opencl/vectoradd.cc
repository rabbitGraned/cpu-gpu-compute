/*
* CPU-GPU-compute examples
* License: GNU GPL v3
* **
* A simple OpenCL application for vector addition.
* 
* ICPX: icpx vectoradd.cc -o vectoradd.exe -lOpenCL
*/

#include <iostream>
#include <vector>
#include <cstdlib>
#include <memory>
#include <string>

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>

//  OpenCL
const char* vectorAddKernel = R"(
__kernel void vector_add(__global const float* A,
                         __global const float* B,
                         __global float* C,
                         const unsigned int n) {
    unsigned int id = get_global_id(0);
    if (id < n) {
        C[id] = A[id] + B[id];
    }
}
)";
//  OpenCL

int main() try {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        std::cerr << "No OpenCL platform found.\n";
        return EXIT_FAILURE;
    }

    cl::Device selectedDevice;
    bool found = false;

    for (auto& platform : platforms) {
        std::vector<cl::Device> devices;
        try {
            platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        }
        catch (const cl::Error& e) {
            if (e.err() == CL_DEVICE_NOT_FOUND) continue;
            throw;
        }

        for (auto& device : devices) {
            if (device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() > 0) {
                selectedDevice = device;
                found = true;
                break;
            }
        }
        if (found) break;
    }

    if (!found) {
        std::cerr << "No suitable GPU device found.\n";
        return EXIT_FAILURE;
    }

    cl::Context context(selectedDevice);
    std::string deviceName = selectedDevice.getInfo<CL_DEVICE_NAME>();
    std::cout << "Selected GPU: " << deviceName << "\n";

    constexpr size_t N = 64; // MIN VALUE
    std::vector<float> hostA(N);
    std::vector<float> hostB(N);
    std::vector<float> hostC(N);

    // Vector's Data
    for (size_t i = 0; i < N; ++i) {
        hostA[i] = static_cast<float>(i);
        hostB[i] = static_cast<float>(i * 2);
    }

    // Creating Buffers
    cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       N * sizeof(float), hostA.data());
    cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       N * sizeof(float), hostB.data());
    cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, N * sizeof(float));

#if CL_HPP_TARGET_OPENCL_VERSION >= 200
    cl::CommandQueue queue(context, selectedDevice, cl::QueueProperties::None);
#else
    cl::CommandQueue queue(context, selectedDevice, 0);
#endif

    // Compiling the program
    cl::Program program(context, vectorAddKernel);
    // cl::Program program(vectorAddKernel); /* OpenCL error: clGetProgramBuildInfo (-33) */
    program.build({selectedDevice});

    cl::Kernel kernel(program, "vector_add");
    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferB);
    kernel.setArg(2, bufferC);
    kernel.setArg(3, static_cast<cl_uint>(N));

    cl::NDRange globalSize(N);
    cl::NDRange localSize(256);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);

    // Device to Host
    cl::copy(queue, bufferC, hostC.begin(), hostC.end());

    queue.finish();
    std::cout << "done. Vector addition completed.\n";

    // (optional) output of the first values
#ifdef OUT
    for (size_t i = 0; i < 10; ++i)
        std::cout << hostA[i] << " + " << hostB[i] << " = " << hostC[i] << "\n";
#endif // OUT

    return EXIT_SUCCESS;
}
catch (const cl::Error& e) {
    std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")\n";
    return EXIT_FAILURE;
}
catch (const std::exception& e) {
    std::cerr << "Standard exception: " << e.what() << "\n";
    return EXIT_FAILURE;
}
catch (...) {
    std::cerr << "Unknown error occurred.\n";
    return EXIT_FAILURE;
}