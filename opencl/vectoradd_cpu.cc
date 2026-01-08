/*
* CPU-GPU-compute examples
* License: GNU GPL v3
* **
* An example of OpenCL offloading compute on both GPU and CPU.
*
* ICPX: icpx vectoradd_cpu.cc -o vectoradd_cpu.exe -O2 -lOpenCL
*/

#include <iostream>
#include <vector>
#include <cstdlib>
#include <memory>
#include <string>
#include <chrono>

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>

// OpenCL
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
// OpenCL

cl::Device findDevice(cl_device_type type, const std::string& typeName) {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        throw std::runtime_error("No OpenCL platforms found.");
    }

    for (auto& platform : platforms) {
        std::vector<cl::Device> devices;
        try {
            platform.getDevices(type, &devices);
        }
        catch (const cl::Error& e) {
            if (e.err() == CL_DEVICE_NOT_FOUND) continue;
            throw;
        }

        for (auto& device : devices) {
            if (device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() > 0) {
                return device;
            }
        }
    }
    throw std::runtime_error("No suitable " + typeName + " device found.");
}

int main() try {
    // Íàéä¸ì óñòðîéñòâà
    cl::Device gpuDevice = findDevice(CL_DEVICE_TYPE_GPU, "GPU");
    cl::Device cpuDevice = findDevice(CL_DEVICE_TYPE_CPU, "CPU");

    std::cout << "Selected GPU: " << gpuDevice.getInfo<CL_DEVICE_NAME>() << "\n";
    std::cout << "Selected CPU: " << cpuDevice.getInfo<CL_DEVICE_NAME>() << "\n";

    constexpr size_t N = 1 << 26;
    std::vector<float> hostA(N);
    std::vector<float> hostB(N);
    for (size_t i = 0; i < N; ++i) {
        hostA[i] = static_cast<float>(i);
        hostB[i] = static_cast<float>(i * 2);
    }

    cl::Context gpuContext(gpuDevice);
    cl::CommandQueue gpuQueue(gpuContext, gpuDevice, cl::QueueProperties::Profiling);

    auto gpuWallStart = std::chrono::high_resolution_clock::now();

    cl::Buffer gpuBufA(gpuContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        N * sizeof(float), hostA.data());
    cl::Buffer gpuBufB(gpuContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        N * sizeof(float), hostB.data());
    cl::Buffer gpuBufC(gpuContext, CL_MEM_WRITE_ONLY, N * sizeof(float));

    cl::Program gpuProgram(gpuContext, vectorAddKernel);
    gpuProgram.build({ gpuDevice });
    cl::Kernel gpuKernel(gpuProgram, "vector_add");
    gpuKernel.setArg(0, gpuBufA);
    gpuKernel.setArg(1, gpuBufB);
    gpuKernel.setArg(2, gpuBufC);
    gpuKernel.setArg(3, static_cast<cl_uint>(N));

    cl::Event gpuEvent;
    cl::NDRange globalSize(N);
    gpuQueue.enqueueNDRangeKernel(gpuKernel, cl::NullRange, globalSize, cl::NullRange, nullptr, &gpuEvent);
    gpuQueue.finish();

    std::vector<float> gpuResult(N);
    cl::copy(gpuQueue, gpuBufC, gpuResult.begin(), gpuResult.end());

    auto gpuWallEnd = std::chrono::high_resolution_clock::now();
    long gpuWallTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(gpuWallEnd - gpuWallStart).count();

    cl_ulong gpuStart = gpuEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong gpuEnd = gpuEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    long gpuKernelTimeMs = (gpuEnd - gpuStart) / 1'000'000.0;

    cl::Context cpuContext(cpuDevice);
    cl::CommandQueue cpuQueue(cpuContext, cpuDevice, cl::QueueProperties::None);

    auto cpuWallStart = std::chrono::high_resolution_clock::now();

    cl::Buffer cpuBufA(cpuContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        N * sizeof(float), hostA.data());
    cl::Buffer cpuBufB(cpuContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        N * sizeof(float), hostB.data());
    cl::Buffer cpuBufC(cpuContext, CL_MEM_WRITE_ONLY, N * sizeof(float));

    cl::Program cpuProgram(cpuContext, vectorAddKernel);
    cpuProgram.build({ cpuDevice });
    cl::Kernel cpuKernel(cpuProgram, "vector_add");
    cpuKernel.setArg(0, cpuBufA);
    cpuKernel.setArg(1, cpuBufB);
    cpuKernel.setArg(2, cpuBufC);
    cpuKernel.setArg(3, static_cast<cl_uint>(N));

    cpuQueue.enqueueNDRangeKernel(cpuKernel, cl::NullRange, globalSize, cl::NullRange);
    cpuQueue.finish();

    std::vector<float> cpuResult(N);
    cl::copy(cpuQueue, cpuBufC, cpuResult.begin(), cpuResult.end());

    auto cpuWallEnd = std::chrono::high_resolution_clock::now();
    long cpuTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(cpuWallEnd - cpuWallStart).count();
    
    std::cout << "GPU wall time:    " << gpuWallTimeMs << " ms\n";
    std::cout << "GPU kernel time:  " << gpuKernelTimeMs << " ms\n";
    std::cout << "CPU time:         " << cpuTimeMs << " ms\n";

    bool match = true;
    for (size_t i = 0; i < std::min(N, size_t(1000)); ++i) {
        if (std::abs(gpuResult[i] - cpuResult[i]) > 1e-4f) {
            match = false;
            break;
        }
    }
    if (!match) {
        std::cerr << "Warning: CPU and GPU results differ!\n";
    }

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

