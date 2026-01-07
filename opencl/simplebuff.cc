/*
* CPU-GPU-compute examples
* License: GNU GPL v3
* **
* A simple application for reading and writing data on an OpenCL device: C++ variation.
* 
* ICPX:    icpx simplebuff.cc - o simplebuff.exe - lOpenCL
*/

#include <iostream>
#include <vector>
#include <cstdlib>
#include <memory>

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>

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

    // Create Context
    cl::Context context(selectedDevice);
    std::string deviceName = selectedDevice.getInfo<CL_DEVICE_NAME>();
    std::cout << "Selected device: " << deviceName << "\n";

    constexpr size_t N = 1024;
    std::vector<float> hostInput(N);
    for (size_t i = 0; i < N; ++i) {
        hostInput[i] = static_cast<float>(i * 2 + 1);
    }
    std::vector<float> hostOutput(N);

    cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        N * sizeof(float), hostInput.data());
    cl::Buffer outputBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(float));

#if CL_HPP_TARGET_OPENCL_VERSION >= 200
    cl::CommandQueue queue(context, selectedDevice, cl::QueueProperties{});
#else
    cl::CommandQueue queue(context, selectedDevice, 0);
#endif

    std::cout << "Buffer has been sent to the device.\n";

    // inputBuffer to outputBuffer
    queue.enqueueCopyBuffer(inputBuffer, outputBuffer, 0, 0, N * sizeof(float));

    // Device to Host
    cl::copy(queue, outputBuffer, hostOutput.begin(), hostOutput.end());

    queue.finish();
    std::cout << "done. The buffer has been on the GPU.\n";

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