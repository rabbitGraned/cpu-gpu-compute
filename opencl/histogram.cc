/*
* CPU-GPU-compute examples
* License: GNU GPL v3
* **
* A simple OpenCL application for histogram calculations on GPU and natively on the CPU.
*
* ICPX:    icpx histogram.cc -o histogram.exe -O2 -std=c++20 -lOpenCL
* Usage:   histogram.exe -kernel=hist_atomic.cl -size=419430400 (as a sample)
*/

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <charconv>
#include <string_view>
#include <fstream>
#include <sstream>
#include <system_error>

#include <cstdlib>

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>

// HELPERS&CONFIG

struct Config {
    unsigned int N = 1'048'576;
    unsigned int Bins = 256;
    std::string kernelPath = "";
};

Config parseArgs(int argc, char* argv[]) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg.starts_with("-size=")) {
            auto res = std::from_chars(arg.data() + 6, arg.data() + arg.size(), cfg.N);
            if (res.ec != std::errc{}) {
                std::cerr << "Invalid -size value\n";
                std::exit(EXIT_FAILURE);
            }
        }
        else if (arg.starts_with("-bins=")) {
            auto res = std::from_chars(arg.data() + 6, arg.data() + arg.size(), cfg.Bins);
            if (res.ec != std::errc{}) {
                std::cerr << "Invalid -bins value\n";
                std::exit(EXIT_FAILURE);
            }
        }
        else if (arg.starts_with("-kernel=")) {
            cfg.kernelPath = std::string(arg.substr(8));
        }
        else {
            std::cerr << "Unknown option: " << arg << "\n";
            std::exit(EXIT_FAILURE);
        }
    }
    return cfg;
}

std::string readKernelFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open kernel file: " << path << "\n";
        std::exit(EXIT_FAILURE);
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

// CPU histogram

void rand_init(std::vector<unsigned int>& v, unsigned int maxVal) {
    static std::mt19937_64 gen;
    std::uniform_int_distribution<unsigned int> dist(0, maxVal - 1);
    for (auto& x : v) x = dist(gen);
}

void histogram_ref(const unsigned int* data, unsigned int* hist, unsigned int N, unsigned int bins) {
    std::fill(hist, hist + bins, 0);
    for (unsigned int i = 0; i < N; ++i) {
        unsigned int val = data[i];
        if (val < bins) {
            hist[val]++;
        }
    }
}

int main(int argc, char* argv[]) try {
    Config cfg = parseArgs(argc, argv);
    const unsigned int N = cfg.N;
    const unsigned int Bins = cfg.Bins;

    std::cout << "Input size: " << N << "\n";
    std::cout << "Histogram bins: " << Bins << "\n";
    std::cout << "Kernel file: " << cfg.kernelPath << "\n\n";

    std::string kernelSource = readKernelFile(cfg.kernelPath); /* Read kernel */
    std::string defines = "#define BINS " + std::to_string(Bins) + "\n";
    kernelSource = defines + kernelSource;

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
    std::cout << "Selected GPU: " << deviceName << "\n\n";

    std::vector<unsigned int> hostData(N);
    std::vector<unsigned int> hostHist_gpu(Bins, 0);
    std::vector<unsigned int> hostHist_cpu(Bins, 0);

    rand_init(hostData, Bins);

    auto cpuStart = std::chrono::high_resolution_clock::now();
    histogram_ref(hostData.data(), hostHist_cpu.data(), N, Bins);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    long cpuTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(cpuEnd - cpuStart).count();

    cl::Buffer bufferData(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        N * sizeof(unsigned int), hostData.data());
    cl::Buffer bufferHist(context, CL_MEM_WRITE_ONLY, Bins * sizeof(unsigned int));

    cl::CommandQueue queue(context, selectedDevice,
        cl::QueueProperties::Profiling | cl::QueueProperties::OutOfOrder);

    cl::Program program(context, kernelSource);
    program.build({ selectedDevice });

    cl::Kernel kernel(program, "histogram");
    kernel.setArg(0, bufferData);
    kernel.setArg(1, bufferHist);
    kernel.setArg(2, N);

    // 1D grid
    cl::NDRange globalSize(N);
    cl::NDRange localSize(256);

    auto gpuWallStart = std::chrono::high_resolution_clock::now();
    cl::Event event;
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize, nullptr, &event);
    queue.finish();
    auto gpuWallEnd = std::chrono::high_resolution_clock::now();
    long gpuWallTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(gpuWallEnd - gpuWallStart).count();

    cl::copy(queue, bufferHist, hostHist_gpu.begin(), hostHist_gpu.end());

    cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    long gpuKernelTimeMs = static_cast<long>((end - start) / 1'000'000);

    std::cout << "GPU wall time:    " << gpuWallTimeMs << " ms\n";
    std::cout << "GPU kernel time:  " << gpuKernelTimeMs << " ms\n";
    std::cout << "CPU time:         " << cpuTimeMs << " ms\n";

    // Simple correctness check
    bool correct = true;
    for (unsigned int i = 0; i < Bins; ++i) {
        if (hostHist_cpu[i] != hostHist_gpu[i]) {
            correct = false;
            break;
        }
    }
    std::cout << "Result correctness: " << (correct ? "PASSED" : "FAILED") << "\n";
    std::cout << "\ndone. Histogram computed.\n";

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