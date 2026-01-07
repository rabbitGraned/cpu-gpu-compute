/*
* CPU-GPU-compute examples
* License: GNU GPL v3
* **
* A simple OpenCL application for matrix multiplication on GPU and natively on the CPU.
*
* ICPX:    icpx matrixmult_cpu_gpu.cc -o matrixmult_cpu_gpu.exe -O2 -std=c++20 -lOpenCL
* Usage:   matrixmult_cpu_gpu.exe -kernel=matrix_localmem.cl -size=1024 (as a sample)
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
    unsigned int N = 256;
    unsigned int Tile = 16;
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
        else if (arg.starts_with("-tile=")) {
            auto res = std::from_chars(arg.data() + 6, arg.data() + arg.size(), cfg.Tile);
            if (res.ec != std::errc{}) {
                std::cerr << "Invalid -tile value\n";
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
        std::cerr << "Failed to open kernel file " << path << "\n";
        std::exit(EXIT_FAILURE);
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

// CPU matrix

void rand_init(std::vector<float>& v, float low, float high) {
    static std::mt19937_64 gen;
    std::uniform_real_distribution<float> dist(low, high);
    for (auto& x : v) x = dist(gen);
}

void transpose_mult_ref(const float* A, const float* B, float* C, unsigned int N) {
    std::vector<float> Bt(N * N);
    for (unsigned int i = 0; i < N; ++i)
        for (unsigned int j = 0; j < N; ++j)
            Bt[j * N + i] = B[i * N + j];

    for (unsigned int i = 0; i < N; ++i) {
        for (unsigned int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (unsigned int k = 0; k < N; ++k)
                sum += A[i * N + k] * Bt[j * N + k];
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char* argv[]) try {
    Config cfg = parseArgs(argc, argv);
    const unsigned int N = cfg.N;
    const size_t matrixSize = N * N;

    std::cout << "Matrix size: " << N << " x " << N << "\n";
    std::cout << "Tile size: " << cfg.Tile << "\n";
    std::cout << "Kernel file: " << cfg.kernelPath << "\n\n";

    std::string kernelSource = readKernelFile(cfg.kernelPath); /* Read kernel */
    std::string defines = "#define TILE " + std::to_string(cfg.Tile) + "\n";
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

    std::vector<float> hostA(matrixSize);
    std::vector<float> hostB(matrixSize);
    std::vector<float> hostC_gpu(matrixSize);
    std::vector<float> hostC_cpu(matrixSize);

    rand_init(hostA, 0.0f, 10.0f);
    rand_init(hostB, 0.0f, 10.0f);

    auto cpuStart = std::chrono::high_resolution_clock::now();
    transpose_mult_ref(hostA.data(), hostB.data(), hostC_cpu.data(), N);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    long cpuTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(cpuEnd - cpuStart).count();

    cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        matrixSize * sizeof(float), hostA.data());
    cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        matrixSize * sizeof(float), hostB.data());
    cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, matrixSize * sizeof(float));

    cl::CommandQueue queue(context, selectedDevice,
        cl::QueueProperties::Profiling | cl::QueueProperties::OutOfOrder);

    cl::Program program(context, kernelSource);
    program.build({ selectedDevice });

    cl::Kernel kernel(program, "matrixmult");
    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferB);
    kernel.setArg(2, bufferC);
    kernel.setArg(3, N);

    cl::NDRange globalSize(N, N);
    cl::NDRange localSize(cfg.Tile, cfg.Tile);

    auto gpuWallStart = std::chrono::high_resolution_clock::now();
    cl::Event event;
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize, nullptr, &event);
    queue.finish();

    auto gpuWallEnd = std::chrono::high_resolution_clock::now();
    long gpuWallTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(gpuWallEnd - gpuWallStart).count();

    cl::copy(queue, bufferC, hostC_gpu.begin(), hostC_gpu.end());

    cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    long gpuKernelTimeMs = static_cast<long>((end - start) / 1'000'000);

    std::cout << "GPU wall time:    " << gpuWallTimeMs << " ms\n";
    std::cout << "GPU kernel time:  " << gpuKernelTimeMs << " ms\n";
    std::cout << "CPU time:         " << cpuTimeMs << " ms\n";

    std::cout << "\ndone. Matrix multiplication completed.\n";

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