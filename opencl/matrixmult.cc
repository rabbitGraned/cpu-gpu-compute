/*
* CPU-GPU-compute examples
* License: GNU GPL v3
* **
* A simple OpenCL application for matrix multiplication.
*
* ICPX:    icpx matrixmult.cc -o matrixmult.exe -O2 -std=c++20 -lOpenCL -DLOCALMEM
*/

#include <iostream>
#include <vector>
#include <memory>
#include <string>

#include <cstdlib>

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>

/* OpenCL */

#ifdef SIMPLE

const char* matmulKernel = R"(
__kernel void matrixmult(__global const float* A,
                           __global const float* B,
                           __global float* C,
                           const unsigned int N) {
    unsigned int row = get_global_id(0);
    unsigned int col = get_global_id(1);

    if (row < N && col < N) {
        float sum = 0.0f;
        for (unsigned int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
)";
#endif // SIMPLE

#ifdef LOCALMEM

const char* matmulKernel = R"(
#define TILE 16

__kernel void matrixmult(__global const float* A,
                         __global const float* B,
                         __global float* C,
                         const unsigned int N)
{
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);

    const int row = get_group_id(0) * TILE + tx;
    const int col = get_group_id(1) * TILE + ty;

    __local float Asub[TILE][TILE];
    __local float Bsub[TILE][TILE];

    float sum = 0.0f;

    const int numTiles = (N + TILE - 1) / TILE; // ceil(N / TILE)
    for (int t = 0; t < numTiles; ++t) {
        const int k = t * TILE;

        if (row < N && (k + ty) < N) {
            Asub[tx][ty] = A[row * N + (k + ty)];
        } else {
            Asub[tx][ty] = 0.0f;
        }

        if ((k + tx) < N && col < N) {
            Bsub[tx][ty] = B[(k + tx) * N + col];
        } else {
            Bsub[tx][ty] = 0.0f;
        }

        // SYNC
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k_local = 0; k_local < TILE; ++k_local) {
            sum += Asub[tx][k_local] * Bsub[k_local][ty];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
)";
#endif // LOCALMEM

/* OpenCL */

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
    std::cout << "Selected device: " << deviceName << "\n";

    constexpr unsigned int N = 256; // SIZE
    const size_t matrixSize = N * N;

    std::vector<float> hostA(matrixSize);
    std::vector<float> hostB(matrixSize);
    std::vector<float> hostC(matrixSize, 0.0f);

    // A[i][j] = i + j, B[i][j] = i * j + 1
    for (unsigned int i = 0; i < N; ++i) {
        for (unsigned int j = 0; j < N; ++j) {
            hostA[i * N + j] = static_cast<float>(i + j);
            hostB[i * N + j] = static_cast<float>(i * j + 1);
        }
    }

    // Creating Buffers
    cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        matrixSize * sizeof(float), hostA.data());
    cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        matrixSize * sizeof(float), hostB.data());
    cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY,
        matrixSize * sizeof(float));

#if CL_HPP_TARGET_OPENCL_VERSION >= 200
    cl::CommandQueue queue(context, selectedDevice, cl::QueueProperties::None);
#else
    cl::CommandQueue queue(context, selectedDevice, 0);
#endif

    // Compiling the program
    cl::Program program(context, matmulKernel);
    program.build({ selectedDevice });

    cl::Kernel kernel(program, "matrixmult");
    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferB);
    kernel.setArg(2, bufferC);
    kernel.setArg(3, N);

    // 2D grid: N x N
    cl::NDRange globalSize(N, N);
    cl::NDRange localSize(16, 16); // 256 work-items per group
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);

    // Device to Host
    cl::copy(queue, bufferC, hostC.begin(), hostC.end());

    queue.finish();
    std::cout << "done. Matrix multiplication completed.\n";

    // (optional) output of the first values
#ifdef OUT
    std::cout << "C[0][0] = " << hostC[0] << "\n";
    std::cout << "C[" << N-1 << "][" << N-1 << "] = " << hostC[N*N - 1] << "\n";
#endif // OUPUT

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