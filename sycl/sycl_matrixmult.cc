/*
* CPU-GPU-compute examples
* License: GNU GPL v3
* **
* A simple SYCL application for matrix multiplication.
*
* ICPX:    icpx sycl_matrixmult.cc -o sycl_matrix_simple.exe -fsycl -std=c++20 -DSIMPLE
*          icpx sycl_matrixmult.cc -o sycl_matrix_private.exe -fsycl -std=c++20 -DPRIVATE
*          icpx sycl_matrixmult.cc -o sycl_matrix_local.exe -fsycl -std=c++20 -DLOCALMEM
* 
*          Add -DCPU for comparison with CPU.
*/

#include <sycl/sycl.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <charconv>
#include <string_view>

#include <cstdlib>

#include <algorithm>
#include <cstring>

struct Config {
    unsigned int N = 256;
    unsigned int Tile = 16;
};

Config parseArgs(int argc, char* argv[]) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg.size() >= 6 && arg.substr(0, 6) == "-size=") {
            auto res = std::from_chars(arg.data() + 6, arg.data() + arg.size(), cfg.N);
            if (res.ec != std::errc{}) {
                std::cerr << "Invalid -size value\n";
                std::exit(EXIT_FAILURE);
            }
        }
        else if (arg.size() >= 6 && arg.substr(0, 6) == "-tile=") {
            auto res = std::from_chars(arg.data() + 6, arg.data() + arg.size(), cfg.Tile);
            if (res.ec != std::errc{}) {
                std::cerr << "Invalid -tile value\n";
                std::exit(EXIT_FAILURE);
            }
        }
        else {
            std::cerr << "Unknown option: " << arg << "\n";
            std::exit(EXIT_FAILURE);
        }
    }
    return cfg;
}

void rand_init(std::vector<float>& v, float low, float high) {
    static std::mt19937_64 gen(42);
    std::uniform_real_distribution<float> dist(low, high);
    for (auto& x : v) x = dist(gen);
}

#ifdef CPU
void tiled_mult_cpu(const float* A, const float* B, float* C, unsigned int N, unsigned int Tile) {
    for (unsigned int i = 0; i < N * N; ++i) C[i] = 0.0f;

    const unsigned int numTiles = (N + Tile - 1) / Tile;

    for (unsigned int ti = 0; ti < numTiles; ++ti) {
        for (unsigned int tj = 0; tj < numTiles; ++tj) {
            for (unsigned int tk = 0; tk < numTiles; ++tk) {

                unsigned int i_start = ti * Tile;
                unsigned int j_start = tj * Tile;
                unsigned int k_start = tk * Tile;

                unsigned int i_end = std::min(i_start + Tile, N);
                unsigned int j_end = std::min(j_start + Tile, N);
                unsigned int k_end = std::min(k_start + Tile, N);

                for (unsigned int i = i_start; i < i_end; ++i) {
                    for (unsigned int j = j_start; j < j_end; ++j) {
                        float sum = C[i * N + j];
                        for (unsigned int k = k_start; k < k_end; ++k) {
                            sum += A[i * N + k] * B[k * N + j];
                        }
                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }
}
#endif

int main(int argc, char* argv[]) {
    try {
        Config cfg = parseArgs(argc, argv);
        const unsigned int N = cfg.N;
        const unsigned int Tile = cfg.Tile;
        const size_t matrixSize = N * N;

        std::cout << "Matrix size: " << N << " x " << N << "\n";
        std::cout << "Tile size: " << Tile << "\n\n";

#if defined(PRIVATE)
        std::cout << "Uses PRIVATE memory.\n\n";
#elif defined(SIMPLE)
        std::cout << "Uses SIMPLE matrix.\n\n";
#elif defined(LOCALMEM) || !(defined(PRIVATE))
        std::cout << "Uses LOCAL memory.\n\n";
#else
        std::cout << "Ifdef occasion.\n";
#endif

        sycl::device selectedDevice;
        bool found = false;
        for (const auto& platform : sycl::platform::get_platforms()) {
            for (const auto& device : platform.get_devices()) {
                if (device.is_gpu() && device.get_info<sycl::info::device::max_compute_units>() > 0) {
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

        std::cout << "Selected GPU: " << selectedDevice.get_info<sycl::info::device::name>() << " [" << std::hex << selectedDevice.get_info<sycl::info::device::vendor_id>() << std::dec << "]\n";
        std::cout << "Driver version:  " << selectedDevice.get_info<sycl::info::device::driver_version>() << "\n\n";

        std::cout << "SYCL runtime: " << selectedDevice.get_platform().get_info<sycl::info::platform::name>() << "\n\n";

        std::vector<float> hostA(matrixSize);
        std::vector<float> hostB(matrixSize);
        std::vector<float> hostC_gpu(matrixSize);
        std::vector<float> hostC_cpu(matrixSize);

        rand_init(hostA, 0.0f, 10.0f);
        rand_init(hostB, 0.0f, 10.0f);

        long cpuTimeMs = 0;
#ifdef CPU
        auto cpuStart = std::chrono::high_resolution_clock::now();
        tiled_mult_cpu(hostA.data(), hostB.data(), hostC_cpu.data(), N, Tile);
        auto cpuEnd = std::chrono::high_resolution_clock::now();
        cpuTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(cpuEnd - cpuStart).count();
#endif

        sycl::buffer<float, 1> bufA(hostA.data(), sycl::range<1>(matrixSize));
        sycl::buffer<float, 1> bufB(hostB.data(), sycl::range<1>(matrixSize));
        sycl::buffer<float, 1> bufC(hostC_gpu.data(), sycl::range<1>(matrixSize));

        sycl::queue q(selectedDevice, sycl::property::queue::enable_profiling{});

        sycl::event event;

#if defined(PRIVATE)
        if (N % Tile != 0) {
            std::cerr << "Error: In PRIVATE mode, N must be divisible by Tile.\n";
            return EXIT_FAILURE;
        }
        sycl::range<2> numGroups(N / Tile, N / Tile);
        sycl::range<2> localRange(Tile, Tile);

        event = q.submit([&](sycl::handler& cgh) {
            auto accA = bufA.get_access<sycl::access::mode::read>(cgh);
            auto accB = bufB.get_access<sycl::access::mode::read>(cgh);
            auto accC = bufC.get_access<sycl::access::mode::write>(cgh);

            sycl::local_accessor<float, 2> Asub(localRange, cgh);
            sycl::local_accessor<float, 2> Bsub(localRange, cgh);

            cgh.parallel_for_work_group<class HierarchicalMatMul>(
                numGroups, localRange,
                [=](sycl::group<2> grp) {
                    sycl::private_memory<float, 2> sum(grp);
                    grp.parallel_for_work_item([&](sycl::h_item<2> it) {
                        sum(it) = 0.0f;
                        });

                    const int numTiles = N / Tile;
                    for (int t = 0; t < numTiles; ++t) {
                        grp.parallel_for_work_item([&](sycl::h_item<2> it) {
                            int row = it.get_global_id(0);
                            int col = it.get_global_id(1);
                            int tx = it.get_local_id(0);
                            int ty = it.get_local_id(1);
                            int k = t * Tile;

                            Asub[tx][ty] = accA[row * N + (k + ty)];
                            Bsub[tx][ty] = accB[(k + tx) * N + col];
                            });

                        grp.parallel_for_work_item([&](sycl::h_item<2> it) {
                            int tx = it.get_local_id(0);
                            int ty = it.get_local_id(1);
                            for (int k = 0; k < Tile; ++k) {
                                sum(it) += Asub[tx][k] * Bsub[k][ty];
                            }
                            });
                    }

                    grp.parallel_for_work_item([&](sycl::h_item<2> it) {
                        int row = it.get_global_id(0);
                        int col = it.get_global_id(1);
                        accC[row * N + col] = sum(it);
                        });
                });
            });

#elif defined(SIMPLE)

        sycl::range<2> globalRange(N, N);

        event = q.submit([&](sycl::handler& cgh) {
            auto accA = bufA.get_access<sycl::access::mode::read>(cgh);
            auto accB = bufB.get_access<sycl::access::mode::read>(cgh);
            auto accC = bufC.get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for<class SimpleMatMul>(
                globalRange,
                [=](sycl::id<2> idx) {
                    int i = idx[0];
                    int j = idx[1];
                    float sum = 0.0f;
                    for (int k = 0; k < N; ++k) {
                        sum += accA[i * N + k] * accB[k * N + j];
                    }
                    accC[i * N + j] = sum;
                });
            });

#else // is default
        sycl::nd_range<2> ndRange(sycl::range<2>(N, N), sycl::range<2>(Tile, Tile));

        event = q.submit([&](sycl::handler& cgh) {
            auto accA = bufA.get_access<sycl::access::mode::read>(cgh);
            auto accB = bufB.get_access<sycl::access::mode::read>(cgh);
            auto accC = bufC.get_access<sycl::access::mode::write>(cgh);

            sycl::local_accessor<float, 2> Asub(sycl::range<2>(Tile, Tile), cgh);
            sycl::local_accessor<float, 2> Bsub(sycl::range<2>(Tile, Tile), cgh);

            cgh.parallel_for<class FlatMatMul>(
                ndRange,
                [=](sycl::nd_item<2> item) {
                    int tx = item.get_local_id(0);
                    int ty = item.get_local_id(1);
                    int row = item.get_group(0) * Tile + tx;
                    int col = item.get_group(1) * Tile + ty;

                    float sum = 0.0f;
                    int numTiles = (N + Tile - 1) / Tile;

                    for (int t = 0; t < numTiles; ++t) {
                        int k = t * Tile;

                        if (row < N && (k + ty) < N) {
                            Asub[tx][ty] = accA[row * N + (k + ty)];
                        }
                        else {
                            Asub[tx][ty] = 0.0f;
                        }

                        if ((k + tx) < N && col < N) {
                            Bsub[tx][ty] = accB[(k + tx) * N + col];
                        }
                        else {
                            Bsub[tx][ty] = 0.0f;
                        }

                        item.barrier(sycl::access::fence_space::local_space);

                        for (int k_local = 0; k_local < Tile; ++k_local) {
                            sum += Asub[tx][k_local] * Bsub[k_local][ty];
                        }

                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    if (row < N && col < N) {
                        accC[row * N + col] = sum;
                    }
                });
            });

#endif

        auto gpuWallStart = std::chrono::high_resolution_clock::now();
        q.wait_and_throw();
        auto gpuWallEnd = std::chrono::high_resolution_clock::now();
        long gpuWallTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(gpuWallEnd - gpuWallStart).count();

        uint64_t start_ns = event.get_profiling_info<sycl::info::event_profiling::command_start>();
        uint64_t end_ns = event.get_profiling_info<sycl::info::event_profiling::command_end>();
        long gpuKernelTimeMs = static_cast<long>((end_ns - start_ns) / 1'000'000);

        std::cout << "GPU wall time:    " << gpuWallTimeMs << " ms\n";
        std::cout << "GPU kernel time:  " << gpuKernelTimeMs << " ms\n";
#ifdef CPU
        std::cout << "CPU time:         " << cpuTimeMs << " ms\n";
#endif
        std::cout << "\ndone. Matrix multiplication completed.\n";

        return EXIT_SUCCESS;
    }
    catch (const sycl::exception& e) {
        std::cerr << "SYCL exception: " << e.what() << " (" << e.code() << ")\n";
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
}
