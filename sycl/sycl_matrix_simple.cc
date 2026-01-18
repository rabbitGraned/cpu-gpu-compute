/*
* CPU-GPU-compute examples
* License: GNU GPL v3
* **
* A simple SYCL application for matrix multiplication.
*
* ICPX:    icpx sycl_matrix.cc -o sycl_matrix.exe -fsycl
*/

#include <sycl/sycl.hpp>

#include <iostream>
#include <vector>

constexpr unsigned int Tile = 16;
constexpr unsigned int N = 256;
constexpr size_t matrixSize = N * N;

int main() try {
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

    std::cout << "Selected device: " << selectedDevice.get_info<sycl::info::device::name>() << "\n";

    std::vector<float> hostA(matrixSize);
    std::vector<float> hostB(matrixSize);
    std::vector<float> hostC(matrixSize, 0.0f);

    for (unsigned int i = 0; i < N; ++i) {
        for (unsigned int j = 0; j < N; ++j) {
            hostA[i * N + j] = static_cast<float>(i + j);
            hostB[i * N + j] = static_cast<float>(i * j + 1);
        }
    }

    sycl::queue q(selectedDevice, sycl::property::queue::in_order{});

    sycl::buffer<float, 1> bufA(hostA.data(), sycl::range<1>(matrixSize));
    sycl::buffer<float, 1> bufB(hostB.data(), sycl::range<1>(matrixSize));
    sycl::buffer<float, 1> bufC(hostC.data(), sycl::range<1>(matrixSize));

    /* SYCL kernel */

    q.submit([&](sycl::handler& cgh) {
        auto accA = bufA.get_access<sycl::access::mode::read>(cgh);
        auto accB = bufB.get_access<sycl::access::mode::read>(cgh);
        auto accC = bufC.get_access<sycl::access::mode::write>(cgh);

        // Local accessors for Tiles
        sycl::local_accessor<float, 2> Asub(sycl::range<2>(Tile, Tile), cgh);
        sycl::local_accessor<float, 2> Bsub(sycl::range<2>(Tile, Tile), cgh);

        cgh.parallel_for(
            sycl::nd_range<2>(
                sycl::range<2>(N, N),         /* global range */
                sycl::range<2>(Tile, Tile)    /* local range (work-group size) */
            ),
            [=](sycl::nd_item<2> item) [[sycl::reqd_work_group_size(Tile, Tile)]] {
                const int tx = item.get_local_id(0);
                const int ty = item.get_local_id(1);

                const int row = item.get_group(0) * Tile + tx;
                const int col = item.get_group(1) * Tile + ty;

                float sum = 0.0f;

                const int numTiles = (N + Tile - 1) / Tile; // ceil(N / Tile)

                for (int t = 0; t < numTiles; ++t) {
                    const int k = t * Tile;

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

    q.wait_and_throw();

    std::cout << "done. Matrix multiplication completed.\n";

#ifdef OUT
    hostC = std::vector<float>(bufC.get_access<sycl::access::mode::read>().get_pointer(),
        bufC.get_access<sycl::access::mode::read>().get_pointer() + matrixSize);
    std::cout << "C[0][0] = " << hostC[0] << "\n";
    std::cout << "C[" << N - 1 << "][" << N - 1 << "] = " << hostC[N * N - 1] << "\n";
#endif

    return EXIT_SUCCESS;
}
catch (const sycl::exception& e) {
    std::cerr << "SYCL exception: " << e.what() << "\n";
    std::cerr << "OpenCL error: " << e.code() << "\n";
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