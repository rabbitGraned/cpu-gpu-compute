/*
* CPU-GPU-compute examples
* License: GNU GPL v3
* **
* A simple SYCL application for vector addition.
*
* ICPX: icpx sycl_vectoradd.cc -o sycl_vectoradd.exe -fsycl
*/

#include <sycl/sycl.hpp>

#include <iostream>
#include <vector>
#include <cstdlib>

constexpr size_t N = 2048;

int main() try {
    std::vector<sycl::platform> platforms = sycl::platform::get_platforms();
    if (platforms.empty()) {
        std::cerr << "No SYCL platform found.\n";
        return EXIT_FAILURE;
    }

    sycl::device selectedDevice;
    bool found = false;

    for (const auto& platform : platforms) {
        std::vector<sycl::device> devices;
        try {
            devices = platform.get_devices(sycl::info::device_type::gpu);
        }
        catch (const sycl::exception& e) {
            continue;
        }

        for (const auto& device : devices) {
            if (device.get_info<sycl::info::device::max_compute_units>() > 0) {
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

    std::string deviceName = selectedDevice.get_info<sycl::info::device::name>();
    std::cout << "Selected GPU: " << deviceName << "\n";

    std::vector<float> hostA(N);
    std::vector<float> hostB(N);
    std::vector<float> hostC(N);

    for (size_t i = 0; i < N; ++i) {
        hostA[i] = static_cast<float>(i);
        hostB[i] = static_cast<float>(i * 2);
    }

    sycl::queue q(selectedDevice);

    sycl::buffer<float, 1> bufferA(hostA.data(), sycl::range<1>(N));
    sycl::buffer<float, 1> bufferB(hostB.data(), sycl::range<1>(N));
    sycl::buffer<float, 1> bufferC(hostC.data(), sycl::range<1>(N));

    q.submit([&](sycl::handler& h) {
        auto accA = bufferA.get_access<sycl::access::mode::read>(h);
        auto accB = bufferB.get_access<sycl::access::mode::read>(h);
        auto accC = bufferC.get_access<sycl::access::mode::write>(h);

        const size_t localSize = 512; /* if localSize does not divide N equally, then SYCL exception:
                                                    Non-uniform work-groups are not supported by the target device (sycl:4) */
                                      /* avoid values above 256 (mostly); sometimes the threshold is 512 on embedded Intel GPUs */

        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(N), sycl::range<1>(localSize)),
            [=](sycl::nd_item<1> item) {
                const size_t id = item.get_global_id(0);
                if (id < N) {
                    accC[id] = accA[id] + accB[id];
                }
            });
        });

    q.wait();

    std::cout << "done. Vector addition completed.\n";

#ifdef OUT
    for (size_t i = 0; i < 10 && i < N; ++i) {
        std::cout << hostA[i] << " + " << hostB[i] << " = " << hostC[i] << "\n";
    }
#endif // OUT

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