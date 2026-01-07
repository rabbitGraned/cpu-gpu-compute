/*
* CPU-GPU-compute examples
* License: GNU GPL v3
* **
* Checking the availability of SYCL on the GPU.
*
* ICPX:    icpx sycl_gpu_check.cc -o sycl_gpu_check.exe -fsycl
*/

#include <sycl/sycl.hpp>

#include <iostream>
#include <string>
#include <vector>

std::string format_memory(size_t bytes) {
    const char* units[] = { "B", "KB", "MB", "GB", "TB" };
    size_t unit = 0;
    double size = static_cast<double>(bytes);
    while (size >= 1024.0 && unit < 4) {
        size /= 1024.0;
        ++unit;
    }
    char buffer[64];
    if (unit == 0)
        snprintf(buffer, sizeof(buffer), "%.0f %s", size, units[unit]);
    else
        snprintf(buffer, sizeof(buffer), "%.1f %s", size, units[unit]);
    return std::string(buffer);
}

int main() {
    try {
        sycl::queue q{ sycl::gpu_selector_v };
        auto device = q.get_device();

        std::cout << "Device: " << device.get_info<sycl::info::device::name>() << "\n";
        std::cout << "Vendor: " << device.get_info<sycl::info::device::vendor>() << "\n";
        std::cout << "Local memory: " << format_memory(device.get_info<sycl::info::device::local_mem_size>()) << "\n";
        std::cout << "Global memory: " << format_memory(device.get_info<sycl::info::device::global_mem_size>()) << "\n";

        auto extensions = device.get_info<sycl::info::device::extensions>();
        if (!extensions.empty()) {
            std::cout << "Extensions: ";
            for (size_t i = 0; i < extensions.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << extensions[i];
            }
            std::cout << "\n";
        }

        q.submit([&](sycl::handler& h) { h.single_task([] {}); }).wait();
        std::cout << "GPU is available!\n";
    }
    catch (...) {
        std::cout << "Failed to run on GPU.\n";
    }
    return 0;
}