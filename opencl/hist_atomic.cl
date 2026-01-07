/*
* CPU-GPU-compute examples
* License: GNU GPL v3
* 
* hist_atomic OpenCL kernel
*/

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

__kernel void histogram(__global const uint* input,
                        __global uint* hist,
                        uint n) {
    uint gid = get_global_id(0);
    if (gid >= n) return;

    uint value = input[gid];
    if (value < BINS) {
        atomic_inc(&hist[value]);
    }
}