/*
* CPU-GPU-compute examples
* License: GNU GPL v3
* 
* matrix_simple OpenCL kernel
*/

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