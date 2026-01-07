/*
* CPU-GPU-compute examples
* License: GNU GPL v3
* 
* matrix_localmem OpenCL kernel
*/

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