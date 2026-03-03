// Stage 3: Vector Addition
// Goal: Two input arrays, one output array. Each thread adds one pair.
// New concepts: multiple arrays, __restrict__, ceiling division for blocks

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
using namespace std;
__global__ void vector_add(const int* __restrict__ a,
                            const int* __restrict__ b,
                            int*       __restrict__ c,
                            int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

void verify(const  vector<int>& a,
            const  vector<int>& b,
            const  vector<int>& c) {
    for (int i = 0; i < (int)a.size(); i++) {
        assert(c[i] == a[i] + b[i]);
    }
}

int main() {
    const int N = 1 << 16;      // 65,536 elements
    const size_t bytes = sizeof(int) * N;

    // ---- Host data ----
     vector<int> h_a(N), h_b(N), h_c(N);
    for (int i = 0; i < N; i++) {
        h_a[i] = rand() % 100;
        h_b[i] = rand() % 100;
    }

    // ---- Device data ----
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // ---- CPU → GPU ----
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    // ---- Launch ----
    const int THREADS = 256;
    const int BLOCKS  = (N + THREADS - 1) / THREADS;   // ceiling division
    vector_add<<<BLOCKS, THREADS>>>(d_a, d_b, d_c, N);

    // ---- GPU → CPU ----
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // ---- Verify ----
    verify(h_a, h_b, h_c);
     cout << "Vector addition correct!\n";

    // ---- Cleanup ----
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}