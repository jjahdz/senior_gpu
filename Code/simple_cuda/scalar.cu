// Stage 4: Scalar Math Per Thread
// Goal: Show that each thread can do complex math, not just addition
// New concepts: sqrtf, multiple operations per thread, float vs int,
//               reading from global memory once into registers

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// -----------------------------------------------------------------------
// This kernel computes: c[i] = sqrt(a[i]^2 + b[i]^2)
// (the hypotenuse of a right triangle — more interesting than just add)
// -----------------------------------------------------------------------
__global__ void hypotenuse(const float* __restrict__ a,
                            const float* __restrict__ b,
                            float*       __restrict__ c,
                            int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;   // early return style boundary check — same as before

    // KEY HABIT: Load from global memory ONCE into local variables (registers)
    // Global memory (VRAM) reads are slow (~400-800 cycles).
    // Registers are instant. Do your math in registers, write back once.
    float a_val = a[tid];
    float b_val = b[tid];

    // Do all the math in registers
    float a_sq  = a_val * a_val;
    float b_sq  = b_val * b_val;
    float sum   = a_sq + b_sq;
    float result = sqrtf(sum);  // sqrtf = float sqrt. Always use 'f' suffix for float math.
                                // sqrtf is faster than sqrt on GPU.

    // Write result back to global memory ONCE
    c[tid] = result;
}

void verify(const std::vector<float>& a,
            const std::vector<float>& b,
            const std::vector<float>& c) {
    for (int i = 0; i < (int)a.size(); i++) {
        float expected = sqrtf(a[i]*a[i] + b[i]*b[i]);
        float diff = fabsf(c[i] - expected);
        // With floats we allow tiny rounding differences
        assert(diff < 1e-4f);
    }
}

int main() {
    const int N = 1 << 16;
    const size_t bytes = sizeof(float) * N;

    std::vector<float> h_a(N), h_b(N), h_c(N);
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)(rand() % 100) + 1.0f;  // +1 to avoid sqrt(0)
        h_b[i] = (float)(rand() % 100) + 1.0f;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    const int THREADS = 256;
    const int BLOCKS  = (N + THREADS - 1) / THREADS;
    hypotenuse<<<BLOCKS, THREADS>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    verify(h_a, h_b, h_c);
    std::cout << "Hypotenuse correct!\n";

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}