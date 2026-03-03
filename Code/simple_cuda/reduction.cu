// Stage 5: Parallel Reduction (Sum)
// Goal: Compute the sum of a large array using all threads cooperatively
// New concepts: shared memory, __syncthreads(), tree reduction, atomicAdd

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// -----------------------------------------------------------------------
// Parallel reduction kernel
//
// The idea: instead of one thread adding up all N elements (slow),
// we have all threads cooperate in a "tournament bracket" style:
//
// Round 1: thread 0 adds elements 0+1, thread 1 adds 2+3, etc.  (N/2 additions)
// Round 2: thread 0 adds results 0+1, thread 1 adds 2+3, etc.  (N/4 additions)
// Round 3: ...and so on until thread 0 has the block's total
//
// This reduces N elements in log2(N) rounds instead of N rounds.
// For 1024 elements: 10 rounds vs 1024 rounds. Huge difference.
// -----------------------------------------------------------------------
__global__ void reduce_sum(const float* __restrict__ input,
                            float* __restrict__ output,
                            int N) {
    // ---- Shared Memory ----
    // This is on-chip memory shared between ALL threads in the same block.
    // Much faster than global memory (~4 cycles vs ~400 cycles).
    // But it only exists for the lifetime of the block and can't be seen
    // by other blocks.
    __shared__ float sdata[256];  // one slot per thread in this block

    int tid    = blockIdx.x * blockDim.x + threadIdx.x;  // global ID
    int local  = threadIdx.x;                              // local ID within block (0-255)

    // ---- Load from global memory into shared memory ----
    // Each thread loads one element. If out of bounds, load 0 (neutral for addition).
    sdata[local] = (tid < N) ? input[tid] : 0.0f;

    // ---- CRITICAL: Synchronize all threads in the block ----
    // We must wait for ALL threads to finish loading before any thread
    // starts reading values that other threads wrote.
    // Without this, thread 0 might try to read sdata[1] before thread 1 wrote it.
    __syncthreads();

    // ---- Tree reduction ----
    // stride starts at 128, halves each round: 128, 64, 32, 16, 8, 4, 2, 1
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (local < stride) {
            // Thread 'local' adds element 'local+stride' into its own slot
            sdata[local] += sdata[local + stride];
        }
        // Sync again: all threads must finish this round before the next
        __syncthreads();
    }

    // ---- Thread 0 writes this block's sum to output ----
    // After the reduction, sdata[0] holds the sum of all elements this block processed
    if (local == 0) {
        // atomicAdd is needed because multiple blocks all try to write to output[0]
        // atomicAdd ensures they don't overwrite each other — each add happens safely
        atomicAdd(output, sdata[0]);
    }
}

int main() {
    const int N = 1 << 16;
    const size_t bytes = sizeof(float) * N;

    std::vector<float> h_input(N);
    float cpu_sum = 0.0f;
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 10);  // small numbers to avoid float overflow
        cpu_sum += h_input[i];
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input,  bytes);
    cudaMalloc(&d_output, sizeof(float));

    cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);

    // Zero out the output on GPU before we start
    // (atomicAdd accumulates INTO this value, so it must start at 0)
    cudaMemset(d_output, 0, sizeof(float));

    const int THREADS = 256;
    const int BLOCKS  = (N + THREADS - 1) / THREADS;
    reduce_sum<<<BLOCKS, THREADS>>>(d_input, d_output, N);

    float gpu_sum = 0.0f;
    cudaMemcpy(&gpu_sum, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Allow some floating point error from different order of operations
    float diff = fabsf(gpu_sum - cpu_sum) / cpu_sum;
    std::cout << "CPU sum: " << cpu_sum << "\n";
    std::cout << "GPU sum: " << gpu_sum << "\n";
    std::cout << "Relative error: " << diff * 100 << "%\n";
    assert(diff < 0.001f);
    std::cout << "Reduction correct!\n";

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
/*

### The tree reduction visualized
```
Initial:  [3, 1, 4, 1, 5, 9, 2, 6]   (8 elements, 4 threads shown)

stride=4: thread 0: sdata[0] += sdata[4]   →  3+5=8
          thread 1: sdata[1] += sdata[5]   →  1+9=10
          thread 2: sdata[2] += sdata[6]   →  4+2=6
          thread 3: sdata[3] += sdata[7]   →  1+6=7
          result: [8, 10, 6, 7, -, -, -, -]

stride=2: thread 0: sdata[0] += sdata[2]   →  8+6=14
          thread 1: sdata[1] += sdata[3]   →  10+7=17
          result: [14, 17, -, -, -, -, -, -]

stride=1: thread 0: sdata[0] += sdata[1]   →  14+17=31
          result: [31, -, -, -, -, -, -, -]

thread 0 writes 31 to output  ✓  (3+1+4+1+5+9+2+6 = 31)
*/