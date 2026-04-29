
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
__global__ void gaf_kernel
(
    float *g_a,
    float *g_b,
    float *final_dot,
    float *final_norm_ga,
    float *final_norm_gb,
    int n
)
{
    __shared__ float dot[256];
    __shared__ float norm_ga[256];
    __shared__ float norm_gb[256];

    int tid = threadIdx.x; 

    int idx = blockIdx.x * blockDim.x + tid;

    if(idx < n)
    {
        dot[tid] = g_a[idx] * g_b[idx];
        norm_ga[tid] = g_a[idx] * g_a[idx];
        norm_gb[tid] = g_b[idx] * g_b[idx];
    }
    else
    {   
        dot[tid] = 0.0f;
        norm_ga[tid] = 0.0f;
        norm_gb[tid] = 0.0f;
    }

    __syncthreads();


    for(int s = blockDim.x / 2; s>0; s >>=1)
    {
        if(tid<s)
        {
            dot[tid] = dot[tid] + dot[tid + s];
            norm_ga[tid] = norm_ga[tid] + norm_ga[tid + s];
            norm_gb[tid] = norm_gb[tid] + norm_gb[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0)
    {
        atomicAdd(final_dot, dot[0]);
        atomicAdd(final_norm_ga, norm_ga[0]);
        atomicAdd(final_norm_gb, norm_gb[0]);
    }

}

