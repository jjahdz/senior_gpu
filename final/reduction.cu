
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "reduction.h"
__global__ void reduce_sum
(
    //input array of gradient contributions
    //either grad_w or grad_b of size N
    //these are the arrays that have been filled with the 
    //previously calcualted gradient error for points x,y
    float *in, 
    //output array to store the block sums
    float *out, 
    //N = 100k
    int n
)
{
    //global memory lives in DRAM, slow to access
    //shared memory lives on the GPU chip, much faster to access
    //shared memory is shared amongst threads in the same block
    //each block gets its own smem array sized 256
    __shared__ float smem[BLOCK_SIZE];

    //local thread ID within the block (0-255)
    int tid = threadIdx.x; 
    //global thread ID across all blocks
    int idx = blockIdx.x * blockDim.x + tid;
    //copies the value from the input array in global memory 
    //to the shared memory array at the index of the local thread ID
    //ex: system[0] = in[0], smem[1] = in[1], until we go through all values [i] for
    if(idx < n)
    {
        //copies the value from input array to smem array
        smem[tid] = in[idx];
    }
    else
    {   
        //if a thread goes over 100k data points, we set its value in smem to 0 so it doesnt contribute to the sum
        smem[tid] = 0.0f;
    }

    //waits for all threads in the block to reach this point before any thread continues
    __syncthreads();


    //parallel reduction to sum the values in the smem array
    //by folding and adding pairs of values, we can sum the array in log₂(BLOCK_SIZE) steps
    //s>>=1 means s = s/2, so we start with s=128 and then we half it to 64,32,16,8,4,3,2,1
    //so instead of 0+128 itd be 0+6
    for(int s = blockDim.x / 2; s>0; s >>=1)
    {
        if(tid<s)
        {
            smem[tid] = smem[tid] + smem[tid + s];
        }
        __syncthreads();
    }

    //after reduction, the first thread (tid=0) contains the sum of this block
    //so we write this blocks sum to the output array in global mem at this blocks index
    if(tid == 0)
    {
        out[blockIdx.x]=smem[0];
    }

}
/*(
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

*/