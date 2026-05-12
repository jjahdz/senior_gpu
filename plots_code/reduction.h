#pragma once
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void reduce_sum(float* in, float* out, int n);