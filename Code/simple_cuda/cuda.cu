#include <algorithm>
#include <cassert>
#include <cstdlib>   // rand
#include <iostream>
#include <vector>

#include <cuda_runtime.h>  // cudaMalloc, cudaMemcpy, cudaFree, cudaDeviceSynchronize
using namespace std;

// CUDA kernel for vector addition
// __global__ = runs on GPU, called from CPU
// Every thread that runs this function executes it simultaneously
__global__ void vectorAdd(const int* a,const int* b,int* c,int N) {

  //tid = example: thread 5 in Block 1, and each block has 32 threads
  //tid = 1 * 32 + 5 = 37
  //blockIdx = the id of the block itself within the grid
  //blockDim = number of threads in each block (32 in this case)
  //threadIdx =  the id of the thread within its block (0-31 in this case)
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Boundary check
  if (tid < N) 
  {
    c[tid] = a[tid] + b[tid];
  }
}

//Check vector add result 
//Runs on CPU
void verify_result(const int* a,const int* b,int* c,int N) {
  for (int i = 0; i < N; i++) 
  {
    assert(c[i] == a[i] + b[i]);
  }
}

int main() {
  // Array size of 2^16 (65536 elements)
  const int N = 1 << 16;                        
                                                //size_t = unsigned integer, can never be negative 
                                                //ex: cant have -5 bytes of memory, used to represent sizes of memory,blocks,arrays 
                                                //int = 32 bits, size_t = 64 bits so its a much larger int
  //sizeof(int) => 1 int = 4 bytes
  //32 ints = 128 bytes
  //4 * 2^16
  const size_t bytes = sizeof(int) * N;

  // Vectors for holding the host-side (CPU-side) data
  // IMPORTANT: c must be sized (resize), not just reserve, before cudaMemcpy writes into it.
  vector<int> a(N);//vector of size N that is assigned for a
  vector<int> b(N);//vector of size N that is assigned for b
  vector<int> c(N);//vector of size N that is assigned for c

  // Initializing random numbers in each array
  for (int i = 0; i < N; i++) {
    a[i] = rand() % 100;
    b[i] = rand() % 100;
  }

  // Allocate memory on the device GPU
  //pointers that will point to GPU memory a,b,c (d_ means device or GPU memory)
  int *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  //Allocate array a,b,c on the gpu with bytes amount of memory
  //&d_ where  the address of where the bytes amount will be assigned
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  // Copy data from the host to the device CPU -> GPU
  //cudaMemcpy(destination, source, number_of_bytes, direction)
  //dest = d_a
  //source = a.data()
  //bytes = sizeof(int) * N
  //direction = cudaMemcpyHostToDevice => upload from CPU to GPU
  cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

  //1024 is the maximum physical limit for a single Block on almost every modern nvidia gpu
  //threads per block
  int NUM_THREADS = 1024;//1 << 10;

  //Divide N by NUM_THREADS and always round up to the next whole number
  //blocks per grid
  int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

  // Launch the kernel on the GPU 
  //<<<NUM_BLOCKS, NUM_THREADS>>> = launch configuration
  //tells the GPU how many blocks and threads to use for this kernel launch
  //CPU does other things while GPU runs kernel simultaneously
  vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);


  // Copy sum vector from device to host
  //cudaMemCpyDeviceToHost => download from GPU to CPU
  //dest = c.data() takes incoming data and writes it into the vector c on the CPU
  //source = d_c is the pointer to the data on the GPU that we want to copy back to the CPU
  cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

  // Check result for errors
  verify_result(a.data(), b.data(), c.data(), N);

  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  cout << "COMPLETED SUCCESSFULLY\n";
  return 0;

  


 // (Optional but recommended) catch launch/runtime errors early:
  // cudaError_t err = cudaGetLastError();
  // assert(err == cudaSuccess);
  // cudaDeviceSynchronize();