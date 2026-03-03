// Goal: Writing a kernel. Each thread writes its ID into an array.
// Concepts: __global__, threadIdx, blockIdx, blockDim, cudaMalloc, 
//               cudaMemcpy, cudaFree

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
using namespace std;
// -----------------------------------------------------------------------
// __global__ = runs on GPU, called from CPU
// Every thread that runs this function executes it simultaneously
// -----------------------------------------------------------------------
__global__ void fill_array(int* a, int N) {
    // Calculate this thread's unique global ID
    // Think of it like: which slot in the array am I responsible for?

    //tid = example: thread 5 in Block 1, and each block has 32 threads
    //tid = 1 * 32 + 5 = 37
    //blockIdx = the id of the block itself within the grid
    //blockDim = number of threads in each block (32 in this case)
    //threadIdx =  the id of the thread within its block (0-31 in this case)
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 

    // Boundary check: don't go past the end of the array
    //Ensures threads don't try to write to memory outside the array, 
    //could cause crashes 
    //a[i] = i ; 0-32
    //a[i] = i*10; 0,10,20...310;
    if (tid < N) {
        a[tid] = tid*10;   // thread 0 writes 0, thread 1 writes 1, etc.
    }
}

int main() {
    // Array size — start small so results are easy to read
    const int N = 32; 
                                                //size_t = unsigned integer, can never be negative 
                                                //ex: cant have -5 bytes of memory, used to represent sizes of memory,blocks,arrays 
                                                //int = 32 bits, size_t = 64 bits so its a much larger int
    //sizeof(int) => 1 int = 4 bytes => 32 ints = 128 bytes
    const size_t bytes = sizeof(int) * N;       /*N is telling the pizza shop you want 32 slices.
                                                bytes is telling the pizza shop how many square inches
                                                 of box space they need to hold all those slices.*/

    // ---- GPU memory ----
    //safety measure, init to nullptr to avoid garbage mem address
    int* d_a = nullptr;                         //int* = this variable will store the memory address of an integer
                                                //d_ = device(GPU) variable, h_ = host(CPU) variable
    
    //allocates 32 ints on the GPU and
    //stores the address of that memory in d_a
    cudaMalloc(&d_a, bytes);                    //d_a = a pointer (holds an address) 
                                                //&d_a = the address of the pointer
                                                /*cudaMalloc needs to change the value 
                                                of d_a to point to the new GPU memmory*/
/*SAFETY CHECK
Because cudaMalloc can fail (e.g., if you 
try to allocate more memory than your GPU has)
cudaError_t err = cudaMalloc(&d_a, bytes);
if (err != cudaSuccess) {
    cout << "GPU Memory Allocation Failed!" << endl;
}*/

    // ---- Launch kernel ----
    // 1 block, 32 threads — enough for our 32 elements
    // Think of this as: "run fill_array 32 times simultaneously"
    //<<<1 block, 32 threads>>> =tells how many blocks and threads to use
    fill_array<<<1, 32>>>(d_a, N);


    // ---- Copy result back to CPU ----
    //h_a(host) vector of 32 ints  
    //Allocates a vector of 32 ints on the CPU to hold the results
    vector<int> h_a(N);                          
    
    
    cudaMemcpy(h_a.data(), d_a, bytes, cudaMemcpyDeviceToHost);     /*h_a.data() (The Destination): This gives the memory address 
                                                                    of the first element in your vector. This is where the data will 
                                                                    be written.

                                                                    d_a (The Source): This is the address on the GPU where your 
                                                                    finished work is sitting.

                                                                    bytes (The Amount): You are telling it to move exactly 128 bytes 
                                                                    (4 bytes per int * 32 ints)
                                                                    
                                                                    cudaMemcpyDeviceToHost (The Direction): This is a specific "flag" 
                                                                    that tells CUDA, "I am pulling data from the GPU to the CPU." 
                                                                    (If you get this direction wrong, the program will usually crash or return zeros).*/

    // ---- Print results ----
    std::cout << "Array contents after GPU fill:\n";
    for (int i = 0; i < N; i++) {
        cout << "  a[" << i << "] = " << h_a[i] << "\n";
    }

    // ---- Cleanup ----
    cudaFree(d_a);

    return 0;
}