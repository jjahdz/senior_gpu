// Stage 1: Hello GPU
// Goal: Confirm CUDA is working and inspect your GPU's properties
// New concepts: cudaGetDeviceCount, cudaGetDeviceProperties

#include <iostream>
#include <cuda_runtime.h>
using namespace std;

int main()
{
//Step 1:Ask CUDA how many GPUs are in this machine
int deviceCount = 0;
cudaGetDeviceCount(&deviceCount);

if(deviceCount==0)
{
    cout<<"No CUDA-Capable GPU found"<<endl;
    return 1;
}

cout<<"Found "<<deviceCount<<"CUDA-Capable GPU(s)"<<endl;

// Step 2: For each GPU, print its properties
// cudaDeviceProp is a struct that holds everything about your GPU
for(int i=0; i<deviceCount; i++)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,i);//Fill the struct for GPU i
    cout << "GPU " << i << " \n";
    cout << "Name:   " << prop.name << " \n";
    cout << "Total VRAM: " << prop.totalGlobalMem / (1024*1024) << " MB\n";
    cout << "SM Count: " << prop.multiProcessorCount << prop.totalGlobalMem << " \n";
    cout << "Max threads per block: " << prop.maxThreadsPerBlock << " \n";
    cout << "Max threads per SM: " << prop.maxThreadsPerMultiProcessor << " \n";
    cout << "Warp size: " << prop.warpSize << " \n";
    cout << "Compute capability: " << prop.major <<"."<< prop.minor << " \n";
    cout << " \n";


}
}