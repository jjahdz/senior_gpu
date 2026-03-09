/*
grdient descent linear regression



CODE: Training a linear regression model using gradient descent 
GOAL: this code applies gradient descent through the trainig of a linear regression model (y = wx + b or y = 3x+2)
to find the accurate w and b, we use parralle computing through reduction to calculate our gradients
 thorugh blocks and these blocks help us later update our w and b per epoch iteration

 linear model y_hat = 3x + 2 from 

MSE = 1/n * summation((y - y_pred)^2)
y hat = w * x + b


*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
using namespace std;

//MACROS FOR CONFIGURATION
#define N 100000 //number of data points/training samples 100k, means we will have arrays of size 100k for x and y
#define EPOCHS 500 //number of training iterations, or how many times we will update our model parameters w and b
#define BLOCK_SIZE 256 //number of threads per block
#define LEARNING_RATE 0.5f//0.01f 
//These are the values we used to generate our data, and the values we are trying to learn with our model through training
//Y = 3x + 2 + noise, so the true weight is 3 and the true bias is 2
#define TRUE_W 3.0f //true weight for synthetic data
#define TRUE_B 2.0f //true bias for synthetic data

/*
CUDA ERROR CHECK 
EVERY CUDA CALL MUST BE CHECKED FOR ERRORS SO WE KNOW IF SOMETHING WENT WRONG
When an error occurs, cudaError_t will print the error message and exit the program
*/
#define CUDA_CHECK(call) { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d — %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
}

//CUDA KERNEL FOR GRADIENT DESCENT 
__global__ void gradient_descent(
    //ex:arrays of different hours studied for test scores
     //input array of x values on (GPU MEMORY),kenrnel will read from this array
    const float * X,
    //ex:arrays of different test scores for the hours studied
    //target array of y values on (GPU MEMORY)
    const float * Y, 
    //array to store the gradient error values for w at index i
    float *grad_w, 
    //array to store the gradient error values for b at index i
    float *grad_b,
    //current model weight
    //current multiplier 
    //maybe each hour is worth 5 points, so w = 5
    float w, 
     //current model bias
     //starting point
     //maybe even if you study 0 hrs you still get 20 poitns, so b = 20
    float b,
    //number of data points x,y pairs
    int n 
)
{
    //calculate the gloabl thread ID
    //global thread ID by block #, # threads in block,and thread # in block
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    //ensure there arent more threads than data points 
    //1 thread per data point, so if we have 100k data points we need 100k threads
    if(i>=n) return; 

    //calculate the prediction for this data y_hat = w * x + b
    //so if we study for 5 hours (w = 5) and we get 20 points just for showing up (b=20)
    //then if we studied for 3 hours, our prediction (y_hat) wouuld be 
    //5*3 + 20 = 35 points
    float y_hat = w * X[i] + b; //prediction for this data point (y hat)
    //but maybe the true test score for studying 3 hours is 40 points,
    //so the error would be 35 - 40 = -5, meaning we underpredicted so we need increase w and b
    float error = y_hat - Y[i]; //error for this data point (ŷ - y)
    //stores the weights output error contribution for current set of points x,y
    grad_w[i] = error * X[i];
    //stores the bias output error contribution for current set of points x,y
    grad_b[i] = error;

}

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
            __syncthreads();
        }
    }

    //after reduction, the first thread (tid=0) contains the sum of this block
    //so we write this blocks sum to the output array in global mem at this blocks index
    if(tid == 0)
    {
        out[blockIdx.x]=smem[0];
    }

}

//ads all the output block sums from th gpu to get the final sum for either grad_w or grad_b
float cpu_sum(float *arr, int len)
{
    float s = 0.0f;
    for(int i=0; i<len; i++)
    {
        s+=arr[i];
    }
    return s;
}

int main()
{
    int n=N;
    size_t bytes = n * sizeof(float);
    //number of blocks needed to cover all threads, we round up to ensure we have enough blocks for all threads
    //almost like doing a ceil_function, but bcs integers truncate we need to add BLOCK_SIZE-1 to ensure we round up when we divide by BLOCK_SIZE
    //instead of 3.99 we need to make sure we have 4.05 to truncate the decimal and get 4 blocks instead of 3
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE; //391

    //Allocating and filling cpu memory
    float *h_x = (float*)malloc(bytes);
    float *h_y = (float*)malloc(bytes);
    float *h_partial = (float*)malloc(num_blocks * sizeof(float));

    //generate synthetic data for training
    for(int i=0; i<n; i++)
    {
        h_x[i] = (float)i/n;// 0/100,00 1/100,00 2/100,00 ... 99,000/100,00
        //y data that we calculate with our known set w and b
        h_y[i] = TRUE_W * h_x[i] + TRUE_B;// + (((float)rand()/RAND_MAX - 0.5f) * 0.1f);
    }

    //allocate gpu memory
    float * d_x;
    float * d_y;
    //holds partial block sums on gpu
    float * d_partial;
    //holds the error for weight and bias for each of the 100k x,y pairs
    float * d_grad_w;
    float * d_grad_b;

    //allocate bytes on the gpu for d_x,d_y,d_partial,d_grad_w,d_grad_b 
    //then goes to the address of the first parameter, non reference
    //sets the variable in d_x, d_y, d_partial, d_grad_w, d_grad_b to point to that gpu memory address (not * variable)
    CUDA_CHECK(cudaMalloc(&d_x,bytes));
    CUDA_CHECK(cudaMalloc(&d_y,bytes));
    CUDA_CHECK(cudaMalloc(&d_partial, num_blocks * sizeof(float)));//391*4
    CUDA_CHECK(cudaMalloc(&d_grad_w, bytes));
    CUDA_CHECK(cudaMalloc(&d_grad_b, bytes));

    //copy input data from cpu to gpu that we filled and initialized randomly earlier
    CUDA_CHECK(cudaMemcpy(d_x,h_x,bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y,h_y,bytes,cudaMemcpyHostToDevice));

    //initialize our w and b parameters to 0
    //starting point
    float w = 0.0f;
    float b = 0.0f;

    //prints the training configuration to the console
    printf("Training: Y = %.1f * X + %.1f | N=%d epochs=%d lr=%.4f\n\n", TRUE_W, TRUE_B, N, EPOCHS, LEARNING_RATE);

    //epoch is the full passing of all the avg gradients, for w and b, having been calculated
    for(int epoch = 0; epoch<EPOCHS; epoch++)
    {
        /*now here we do the first step which is the step of gradient descent for w and b,
         then we reduce w, then we pass that partial sum of w to the host once its been computed in parallel by the gpu
         then the block sums arrays with their single index's are all added with our cpu sum helper function
         the same is done for bias afterwards and sent to host cpu*/
        //launches the gradient descent kernel to compute the gradient contributions for each data point
        gradient_descent<<<num_blocks, BLOCK_SIZE>>>(d_x,d_y,d_grad_w,d_grad_b,w,b,n);//319,256

        //launches the reduction kernel to sum the gradient contributions for w and b across all blocks
        reduce_sum<<<num_blocks, BLOCK_SIZE>>>(d_grad_w, d_partial, n);

        //copies the block sums from the gpu to the cpu so we can finish summing them to get the final gradient for w
        CUDA_CHECK(cudaMemcpy(h_partial,d_partial,num_blocks * sizeof(float), cudaMemcpyDeviceToHost));

        //sums the block sums on the cpu to get the final gradient for w, and divides by n to get the average gradient
        float grad_w = cpu_sum(h_partial,num_blocks) / n;

        //launches the reduction kernel to sum the gradient contributions for w and b across all blocks
        reduce_sum<<<num_blocks, BLOCK_SIZE>>>(d_grad_b, d_partial, n);

        //copies the block sums from the gpu to the cpu so we can finish summing them to get the final gradient for b
        CUDA_CHECK(cudaMemcpy(h_partial,d_partial,num_blocks * sizeof(float), cudaMemcpyDeviceToHost));

        //sums the block sums on the cpu to get the final gradient for b, and divides by n to get the average gradient
        float grad_b = cpu_sum(h_partial,num_blocks) / n;

        /*this is where we would update our w and b instead of 0 it would be a new 
        value we continually update until all epochs have passed
        we use our new grad_w and new grad_b to update */
        //updates parameters w and b using the computed gradients and the learning rate
        w -= LEARNING_RATE * grad_w;//w = w - lr * grad_w
        b -= LEARNING_RATE * grad_b;//w = w - lr * grad_b

        //prints the current epoch and the values of w and b every 50 epochs
        if(epoch % 50 == 0 || epoch == EPOCHS -1)
        {
            printf("[Epoch %3d] w=%.5f b=%.5f\n", epoch, w, b);
        }
    }

    //prinig the final learned parameters w and b after training
    printf("\n---- Results ----\n");
    printf("Learned:  w=%.5f  b=%.5f\n", w, b);
    printf("True:     w=%.5f  b=%.5f\n", TRUE_W, TRUE_B);
    printf("Error:    w=%.6f  b=%.6f\n", fabs(w-TRUE_W), fabs(b-TRUE_B));

    //free gpu memory
    free(h_x); free(h_y); free(h_partial);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_partial); cudaFree(d_grad_w); cudaFree(d_grad_b);

    return 0;
}