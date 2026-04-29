//CUDA KERNEL FOR GRADIENT DESCENT y_hat = w1*x1 + w2*x2 + w3*x3 + b
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void gradient_descent(
    //ex:arrays of different hours studied for test scores
     //input array of x values on (GPU MEMORY),kenrnel will read from this array
    const float * X,
    //ex:arrays of different test scores for the hours studied
    //target array of y values on (GPU MEMORY)
    const float * Y,
    //array to store the gradient error values for w at index i
    float *grad_a,
    float *grad_b,
    float *grad_c, 
    //array to store the gradient error values for b at index i
    float *grad_bias,
    //current model weight
    //current multiplier 
    //maybe each hour is worth 5 points, so w = 5
    float a,
    float b,
    float c, 
     //current model bias
     //starting point
     //maybe even if you study 0 hrs you still get 20 poitns, so b = 20
    float bias,
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

    float x = X[i];
    float x2 = x * x;
    float x3 = x * x * x;
    //calculate the prediction for this data y_hat = w * x + b
    //so if we study for 5 hours (w = 5) and we get 20 points just for showing up (b=20)
    //then if we studied for 3 hours, our prediction (y_hat) wouuld be 
    //5*3 + 20 = 35 points
    float y_hat = a * x3 + b * x2 + c * x + d; //prediction for this data point (y hat)
    //but maybe the true test score for studying 3 hours is 40 points,
    //so the error would be 35 - 40 = -5, meaning we underpredicted so we need increase w and b
    float error = y_hat - Y[i]; //error for this data point (ŷ - y)
    //stores the weights output error contribution for current set of points x,y
    grad_a[i] = error * x3;
    grad_b[i] = error * x2[i];
    grad_c[i] = error * x;
    //stores the bias output error contribution for current set of points x,y
    grad_bias[i] = error;

}