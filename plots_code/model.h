#pragma once
#define N             1024//93239//9568//100000///9568//1024
#define EPOCHS        200//500//200//2000//1000
#define BLOCK_SIZE    256
#define LEARNING_RATE 0.01f//0.0010f
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
    float *grad_d,
    //current model weight
    //current multiplier 
    //maybe each hour is worth 5 points, so w = 5
    float a,
    float b,
    float c, 
     //current model bias
     //starting point
     //maybe even if you study 0 hrs you still get 20 poitns, so b = 20
    float d,
    //number of data points x,y pairs
    int n 
);