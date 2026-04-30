
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "model.h"
#include "reduction.h"
#include "adamw.h"
#include "data_gen.h"
#include "cuda_check.h"
int main()
{
    int n=N;
    size_t bytes = n * sizeof(float);
    //process 256 points in groups of 256
    //goes through the gradient descent and reduce sum
    //stores the block avg
    int batch_size = 256;
    int points_per_batch = batch_size;
    int blocks_per_batch = 1;
    //Adam variables
    //first and second moments m_w,v_w,m_b,v_b
    float m_a = 0.0f, v_a = 0.0f;
    float m_b = 0.0f, v_b = 0.0f;
    float m_c = 0.0f, v_c = 0.0f;
    float m_d = 0.0f, v_d = 0.0f;
    //decides how much of the previous direction to keep
    //set to keep 90% of previous data and 10% of current gradient
    //limits noisy outliers
    float beta1 = 0.9f;
    //scales the learning rate
    //0.999 creates smoother traversal and ignores jumpy 
    //mini batch outliers compared to beta2 = 0.9
    float beta2 = 0.999f;
    //ensures no 0 division while keeping the denominator for 
    //sqrt(v) small enough to not have a large enough impact
    float eps = 1e-8f;
    //00keeps the daisy as smooth as possible
    float weight_decay = 0.01f; // The "W" in AdamW
    //corrects bias accordingly
    int timestep = 0;

    //number of blocks needed to cover all threads, we round up to ensure we have enough blocks for all threads
    //almost like doing a ceil_function, but bcs integers truncate we need to add BLOCK_SIZE-1 to ensure we round up when we divide by BLOCK_SIZE
    //instead of 3.99 we need to make sure we have 4.05 to truncate the decimal and get 4 blocks instead of 3
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    //Allocating and filling cpu memory
    float *h_x = (float*)malloc(bytes);
    float *h_y = (float*)malloc(bytes);
    float *h_partial = (float*)malloc(num_blocks * sizeof(float));



    //allocate gpu memory
    float * d_x;
    float * d_y;
    float * d_partial;
    float * d_grad_a;
    float * d_grad_b;
    float * d_grad_c;
    float * d_grad_d;

    //allocate bytes on the gpu for d_x,d_y,d_partial,d_grad_w,d_grad_b 
    //then return the pointer to that memory in d_x, d_y, d_partial, d_grad_w, d_grad_b
    CUDA_CHECK(cudaMalloc(&d_x,bytes));
    //CUDA_CHECK(cudaMalloc(&d_x2,bytes));
    //CUDA_CHECK(cudaMalloc(&d_x3,bytes));
    CUDA_CHECK(cudaMalloc(&d_y,bytes));
    CUDA_CHECK(cudaMalloc(&d_partial, num_blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_grad_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_grad_c, bytes));
    CUDA_CHECK(cudaMalloc(&d_grad_d, bytes));

    //data fill/
    data_gen(h_x,h_y,n);

    //copy input data from cpu to gpu
    CUDA_CHECK(cudaMemcpy(d_x,h_x,bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y,h_y,bytes,cudaMemcpyHostToDevice));

    //initialize our w and b parameters to 0
    float a = 0.0f;
    float b = 0.0f;
    float c = 0.0f;
    float d = 0.0f;

    printf("Training: Y = %.1fx^3 + %.1fx^2 + %.1fx + %.1f | N=%d epochs=%d lr=%.4f\n\n",
       TRUE_A, TRUE_B, TRUE_C, TRUE_BIAS, N, EPOCHS, LEARNING_RATE);

    for(int epoch = 0; epoch<EPOCHS; epoch++)
    {
        for(int i=0; i<n; i+= points_per_batch)
        {
            timestep++; //incremeents the timestep for adam updates
            //launches the gradient descent kernel to compute the gradient contributions for each data point
            gradient_descent<<<blocks_per_batch, BLOCK_SIZE>>>(d_x + i,
                                                            d_y + i, 
                                                            d_grad_a,
                                                            d_grad_b,
                                                            d_grad_c,   
                                                            d_grad_d, 
                                                            a, 
                                                            b,
                                                            c,
                                                            d,
                                                             points_per_batch);
            reduce_sum<<<blocks_per_batch, BLOCK_SIZE>>>(d_grad_a, d_partial, points_per_batch);
            CUDA_CHECK(cudaMemcpy(h_partial, d_partial, sizeof(float), cudaMemcpyDeviceToHost));
            //sums the block sums on the cpu to get the final gradient for w, and divides by n to get the average gradient
            float grad_a = h_partial[0] / points_per_batch;

            reduce_sum<<<blocks_per_batch, BLOCK_SIZE>>>(d_grad_b, d_partial, points_per_batch);
            CUDA_CHECK(cudaMemcpy(h_partial, d_partial, sizeof(float), cudaMemcpyDeviceToHost));
            
            //sums the block sums on the cpu to get the final gradient for w, and divides by n to get the average gradient
            float grad_b = h_partial[0] / points_per_batch;

            reduce_sum<<<blocks_per_batch, BLOCK_SIZE>>>(d_grad_c, d_partial, points_per_batch);
            CUDA_CHECK(cudaMemcpy(h_partial, d_partial, sizeof(float), cudaMemcpyDeviceToHost));
            //sums the block sums on the cpu to get the final gradient for w, and divides by n to get the average gradient
            float grad_c = h_partial[0] / points_per_batch;

            reduce_sum<<<blocks_per_batch, BLOCK_SIZE>>>(d_grad_d, d_partial, points_per_batch);
            CUDA_CHECK(cudaMemcpy(h_partial, d_partial, sizeof(float), cudaMemcpyDeviceToHost));
            //sums the block sums on the cpu to get the final gradient for w, and divides by n to get the average gradient
            float grad_d = h_partial[0] / points_per_batch;


            adamw_update(&a,grad_a,&m_a,&v_a,beta1,beta2,weight_decay,LEARNING_RATE,eps,timestep);
            adamw_update(&b,grad_b,&m_b,&v_b,beta1,beta2,weight_decay,LEARNING_RATE,eps,timestep);
            adamw_update(&c,grad_c,&m_c,&v_c,beta1,beta2,weight_decay,LEARNING_RATE,eps,timestep);
            adamw_update(&d,grad_d,&m_d,&v_d,beta1,beta2,weight_decay,LEARNING_RATE,eps,timestep);

        }
        //prints the current epoch and the values of w and b every 50 epochs and on the last epoch
        if(epoch % 50 == 0 || epoch == EPOCHS -1)
        {
            printf("[Epoch %3d] a=%.5f b=%.5f c=%.5f d=%.5f\n", epoch, a, b, c, d);
        }
    }

    //prinig the final learned parameters w and b after training
    printf("\n---- Results ----\n");
    printf("Learned: a=%.5f b=%.5f c=%.5f d=%.5f\n", a, b, c, d);
    printf("True:    a=%.5f b=%.5f c=%.5f d=%.5f\n", TRUE_A, TRUE_B, TRUE_C, TRUE_BIAS);
    printf("Error:   a=%.6f b=%.6f c=%.6f d=%.6f\n",
        fabsf(a-TRUE_A), fabsf(b-TRUE_B),
        fabsf(c-TRUE_C), fabsf(d-TRUE_BIAS));

    //free gpu memory
    free(h_x); free(h_y); free(h_partial);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_partial);
    cudaFree(d_grad_a); cudaFree(d_grad_b); cudaFree(d_grad_c); cudaFree(d_grad_d);

    return 0;
}