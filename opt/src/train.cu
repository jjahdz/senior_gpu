
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "model.h"
#include "reduce.h"
#include "adamw.h"
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
    //float *h_x2 = (float*)malloc(bytes);
    //float *h_x3 = (float*)malloc(bytes);
    float *h_y = (float*)malloc(bytes);
    float *h_partial = (float*)malloc(num_blocks * sizeof(float));



    //allocate gpu memory
    float * d_x;
    //float * d_x2;
    //float * d_x3;
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

    //copy input data from cpu to gpu
    CUDA_CHECK(cudaMemcpy(d_x,h_x,bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x2,h_x2,bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x3,h_x3,bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y,h_y,bytes,cudaMemcpyHostToDevice));

    //initialize our w and b parameters to 0
    float a = 0.0f;
    float b = 0.0f;
    float c = 0.0f;
    float d = 0.0f;

    //prints the training configuration to the console
    printf("Training: Y = %.1f * X + %.1f | N=%d epochs=%d lr=%.4f\n\n", TRUE_W, TRUE_B, N, EPOCHS, LEARNING_RATE);

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
            //launches the reduction kernel to sum the gradient contributions for w and b across all blocks
            reduce_sum<<<blocks_per_batch, BLOCK_SIZE>>>(d_grad_a, d_partial, points_per_batch);
            reduce_sum<<<blocks_per_batch, BLOCK_SIZE>>>(d_grad_b, d_partial, points_per_batch);
            reduce_sum<<<blocks_per_batch, BLOCK_SIZE>>>(d_grad_c, d_partial, points_per_batch);
            reduce_sum<<<blocks_per_batch, BLOCK_SIZE>>>(d_grad_d, d_partial, points_per_batch);

            //copies the block sums from the gpu to the cpu so we can finish summing them to get the final gradient for w
            CUDA_CHECK(cudaMemcpy(h_partial, d_partial, sizeof(float), cudaMemcpyDeviceToHost));

            //sums the block sums on the cpu to get the final gradient for w, and divides by n to get the average gradient
            float grad_a = h_partial[0] / points_per_batch;
            float grad_b = h_partial[0] / points_per_batch;
            float grad_c = h_partial[0] / points_per_batch;
            
 m_a = beta1 * m_a + (1.0f - beta1) * grad_a;
        v_a = beta2 * v_a + (1.0f - beta2) * grad_a * grad_a;
        float m_a_hat = m_a / (1.0f - powf(beta1, timestep));
        float v_a_hat = v_a / (1.0f - powf(beta2, timestep));
        a -= LEARNING_RATE * (m_a_hat / (sqrtf(v_a_hat) + eps) + weight_decay * a);
        
        // b
        m_b = beta1 * m_b + (1.0f - beta1) * grad_b;

            adamw_update_kernel(&a,)
            adamw__global__ void adam_update_kernel
(
    float *grad,
    float *params,
    float *m,
    float *v,
    float beta1,
    float beta2,
    float weight_decay,
    float lr,
    float eps,
    int timestep,
    int n
) 

            //launches the reduction kernel to sum the gradient contributions for w and b across all blocks
            reduce_sum<<<blocks_per_batch, BLOCK_SIZE>>>(d_grad_a, d_partial, points_per_batch);
            reduce_sum<<<blocks_per_batch, BLOCK_SIZE>>>(d_grad_b, d_partial, points_per_batch);
            reduce_sum<<<blocks_per_batch, BLOCK_SIZE>>>(d_grad_c, d_partial, points_per_batch);
            reduce_sum<<<blocks_per_batch, BLOCK_SIZE>>>(d_grad_d, d_partial, points_per_batch);

            //copies the block sums from the gpu to the cpu so we can finish summing them to get the final gradient for b
            CUDA_CHECK(cudaMemcpy(h_partial, d_partial, sizeof(float), cudaMemcpyDeviceToHost));
            //sums the block sums on the cpu to get the final gradient for b, and divides by n to get the average gradient
            float grad_d = h_partial[0] / points_per_batch;

            m_d = beta1 * m_d + (1.0f - beta1) * grad_d;
            v_d = beta2 * v_d + (1.0f - beta2) * (grad_d * grad_d);

            float m_d_hat = m_d / (1.0f - (powf(beta1,timestep)));
            float v_d_hat = v_d / (1.0f - powf(beta2,timestep));

            //updates the bias 
            d -= LEARNING_RATE * (m_d_hat / (sqrtf(v_d_hat) + eps) + weight_decay * d);

        }
        //prints the current epoch and the values of w and b every 50 epochs and on the last epoch
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
    free(h_x); free(h_x2); free(h_x3); free(h_y); free(h_partial);
    cudaFree(d_x); free(d_x2); free(d_x3); cudaFree(d_y); cudaFree(d_partial); cudaFree(d_grad_a); cudaFree(d_grad_b);
    cudaFree(d_grad_c); cudaFree(d_grad_b); 

    return 0;
}