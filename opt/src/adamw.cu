/*
grdient descent linear regression

CODE: Training a linear regression model using gradient descent 
GOAL: To learn the parameters w and b of the linear model y_hat = 3x + 2 from 

MSE = 1/n * summation((y - y_pred)^2)
y hat = w * x + b


*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void adam_update_kernel
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
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    m[i] = beta1 * m[i] + (1.0f - beta1) * grad[i];
    v[i] = beta2 * v[i] + (1.0f - beta2) * (grad[i] * grad[i]);
    float m_param_hat = m[i] / (1.0f - powf(beta1, (float)timestep));
    float v_param_hat = v[i] / (1.0f - powf(beta2, (float)timestep));
    params[i] -= lr * (m_param_hat / (sqrtf(v_param_hat) + eps) + weight_decay * params[i]);
}
