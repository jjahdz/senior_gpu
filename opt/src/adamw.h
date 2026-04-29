void adam_update_kernel
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
);