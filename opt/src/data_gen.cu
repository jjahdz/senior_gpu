#define TRUE_A 3.0f
#define TRUE_B -2.0f
#define TRUE_C 1.5f
#define TRUE_BIAS 2.0f

//generate synthetic data for training
    for(int i=0; i<n; i++)
    {
        float x = (float)i/n;
        h_x[i] = x;
        h_x2[i] = x * x;
        h_x3[i] = x * x * x;
        float noise = (((float)rand()/RAND_MAX - 0.5f) * 0.5f);
        h_y[i] = TRUE_A * h_x3[i] + TRUE_B * h_x2[i] + TRUE_C * h_x[i] + TRUE_BIAS + noise; //TRUE_W * h_x[i] + TRUE_B;// + (((float)rand()/RAND_MAX - 0.5f) * 0.1f);
    }