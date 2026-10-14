#include "timer.h"

void vecadd_cpu(float *x, float *y, float *z, int N){
    for (unsigned int i = 0; i<N; i++){
        z[i] = x[i] + y[i];
    }
}

__global__ void vector_kernel(float *x, float *y, float *z, int N){
    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i<N){
        z[i] = x[i] + y[i];
    }
}

void vecadd_gpu(float *x, float *y, float *z, int N){
    float *x_d, *y_d, *z_d;
    size_t size = N * sizeof(float);
    cudaMalloc((void**) &x_d, N*sizeof(float));
    cudaMalloc((void**) &y_d, N*sizeof(float));
    cudaMalloc((void**) &z_d, N*sizeof(float));

    cudaMemcpy(x_d, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, size, cudaMemcpyHostToDevice);

    const unsigned int numThreadsPerBlock = 512;
    const unsigned int numBlocks = (N+numThreadsPerBlock-1)/numThreadsPerBlock;

    vector_kernel<<<numBlocks, numThreadsPerBlock>>>(x_d, y_d, z_d, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(z, z_d, size, cudaMemcpyDeviceToHost);

    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
}


int main(int argc, char**argv) {

    cudaDeviceSynchronize();

    // Allocate memory and initialize data
    Timer timer;
    unsigned int N = (argc > 1)?(atoi(argv[1])):(1 << 25);
    float* x = (float*) malloc(N*sizeof(float));
    float* y = (float*) malloc(N*sizeof(float));
    float* z = (float*) malloc(N*sizeof(float));
    for (unsigned int i = 0; i < N; ++i) {
        x[i] = rand();
        y[i] = rand();
    }

    // Vector addition on CPU
    startTime(&timer);
    vecadd_cpu(x, y, z, N);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", CYAN);

    // Vector addition on GPU
    startTime(&timer);
    vecadd_gpu(x, y, z, N);
    stopTime(&timer);
    printElapsedTime(timer, "GPU time", DGREEN);

    // Free memory
    free(x);
    free(y);
    free(z);

    return 0;
}