#include "../timer.h"

void matmul_cpu(float *A, float *B, float *C, unsigned int N){
    for (unsigned int row = 0; row<N; row++){
        for (unsigned int col = 0; col<N; col++){
            float sum = 0.0f;
            for (unsigned int k = 0; k<N; k++){
                sum += A[row*N + k]*B[k*N + col];
            }
            C[row*N + col] = sum;
        }
    }
}

void matmul_gpu(float *A, float *B, float *C, unsigned int N){
    // TODO
}

int main(int argc, char**argv) {

    cudaDeviceSynchronize();

    // Allocate memory and initialize data
    Timer timer;
    unsigned int N = (argc > 1)?(atoi(argv[1])):(1 << 10);     // A, B, C are N x N

    float* A = (float*) malloc(N*N*sizeof(float));
    float* B = (float*) malloc(N*N*sizeof(float));
    float* C = (float*) malloc(N*N*sizeof(float));
    float* C_cpu = (float*) malloc(N*N*sizeof(float));

    for (unsigned int i = 0; i < N*N; ++i) {
        A[i] = rand()/(float)RAND_MAX;
        B[i] = rand()/(float)RAND_MAX;
    }

    // Matmul on CPU
    startTime(&timer);
    matmul_cpu(A, B, C_cpu, N);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", CYAN);

    // Matmul on GPU
    startTime(&timer);
    matmul_gpu(A, B, C, N);
    stopTime(&timer);
    printElapsedTime(timer, "GPU time", DGREEN);

    // Verify GPU result against CPU (floats: allow small relative error)
    unsigned int mismatches = 0;
    for (unsigned int i = 0; i < N*N; ++i) {
        float diff = C[i] - C_cpu[i];
        if (diff < 0) diff = -diff;
        if (diff > 1e-3f*C_cpu[i]) ++mismatches;
    }
    printf("Mismatches: %u / %u\n", mismatches, N*N);

    // Free memory
    free(A);
    free(B);
    free(C);
    free(C_cpu);

    return 0;
}
