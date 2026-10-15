#include "../timer.h"

#define BLUR_SIZE 1     // radius: each output pixel averages a (2*BLUR_SIZE+1)^2 window

void blur_cpu(unsigned char *image, unsigned char *blurred, unsigned int width, unsigned int height){
    for (int outRow = 0; outRow<(int)height; outRow++){
        for (int outCol = 0; outCol<(int)width; outCol++){
            unsigned int sum = 0;
            unsigned int count = 0;
            for (int inRow = outRow-BLUR_SIZE; inRow<=outRow+BLUR_SIZE; inRow++){
                for (int inCol = outCol-BLUR_SIZE; inCol<=outCol+BLUR_SIZE; inCol++){
                    if (inRow>=0 && inRow<(int)height && inCol>=0 && inCol<(int)width){
                        sum += image[inRow*width + inCol];
                        count++;
                    }
                }
            }
            blurred[outRow*width + outCol] = sum/count;
        }
    }
}

__global__ void blur_kernel(unsigned char *image, unsigned char *blurred, unsigned int width, unsigned int height){
    // TODO: implement blur
    int outCol = blockDim.x*blockIdx.x + threadIdx.x;
    int outRow = blockDim.y*blockIdx.y + threadIdx.y;

    if(outRow< height && outCol < width){
        unsigned int average = 0;
        unsigned int count = 0;
        for (int inRow = outRow-BLUR_SIZE; inRow <= outRow + BLUR_SIZE; inRow++){
            for (int inCol = outCol-BLUR_SIZE; inCol <= outCol + BLUR_SIZE; inCol++){
                if (inRow<height && inRow >= 0 && inCol<width && inCol >= 0 ){
                    average += image[inRow*width + inCol];
                    count++;
                }
            }
        }
        blurred[outRow*width+outCol] = (unsigned char)(average/count);
    }
}

void blur_gpu(unsigned char *image, unsigned char *blurred, unsigned int width, unsigned int height){
    // Allocating mem on gpu
    unsigned char *image_d, *blurred_d;
    size_t size = height * width * sizeof(unsigned char);
    cudaMalloc((void**) &image_d, size);
    cudaMalloc((void**) &blurred_d, size);

    // Copy pointers
    cudaMemcpy(image_d, image, size, cudaMemcpyHostToDevice);

    // Kernel logic
    dim3 numThreadsPerBlock(32, 32);
    dim3 numBlocks((width+numThreadsPerBlock.x-1)/numThreadsPerBlock.x, (height+numThreadsPerBlock.y-1)/numThreadsPerBlock.y);

    Timer timer;
    startTime(&timer);
    blur_kernel<<<numBlocks, numThreadsPerBlock>>>(image_d, blurred_d, width, height);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);
    // Copy back to cpu
    cudaMemcpy(blurred, blurred_d, size, cudaMemcpyDeviceToHost);

    // Free GPU mem
    cudaFree(image_d);
    cudaFree(blurred_d);
}


int main(int argc, char**argv) {

    cudaDeviceSynchronize();

    // Allocate memory and initialize data
    Timer timer;
    unsigned int width = (argc > 1)?(atoi(argv[1])):(1 << 12);
    unsigned int height = (argc > 2)?(atoi(argv[2])):(1 << 12);
    unsigned int N = width*height;

    unsigned char* image = (unsigned char*) malloc(N*sizeof(unsigned char));
    unsigned char* blurred = (unsigned char*) malloc(N*sizeof(unsigned char));
    unsigned char* blurred_cpu = (unsigned char*) malloc(N*sizeof(unsigned char));

    for (unsigned int i = 0; i < N; ++i) {
        image[i] = rand()%256;
    }

    // Blur on CPU
    startTime(&timer);
    blur_cpu(image, blurred_cpu, width, height);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", CYAN);

    // Blur on GPU
    startTime(&timer);
    blur_gpu(image, blurred, width, height);
    stopTime(&timer);
    printElapsedTime(timer, "GPU time", DGREEN);

    // Verify GPU result against CPU
    unsigned int mismatches = 0;
    for (unsigned int i = 0; i < N; ++i) {
        if (blurred[i] != blurred_cpu[i]) ++mismatches;
    }
    printf("Mismatches: %u / %u\n", mismatches, N);

    // Free memory
    free(image);
    free(blurred);
    free(blurred_cpu);

    return 0;
}
