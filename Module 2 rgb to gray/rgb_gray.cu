#include "../timer.h"

void rgbtogray_cpu(unsigned char *red, unsigned char *green, unsigned char *blue,
    unsigned char *gray, unsigned int width, unsigned int height){
    for (unsigned int row = 0; row<height; row++){
        for (unsigned int col = 0; col<width; col++){
            unsigned int i = row*width + col;
            gray[i] = red[i]*3/10 + green[i]*6/10 + blue[i]*1/10;
        }
    }
}

__global__ void rgbgray_kernel(unsigned char *red, unsigned char *green, unsigned char *blue, 
    unsigned char *gray, unsigned int width, unsigned int height){
    unsigned int col = blockDim.x*blockIdx.x + threadIdx.x;
    unsigned int row = blockDim.y*blockIdx.y + threadIdx.y;

    unsigned int i = row*width + col;
    if (row<height && col<width){
        gray[i] = red[i]*3/10 + green[i]*6/10 + blue[i]*1/10;
    }
}

void rgbtogray(unsigned char *red, unsigned char *green, unsigned char *blue, 
    unsigned char *gray, unsigned int width, unsigned int height){
    // Allocating mem on gpu 
    unsigned char *red_d, *green_d, *blue_d, *gray_d;
    size_t size = height * width * sizeof(unsigned char);
    cudaMalloc((void**) &red_d, size);
    cudaMalloc((void**) &green_d, size);
    cudaMalloc((void**) &blue_d, size);
    cudaMalloc((void**) &gray_d, size);

    // Copy pointers 
    cudaMemcpy(red_d, red, size, cudaMemcpyHostToDevice);
    cudaMemcpy(green_d, green, size, cudaMemcpyHostToDevice);
    cudaMemcpy(blue_d, blue, size, cudaMemcpyHostToDevice);

    // Kernel logic
    dim3 numThreadsPerBlock(32, 32);
    dim3 numBlocks((width+numThreadsPerBlock.x-1)/numThreadsPerBlock.x, (height+numThreadsPerBlock.y-1)/numThreadsPerBlock.y);

    Timer timer;
    startTime(&timer);
    rgbgray_kernel<<<numBlocks, numThreadsPerBlock>>>(red_d, green_d, blue_d, gray_d, width, height);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);
    // Copy back to cpu
    cudaMemcpy(gray, gray_d, size, cudaMemcpyDeviceToHost);

    // Free GPU mem
    cudaFree(red_d);
    cudaFree(green_d);
    cudaFree(blue_d);
    cudaFree(gray_d);
}


int main(int argc, char**argv) {

    cudaDeviceSynchronize();

    // Allocate memory and initialize data
    Timer timer;
    unsigned int width = (argc > 1)?(atoi(argv[1])):(1 << 12);
    unsigned int height = (argc > 2)?(atoi(argv[2])):(1 << 12);
    unsigned int N = width*height;
    
    unsigned char* red = (unsigned char*) malloc(N*sizeof(unsigned char));
    unsigned char* green = (unsigned char*) malloc(N*sizeof(unsigned char));
    unsigned char* blue = (unsigned char*) malloc(N*sizeof(unsigned char));
    unsigned char* gray = (unsigned char*) malloc(N*sizeof(unsigned char));
    unsigned char* gray_cpu = (unsigned char*) malloc(N*sizeof(unsigned char));
    
    for (unsigned int i = 0; i < N; ++i) {
        red[i] = rand()%256;
        green[i] = rand()%256;
        blue[i] = rand()%256;
    }

    // RGB to gray on CPU
    startTime(&timer);
    rgbtogray_cpu(red, green, blue, gray_cpu, width, height);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", CYAN);

    // RGB to gray on GPU
    startTime(&timer);
    rgbtogray(red, green, blue, gray, width, height);
    stopTime(&timer);
    printElapsedTime(timer, "GPU time", DGREEN);

    // Verify GPU result against CPU
    unsigned int mismatches = 0;
    for (unsigned int i = 0; i < N; ++i) {
        if (gray[i] != gray_cpu[i]) ++mismatches;
    }
    printf("Mismatches: %u / %u\n", mismatches, N);

    // Free memory
    free(red);
    free(green);
    free(blue);
    free(gray);
    free(gray_cpu);

    return 0;
}