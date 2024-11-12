#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define SIZE 1000000
#define RANGE 10

// Time function
double get_clock() {
    struct timeval tv;
    int ok;
    ok = gettimeofday(&tv, NULL);
    if (ok < 0) { 
        printf("gettimeofday error\n");
        exit(1);
    }
    return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

// Kernel to initialize the histogram to zero
__global__ void initHistogram(int *histogram, int range) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < range) {
        histogram[idx] = 0;  // Set each element to zero
    }
}

// GPU kernel to compute histogram with strided access
__global__ void computeHistogramGPU(int *data, int *histogram, int size, int range) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  // Global thread index
    int stride = blockDim.x * gridDim.x;  // Stride for each thread to process multiple elements

    for (int i = idx; i < size; i += stride) {
        // Atomic add to ensure thread-safe updates to the histogram
        atomicAdd(&histogram[data[i]], 1);
    }
}

int main() {
    int *data = (int *)malloc(sizeof(int) * SIZE);
    int *histogram = (int *)malloc(sizeof(int) * RANGE);
    int *d_data, *d_histogram;

    // Generate random data
    printf("Generated Data:\n");
    for (int i = 0; i < SIZE; i++) {
        data[i] = rand() % RANGE;
        printf("%d ", data[i]);
    }
    printf("\n");

    // Allocate memory on the device
    cudaMalloc((void **)&d_data, sizeof(int) * SIZE);
    cudaMalloc((void **)&d_histogram, sizeof(int) * RANGE);

    // Launch the initialization kernel
    int blockSize = 256; // Number of threads per block
    int numBlocks = (RANGE + blockSize - 1) / blockSize; // Number of blocks for initialization
    initHistogram<<<numBlocks, blockSize>>>(d_histogram, RANGE);

    // Copy data to the device
    cudaMemcpy(d_data, data, sizeof(int) * SIZE, cudaMemcpyHostToDevice);

    // Measure time
    double t0 = get_clock();

    // Launch the histogram computation kernel
    numBlocks = (SIZE + blockSize - 1) / blockSize; // Number of blocks for main kernel
    computeHistogramGPU<<<numBlocks, blockSize>>>(d_data, d_histogram, SIZE, RANGE);

    // Synchronize to ensure kernel execution is done
    cudaDeviceSynchronize();

    double t1 = get_clock();

    // Copy the histogram back to the host
    cudaMemcpy(histogram, d_histogram, sizeof(int) * RANGE, cudaMemcpyDeviceToHost);

    // Print the histogram
    printf("Histogram:\n");
    for (int i = 0; i < RANGE; i++) {
        printf("%d: %d\n", i, histogram[i]);
    }

    // Print time taken
    printf("Time: %f ns\n", 1000000000.0 * (t1 - t0));

    // Free memory
    free(data);
    free(histogram);
    cudaFree(d_data);
    cudaFree(d_histogram);

    return 0;
}
