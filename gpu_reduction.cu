#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>
#include <limits.h>

#define SIZE 10000
#define BLOCKSIZE 16

// Host function to get time
double get_clock() {
    struct timeval tv;
    int ok;
    ok = gettimeofday(&tv, (void *) 0);
    if (ok < 0) { printf("gettimeofday error"); }
    return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

// GPU kernel function with embedded reduction logic
__global__ void reductionKernel(int *data, int length, char op) {
    __shared__ int partialSum[2 * BLOCKSIZE];
    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;

    // Initialize shared memory based on operation type
    switch (op) {
        case '+':
            partialSum[t] = (start + t < length) ? data[start + t] : 0;
            partialSum[blockDim.x + t] = (start + blockDim.x + t < length) ? data[start + blockDim.x + t] : 0;
            break;
        case '*':
            partialSum[t] = (start + t < length) ? data[start + t] : 1;
            partialSum[blockDim.x + t] = (start + blockDim.x + t < length) ? data[start + blockDim.x + t] : 1;
            break;
        case 'm':
            partialSum[t] = (start + t < length) ? data[start + t] : INT_MAX;
            partialSum[blockDim.x + t] = (start + blockDim.x + t < length) ? data[start + blockDim.x + t] : INT_MAX;
            break;
        case 'M':
            partialSum[t] = (start + t < length) ? data[start + t] : INT_MIN;
            partialSum[blockDim.x + t] = (start + blockDim.x + t < length) ? data[start + blockDim.x + t] : INT_MIN;
            break;
        default:
            printf("Unsupported operation.\n");
            return;
    }
    
    // Perform reduction in shared memory
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        if (t % stride == 0 && t + stride < 2 * blockDim.x) {
            switch (op) {
                case '+':
                    partialSum[2 * t] += partialSum[2 * t + stride];
                    break;
                case '*':
                    partialSum[2 * t] *= partialSum[2 * t + stride];
                    break;
                case 'm':
                    partialSum[2 * t] = min(partialSum[2 * t], partialSum[2 * t + stride]);
                    break;
                case 'M':
                    partialSum[2 * t] = max(partialSum[2 * t], partialSum[2 * t + stride]);
                    break;
            }
        }
    }

    // Write the result of this block to global memory
    if (t == 0) {
        data[blockIdx.x] = partialSum[0];
    }
}

int main() {
    int *data = (int *)malloc(sizeof(int) * SIZE);
    int *d_data;
    char op;

    // Initialize data on host
    for (int i = 0; i < SIZE; i++) {
        data[i] = i;
        printf(" %d", data[i]);
    }
    printf("\n");
    printf("Enter the reduction operation (+, *, m for min, M for max): ");
    scanf(" %c", &op);

    // Allocate device memory
    cudaMalloc((void **)&d_data, sizeof(int) * SIZE);

    // Copy data to device
    cudaMemcpy(d_data, data, sizeof(int) * SIZE, cudaMemcpyHostToDevice);

    // Time the GPU computation
    double t0 = get_clock();

    // Launch the kernel with appropriate number of blocks
    int numBlocks = (SIZE + (2 * BLOCKSIZE) - 1) / (2 * BLOCKSIZE);
    reductionKernel<<<numBlocks, BLOCKSIZE>>>(d_data, SIZE, op);
    cudaDeviceSynchronize();

    // Check for errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copy partial results back to host
    cudaMemcpy(data, d_data, sizeof(int) * numBlocks, cudaMemcpyDeviceToHost);
    double t1 = get_clock();

    // Sum up partial results on host
    int final_result = data[0];
    for (int i = 1; i < numBlocks; i++) {
        switch (op) {
            case '+': final_result += data[i]; break;
            case '*': final_result *= data[i]; break;
            case 'm': final_result = (final_result < data[i]) ? final_result : data[i]; break;
            case 'M': final_result = (final_result > data[i]) ? final_result : data[i]; break;
        }
    }

    // Print result and time taken
    printf("Reduction result: %d\n", final_result);
    printf("Time: %f ns\n", 1000000000.0 * (t1 - t0));

    // Clean up
    cudaFree(d_data);
    free(data);
    return 0;
}
