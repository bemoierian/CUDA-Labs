
// %%cuda
#include <stdio.h>

// Kernel function to calculate sum of array elements
__global__ void sumArray(float *d_array, int size, float *d_result) {
    int tid = threadIdx.x;

    // Each thread sums up a portion of the array
    float sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        sum += d_array[i];
    }

    // Store partial sum into shared memory
    extern __shared__ float shared[];
    shared[tid] = sum;
    __syncthreads();

    // Reduction within block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    // Store the block sum to global memory
    if (tid == 0) {
        d_result[blockIdx.x] = shared[0];
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        // printf("Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    char *input_file = argv[1];

    // Read array size and elements from external file
    FILE *file;
    file = fopen(input_file, "r");
    if (file == NULL) {
        printf("Error opening file %s.\n", input_file);
        return 1;
    }

    int size = 0;
    float value;
    while (fscanf(file, "%f", &value) == 1) {
        ++size;
    }
    fseek(file, 0, SEEK_SET); // Reset file pointer to the beginning

    float *h_array = (float*)malloc(size * sizeof(float));
    // printf("Array content:\n");
    for (int i = 0; i < size; ++i) {
        fscanf(file, "%f", &h_array[i]);
        // printf("%.1f\n", h_array[i]); // Print each element
    }
    fclose(file);
    printf("\n");

    // Allocate memory on device and copy array
    float *d_array, *d_result;
    cudaMalloc((void**)&d_array, size * sizeof(float));
    cudaMalloc((void**)&d_result, sizeof(float));

    cudaMemcpy(d_array, h_array, size * sizeof(float), cudaMemcpyHostToDevice);

    // Invoke kernel
    int blockSize = 1024; // Choose a block size according to the device capabilities
    int numBlocks = 1; // Only one block
    sumArray<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_array, size, d_result);

    // Copy result back to host and print
    float h_result;
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%.1f", h_result);

    // Free device memory
    cudaFree(d_array);
    cudaFree(d_result);

    // Free host memory
    free(h_array);

    return 0;
}
