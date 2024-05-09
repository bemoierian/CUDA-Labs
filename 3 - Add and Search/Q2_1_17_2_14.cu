
// %%cuda

#include <stdio.h>
#include <stdlib.h>
// Kernel function to perform binary search
__global__ void search(float *array, int size, float target, int *result) {
    int tid = threadIdx.x; // Thread ID within the block

    // Allocate shared memory for local results
    __shared__ int rangeTid;

   // localResults[tid] = false;
    if (tid == 0) {
        *result = -1;
    }
    __syncthreads();

    // Calculate segment size
    int segmentSize = ceilf((float)size / blockDim.x);
    int start = tid * segmentSize;
    int end = min(start + segmentSize, size) - 1;
    while (end - start >= 1) {
        if (array[start] <= target && array[end] >= target) {
            rangeTid = start;
        }

        __syncthreads();
        segmentSize = ceilf((float)segmentSize / blockDim.x);
        start = rangeTid + tid * segmentSize;
        end = min(start + segmentSize, size) - 1;
    }

    // Let one thread write the final result to global memory
    if (array[start] == target) {
        *result = start; // Fixed variable name from i to rangeTid
    }
}



int main(int argc, char *argv[]) {
    // Check if the correct number of arguments is provided
    if (argc != 3) {
        printf("Usage: %s <input_file> <target_number>\n", argv[0]);
        return 1;
    }

    // Open and read the input file
    FILE *file = fopen(argv[1], "r");
    if (file == NULL) {
        printf("Error opening file.\n");
        return 1;
    }

    int size = 0;
    float value;
    while (fscanf(file, "%f", &value) == 1) {
        ++size;
    }
    fseek(file, 0, SEEK_SET); // Reset file pointer to the beginning

    // Allocate memory for the array
    float *h_array = (float *)malloc(size * sizeof(float));

    // Read the array elements
    for (int i = 0; i < size; ++i) {
        fscanf(file, "%f", &h_array[i]);
        // printf("%.1f\n", h_array[i]); // Print each element
    }
    fclose(file);

    // Convert the target number to float
    float target = atof(argv[2]);

    // Allocate memory for the result on the device
    int *d_result;
    cudaMalloc((void **)&d_result, sizeof(int));

    // Allocate memory for the array on the device and copy data
    float *d_array;
    cudaMalloc((void **)&d_array, size * sizeof(float));
    cudaMemcpy(d_array, h_array, size * sizeof(float), cudaMemcpyHostToDevice);

    // Invoke kernel
    int blockSize = 256;
    search<<<1, blockSize>>>(d_array, size, target, d_result);

    // Copy result back to host
    int result;
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    printf("%d\n", result);

    // Free device memory
    cudaFree(d_array);
    cudaFree(d_result);

    // Free host memory
    free(h_array);

    return 0;
}
