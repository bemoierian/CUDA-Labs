// Kernel 3: Each thread computes one column of the output matrix

#include <stdio.h>

#define N 10000

__global__ void MatAdd(float *A, float *B, float *C, int height, int width)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < width)
    {
        for (int row = 0; row < height; ++row)
        {
            C[row * width + col] = A[row * width + col] + B[row * width + col];
        }
    }
}

int main(int argc, char *argv[])
{
    // read testcases from input file
    FILE *in_fptr;
    in_fptr = fopen(argv[1], "r");
    // Create output file to write the results
    FILE *out_fptr;
    out_fptr = fopen(argv[2], "w");
    // Read the number of testcases
    int numOfTestcases;
    fscanf(in_fptr, "%d", &numOfTestcases);
    for (int i = 0; i < numOfTestcases; i++)
    {
        int rows, columns;
        fscanf(in_fptr, "%d %d", &rows, &columns);
        // Size of matrices
        int size = rows * columns * sizeof(float);
        // Matrix initialization
        float *h_A = (float *)malloc(size);
        float *h_B = (float *)malloc(size);
        float *h_C = (float *)malloc(size);
        for (int j = 0; j < rows; j++)
        {
            for (int k = 0; k < columns; k++)
            {
                fscanf(in_fptr, "%f", &h_A[j * columns + k]);
            }
        }
        for (int j = 0; j < rows; j++)
        {
            for (int k = 0; k < columns; k++)
            {
                fscanf(in_fptr, "%f", &h_B[j * columns + k]);
            }
        }
        // Device memory allocation
        float *d_A, *d_B, *d_C;
        cudaMalloc((void **)&d_A, size);
        cudaMalloc((void **)&d_B, size);
        cudaMalloc((void **)&d_C, size);

        // Data transfer: Host to Device
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

        // Kernel invocation
        int threadsPerBlock = 16;
        int numBlocks = (columns - 1) / threadsPerBlock + 1;
        MatAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, rows, columns);

        // Data transfer: Device to Host
        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        // Append the result to the output file
        for (int j = 0; j < rows; j++)
        {
            for (int k = 0; k < columns; k++)
            {
                fprintf(out_fptr, "%.1f ", h_C[j * columns + k]);
            }
            fprintf(out_fptr, "\n");
        }

        // Free host memory
        free(h_A);
        free(h_B);
        free(h_C);
    }
    fclose(in_fptr);
    fclose(out_fptr);

    return 0;
}
