// Kernel 1: Each thread computes one element of C

#include <stdio.h>

__global__ void matrixVectorMul(float *matrix, float *vector, float *result, int N, int M)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N)
    {
        float sum = 0.0f;
        for (int i = 0; i < M; ++i)
        {
            sum += matrix[tid * M + i] * vector[i];
        }
        result[tid] = sum;
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
        int matrix_size = rows * columns * sizeof(float);
        int vector_size = columns * sizeof(float);
        int result_size = rows * sizeof(float);
        // Matrix initialization
        float *h_A = (float *)malloc(matrix_size);
        float *h_B = (float *)malloc(vector_size);
        float *h_C = (float *)malloc(result_size);
        for (int j = 0; j < rows; j++)
        {
            for (int k = 0; k < columns; k++)
            {
                fscanf(in_fptr, "%f", &h_A[j * rows + k]);
            }
        }
        for (int j = 0; j < columns; j++)
        {
            fscanf(in_fptr, "%f", &h_B[j]);
        }
        // Device memory allocation
        float *d_A, *d_B, *d_C;
        cudaMalloc((void **)&d_A, matrix_size);
        cudaMalloc((void **)&d_B, vector_size);
        cudaMalloc((void **)&d_C, result_size);

        // Data transfer: Host to Device
        cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, vector_size, cudaMemcpyHostToDevice);

        // Launch kernel
        int block_size = 256;
        int grid_size = (rows + block_size - 1) / block_size;
        matrixVectorMul<<<grid_size, block_size>>>(d_A, d_B, d_C, rows, columns);

        // Data transfer: Device to Host
        cudaMemcpy(h_C, d_C, result_size, cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        // Append the result to the output file
        for (int j = 0; j < rows; j++)
        {
            fprintf(out_fptr, "%f\n", h_C[j]);
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
