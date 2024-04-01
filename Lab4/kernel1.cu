#include <iostream>
#include <fstream>

// Define image dimensions
#define IMAGE_WIDTH 256
#define IMAGE_HEIGHT 256
#define IMAGE_CHANNELS 3

// Define convolution kernel dimensions
#define MASK_WIDTH 3

// Kernel function for 3D convolution
__global__ void convolution3D(float *input, int width, int height, int depth, float *output, float *mask, int maskWidth, int batch_size)
{
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Channel = blockIdx.z * blockDim.z + threadIdx.z;
    int N_start_col = Col - maskWidth / 2;
    int N_start_row = Row - maskWidth / 2;
    int N_start_channel = Channel - maskWidth / 2;

    if (Col < width && Row < height && Channel < depth)
    {
        float pixValue = 0.0;
        for (int j = 0; j < maskWidth; ++j)
        {
            for (int k = 0; k < maskWidth; ++k)
            {
                for (int l = 0; l < maskWidth; ++l)
                {
                    int currRow = N_start_row + j;
                    int currCol = N_start_col + k;
                    int currChannel = N_start_channel + l;
                    if (currRow > -1 && currRow < height && currCol > -1 && currCol < width && currChannel > -1 && currChannel < depth)
                    {
                        pixValue += input[(currRow * width + currCol) * depth + currChannel] * mask[(j * maskWidth + k) * depth + l];
                    }
                }
            }
        }
        int out_index = (Row * width + Col) * depth + Channel;
        output[out_index] = pixValue;
    }
}

float *createMask(int maskWidth, int depth);
float *readImage(std::string inputPath);
void saveImage(std::string outputPath, float *image);
int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <input_folder> <output_folder> <batch_size>" << std::endl;
        return 1;
    }

    // Load input images from folder
    std::string inputFolder = argv[1];
    std::string outputFolder = argv[2];
    int batch_size = std::atoi(argv[3]);

    // Iterate through images in the input folder
    for (int i = 0; i < batch_size; ++i)
    {
        std::string inputPath = inputFolder + "/image_" + std::to_string(i) + ".jpg";
        std::string outputPath = outputFolder + "/output_" + std::to_string(i) + ".jpg";

        float *inputImage = readImage(inputPath);
        std::cout << "Pixel 0: " << inputImage[0] << std::endl;

        // Create edge detection mask
        float *mask = createMask(MASK_WIDTH, IMAGE_CHANNELS);
        // Allocate memory for output image on GPU
        float *d_input, *d_output, *d_mask;
        size_t imageSize = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS * sizeof(float);
        size_t maskSize = MASK_WIDTH * MASK_WIDTH * IMAGE_CHANNELS * sizeof(float);
        cudaMalloc(&d_input, imageSize);
        cudaMalloc(&d_output, imageSize);
        cudaMalloc(&d_mask, maskSize);

        // Copy input image data from host to device
        cudaMemcpy(d_input, inputImage, imageSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_mask, mask, maskSize, cudaMemcpyHostToDevice);

        // Define grid and block dimensions
        dim3 threadsPerBlock(16, 16, 1);
        dim3 numBlocks((IMAGE_WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (IMAGE_HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (IMAGE_CHANNELS + threadsPerBlock.z - 1) / threadsPerBlock.z);

        // Launch kernel
        convolution3D<<<numBlocks, threadsPerBlock>>>(d_input, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS, d_output, d_mask, MASK_WIDTH, batch_size);
        cudaDeviceSynchronize();
        // Copy output image data from device to host
        float *h_output = new float[imageSize];
        cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);

        // Save output image to binary file
        saveImage(outputPath, h_output);

        // Free device memory
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_mask);
        delete[] inputImage;
        delete[] h_output;
        delete[] mask;
    }

    return 0;
}
float *createMask(int maskWidth, int depth)
{
    // Allocate memory for the mask
    float *mask = new float[maskWidth * maskWidth * depth];

    // Sobel filter for edge detection (example)
    // float sobelX[maskWidth][maskWidth] = {{-1, 0, 1},
    //                                       {-2, 0, 2},
    //                                       {-1, 0, 1}};
    // 3*3 (1/9) filter
    float sobelX[maskWidth][maskWidth] = {{1 / 9, 1 / 9, 1 / 9},
                                          {1 / 9, 1 / 9, 1 / 9},
                                          {1 / 9, 1 / 9, 1 / 9}};

    // Fill the mask with the Sobel filter values
    for (int i = 0; i < maskWidth; ++i)
    {
        for (int j = 0; j < maskWidth; ++j)
        {
            for (int k = 0; k < depth; ++k)
            {
                mask[(i * maskWidth + j) * depth + k] = sobelX[i][j];
            }
        }
    }

    return mask;
}
float *readImage(std::string inputPath)
{
    // Load image from binary file
    std::ifstream inputFile(inputPath, std::ios::binary);
    if (!inputFile)
    {
        std::cerr << "Could not read image: " << inputPath << std::endl;
        return nullptr;
    }
    float *inputImage = new float[IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS];
    inputFile.read(reinterpret_cast<char *>(inputImage), sizeof(float) * IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS);
    inputFile.close();

    return inputImage;
}
void saveImage(std::string outputPath, float *image)
{

    // Save image to binary file
    std::ofstream outputFile(outputPath, std::ios::binary);
    outputFile.write(reinterpret_cast<char *>(image), sizeof(float) * IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS);
    outputFile.close();
}
