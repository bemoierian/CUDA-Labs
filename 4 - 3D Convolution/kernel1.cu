#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

// Define image dimensions
#define IMAGE_WIDTH 600
#define IMAGE_HEIGHT 350
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
                        pixValue += input[(currRow * width + currCol) * depth + currChannel] * mask[(j * maskWidth + k) * maskWidth + l];
                    }
                }
            }
        }
        int out_index = (Row * width + Col) * depth + Channel;
        output[out_index] = pixValue;
    }
}

float *createMask(int maskWidth, int depth);
float *readImageToFloat(const std::string &imagePath);
void saveFloatImage(const std::string &outputPath, float *imageData, int width, int height);
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
        std::string outputPath = "./" + outputFolder + "/output_new_" + std::to_string(i) + ".jpg";

        float *inputImage = readImageToFloat(inputPath);
        std::cout << "Pixel 50,50,0: " << inputImage[(50 * IMAGE_WIDTH + 50) * IMAGE_CHANNELS + 0] << std::endl;
        std::cout << "Pixel 50,50,1: " << inputImage[(50 * IMAGE_WIDTH + 50) * IMAGE_CHANNELS + 1] << std::endl;
        std::cout << "Pixel 50,50,2: " << inputImage[(50 * IMAGE_WIDTH + 50) * IMAGE_CHANNELS + 2] << std::endl;
        std::string outputPath2 = "./" + outputFolder + "/original_" + std::to_string(i) + ".jpg";
        saveFloatImage(outputPath2, inputImage, IMAGE_WIDTH, IMAGE_HEIGHT);

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
        dim3 threadsPerBlock(16, 16, 3);
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
        std::cout << "Output Pixel 50,50,0: " << h_output[(50 * IMAGE_WIDTH + 50) * IMAGE_CHANNELS + 0] << std::endl;
        std::cout << "Output Pixel 50,50,1: " << h_output[(50 * IMAGE_WIDTH + 50) * IMAGE_CHANNELS + 1] << std::endl;
        std::cout << "Output Pixel 50,50,2: " << h_output[(50 * IMAGE_WIDTH + 50) * IMAGE_CHANNELS + 2] << std::endl;

        saveFloatImage(outputPath, h_output, IMAGE_WIDTH, IMAGE_HEIGHT);

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
    float smoothFactor = 1.0 / (3.0 * 3.0 * 3.0);
    float maskVal[maskWidth][maskWidth] = {{smoothFactor, smoothFactor, smoothFactor},
                                           {smoothFactor, smoothFactor, smoothFactor},
                                           {smoothFactor, smoothFactor, smoothFactor}};

    // Fill the mask with the Sobel filter values
    for (int i = 0; i < maskWidth; ++i)
    {
        for (int j = 0; j < maskWidth; ++j)
        {
            for (int k = 0; k < depth; ++k)
            {
                mask[(i * maskWidth + j) * depth + k] = maskVal[i][j];
            }
        }
    }

    return mask;
}

// Function to read an image and convert it to float*
float *readImageToFloat(const std::string &imagePath)
{
    // Read an image using OpenCV as RGB
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cerr << "Error: Unable to load image" << std::endl;
        return nullptr;
    }

    // Convert the image to float
    cv::Mat imageFloat;
    image.convertTo(imageFloat, CV_32F);

    // Allocate memory for float* to store the image data
    float *imageData = new float[image.rows * image.cols * image.channels()];

    // Copy the image data to a float*
    memcpy(imageData, imageFloat.data, image.rows * image.cols * image.channels() * sizeof(float));

    return imageData;
}

void saveFloatImage(const std::string &outputPath, float *imageData, int width, int height)
{
    // Create a cv::Mat from the float* data
    cv::Mat imageFloat(height, width, CV_32FC3, imageData);

    // Convert the image back to 8-bit unsigned char
    cv::Mat image;
    imageFloat.convertTo(image, CV_8UC3);

    // Save the image using OpenCV
    if (!cv::imwrite(outputPath, image))
    {
        std::cout << "Error: Unable to save image" << std::endl;
    }
}