// nvcc -w K2_1_17_2_14.cu -o kernel2.out
// ./kernel2.out Input "Output 2" 4 mask.txt
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "dirent.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Define tile dimensions
#define O_TILE_WIDTH 16

#define MAX_MASK_WIDTH 25
__constant__ float c_mask[MAX_MASK_WIDTH];

// Kernel function for 3D convolution with tiling
__global__ void convolution3D(const uint8_t *input, int width, int height, int depth, float *output, float *mask, int maskWidth)
{
    // Define shared memory size dynamically
    extern __shared__ float inputTile[];

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Output indices
    int col_o = blockIdx.x * O_TILE_WIDTH + tx;
    int row_o = blockIdx.y * O_TILE_WIDTH + ty;

    // Input indices
    int col_i = col_o - maskWidth / 2;
    int row_i = row_o - maskWidth / 2;

    int batch_index = blockIdx.z;
    int image_index_in_batch = batch_index * width * height * depth;

    float pixelValue = 0.0;
    for (int channel = 0; channel < depth; channel++)
    {
        // Load input tile into shared memory
        if (col_i >= 0 && col_i < width && row_i >= 0 && row_i < height)
            inputTile[ty * (O_TILE_WIDTH + maskWidth - 1) + tx] = static_cast<float>(input[(row_i * width + col_i) * depth + channel + image_index_in_batch]) / 255.0f;
        else
            inputTile[ty * (O_TILE_WIDTH + maskWidth - 1) + tx] = 0.0f;

        __syncthreads();

        // Compute convolution
        if (tx < O_TILE_WIDTH && ty < O_TILE_WIDTH)
        {
            for (int j = 0; j < maskWidth; j++)
                for (int k = 0; k < maskWidth; k++)
                    pixelValue += inputTile[(ty + j) * (O_TILE_WIDTH + maskWidth - 1) + (tx + k)] * c_mask[j * maskWidth + k];
        }

        __syncthreads();
    }
    if (tx < O_TILE_WIDTH && ty < O_TILE_WIDTH)
    {
        // Write output
        if (col_o < width && row_o < height)
        {
            int outIndex = (row_o * width + col_o) + batch_index * width * height;
            output[outIndex] = pixelValue;
        }
    }
}

void process_batch(const char *input_folder, const char *output_folder, float *mask, int maskWidth, int batch_size);
void launch_convolution_kernel(std::vector<uint8_t *> images, int width, int height, int depth, float *output_imgs, float *mask, int maskWidth, int batch_size, int batch_index, int current_batch_size);
void load_images(std::vector<uint8_t *> &images, const char *input_folder, int batch_size, int &width, int &height, int &depth);
float *createMask(float **maskVal, int maskWidth);
void normalizeFloatImage(float *img, uint8_t *output_img, int width, int height);

int main(int argc, char **argv)
{
    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0] << " <input_folder> <output_folder> <batch_size> <mask_file>" << std::endl;
        return 1;
    }

    // Load input images from folder
    const char *inputFolder = argv[1];
    const char *outputFolder = argv[2];
    int batch_size = std::atoi(argv[3]);
    const char *maskfilePath = argv[4];
    std::ifstream inFile(maskfilePath); // Open the mask file
    if (!inFile)
    {
        std::cerr << "Unable to open file mask.txt" << std::endl;
        return 1;
    }

    int maskWidth;
    inFile >> maskWidth; // Read mask width from the first line

    float **maskVal = new float *[maskWidth]; // Allocate memory for rows
    for (int i = 0; i < maskWidth; ++i)
    {
        maskVal[i] = new float[maskWidth]; // Allocate memory for columns
    }

    // Read mask values from the file
    for (int i = 0; i < maskWidth; ++i)
    {
        for (int j = 0; j < maskWidth; ++j)
        {
            std::string maskValue;
            if (!(inFile >> maskValue))
            {
                std::cerr << "Error reading mask value" << std::endl;
                return 1;
            }

            // Parse the fraction if it contains '/'
            size_t pos = maskValue.find('/');
            if (pos != std::string::npos)
            {
                int numerator = std::atoi(maskValue.substr(0, pos).c_str());
                int denominator = std::atoi(maskValue.substr(pos + 1).c_str());
                if (denominator == 0)
                {
                    std::cerr << "Invalid denominator in mask value" << std::endl;
                    return 1;
                }
                maskVal[i][j] = static_cast<double>(numerator) / denominator;
            }
            else
            {
                // If there's no '/', parse the value directly
                maskVal[i][j] = std::atof(maskValue.c_str());
            }
        }
    }
    inFile.close();

    // Create mask using the provided function
    float *mask = createMask(maskVal, maskWidth);

    process_batch(inputFolder, outputFolder, mask, maskWidth, batch_size);

    delete[] mask;
    return 0;
}

void process_batch(const char *input_folder, const char *output_folder, float *mask, int maskWidth, int batch_size)
{
    // Load images from input folder
    int width, height, depth;
    std::vector<uint8_t *> images;
    std::vector<uint8_t *> imagesOutput;
    load_images(images, input_folder, batch_size, width, height, depth);

    int imageSize = width * height * depth;
    int outputImageSize = width * height;
    int number_of_images = images.size();
    int num_iterations = (number_of_images + batch_size - 1) / batch_size;
    int remaining_images = number_of_images;
    uint8_t *output_img_uint8;
    float *output_imgs = (float *)malloc(batch_size * outputImageSize * sizeof(float));

    for (int batch_index = 0; batch_index < num_iterations; ++batch_index)
    {
        // Start convolution batch processing
        int current_batch_size = min(batch_size, remaining_images);
        printf("Started processing %d images with dimensions : (%d, %d, %d) \n", current_batch_size, width, height, depth);
        launch_convolution_kernel(images, width, height, depth, output_imgs, mask, maskWidth, batch_size, batch_index, current_batch_size);
        remaining_images -= batch_size;
        // Push output images into output vector
        for (int i = 0; i < current_batch_size; ++i)
        {
            output_img_uint8 = (uint8_t *)malloc(outputImageSize * sizeof(uint8_t));
            float *output_image_ptr = (float *)malloc(outputImageSize * sizeof(float));
            memcpy(output_image_ptr, output_imgs + i * outputImageSize, outputImageSize * sizeof(float));
            normalizeFloatImage(output_image_ptr, output_img_uint8, width, height);
            imagesOutput.push_back(output_img_uint8);
            free(output_image_ptr);
        }
        printf("Batch %d completed\n", batch_index + 1);
    }

    // Write output
    printf("\nWriting %d outputs ... \n", imagesOutput.size());
    for (int i = 0; i < imagesOutput.size(); ++i)
    {
        // std::string outputPath2 = std::string(output_folder) + "/original_" + std::to_string(i) + ".jpg";
        // stbi_write_jpg(outputPath2.c_str(), width, height, depth, images[i], 100); // original

        std::string outputPath = std::string(output_folder) + "/output_" + std::to_string(i) + ".jpg";
        stbi_write_jpg(outputPath.c_str(), width, height, 1, imagesOutput[i], 100); // convolution
        // printf("%d - Path : %s \n", i, outputPath.c_str());
    }
    printf("----------------------------------------------------------------------------------\n");

    // Free allocated memory
    for (int i = 0; i < images.size(); ++i)
    {
        stbi_image_free(images[i]);
        stbi_image_free(imagesOutput[i]);
    }

    free(output_imgs);
}

void launch_convolution_kernel(std::vector<uint8_t *> images, int width, int height, int depth, float *output_imgs, float *mask, int maskWidth, int batch_size, int batch_index, int current_batch_size)
{
    float *d_mask, *d_output;
    uint8_t *d_input;
    int imageSize = width * height * depth;
    int outputImageSize = width * height;

    cudaMalloc((void **)&d_input, imageSize * batch_size * sizeof(uint8_t));
    cudaMalloc((void **)&d_output, outputImageSize * batch_size * sizeof(float));
    cudaMalloc((void **)&d_mask, maskWidth * maskWidth * sizeof(float));

    uint8_t *input_imgs = (uint8_t *)malloc(batch_size * imageSize * sizeof(uint8_t));
    // Copy input images data from host to device
    for (int i = 0; i < current_batch_size; ++i)
    {
        memcpy(input_imgs + i * imageSize, images[i + batch_index * batch_size], imageSize * sizeof(uint8_t));
    }
    cudaMemcpy(d_input, input_imgs, batch_size * imageSize * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, maskWidth * maskWidth * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_mask, mask, maskWidth * maskWidth * sizeof(float));

    // Define grid and block dimensions
    dim3 threadsPerBlock(O_TILE_WIDTH + maskWidth - 1, O_TILE_WIDTH + maskWidth - 1, 1);
    dim3 numBlocks((width - 1) / O_TILE_WIDTH + 1,
                   (height - 1) / O_TILE_WIDTH + 1,
                   current_batch_size);
    int sharedMemorySize = sizeof(float) * (O_TILE_WIDTH + maskWidth - 1) * (O_TILE_WIDTH + maskWidth - 1);
    // Launch kernel
    convolution3D<<<numBlocks, threadsPerBlock, sharedMemorySize>>>(d_input, width, height, depth, d_output, d_mask, maskWidth);
    cudaDeviceSynchronize();

    cudaMemcpy(output_imgs, d_output, batch_size * outputImageSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mask);
}

// Define a struct to hold image data along with its filename
struct ImageData
{
    std::string filename;
    uint8_t *data;
};

// Comparator function to sort ImageData based on filenames
bool compare_filenames(const ImageData &a, const ImageData &b)
{
    return a.filename < b.filename;
}
void load_images(std::vector<uint8_t *> &images, const char *input_folder, int batch_size, int &width, int &height, int &depth)
{
    std::vector<ImageData> images_ds;
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(input_folder)) != NULL)
    {
        while ((ent = readdir(dir)) != NULL)
        {
            if (ent->d_type == DT_REG)
            { // Check if it's a regular file
                char inputPath[512];
                snprintf(inputPath, sizeof(inputPath), "%s/%s", input_folder, ent->d_name);

                uint8_t *inputImage = stbi_load(inputPath, &width, &height, &depth, 0);
                if (!inputImage)
                {
                    printf("Failed to load image: %s\n", inputPath);
                    continue; // Skip to the next image
                }

                // Store the filename and image data into the vector
                images_ds.push_back({ent->d_name, inputImage});
            }
        }
        closedir(dir);
    }
    else
    {
        perror("Failed to open directory");
        exit(EXIT_FAILURE);
    }

    // Sort the vector based on filenames
    std::sort(images_ds.begin(), images_ds.end(), compare_filenames);
    for (int i = 0; i < images_ds.size(); ++i)
    {
        images.push_back(images_ds[i].data);
    }
}

float *createMask(float **maskVal, int maskWidth)
{

    // Allocate memory for the mask
    float *mask = new float[maskWidth * maskWidth];

    // Fill the mask with the Sobel filter values
    for (int i = 0; i < maskWidth; ++i)
    {
        for (int j = 0; j < maskWidth; ++j)
        {
            mask[i * maskWidth + j] = maskVal[i][j];
        }
    }
    return mask;
}

// Function to normalize float image to uint8_t
void normalizeFloatImage(float *img, uint8_t *output_img, int width, int height)
{
    float min_pixel = FLT_MAX;
    float max_pixel = -FLT_MAX;

    // Find min and max pixel values
    for (int i = 0; i < width * height; ++i)
    {
        if (img[i] < min_pixel)
        {
            min_pixel = img[i];
        }
        if (img[i] > max_pixel)
        {
            max_pixel = img[i];
        }
    }

    // Normalize and convert pixel values to uint8_t
    for (int i = 0; i < width * height; ++i)
    {
        float normalized_pixel = ((img[i] - min_pixel) / (max_pixel - min_pixel)) * 255.0f;
        output_img[i] = (uint8_t)normalized_pixel;
    }
}
