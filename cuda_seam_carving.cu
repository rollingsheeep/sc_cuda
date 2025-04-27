#include <stdio.h>
#include <stdint.h>
#include <string>
#include <cmath>
#include <algorithm>
#include "common.h"
#include <chrono>

using namespace std;

// CUDA error checking macro
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d in file %s\n", cudaGetErrorString(err), __LINE__, __FILE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Forward declarations
void readPnm(char * fileName, int &width, int &height, uchar3 * &pixels);
void writePnm(uchar3 *pixels, int width, int height, int originalWidth, char *fileName);
int backwardEnergy(uint8_t * grayPixels, int row, int col, int width, int height, int originalWidth);
void seamCarvingByCuda(uchar3 *inPixels, int width, int height, int targetWidth, uchar3* outPixels);

// Sobel filter kernels
__constant__ int d_xSobel[3][3] = {{1,0,-1},{2,0,-2},{1,0,-1}};
__constant__ int d_ySobel[3][3] = {{1,2,1},{0,0,0},{-1,-2,-1}};

// Safely retrieve a pixel value from the image
__device__ uint8_t getClosestDevice(uint8_t *pixels, int r, int c, int width, int height, int originalWidth)
{
    if (r < 0) {
        r = 0;
    } else if (r >= height) {
        r = height - 1;
    }

    if (c < 0) {
        c = 0;
    } else if (c >= width) {
        c = width - 1;
    }

    return pixels[r * originalWidth + c];
}

// Calculate backward energy using Sobel operators
__device__ int backwardEnergyDevice(uint8_t * grayPixels, int row, int col, int width, int height, int originalWidth) {
    int x = 0, y = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            uint8_t closest = getClosestDevice(grayPixels, row - 1 + i, col - 1 + j, width, height, originalWidth);
            x += closest * d_xSobel[i][j];
            y += closest * d_ySobel[i][j];
        }
    }
    return abs(x) + abs(y);
}

// Convert RGB to grayscale kernel
__global__ void RGB2GrayKernel(uchar3 *inPixels, int width, int height, uint8_t *grayPixels) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (r < height && c < width) {
        int i = r * width + c;
        grayPixels[i] = 0.299f * inPixels[i].x + 0.587f * inPixels[i].y + 0.114f * inPixels[i].z;
    }
}

// Calculate backward energy kernel
__global__ void backwardEnergyKernel(uint8_t *grayPixels, int width, int height, int originalWidth, int *importants) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (r < height && c < width) {
        int idx = r * originalWidth + c;
        importants[idx] = backwardEnergyDevice(grayPixels, r, c, width, height, originalWidth);
    }
}

// Calculate forward energy kernel
__global__ void forwardEnergyKernel(uint8_t *grayPixels, int width, int height, float *energy, int originalWidth) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    
    // First row has zero energy
    if (blockIdx.y == 0 && c < width) {
        energy[c] = 0.0f;
        return;
    }
    
    int r = blockIdx.y;
    if (r < height && c < width) {
        int idx = r * originalWidth + c;
        
        // Get neighboring pixel values safely with bounds
        float left = (c > 0) ? static_cast<float>(grayPixels[idx - 1]) : static_cast<float>(grayPixels[idx]);
        float right = (c < width - 1) ? static_cast<float>(grayPixels[idx + 1]) : static_cast<float>(grayPixels[idx]);
        float up = static_cast<float>(grayPixels[idx - originalWidth]);
        
        // Compute directional costs using floating-point
        float cU = fabsf(right - left);  // Cost for going straight up
        float cL = cU + fabsf(up - left);  // Cost for going up-left
        float cR = cU + fabsf(up - right);  // Cost for going up-right
        
        // Get minimum previous path cost
        float min_energy = energy[idx - originalWidth] + cU;
        if (c > 0) {
            min_energy = fminf(min_energy, energy[idx - originalWidth - 1] + cL);
        }
        if (c < width - 1) {
            min_energy = fminf(min_energy, energy[idx - originalWidth + 1] + cR);
        }
        
        energy[idx] = min_energy;
    }
}

// Combine backward and forward energy
__global__ void hybridEnergyKernel(int *backwardEnergy, float *forwardEnergy, float *hybridEnergy, int width, int height, int originalWidth) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (r < height && c < width) {
        int idx = r * originalWidth + c;
        
        // Normalize both energy values to 0-1 range
        float backwardNorm = static_cast<float>(backwardEnergy[idx]) / 255.0f;
        float forwardNorm = forwardEnergy[idx] / 255.0f;
        
        // Choose the higher energy value
        float hybridVal = fmaxf(backwardNorm, forwardNorm);
        
        // Store the normalized hybrid energy
        hybridEnergy[idx] = hybridVal;
    }
}

// Convert hybrid energy to importance values
__global__ void hybridToImportanceKernel(float *hybridEnergy, int *importants, int width, int height, int originalWidth) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (r < height && c < width) {
        int idx = r * originalWidth + c;
        importants[idx] = static_cast<int>(hybridEnergy[idx] * 255.0f);
    }
}

// Dynamic programming kernel for each row
__global__ void seamsScoreKernel(int *importants, int *score, int width, int height, int originalWidth, int currentRow) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (currentRow == 0) {
        // Initialize the first row
        if (c < width) {
            score[c] = importants[c];
        }
    } else if (c < width) {
        int idx = currentRow * originalWidth + c;
        int aboveIdx = (currentRow - 1) * originalWidth + c;
        
        int min = score[aboveIdx];
        if (c > 0 && score[aboveIdx - 1] < min) {
            min = score[aboveIdx - 1];
        }
        if (c < width - 1 && score[aboveIdx + 1] < min) {
            min = score[aboveIdx + 1];
        }
        
        score[idx] = min + importants[idx];
    }
}

// Find minimal seam starting position
__global__ void findMinSeamStartKernel(int *score, int width, int height, int originalWidth, int *minCol) {
    extern __shared__ int sdata[];
    
    int tid = threadIdx.x;
    int idx = (height - 1) * originalWidth + tid;
    
    // Load score into shared memory
    if (tid < width) {
        sdata[tid] = score[idx];
        sdata[tid + width] = tid; // Store column index
    } else {
        sdata[tid] = INT_MAX;
        sdata[tid + width] = -1;
    }
    __syncthreads();
    
    // Reduction to find minimum score and its column
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid] > sdata[tid + s]) {
                sdata[tid] = sdata[tid + s];
                sdata[tid + width] = sdata[tid + s + width];
            }
        }
        __syncthreads();
    }
    
    // Write result to global memory
    if (tid == 0) {
        *minCol = sdata[width]; // Store the column with minimum score
    }
}

// Trace seam and remove pixels
__global__ void traceSeamKernel(int *score, int *seam, int width, int height, int originalWidth) {
    // This is a sequential operation that will be executed by a single thread
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int r = height - 1;
        
        // Find minimum score in the last row
        int minCol = 0;
        for (int c = 1; c < width; ++c) {
            if (score[r * originalWidth + c] < score[r * originalWidth + minCol]) {
                minCol = c;
            }
        }
        
        // Save the seam path
        seam[r] = minCol;
        
        // Backtrack to find the seam
        for (r = height - 1; r > 0; --r) {
            int aboveIdx = (r - 1) * originalWidth + minCol;
            int min = score[aboveIdx], minColCpy = minCol;
            
            if (minColCpy > 0 && score[aboveIdx - 1] < min) {
                min = score[aboveIdx - 1];
                minCol = minColCpy - 1;
            }
            
            if (minColCpy < width - 1 && score[aboveIdx + 1] < min) {
                minCol = minColCpy + 1;
            }
            
            seam[r - 1] = minCol;
        }
    }
}

// Remove seam from the image
__global__ void removeSeamKernel(uchar3 *pixels, uint8_t *grayPixels, int *seam, int width, int height, int originalWidth) {
    int r = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (r < height && c < width - 1) {
        int seamCol = seam[r];
        
        if (c >= seamCol) {
            // Shift pixels left to remove the seam
            pixels[r * originalWidth + c] = pixels[r * originalWidth + c + 1];
            grayPixels[r * originalWidth + c] = grayPixels[r * originalWidth + c + 1];
        }
    }
}

// Update local importance after seam removal
__global__ void updateImportanceKernel(uint8_t *grayPixels, int *importants, int *seam, int width, int height, int originalWidth) {
    int r = blockIdx.y;
    
    if (r < height) {
        int seamCol = seam[r];
        
        // Update importance for pixels near the seam
        for (int c = max(0, seamCol - 2); c <= min(seamCol + 2, width - 2); ++c) {
            int idx = r * originalWidth + c;
            importants[idx] = backwardEnergyDevice(grayPixels, r, c, width - 1, height, originalWidth);
        }
    }
}

/**
 * Main function implementing CUDA seam carving
 * Removes vertical seams from the image until target width is reached
 * @param inPixels - Input RGB image
 * @param width - Initial image width
 * @param height - Image height
 * @param targetWidth - Desired final width
 * @param outPixels - Output image after seam removal
 */
void seamCarvingByCuda(uchar3 *inPixels, int width, int height, int targetWidth, uchar3* outPixels) {
    // Start total timer
    auto totalStart = std::chrono::high_resolution_clock::now();
    
    // Initialize timing variables
    double totalGrayscaleTime = 0.0;
    double totalBackwardEnergyTime = 0.0;
    double totalForwardEnergyTime = 0.0;
    double totalHybridEnergyTime = 0.0;
    double totalDpTime = 0.0;
    double totalSeamTracingTime = 0.0;

    // Copy input to output initially
    CHECK_CUDA_ERROR(cudaMemcpy(outPixels, inPixels, width * height * sizeof(uchar3), cudaMemcpyHostToHost));
    const int originalWidth = width;

    // Allocate device memory
    uchar3 *d_pixels;
    uint8_t *d_grayPixels;
    int *d_importants, *d_score, *d_seam, *d_minCol;
    float *d_forwardEnergy, *d_hybridEnergy;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_pixels, width * height * sizeof(uchar3)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_grayPixels, width * height * sizeof(uint8_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_importants, width * height * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_score, width * height * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_forwardEnergy, width * height * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_hybridEnergy, width * height * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_seam, height * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_minCol, sizeof(int)));
    
    // Copy input image to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_pixels, inPixels, width * height * sizeof(uchar3), cudaMemcpyHostToDevice));
    
    // Define thread block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    
    // Convert to grayscale
    auto grayscaleStart = std::chrono::high_resolution_clock::now();
    RGB2GrayKernel<<<gridDim, blockDim>>>(d_pixels, width, height, d_grayPixels);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    auto grayscaleEnd = std::chrono::high_resolution_clock::now();
    totalGrayscaleTime = std::chrono::duration_cast<std::chrono::microseconds>(grayscaleEnd - grayscaleStart).count() / 1000.0;

    while (width > targetWidth) {
        // Calculate backward energy
        auto backwardStart = std::chrono::high_resolution_clock::now();
        backwardEnergyKernel<<<gridDim, blockDim>>>(d_grayPixels, width, height, originalWidth, d_importants);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        auto backwardEnd = std::chrono::high_resolution_clock::now();
        totalBackwardEnergyTime += std::chrono::duration_cast<std::chrono::microseconds>(backwardEnd - backwardStart).count() / 1000.0;

        // Calculate forward energy
        auto forwardStart = std::chrono::high_resolution_clock::now();
        dim3 forwardGridDim(width, height);
        forwardEnergyKernel<<<forwardGridDim, 1>>>(d_grayPixels, width, height, d_forwardEnergy, originalWidth);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        auto forwardEnd = std::chrono::high_resolution_clock::now();
        totalForwardEnergyTime += std::chrono::duration_cast<std::chrono::microseconds>(forwardEnd - forwardStart).count() / 1000.0;

        // Calculate hybrid energy
        auto hybridStart = std::chrono::high_resolution_clock::now();
        hybridEnergyKernel<<<gridDim, blockDim>>>(d_importants, d_forwardEnergy, d_hybridEnergy, width, height, originalWidth);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Convert hybrid energy to importance values
        hybridToImportanceKernel<<<gridDim, blockDim>>>(d_hybridEnergy, d_importants, width, height, originalWidth);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        auto hybridEnd = std::chrono::high_resolution_clock::now();
        totalHybridEnergyTime += std::chrono::duration_cast<std::chrono::microseconds>(hybridEnd - hybridStart).count() / 1000.0;

        // Seam score calculation using dynamic programming
        auto dpStart = std::chrono::high_resolution_clock::now();
        dim3 dpBlockDim(256);
        dim3 dpGridDim((width + dpBlockDim.x - 1) / dpBlockDim.x);
        
        for (int r = 0; r < height; ++r) {
            seamsScoreKernel<<<dpGridDim, dpBlockDim>>>(d_importants, d_score, width, height, originalWidth, r);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        }
        auto dpEnd = std::chrono::high_resolution_clock::now();
        totalDpTime += std::chrono::duration_cast<std::chrono::microseconds>(dpEnd - dpStart).count() / 1000.0;

        // Find min seam start position and trace seam
        auto seamTracingStart = std::chrono::high_resolution_clock::now();
        int blockSize = 256;
        findMinSeamStartKernel<<<1, blockSize, 2 * blockSize * sizeof(int)>>>(d_score, width, height, originalWidth, d_minCol);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Trace seam
        traceSeamKernel<<<1, 1>>>(d_score, d_seam, width, height, originalWidth);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Remove seam from both RGB and grayscale images
        dim3 removeBlockDim(256);
        dim3 removeGridDim((width + removeBlockDim.x - 1) / removeBlockDim.x, height);
        removeSeamKernel<<<removeGridDim, removeBlockDim>>>(d_pixels, d_grayPixels, d_seam, width, height, originalWidth);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Update importance values for pixels around the seam
        updateImportanceKernel<<<height, 1>>>(d_grayPixels, d_importants, d_seam, width, height, originalWidth);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        auto seamTracingEnd = std::chrono::high_resolution_clock::now();
        totalSeamTracingTime += std::chrono::duration_cast<std::chrono::microseconds>(seamTracingEnd - seamTracingStart).count() / 1000.0;

        --width;
    }
    
    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(outPixels, d_pixels, originalWidth * height * sizeof(uchar3), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_pixels));
    CHECK_CUDA_ERROR(cudaFree(d_grayPixels));
    CHECK_CUDA_ERROR(cudaFree(d_importants));
    CHECK_CUDA_ERROR(cudaFree(d_score));
    CHECK_CUDA_ERROR(cudaFree(d_forwardEnergy));
    CHECK_CUDA_ERROR(cudaFree(d_hybridEnergy));
    CHECK_CUDA_ERROR(cudaFree(d_seam));
    CHECK_CUDA_ERROR(cudaFree(d_minCol));

    // Calculate total time
    auto totalEnd = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration_cast<std::chrono::microseconds>(totalEnd - totalStart).count() / 1000.0;

    // Print performance metrics
    printf("\nCUDA Seam Carving Performance Analysis:\n");
    printf("---------------------------------\n");
    printf("Grayscale conversion: %.2f ms\n", totalGrayscaleTime);
    printf("Backward energy (Sobel): %.2f ms\n", totalBackwardEnergyTime);
    printf("Forward energy: %.2f ms\n", totalForwardEnergyTime);
    printf("Hybrid energy: %.2f ms\n", totalHybridEnergyTime);
    printf("Dynamic programming: %.2f ms\n", totalDpTime);
    printf("Seam tracing and removal: %.2f ms\n", totalSeamTracingTime);
    printf("---------------------------------\n");
    printf("Total seam carving time: %.2f ms\n\n", totalTime);
}

/**
 * Main program entry point
 * Handles command line arguments, image processing, and output
 * @param argc - Number of command line arguments
 * @param argv - Command line arguments array
 * @return int - Program exit status
 */
int main(int argc, char ** argv)
{   
    if (argc != 4)
    {
        printf("The number of arguments is invalid\n");
        return EXIT_FAILURE;
    }

    // Read input RGB image file
    int width, height;
    uchar3 *inPixels;
    readPnm(argv[1], width, height, inPixels);
    printf("Image size (width x height): %i x %i\n\n", width, height);

    int numSeamRemoved = stoi(argv[3]);
    if (numSeamRemoved <= 0 || numSeamRemoved >= width)
        return EXIT_FAILURE; // invalid ratio
    printf("Number of seam removed: %d\n\n", numSeamRemoved);

    int targetWidth = width - numSeamRemoved;

    // seam carving using CUDA
    uchar3 * outPixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
    seamCarvingByCuda(inPixels, width, height, targetWidth, outPixels);
    printf("Image size after seam carving (new_width x height): %i x %i\n\n", targetWidth, height);
    
    // Write results to files
    char *outFileNameBase = strtok(argv[2], "."); // Get rid of extension
    writePnm(outPixels, targetWidth, height, width, concatStr(outFileNameBase, "_cuda.pnm"));

    // Free memories
    free(inPixels);
    free(outPixels);
} 