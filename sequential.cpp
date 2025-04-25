#include <stdio.h>
#include <stdint.h>
#include <string>
#include <cmath>
#include <algorithm>
#include "common.h"
#include <chrono>

using namespace std;

// Forward declarations
void readPnm(char * fileName, int &width, int &height, uchar3 * &pixels);
void writePnm(uchar3 *pixels, int width, int height, int originalWidth, char *fileName);
uint8_t getClosest(uint8_t *pixels, int r, int c, int width, int height, int originalWidth);
int pixelsImportant(uint8_t * grayPixels, int row, int col, int width, int height, int originalWidth);
void RGB2Gray(uchar3 * inPixels, int width, int height, uint8_t * outPixels);
void seamsScore(int *importants, int *score, int width, int height, int originalWidth);
void forwardEnergy(uint8_t *grayPixels, int width, int height, float *energy, int originalWidth);
void hybridEnergy(int *backwardEnergy, float *forwardEnergy, float *hybridEnergy, int width, int height, int originalWidth);
void seamCarvingByHost(uchar3 *inPixels, int width, int height, int targetWidth, uchar3* outPixels);

// Sobel filter kernels
int xSobel[3][3] = {{1,0,-1},{2,0,-2},{1,0,-1}};
int ySobel[3][3] = {{1,2,1},{0,0,0},{-1,-2,-1}};

/**
 * Safely retrieves a pixel value from the image, handling boundary conditions
 * @param pixels - Pointer to the image pixel array
 * @param r - Row index (can be out of bounds)
 * @param c - Column index (can be out of bounds)
 * @param width - Current image width
 * @param height - Image height
 * @param originalWidth - Original image width (for proper memory access)
 * @return uint8_t - The pixel value at (r,c) or the closest valid pixel if out of bounds
 */
uint8_t getClosest(uint8_t *pixels, int r, int c, int width, int height, int originalWidth)
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

/**
 * Calculates the importance of a pixel using Sobel edge detection
 * Applies 3x3 Sobel filters to detect edges in both x and y directions
 * @param grayPixels - Grayscale image data
 * @param row - Row index of the pixel
 * @param col - Column index of the pixel
 * @param width - Current image width
 * @param height - Image height
 * @param originalWidth - Original image width
 * @return int - Importance value (higher means more important/edge-like)
 */
int pixelsImportant(uint8_t * grayPixels, int row, int col, int width, int height, int originalWidth) {
    int x = 0, y = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            uint8_t closest = getClosest(grayPixels, row - 1 + i, col - 1 + j, width, height, originalWidth);
            x += closest * xSobel[i][j];
            y += closest * ySobel[i][j];
        }
    }
    return abs(x) + abs(y);
}

/**
 * Converts an RGB image to grayscale using the standard luminance formula
 * Uses the formula: 0.299R + 0.587G + 0.114B
 * @param inPixels - Input RGB image
 * @param width - Image width
 * @param height - Image height
 * @param outPixels - Output grayscale image
 */
void RGB2Gray(uchar3 * inPixels, int width, int height, uint8_t * outPixels) {
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            int i = r * width + c;
            outPixels[i] = 0.299f * inPixels[i].x + 0.587f * inPixels[i].y + 0.114f * inPixels[i].z;
        }
    }
}

/**
 * Performs dynamic programming to find the optimal seam path
 * Calculates cumulative energy scores for each pixel
 * @param importants - Pixel importance values
 * @param score - Output array for cumulative scores
 * @param width - Current image width
 * @param height - Image height
 * @param originalWidth - Original image width
 */
void seamsScore(int *importants, int *score, int width, int height, int originalWidth) {
    for (int c = 0; c < width; ++c) {
        score[c] = importants[c];
    }
    for (int r = 1; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            int idx = r * originalWidth + c;
            int aboveIdx = (r - 1) * originalWidth + c;

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
}

/**
 * Calculates forward energy for each pixel
 * Measures the cost of removing a pixel based on its neighbors
 * @param grayPixels - Grayscale image data
 * @param width - Current image width
 * @param height - Image height
 * @param energy - Output array for energy values
 * @param originalWidth - Original image width
 */
void forwardEnergy(uint8_t *grayPixels, int width, int height, float *energy, int originalWidth) {
    // First row has zero energy
    for (int c = 0; c < width; ++c) {
        energy[c] = 0.0f;
    }

    // Calculate energy for remaining rows
    for (int r = 1; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            int idx = r * originalWidth + c;
            
            // Get neighboring pixel values safely with bounds
            float left = (c > 0) ? static_cast<float>(grayPixels[idx - 1]) : static_cast<float>(grayPixels[idx]);
            float right = (c < width - 1) ? static_cast<float>(grayPixels[idx + 1]) : static_cast<float>(grayPixels[idx]);
            float up = static_cast<float>(grayPixels[idx - originalWidth]);
            
            // Compute directional costs using floating-point
            float cU = fabs(right - left);  // Cost for going straight up
            float cL = cU + fabs(up - left);  // Cost for going up-left
            float cR = cU + fabs(up - right);  // Cost for going up-right
            
            // Get minimum previous path cost
            float min_energy = energy[idx - originalWidth] + cU;
            if (c > 0) {
                min_energy = fmin(min_energy, energy[idx - originalWidth - 1] + cL);
            }
            if (c < width - 1) {
                min_energy = fmin(min_energy, energy[idx - originalWidth + 1] + cR);
            }
            
            energy[idx] = min_energy;
        }
    }
}

/**
 * Combines backward and forward energy measures
 * Normalizes both energies and takes the maximum
 * @param backwardEnergy - Sobel-based energy values
 * @param forwardEnergy - Forward energy values
 * @param hybridEnergy - Output array for combined energy
 * @param width - Current image width
 * @param height - Image height
 * @param originalWidth - Original image width
 */
void hybridEnergy(int *backwardEnergy, float *forwardEnergy, float *hybridEnergy, int width, int height, int originalWidth) {
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            int idx = r * originalWidth + c;
            
            // Normalize both energy values to 0-1 range
            float backwardNorm = static_cast<float>(backwardEnergy[idx]) / 255.0f;
            float forwardNorm = forwardEnergy[idx] / 255.0f;
            
            // Choose the higher energy value
            float hybridVal = fmax(backwardNorm, forwardNorm);
            
            // Store the normalized hybrid energy
            hybridEnergy[idx] = hybridVal;
        }
    }
}

/**
 * Main function implementing sequential seam carving
 * Removes vertical seams from the image until target width is reached
 * @param inPixels - Input RGB image
 * @param width - Initial image width
 * @param height - Image height
 * @param targetWidth - Desired final width
 * @param outPixels - Output image after seam removal
 */
void seamCarvingByHost(uchar3 *inPixels, int width, int height, int targetWidth, uchar3* outPixels) {
    // Start total timer
    auto totalStart = std::chrono::high_resolution_clock::now();
    
    // Initialize timing variables
    double totalGrayscaleTime = 0.0;
    double totalBackwardEnergyTime = 0.0;
    double totalForwardEnergyTime = 0.0;
    double totalHybridEnergyTime = 0.0;
    double totalDpTime = 0.0;
    double totalSeamTracingTime = 0.0;
    double totalLocalUpdateTime = 0.0;

    memcpy(outPixels, inPixels, width * height * sizeof(uchar3));
    const int originalWidth = width;

    // allocate memory
    int *importants = (int *)malloc(width * height * sizeof(int));
    int *score = (int *)malloc(width * height * sizeof(int));
    uint8_t *grayPixels = (uint8_t *)malloc(width * height * sizeof(uint8_t));
    float *forwardEnergyArray = (float *)malloc(width * height * sizeof(float));
    float *hybridEnergyArray = (float *)malloc(width * height * sizeof(float));
    
    // Convert to grayscale image
    auto grayscaleStart = std::chrono::high_resolution_clock::now();
    RGB2Gray(inPixels, width, height, grayPixels);
    auto grayscaleEnd = std::chrono::high_resolution_clock::now();
    totalGrayscaleTime = std::chrono::duration_cast<std::chrono::microseconds>(grayscaleEnd - grayscaleStart).count() / 1000.0;

    while (width > targetWidth) {
        // Calculate backward energy (Sobel filter importance)
        auto backwardStart = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < height; ++r) {
            for (int c = 0; c < width; ++c) {
                importants[r * originalWidth + c] = pixelsImportant(grayPixels, r, c, width, height, originalWidth);
            }
        }
        auto backwardEnd = std::chrono::high_resolution_clock::now();
        totalBackwardEnergyTime += std::chrono::duration_cast<std::chrono::microseconds>(backwardEnd - backwardStart).count() / 1000.0;

        // Calculate forward energy
        auto forwardStart = std::chrono::high_resolution_clock::now();
        forwardEnergy(grayPixels, width, height, forwardEnergyArray, originalWidth);
        auto forwardEnd = std::chrono::high_resolution_clock::now();
        totalForwardEnergyTime += std::chrono::duration_cast<std::chrono::microseconds>(forwardEnd - forwardStart).count() / 1000.0;

        // Calculate hybrid energy
        auto hybridStart = std::chrono::high_resolution_clock::now();
        hybridEnergy(importants, forwardEnergyArray, hybridEnergyArray, width, height, originalWidth);
        auto hybridEnd = std::chrono::high_resolution_clock::now();
        totalHybridEnergyTime += std::chrono::duration_cast<std::chrono::microseconds>(hybridEnd - hybridStart).count() / 1000.0;

        // Convert hybrid energy to integer for seam finding
        for (int r = 0; r < height; ++r) {
            for (int c = 0; c < width; ++c) {
                int idx = r * originalWidth + c;
                importants[idx] = static_cast<int>(hybridEnergyArray[idx] * 255.0f);
            }
        }

        // Dynamic programming to find minimal seam
        auto dpStart = std::chrono::high_resolution_clock::now();
        seamsScore(importants, score, width, height, originalWidth);
        auto dpEnd = std::chrono::high_resolution_clock::now();
        totalDpTime += std::chrono::duration_cast<std::chrono::microseconds>(dpEnd - dpStart).count() / 1000.0;

        // Find where seam starts
        int minCol = 0, r = height - 1, prevMinCol;
        for (int c = 1; c < width; ++c) {
            if (score[r * originalWidth + c] < score[r * originalWidth + minCol])
                minCol = c;
        }

        // Seam tracing and removal
        auto seamTracingStart = std::chrono::high_resolution_clock::now();
        for (; r >= 0; --r) {
            for (int i = minCol; i < width - 1; ++i) {
                outPixels[r * originalWidth + i] = outPixels[r * originalWidth + i + 1];
                grayPixels[r * originalWidth + i] = grayPixels[r * originalWidth + i + 1];
                importants[r * originalWidth + i] = importants[r * originalWidth + i + 1];
            }

            if (r < height - 1) {
                // Local update of importance map
                auto localUpdateStart = std::chrono::high_resolution_clock::now();
                for (int affectedCol = max(0, prevMinCol - 2); affectedCol <= prevMinCol + 2 && affectedCol < width - 1; ++affectedCol) {
                    importants[(r + 1) * originalWidth + affectedCol] = pixelsImportant(grayPixels, r + 1, affectedCol, width - 1, height, originalWidth);
                }
                auto localUpdateEnd = std::chrono::high_resolution_clock::now();
                totalLocalUpdateTime += std::chrono::duration_cast<std::chrono::microseconds>(localUpdateEnd - localUpdateStart).count() / 1000.0;
            }

            if (r > 0) {
                prevMinCol = minCol;
                int aboveIdx = (r - 1) * originalWidth + minCol;
                int min = score[aboveIdx], minColCpy = minCol;
                if (minColCpy > 0 && score[aboveIdx - 1] < min) {
                    min = score[aboveIdx - 1];
                    minCol = minColCpy - 1;
                }
                if (minColCpy < width - 1 && score[aboveIdx + 1] < min) {
                    minCol = minColCpy + 1;
                }
            }
        }
        auto seamTracingEnd = std::chrono::high_resolution_clock::now();
        totalSeamTracingTime += std::chrono::duration_cast<std::chrono::microseconds>(seamTracingEnd - seamTracingStart).count() / 1000.0;

        // Update importance map for first row
        auto firstRowUpdateStart = std::chrono::high_resolution_clock::now();
        for (int affectedCol = max(0, minCol - 2); affectedCol <= minCol + 2 && affectedCol < width - 1; ++affectedCol) {
            importants[affectedCol] = pixelsImportant(grayPixels, 0, affectedCol, width - 1, height, originalWidth);
        }
        auto firstRowUpdateEnd = std::chrono::high_resolution_clock::now();
        totalLocalUpdateTime += std::chrono::duration_cast<std::chrono::microseconds>(firstRowUpdateEnd - firstRowUpdateStart).count() / 1000.0;

        --width;
    }
    
    // Free memory
    free(grayPixels);
    free(score);
    free(importants);
    free(forwardEnergyArray);
    free(hybridEnergyArray);

    // Calculate total time
    auto totalEnd = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration_cast<std::chrono::microseconds>(totalEnd - totalStart).count() / 1000.0;

    // Print performance metrics
    printf("\nSeam Carving Performance Analysis:\n");
    printf("---------------------------------\n");
    printf("Grayscale conversion: %.2f ms\n", totalGrayscaleTime);
    printf("Backward energy (Sobel): %.2f ms\n", totalBackwardEnergyTime);
    printf("Forward energy: %.2f ms\n", totalForwardEnergyTime);
    printf("Hybrid energy: %.2f ms\n", totalHybridEnergyTime);
    printf("Dynamic programming: %.2f ms\n", totalDpTime);
    printf("Seam tracing and removal: %.2f ms\n", totalSeamTracingTime);
    printf("Local importance map updates: %.2f ms\n", totalLocalUpdateTime);
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

    // seam carving using host
    uchar3 * outPixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
    seamCarvingByHost(inPixels, width, height, targetWidth, outPixels);
    printf("Image size after seam carving (new_width x height): %i x %i\n\n", targetWidth, height);
    
    // Write results to files
    char *outFileNameBase = strtok(argv[2], "."); // Get rid of extension
    writePnm(outPixels, targetWidth, height, width, concatStr(outFileNameBase, "_seq.pnm"));

    // Free memories
    free(inPixels);
    free(outPixels);
} 