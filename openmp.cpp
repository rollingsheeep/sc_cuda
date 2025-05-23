#include <stdio.h>
#include <stdint.h>
#include <string>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include "common.h"
#include <chrono>
#include <fstream>  // For file output
#include <vector>   // For potential buffer usage (not strictly needed here)
#include <iomanip>  // For formatting float output

using namespace std;

// Helper function to write array data to a file
template<typename T>
void writeArrayToFile(const std::string& filename, const T* data, int width, int height, int allocatedWidth) {
    std::ofstream outFile(filename);
    if (!outFile) {
        fprintf(stderr, "Error: Cannot open file %s for writing.\n", filename.c_str());
        return;
    }
    outFile << std::fixed << std::setprecision(10); // Set precision for floats
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            outFile << data[r * allocatedWidth + c] << (c == width - 1 ? "" : " ");
        }
        outFile << "\n";
    }
    printf("Debug data written to %s\n", filename.c_str());
}

// Forward declarations
void readPnm(char * fileName, int &width, int &height, uchar3 * &pixels);
void writePnm(uchar3 *pixels, int width, int height, int originalWidth, char *fileName);
uint8_t getClosest(uint8_t *pixels, int r, int c, int width, int height, int originalWidth);
int backwardEnergy(uint8_t * grayPixels, int row, int col, int width, int height, int originalWidth);
void RGB2Gray(uchar3 * inPixels, int width, int height, uint8_t * outPixels);
void seamsScore(int *importants, int *score, int width, int height, int originalWidth);
void forwardEnergy(uint8_t *grayPixels, int width, int height, float *energy, int originalWidth);
void hybridEnergy(int *backwardEnergy, float *forwardEnergy, float *hybridEnergy, int width, int height, int originalWidth);
void seamCarvingByOpenMP(uchar3 *inPixels, int width, int height, int targetWidth, uchar3* outPixels);

// Sobel filter kernels
int xSobel[3][3] = {{1,0,-1},{2,0,-2},{1,0,-1}};
int ySobel[3][3] = {{1,2,1},{0,0,0},{-1,-2,-1}};

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

int backwardEnergy(uint8_t * grayPixels, int row, int col, int width, int height, int originalWidth) {
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

void RGB2Gray(uchar3 * inPixels, int width, int height, uint8_t * outPixels) {
    #pragma omp parallel for
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            int i = r * width + c;
            outPixels[i] = 0.299f * inPixels[i].x + 0.587f * inPixels[i].y + 0.114f * inPixels[i].z;
        }
    }
}

void seamsScore(int *importants, int *score, int width, int height, int originalWidth) {
    // First row
    #pragma omp parallel for
    for (int c = 0; c < width; ++c) {
        score[c] = importants[c];
    }

    // Remaining rows - sequential due to dependencies
    for (int r = 1; r < height; ++r) {
        #pragma omp parallel for
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

void forwardEnergy(uint8_t *grayPixels, int width, int height, float *energy, int originalWidth) {
    // First row has zero energy
    #pragma omp parallel for
    for (int c = 0; c < width; ++c) {
        energy[c] = 0.0f;
    }

    // Calculate energy for remaining rows - sequential due to dependencies
    for (int r = 1; r < height; ++r) {
        #pragma omp parallel for
        for (int c = 0; c < width; ++c) {
            int idx = r * originalWidth + c;
            
            float left = (c > 0) ? static_cast<float>(grayPixels[idx - 1]) : static_cast<float>(grayPixels[idx]);
            float right = (c < width - 1) ? static_cast<float>(grayPixels[idx + 1]) : static_cast<float>(grayPixels[idx]);
            float up = static_cast<float>(grayPixels[idx - originalWidth]);
            
            float cU = fabsf(right - left);
            float cL = cU + fabsf(up - left);
            float cR = cU + fabsf(up - right);
            
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
}

void hybridEnergy(int *backwardEnergy, float *forwardEnergy, float *hybridEnergy, int width, int height, int originalWidth) {
    #pragma omp parallel for
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            int idx = r * originalWidth + c;
            float backwardNorm = static_cast<float>(backwardEnergy[idx]) / 255.0f;
            float forwardNorm = forwardEnergy[idx] / 255.0f;
            hybridEnergy[idx] = fmaxf(backwardNorm, forwardNorm);
        }
    }
}

void seamCarvingByOpenMP(uchar3 *inPixels, int width, int height, int targetWidth, uchar3* outPixels) {
    // Log OpenMP thread count
    int num_threads = omp_get_max_threads();
    printf("OpenMP running with %d threads\n", num_threads);

    // Start total timer
    auto totalStart = std::chrono::high_resolution_clock::now();
    
    // Initialize timing variables
    double totalGrayscaleTime = 0.0;
    double totalBackwardEnergyTime = 0.0;
    double totalForwardEnergyTime = 0.0;
    double totalHybridEnergyTime = 0.0;
    double totalDpTime = 0.0;
    double totalSeamTracingTime = 0.0;

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
        // Calculate backward energy (pixel importance) -> stored in importants
        auto backwardStart = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for (int r = 0; r < height; ++r) {
            for (int c = 0; c < width; ++c) {
                importants[r * originalWidth + c] = backwardEnergy(grayPixels, r, c, width, height, originalWidth);
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

        // Dynamic programming uses backward energy directly from 'importants'
        auto dpStart = std::chrono::high_resolution_clock::now();
        seamsScore(importants, score, width, height, originalWidth);
        auto dpEnd = std::chrono::high_resolution_clock::now();
        totalDpTime += std::chrono::duration_cast<std::chrono::microseconds>(dpEnd - dpStart).count() / 1000.0;

        // Find where seam starts
        auto seamTracingStart = std::chrono::high_resolution_clock::now();
        int minCol = 0, r = height - 1, prevMinCol;
        for (int c = 1; c < width; ++c) {
            if (score[r * originalWidth + c] < score[r * originalWidth + minCol])
                minCol = c;
        }

        // Trace and remove seams
        for (; r >= 0; --r) {
            for (int i = minCol; i < width - 1; ++i) {
                outPixels[r * originalWidth + i] = outPixels[r * originalWidth + i + 1];
                grayPixels[r * originalWidth + i] = grayPixels[r * originalWidth + i + 1];
            }

            if (r > 0) {
                prevMinCol = minCol;
                int aboveIdx = (r - 1) * originalWidth + minCol;
                int min_score = score[aboveIdx], minColCpy = minCol;
                if (prevMinCol > 0 && score[aboveIdx - 1] < min_score) {
                    min_score = score[aboveIdx - 1];
                    minCol = prevMinCol - 1;
                }
                if (prevMinCol < width - 1 && score[aboveIdx + 1] < min_score) {
                    minCol = prevMinCol + 1;
                }
            }
        }
        auto seamTracingEnd = std::chrono::high_resolution_clock::now();
        totalSeamTracingTime += std::chrono::duration_cast<std::chrono::microseconds>(seamTracingEnd - seamTracingStart).count() / 1000.0;

        // ---- Intermediate Update ----
        int current_width = width; // Store width before decrementing
        int next_width = width - 1;

        #pragma omp parallel for
        for (int row_idx = 0; row_idx < height; ++row_idx) {
            for (int col_idx = 0; col_idx < next_width; ++col_idx) {
                 importants[row_idx * originalWidth + col_idx] = backwardEnergy(grayPixels, row_idx, col_idx, next_width, height, originalWidth);
            }
        }
        seamsScore(importants, score, next_width, height, originalWidth);
        // ---- End Intermediate Update ----

        // ---- Debug Output after first seam removal ----
        // if (width == originalWidth - 1) {
        //     printf("\n--- Writing OpenMP intermediate arrays (width=%d) ---\n", next_width);
        //     writeArrayToFile("omp_importants_1.txt", importants, next_width, height, originalWidth); 
        //     writeArrayToFile("omp_forward_1.txt", forwardEnergyArray, next_width, height, originalWidth); 
        //     writeArrayToFile("omp_score_1.txt", score, next_width, height, originalWidth); 
        //     printf("--- Finished writing OpenMP intermediate arrays ---\n\n");
        // }
        // ---- End Debug Output ----

        --width;
    } // End while loop
    
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
    printf("\nOpenMP Implementation Performance Analysis:\n");
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

    // seam carving using OpenMP
    uchar3 * outPixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
    seamCarvingByOpenMP(inPixels, width, height, targetWidth, outPixels);
    printf("Image size after seam carving (new_width x height): %i x %i\n\n", targetWidth, height);
    
    // Write results to files
    char *outFileNameBase = strtok(argv[2], "."); // Get rid of extension
    writePnm(outPixels, targetWidth, height, width, concatStr(outFileNameBase, "_omp.pnm"));

    // Free memories
    free(inPixels);
    free(outPixels);
} 