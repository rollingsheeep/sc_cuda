#include <stdio.h>
#include <stdint.h>
#include <string>
#include <cmath>
#include <algorithm>
#include <mpi.h>
#include "common.h"
#include <chrono>

using namespace std;

// Forward declarations
void readPnm(char * fileName, int &width, int &height, uchar3 * &pixels);
void writePnm(uchar3 *pixels, int width, int height, int originalWidth, char *fileName);
uint8_t getClosest(uint8_t *pixels, int r, int c, int width, int height, int originalWidth);
int backwardEnergy(uint8_t * grayPixels, int row, int col, int width, int height, int originalWidth);
void RGB2Gray(uchar3 * inPixels, int width, int height, uint8_t * outPixels);
void seamsScore(int *importants, int *score, int width, int height, int originalWidth);
void forwardEnergy(uint8_t *grayPixels, int width, int height, float *energy, int originalWidth);
void hybridEnergy(int *backwardEnergy, float *forwardEnergy, float *hybridEnergy, int width, int height, int originalWidth);
void forwardEnergyLocal(uint8_t *grayPixels, int width, int height, float *energy, int originalWidth, 
                       int startRow, int localHeight, int rank, int size);
void hybridEnergyLocal(int *backwardEnergy, float *forwardEnergy, float *hybridEnergy, 
                      int width, int height, int originalWidth,
                      int startRow, int localHeight);

// Sobel filter kernels
int xSobel[3][3] = {{1,0,-1},{2,0,-2},{1,0,-1}};
int ySobel[3][3] = {{1,2,1},{0,0,0},{-1,-2,-1}};

// Helper functions from sequential implementation
uint8_t getClosest(uint8_t *pixels, int r, int c, int width, int height, int originalWidth) {
    if (r < 0) r = 0;
    else if (r >= height) r = height - 1;
    if (c < 0) c = 0;
    else if (c >= width) c = width - 1;
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
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            int i = r * width + c;
            outPixels[i] = 0.299f * inPixels[i].x + 0.587f * inPixels[i].y + 0.114f * inPixels[i].z;
        }
    }
}

void forwardEnergy(uint8_t *grayPixels, int width, int height, float *energy, int originalWidth) {
    for (int c = 0; c < width; ++c) {
        energy[c] = 0.0f;
    }

    
    for (int r = 1; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            int idx = r * originalWidth + c;
            float left = (c > 0) ? static_cast<float>(grayPixels[idx - 1]) : static_cast<float>(grayPixels[idx]);
            float right = (c < width - 1) ? static_cast<float>(grayPixels[idx + 1]) : static_cast<float>(grayPixels[idx]);
            float up = static_cast<float>(grayPixels[idx - originalWidth]);
            
            float cU = fabs(right - left);
            float cL = cU + fabs(up - left);
            float cR = cU + fabs(up - right);
            
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

void hybridEnergy(int *backwardEnergy, float *forwardEnergy, float *hybridEnergy, int width, int height, int originalWidth) {
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            int idx = r * originalWidth + c;
            float backwardNorm = static_cast<float>(backwardEnergy[idx]) / 255.0f;
            float forwardNorm = forwardEnergy[idx] / 255.0f;
            float hybridVal = fmax(backwardNorm, forwardNorm);
            hybridEnergy[idx] = hybridVal;
        }
    }
}

void hybridEnergyLocal(int *backwardEnergy, float *forwardEnergy, float *hybridEnergy, 
                      int width, int height, int originalWidth,
                      int startRow, int localHeight) {
    // Only process assigned rows
    for (int r = startRow; r < startRow + localHeight; ++r) {
        for (int c = 0; c < width; ++c) {
            int idx = r * originalWidth + c;
            float backwardNorm = static_cast<float>(backwardEnergy[idx]) / 255.0f;
            float forwardNorm = forwardEnergy[idx] / 255.0f;
            float hybridVal = fmax(backwardNorm, forwardNorm);
            hybridEnergy[idx] = hybridVal;
        }
    }
}

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

void forwardEnergyLocal(uint8_t *grayPixels, int width, int height, float *energy, int originalWidth, 
                       int startRow, int localHeight, int rank, int size) {
    // First row has zero energy (only rank 0)
    if (rank == 0) {
        for (int c = 0; c < width; ++c) {
            energy[c] = 0.0f;
        }
    }

    // Allocate buffers for halo rows
    uint8_t *haloPixels = nullptr;
    float *haloEnergy = nullptr;
    if (rank > 0) {
        haloPixels = (uint8_t *)malloc(originalWidth * sizeof(uint8_t));
        haloEnergy = (float *)malloc(originalWidth * sizeof(float));
    }

    // First send data to next rank (if not last rank)
    if (rank < size - 1) {
        int lastRow = startRow + localHeight - 1;
        // Send last row of pixels
        MPI_Send(grayPixels + lastRow * originalWidth, 
                originalWidth, MPI_BYTE, 
                rank + 1, 0, MPI_COMM_WORLD);
        // Send last row of energy
        MPI_Send(energy + lastRow * originalWidth, 
                originalWidth, MPI_FLOAT, 
                rank + 1, 1, MPI_COMM_WORLD);
    }

    // Then receive data from previous rank (if not first rank)
    if (rank > 0) {
        // Receive halo row pixels
        MPI_Recv(haloPixels, originalWidth, MPI_BYTE, 
                rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // Receive halo row energy
        MPI_Recv(haloEnergy, originalWidth, MPI_FLOAT, 
                rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Now that we have both halo pixels and energies, calculate forward energy
    // Start from first row for rank 0, and from startRow for other ranks
    for (int r = (rank == 0 ? 1 : startRow); r < startRow + localHeight; ++r) {
        for (int c = 0; c < width; ++c) {
            // Get pixel values for current position
            float left = (c > 0) ? 
                static_cast<float>(grayPixels[r * originalWidth + c - 1]) : 
                static_cast<float>(grayPixels[r * originalWidth + c]);
            
            float right = (c < width - 1) ? 
                static_cast<float>(grayPixels[r * originalWidth + c + 1]) : 
                static_cast<float>(grayPixels[r * originalWidth + c]);
            
            // Get up pixel value - either from halo or local data
            float up;
            if (r == startRow && rank > 0) {
                up = static_cast<float>(haloPixels[c]);
            } else {
                up = static_cast<float>(grayPixels[(r - 1) * originalWidth + c]);
            }

            // Calculate costs
            float cU = fabsf(right - left);
            float cL = cU + fabsf(up - left);
            float cR = cU + fabsf(up - right);

            // Calculate minimum energy from previous row
            float min_energy;
            if (r == startRow && rank > 0) {
                // Use halo energy for first row of non-zero ranks
                min_energy = haloEnergy[c] + cU;
                if (c > 0) {
                    min_energy = fminf(min_energy, haloEnergy[c - 1] + cL);
                }
                if (c < width - 1) {
                    min_energy = fminf(min_energy, haloEnergy[c + 1] + cR);
                }
            } else {
                // Use local energy for other rows
                int prevRow = (r - 1) * originalWidth;
                min_energy = energy[prevRow + c] + cU;
                if (c > 0) {
                    min_energy = fminf(min_energy, energy[prevRow + c - 1] + cL);
                }
                if (c < width - 1) {
                    min_energy = fminf(min_energy, energy[prevRow + c + 1] + cR);
                }
            }

            energy[r * originalWidth + c] = min_energy;
        }
    }

    // Cleanup halo buffers
    if (rank > 0) {
        free(haloPixels);
        free(haloEnergy);
    }

    // Synchronize all processes before proceeding
    MPI_Barrier(MPI_COMM_WORLD);
}

void seamCarvingByMPI(uchar3 *inPixels, int width, int height, int targetWidth, uchar3* outPixels) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Timing variables
    double totalGrayscaleTime = 0.0;
    double totalBackwardEnergyTime = 0.0;
    double totalForwardEnergyTime = 0.0;
    double totalHybridEnergyTime = 0.0;
    double totalDpTime = 0.0;
    double totalSeamTracingTime = 0.0;
    auto totalStart = std::chrono::high_resolution_clock::now();

    const int originalWidth = width;
    memcpy(outPixels, inPixels, width * height * sizeof(uchar3));

    // Allocate memory
    int *importants = (int *)malloc(width * height * sizeof(int));
    int *score = (int *)malloc(width * height * sizeof(int));
    uint8_t *grayPixels = (uint8_t *)malloc(width * height * sizeof(uint8_t));
    float *forwardEnergyArray = (float *)malloc(width * height * sizeof(float));
    float *hybridEnergyArray = (float *)malloc(width * height * sizeof(float));
    float *gatheredHybridEnergy = nullptr;  // Only allocated on rank 0
    int *gatheredImportants = nullptr; // Only allocated on rank 0

    // Calculate rows per process and displacements for MPI_Gatherv
    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    int rowsPerProcess = height / size;
    int remainingRows = height % size;
    
    // Calculate send counts and displacements
    int totalDisplacement = 0;
    for (int i = 0; i < size; i++) {
        sendcounts[i] = (rowsPerProcess + (i < remainingRows ? 1 : 0)) * originalWidth;
        displs[i] = totalDisplacement;
        totalDisplacement += sendcounts[i];
    }

    int startRow = rank * rowsPerProcess + (rank < remainingRows ? rank : remainingRows);
    int localHeight = rowsPerProcess + (rank < remainingRows ? 1 : 0);

    if (rank == 0) {
        gatheredHybridEnergy = (float *)malloc(width * height * sizeof(float));
        gatheredImportants = (int *)malloc(width * height * sizeof(int)); // Allocate on rank 0
    }

    // Convert to grayscale
    auto grayscaleStart = std::chrono::high_resolution_clock::now();
    RGB2Gray(inPixels, width, height, grayPixels);
    auto grayscaleEnd = std::chrono::high_resolution_clock::now();
    totalGrayscaleTime = std::chrono::duration_cast<std::chrono::microseconds>(grayscaleEnd - grayscaleStart).count() / 1000.0;

    while (width > targetWidth) {
        // Calculate backward energy
        auto backwardStart = std::chrono::high_resolution_clock::now();
        for (int r = startRow; r < startRow + localHeight; ++r) {
            for (int c = 0; c < width; ++c) {
                importants[r * originalWidth + c] = backwardEnergy(grayPixels, r, c, width, height, originalWidth);
            }
        }
        auto backwardEnd = std::chrono::high_resolution_clock::now();
        totalBackwardEnergyTime += std::chrono::duration_cast<std::chrono::microseconds>(backwardEnd - backwardStart).count() / 1000.0;

        // Gather backward energy (importants) onto rank 0
        MPI_Gatherv(importants + startRow * originalWidth, // Send local chunk
                   localHeight * originalWidth,          // Size of local chunk
                   MPI_INT,                              // Type
                   gatheredImportants,                   // Receive buffer on rank 0
                   sendcounts,                           // Count per rank
                   displs,                               // Displacement per rank
                   MPI_INT,                              // Type
                   0,                                    // Root rank
                   MPI_COMM_WORLD);

        // Calculate forward energy locally
        auto forwardStart = std::chrono::high_resolution_clock::now();
        forwardEnergyLocal(grayPixels, width, height, forwardEnergyArray, originalWidth, startRow, localHeight, rank, size);
        auto forwardEnd = std::chrono::high_resolution_clock::now();
        totalForwardEnergyTime += std::chrono::duration_cast<std::chrono::microseconds>(forwardEnd - forwardStart).count() / 1000.0;

        // Calculate hybrid energy locally
        auto hybridStart = std::chrono::high_resolution_clock::now();
        hybridEnergyLocal(importants, forwardEnergyArray, hybridEnergyArray, width, height, originalWidth, startRow, localHeight);
        auto hybridEnd = std::chrono::high_resolution_clock::now();
        totalHybridEnergyTime += std::chrono::duration_cast<std::chrono::microseconds>(hybridEnd - hybridStart).count() / 1000.0;

        // Gather all hybrid energies at rank 0 using MPI_Gatherv
        MPI_Gatherv(hybridEnergyArray + startRow * originalWidth,
                   localHeight * originalWidth,
                   MPI_FLOAT,
                   gatheredHybridEnergy,
                   sendcounts,
                   displs,
                   MPI_FLOAT,
                   0,
                   MPI_COMM_WORLD);

        if (rank == 0) {
            // Dynamic programming to find minimal seam
            auto dpStart = std::chrono::high_resolution_clock::now();
            // Use the gathered importants array
            seamsScore(gatheredImportants, score, width, height, originalWidth);
            auto dpEnd = std::chrono::high_resolution_clock::now();
            totalDpTime += std::chrono::duration_cast<std::chrono::microseconds>(dpEnd - dpStart).count() / 1000.0;

            // Find and trace seam
            auto seamTracingStart = std::chrono::high_resolution_clock::now();
            int minCol = 0;
            for (int c = 1; c < width; ++c) {
                if (score[(height - 1) * originalWidth + c] < score[(height - 1) * originalWidth + minCol]) {
                    minCol = c;
                }
            }

            // Remove seam
            for (int r = height - 1; r >= 0; --r) {
                for (int c = minCol; c < width - 1; ++c) {
                    outPixels[r * originalWidth + c] = outPixels[r * originalWidth + c + 1];
                    grayPixels[r * originalWidth + c] = grayPixels[r * originalWidth + c + 1];
                }

                if (r > 0) {
                    int prev_minCol = minCol; // Store column from current row r
                    int base_check_idx = (r - 1) * originalWidth + prev_minCol; // Index directly above in row r-1

                    int best_col_prev_row = prev_minCol;          // Start by assuming the middle column is best
                    int min_score_prev_row = score[base_check_idx]; // Get score directly above

                    // Check score above-left
                    if (prev_minCol > 0 && score[base_check_idx - 1] < min_score_prev_row) {
                        min_score_prev_row = score[base_check_idx - 1];
                        best_col_prev_row = prev_minCol - 1;
                    }

                    // Check score above-right (compare with the current minimum found so far)
                    if (prev_minCol < width - 1 && score[base_check_idx + 1] < min_score_prev_row) {
                        // No need to update min_score_prev_row, just the best column
                        best_col_prev_row = prev_minCol + 1;
                    }

                    minCol = best_col_prev_row; // Update minCol for the next iteration (row r-1)
                }
            }

            // Intermediate Update: Recalculate energy/score on rank 0
            // after seam removal, mimicking the OpenMP version's logic
            int next_width = width - 1;
            if (next_width > targetWidth) {
                // Recalculate backward energy (importants) sequentially on rank 0
                for (int r_update = 0; r_update < height; ++r_update) {
                    for (int c_update = 0; c_update < next_width; ++c_update) {
                        gatheredImportants[r_update * originalWidth + c_update] = 
                            backwardEnergy(grayPixels, r_update, c_update, next_width, height, originalWidth);
                    }
                }

                // Recalculate scores sequentially on rank 0 using the updated importants
                seamsScore(gatheredImportants, score, next_width, height, originalWidth);
            }

            auto seamTracingEnd = std::chrono::high_resolution_clock::now();
            totalSeamTracingTime += std::chrono::duration_cast<std::chrono::microseconds>(seamTracingEnd - seamTracingStart).count() / 1000.0;
        }

        // Broadcast updated image and grayscale to all ranks
        MPI_Bcast(outPixels, originalWidth * height * sizeof(uchar3), MPI_BYTE, 0, MPI_COMM_WORLD);
        MPI_Bcast(grayPixels, originalWidth * height * sizeof(uint8_t), MPI_BYTE, 0, MPI_COMM_WORLD);

        --width;
    }

    // Free memory
    free(grayPixels);
    free(score);
    free(importants);
    free(forwardEnergyArray);
    free(hybridEnergyArray);
    free(sendcounts);
    free(displs);
    if (rank == 0) {
        free(gatheredHybridEnergy);
        free(gatheredImportants); // Free the gathered array
    }

    // Print timing information on rank 0
    if (rank == 0) {
        auto totalEnd = std::chrono::high_resolution_clock::now();
        double totalTime = std::chrono::duration_cast<std::chrono::microseconds>(totalEnd - totalStart).count() / 1000.0;

        printf("\nSeam Carving Performance Analysis (MPI):\n");
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
}

int main(int argc, char ** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("MPI initialized with %d processes\n\n", size);
    }

    if (argc != 4) {
        if (rank == 0) {
            printf("The number of arguments is invalid\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Read input RGB image file (only on rank 0)
    int width, height;
    uchar3 *inPixels = nullptr;
    if (rank == 0) {
        readPnm(argv[1], width, height, inPixels);
        printf("Image size (width x height): %i x %i\n\n", width, height);
    }

    // Broadcast image dimensions to all ranks
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for input image on all ranks
    if (rank != 0) {
        inPixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
    }

    // Broadcast the image data to all ranks
    MPI_Bcast(inPixels, width * height * sizeof(uchar3), MPI_BYTE, 0, MPI_COMM_WORLD);

    int numSeamRemoved = stoi(argv[3]);
    if (numSeamRemoved <= 0 || numSeamRemoved >= width) {
        if (rank == 0) {
            printf("Invalid number of seams to remove\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (rank == 0) {
        printf("Number of seams to remove: %d\n\n", numSeamRemoved);
    }

    int targetWidth = width - numSeamRemoved;

    // Allocate memory for output image
    uchar3 *outPixels = (uchar3 *)malloc(width * height * sizeof(uchar3));

    // Perform seam carving
    seamCarvingByMPI(inPixels, width, height, targetWidth, outPixels);

    // Write results (only on rank 0)
    if (rank == 0) {
        printf("Image size after seam carving (new_width x height): %i x %i\n\n", targetWidth, height);
        char *outFileNameBase = strtok(argv[2], ".");
        writePnm(outPixels, targetWidth, height, width, concatStr(outFileNameBase, "_mpi.pnm"));
    }

    // Free memory
    free(inPixels);
    free(outPixels);

    MPI_Finalize();
    return 0;
} 