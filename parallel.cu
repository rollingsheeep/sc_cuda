#include <stdio.h>
#include <stdint.h>
#include <string>
#include <cmath>
#include <algorithm>

using namespace std;

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(EXIT_FAILURE);\
    }\
}

// Forward declarations of CUDA kernels
__global__ void Rgb2GrayKernel(uchar3 * inPixels, int width, int height, uint8_t * outPixels);
__global__ void pixelsImportantKernel(uint8_t * inPixels, int width, int height, int* xfilter, int* yfilter, int * importants);
__global__ void carvingKernel(int *leastSignificantPixel, uchar3 *outPixels, uint8_t *grayPixels, int * importants, int width);
__global__ void seamsScoreKernel(int *importants, int *score, int width, int height, int fromRow);
__global__ void forwardEnergyKernel(uint8_t *grayPixels, int width, int height, float *energy);
__global__ void hybridEnergyKernel(int *backwardEnergy, float *forwardEnergy, float *hybridEnergy, int width, int height);
__global__ void convertToIntKernel(float *floatEnergy, int *intEnergy, int width, int height);

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);                                                                 
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

void readPnm(char * fileName, int &width, int &height, uchar3 * &pixels)
{
    FILE * f = fopen(fileName, "r");
    if (f == NULL)
    {
        printf("Cannot read %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    char type[3];
    fscanf(f, "%s", type);
    
    if (strcmp(type, "P3") != 0) 
    {
        fclose(f);
        printf("Cannot read %s\n", fileName); 
        exit(EXIT_FAILURE); 
    }

    fscanf(f, "%i", &width);
    fscanf(f, "%i", &height);
    
    int max_val;
    fscanf(f, "%i", &max_val);
    if (max_val > 255) 
    {
        fclose(f);
        printf("Cannot read %s\n", fileName); 
        exit(EXIT_FAILURE); 
    }

    pixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
    for (int i = 0; i < width * height; i++)
        fscanf(f, "%hhu%hhu%hhu", &pixels[i].x, &pixels[i].y, &pixels[i].z);

    fclose(f);
}

void writePnm(uchar3 *pixels, int width, int height, int originalWidth, char *fileName)
{
    FILE * f = fopen(fileName, "w");
    if (f == NULL)
    {
        printf("Cannot write %s\n", fileName);
        exit(EXIT_FAILURE);
    }   

    fprintf(f, "P3\n%i\n%i\n255\n", width, height); 

    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            int i = r * originalWidth + c;
            fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z);
        }
    }
    
    fclose(f);
}

int xSobel[3][3] = {{1,0,-1},{2,0,-2},{1,0,-1}};
int ySobel[3][3] = {{1,2,1},{0,0,0},{-1,-2,-1}};

__device__ int d_originalWidth;

__global__ void Rgb2GrayKernel(uchar3 * inPixels, int width, int height, uint8_t * outPixels) {
    size_t r = blockIdx.y * blockDim.y + threadIdx.y;
    size_t c = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = r * width + c;
    if (r < height && c < width) {
        outPixels[i] = 0.299f * inPixels[i].x
                    + 0.587f * inPixels[i].y
                    + 0.114f * inPixels[i].z;
    }
}

__global__ void pixelsImportantKernel(uint8_t * inPixels, int width, int height, int* xfilter, int* yfilter, int * importants) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	if (px >= width || py >= height) 
	{
		return;
	}

    int x = 0, y = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {

            // Handling the case of pixels falling outside the boundary
            int r = (py - 1) + i;
            int c = (px - 1) + j;
            if (r < 0) 
                r = 0;
            else if (r >= height) 
                r = height - 1;
        
            if (c < 0) {
                c = 0;
            } else if (c >= width) {
                c = width - 1;
            }
            int idx = i * 3 + j;
            uint8_t closest = inPixels[r * d_originalWidth + c];
            x += closest * xfilter[idx];
            y += closest * yfilter[idx];
        }
    }
    importants[py * d_originalWidth + px] = abs(x) + abs(y);
}

__global__ void carvingKernel(int *leastSignificantPixel, uchar3 *outPixels, uint8_t *grayPixels, int * importants, int width) {
    int row = blockIdx.x;
    int baseIdx = row * d_originalWidth;
    for (int i = leastSignificantPixel[row]; i < width - 1; ++i) {
        outPixels[baseIdx + i] = outPixels[baseIdx + i + 1];
        grayPixels[baseIdx + i] = grayPixels[baseIdx + i + 1];
        importants[baseIdx + i] = importants[baseIdx + i + 1];
    }
}

void trace(int *score, int *leastSignificantPixel, int width, int height, int originalWidth) {
    int minCol = 0, r = height - 1;
    for (int c = 1; c < width; ++c) {
        if (score[r * originalWidth + c] < score[r * originalWidth + minCol])
            minCol = c;
    }
    for (; r >= 0; --r) {
        leastSignificantPixel[r] = minCol;
        if (r > 0) {
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
}

__global__ void seamsScoreKernel(int *importants, int *score, int width, int height, int fromRow) {
    size_t halfBlock = blockDim.x >> 1;

    int col = blockIdx.x * halfBlock - halfBlock + threadIdx.x;

    if (fromRow == 0 && col >= 0 && col < width) {
        score[col] = importants[col];
    }
    __syncthreads();

    for (int stride = fromRow != 0 ? 0 : 1; stride < halfBlock && fromRow + stride < height; ++stride) {
        if (threadIdx.x < blockDim.x - (stride << 1)) {
            int curRow = fromRow + stride;
            int curCol = col + stride;

            if (curCol >= 0 && curCol < width) {
                int idx = curRow * d_originalWidth + curCol;
                int aboveIdx = (curRow - 1) * d_originalWidth + curCol;

                int min = score[aboveIdx];
                if (curCol > 0 && score[aboveIdx - 1] < min) {
                    min = score[aboveIdx - 1];
                }
                if (curCol < width - 1 && score[aboveIdx + 1] < min) {
                    min = score[aboveIdx + 1];
                }

                score[idx] = min + importants[idx];
            }
        }
        __syncthreads();
    }
}

__global__ void forwardEnergyKernel(uint8_t *grayPixels, int width, int height, float *energy) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col >= width || row >= height) return;
    
    int idx = row * d_originalWidth + col;
    
    if (row == 0) {
        energy[idx] = 0.0f;
        return;
    }
    
    // Get neighboring pixel values safely with bounds
    float left = (col > 0) ? static_cast<float>(grayPixels[idx - 1]) : static_cast<float>(grayPixels[idx]);
    float right = (col < width - 1) ? static_cast<float>(grayPixels[idx + 1]) : static_cast<float>(grayPixels[idx]);
    float up = static_cast<float>(grayPixels[idx - d_originalWidth]);
    float upLeft = (col > 0) ? static_cast<float>(grayPixels[idx - d_originalWidth - 1]) : up;
    float upRight = (col < width - 1) ? static_cast<float>(grayPixels[idx - d_originalWidth + 1]) : up;
    
    // Compute directional costs using floating-point
    float cU = fabsf(right - left);  // Cost for going straight up
    float cL = cU + fabsf(up - left);  // Cost for going up-left
    float cR = cU + fabsf(up - right);  // Cost for going up-right
    
    // Get minimum previous path cost
    float min_energy = energy[idx - d_originalWidth] + cU;
    if (col > 0) {
        min_energy = fminf(min_energy, energy[idx - d_originalWidth - 1] + cL);
    }
    if (col < width - 1) {
        min_energy = fminf(min_energy, energy[idx - d_originalWidth + 1] + cR);
    }
    
    energy[idx] = min_energy;
}

__global__ void hybridEnergyKernel(int *backwardEnergy, float *forwardEnergy, float *hybridEnergy, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col >= width || row >= height) return;
    
    int idx = row * d_originalWidth + col;
    
    // Normalize both energy values to 0-1 range
    float backwardNorm = static_cast<float>(backwardEnergy[idx]) / 255.0f;
    float forwardNorm = forwardEnergy[idx] / 255.0f;  // forwardEnergy is already in float
    
    // Choose the higher energy value
    float hybridVal = fmaxf(backwardNorm, forwardNorm);
    
    // Store the normalized hybrid energy
    hybridEnergy[idx] = hybridVal;
}

__global__ void convertToIntKernel(float *floatEnergy, int *intEnergy, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col >= width || row >= height) return;
    
    int idx = row * d_originalWidth + col;
    intEnergy[idx] = static_cast<int>(floatEnergy[idx] * 255.0f);
}

void seamCarvingByDevice(uchar3 *inPixels, int width, int height, int targetWidth, uchar3* outPixels, dim3 blockSize) {
    GpuTimer timer;
    timer.Start();

    // Memory allocation
    uchar3 *d_inPixels;
    CHECK(cudaMalloc(&d_inPixels, width * height * sizeof(uchar3)));
    uint8_t * d_grayPixels;
    CHECK(cudaMalloc(&d_grayPixels, width * height * sizeof(uint8_t)));
    int * d_importants;
    CHECK(cudaMalloc(&d_importants, width * height * sizeof(int)));
    int * d_leastSignificantPixel;
    CHECK(cudaMalloc(&d_leastSignificantPixel, height * sizeof(int)));
    int * d_score;
    CHECK(cudaMalloc(&d_score, width * height * sizeof(int)));
    float * d_forwardEnergy;
    CHECK(cudaMalloc(&d_forwardEnergy, width * height * sizeof(float)));
    float * d_hybridEnergy;
    CHECK(cudaMalloc(&d_hybridEnergy, width * height * sizeof(float)));

    int * importants = (int *)malloc(width * height * sizeof(int));
    int * leastSignificantPixel = (int *)malloc(height * sizeof(int));
    int * score = (int *)malloc(width * height * sizeof(int));

    // Allocate to shared memory
    size_t smemSize = ((blockSize.x + 3 - 1) * (blockSize.y + 3 - 1)) * sizeof(uint8_t);
    
    // block size use to calculate seam score table
    int blockSizeDp = 256;
    int gridSizeDp = (((width - 1) / blockSizeDp + 1) << 1) + 1;
    int stripHeight = (blockSizeDp >> 1) + 1;

    CHECK(cudaMemcpyToSymbol(d_originalWidth, &width, sizeof(int)));
    const int originalWidth = width;

    // copy input to device
    CHECK(cudaMemcpy(d_inPixels, inPixels, width * height * sizeof(uchar3), cudaMemcpyHostToDevice));

    // Step 1: Convert RGB image to grayscale
    dim3 gridSize((width-1)/blockSize.x + 1, (height-1)/blockSize.y + 1);
    Rgb2GrayKernel<<<gridSize, blockSize>>>(d_inPixels, width, height, d_grayPixels);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    // Allocate and initialize values ​​for 2 filters
    int _xSobel[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    int _ySobel[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
    int* xfilter, *yfilter;
    int xysize = 9*sizeof(int);
    CHECK(cudaMalloc(&xfilter, xysize))
    CHECK(cudaMalloc(&yfilter, xysize))

    CHECK(cudaMemcpy(xfilter, _xSobel, xysize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(yfilter, _ySobel, xysize, cudaMemcpyHostToDevice));

    // Loop to delete each seam
    while (width > targetWidth) {
        // Step 2: Calculate backward energy
        pixelsImportantKernel<<<gridSize, blockSize, smemSize>>>(d_grayPixels, width, height, xfilter, yfilter, d_importants);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());

        // Step 2.1: Calculate forward energy
        forwardEnergyKernel<<<gridSize, blockSize>>>(d_grayPixels, width, height, d_forwardEnergy);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());

        // Step 2.2: Combine energies using hybrid approach
        hybridEnergyKernel<<<gridSize, blockSize>>>(d_importants, d_forwardEnergy, d_hybridEnergy, width, height);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());

        // Convert hybrid energy to integer for seam finding
        dim3 convertGrid((width-1)/blockSize.x + 1, (height-1)/blockSize.y + 1);
        convertToIntKernel<<<convertGrid, blockSize>>>(d_hybridEnergy, d_importants, width, height);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());

        // Step 3: Calculate the seam table to find the seam with the smallest value
        for (int i = 0; i < height; i += (stripHeight >> 1)) {
            seamsScoreKernel<<<gridSizeDp, blockSizeDp>>>(d_importants, d_score, width, height, i);
            cudaDeviceSynchronize();
            CHECK(cudaGetLastError());
        }

        // From the bottom smallest pixel, trace up the first line to find the seam.
        CHECK(cudaMemcpy(score, d_score, originalWidth * height * sizeof(int), cudaMemcpyDeviceToHost));
        trace(score, leastSignificantPixel, width, height, originalWidth);

        // Step 4: Delete the seam found
        CHECK(cudaMemcpy(d_leastSignificantPixel, leastSignificantPixel, height * sizeof(int), cudaMemcpyHostToDevice));
        carvingKernel<<<height, 1>>>(d_leastSignificantPixel, d_inPixels, d_grayPixels, d_importants, width);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());
        
        --width;
    }

    CHECK(cudaMemcpy(outPixels, d_inPixels, originalWidth * height * sizeof(uchar3), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_inPixels));
    CHECK(cudaFree(d_grayPixels));
    CHECK(cudaFree(d_importants));
    CHECK(cudaFree(d_leastSignificantPixel));
    CHECK(cudaFree(d_score));
    CHECK(cudaFree(d_forwardEnergy));
    CHECK(cudaFree(d_hybridEnergy));
    CHECK(cudaFree(xfilter));
    CHECK(cudaFree(yfilter));
    free(score);
    free(leastSignificantPixel);
    free(importants);

    timer.Stop();
    float time = timer.Elapsed();
    printf("Processing time (use device): %f ms\n\n", time);    
}

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

void RGB2Gray(uchar3 * inPixels, int width, int height, uint8_t * outPixels) {
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            int i = r * width + c;
            outPixels[i] = 0.299f * inPixels[i].x + 0.587f * inPixels[i].y + 0.114f * inPixels[i].z;
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

void seamCarvingByHost(uchar3 *inPixels, int width, int height, int targetWidth, uchar3* outPixels) {
    GpuTimer timer;
    timer.Start();

    memcpy(outPixels, inPixels, width * height * sizeof(uchar3));

    const int originalWidth = width;

    // allocate memory
    int *importants = (int *)malloc(width * height * sizeof(int));
    int *score = (int *)malloc(width * height * sizeof(int));
    uint8_t *grayPixels= (uint8_t *)malloc(width * height * sizeof(uint8_t));
    
    // Convert to grayscale image
    RGB2Gray(inPixels, width, height, grayPixels);

    // Calculate pixel importance
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            importants[r * originalWidth + c] = pixelsImportant(grayPixels, r, c, width, height, width);
        }
    }

    while (width > targetWidth) {
        seamsScore(importants, score, width, height, originalWidth);

        // find where seam starts
        int minCol = 0, r = height - 1, prevMinCol;
        for (int c = 1; c < width; ++c) {
            if (score[r * originalWidth + c] < score[r * originalWidth + minCol])
                minCol = c;
        }

        // trace and remove seams
        for (; r >= 0; --r) {
            for (int i = minCol; i < width - 1; ++i) {
                outPixels[r * originalWidth + i] = outPixels[r * originalWidth + i + 1];
                grayPixels[r * originalWidth + i] = grayPixels[r * originalWidth + i + 1];
                importants[r * originalWidth + i] = importants[r * originalWidth + i + 1];
            }

            
            if (r < height - 1) {
                for (int affectedCol = max(0, prevMinCol - 2); affectedCol <= prevMinCol + 2 && affectedCol < width - 1; ++affectedCol) {
                    importants[(r + 1) * originalWidth + affectedCol] = pixelsImportant(grayPixels, r + 1, affectedCol, width - 1, height, originalWidth);
                }
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

        for (int affectedCol = max(0, minCol - 2); affectedCol <= minCol + 2 && affectedCol < width - 1; ++affectedCol) {
            importants[affectedCol] = pixelsImportant(grayPixels, 0, affectedCol, width - 1, height, originalWidth);
        }

        --width;
    }
    
    free(grayPixels);
    free(score);
    free(importants);

    timer.Stop();
    float time = timer.Elapsed();
    printf("Processing time (use host): %f ms\n\n", time);
}

float computeError(uchar3 * a1, uchar3 * a2, int n)
{
    float err = 0;
    for (int i = 0; i < n; i++)
    {
        err += abs((int)a1[i].x - (int)a2[i].x);
        err += abs((int)a1[i].y - (int)a2[i].y);
        err += abs((int)a1[i].z - (int)a2[i].z);
    }
    err /= (n * 3);
    return err;
}

char *concatStr(const char * s1, const char * s2)
{
    char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

void printDeviceInfo()
{
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
    printf("CMEM: %lu bytes\n", devProv.totalConstMem);
    printf("L2 cache: %i bytes\n", devProv.l2CacheSize);
    printf("SMEM / one SM: %lu bytes\n", devProv.sharedMemPerMultiprocessor);

    printf("****************************\n\n");

}

int main(int argc, char ** argv)
{   
    if (argc != 4 && argc != 6)
    {
        printf("The number of arguments is invalid\n");
        return EXIT_FAILURE;
    }

    printDeviceInfo();

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
    uchar3 * correctOutPixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
    // seamCarvingByHost(inPixels, width, height, targetWidth, correctOutPixels);

    // seam carving using device
    uchar3 * outPixels= (uchar3 *)malloc(width * height * sizeof(uchar3));
    dim3 blockSize(16, 16); // Default
    if (argc == 6)
    {
        blockSize.x = atoi(argv[4]);
        blockSize.y = atoi(argv[5]);
    } 
    seamCarvingByDevice(inPixels, width, height, targetWidth, outPixels, blockSize);
    printf("Image size after seam carving (new_width x height): %i x %i\n\n", targetWidth, height);
    // Compute mean absolute error between host result and device result
    // float err = computeError(outPixels, correctOutPixels, width * height);
    // printf("Error between device result and host result: %f\n", err);
    
    // Write results to files
    char *outFileNameBase = strtok(argv[2], "."); // Get rid of extension
    // writePnm(correctOutPixels, targetWidth, height, width, concatStr(outFileNameBase, "_host.pnm"));
    writePnm(outPixels, targetWidth, height, width, concatStr(outFileNameBase, "_device.pnm"));

    // Free memories
    free(inPixels);
    free(correctOutPixels);
    free(outPixels);
}