#ifndef COMMON_H
#define COMMON_H

#include <cstdint>
#include <string>

// RGB pixel structure aligned with CUDA's uchar3
typedef struct {
    uint8_t x;  // R
    uint8_t y;  // G
    uint8_t z;  // B
} uchar3;

// Common function declarations
void readPnm(char * fileName, int &width, int &height, uchar3 * &pixels);
void writePnm(uchar3 *pixels, int width, int height, int originalWidth, char *fileName);
char* concatStr(char* s1, const char* s2);

#endif // COMMON_H 