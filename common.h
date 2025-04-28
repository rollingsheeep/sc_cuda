#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdint.h>
#include <string>
#include <cmath>
#include <algorithm>

using namespace std;

// Common type definitions
struct uchar3 {
    uint8_t x, y, z;
};

// Common function declarations
void readPnm(char * fileName, int &width, int &height, uchar3 * &pixels);
void writePnm(uchar3 *pixels, int width, int height, int originalWidth, char *fileName);
char *concatStr(const char * s1, const char * s2);

#endif // COMMON_H 