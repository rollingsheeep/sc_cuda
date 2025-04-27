#include <stdio.h>
#include <stdint.h>
#include <string>
#include <cstring>
#include <cstdlib>
#include "common.h"

/**
 * Concatenates two strings
 * @param s1 - First string (will be modified)
 * @param s2 - Second string
 * @return char* - Concatenated string
 */
char* concatStr(char* s1, const char* s2) {
    char* result = (char*)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

/**
 * Reads a PNM image file
 * @param fileName - Input file name
 * @param width - Output width of the image
 * @param height - Output height of the image
 * @param pixels - Output pixel data (allocated in this function)
 */
void readPnm(char* fileName, int &width, int &height, uchar3* &pixels) {
    FILE* f = fopen(fileName, "r");
    if (f == NULL) {
        printf("Cannot read %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    char type[3];
    fscanf(f, "%s", type);
    
    // Skip comments
    char c = getc(f);
    while (c == '#') {
        while (c != '\n') {
            c = getc(f);
        }
        c = getc(f);
    }
    ungetc(c, f);
    
    fscanf(f, "%d %d", &width, &height);
    
    int maxVal;
    fscanf(f, "%d", &maxVal);
    
    // Skip newline
    fgetc(f);
    
    // Allocate memory for pixels
    pixels = (uchar3*)malloc(width * height * sizeof(uchar3));
    
    // Read pixel data
    fread(pixels, sizeof(uchar3), width * height, f);
    
    fclose(f);
}

/**
 * Writes a PNM image file
 * @param pixels - Input pixel data
 * @param width - Width of the image
 * @param height - Height of the image
 * @param originalWidth - Original width for stride calculation
 * @param fileName - Output file name
 */
void writePnm(uchar3* pixels, int width, int height, int originalWidth, char* fileName) {
    FILE* f = fopen(fileName, "w");
    if (f == NULL) {
        printf("Cannot write %s\n", fileName);
        exit(EXIT_FAILURE);
    }
    
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    
    // Write pixel data with proper stride handling
    for (int r = 0; r < height; ++r) {
        fwrite(&pixels[r * originalWidth], sizeof(uchar3), width, f);
    }
    
    fclose(f);
} 