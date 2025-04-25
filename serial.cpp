#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <chrono>
#include <string>

// Custom image structure to replace OpenCV Mat
struct Image {
    int width;
    int height;
    int channels;
    std::vector<unsigned char> data;

    Image(int w, int h, int c) : width(w), height(h), channels(c) {
        data.resize(w * h * c);
    }

    unsigned char& at(int y, int x, int c) {
        return data[(y * width + x) * channels + c];
    }

    const unsigned char& at(int y, int x, int c) const {
        return data[(y * width + x) * channels + c];
    }
};

// Sequential conversion from OpenCV Mat to our Image structure
Image matToImage(const cv::Mat& mat) {
    Image img(mat.cols, mat.rows, mat.channels());
    // Process pixels sequentially row by row
    for (int y = 0; y < mat.rows; y++) {
        for (int x = 0; x < mat.cols; x++) {
            for (int c = 0; c < mat.channels(); c++) {
                img.at(y, x, c) = mat.at<cv::Vec3b>(y, x)[c];
            }
        }
    }
    return img;
}

// Sequential conversion from our Image structure to OpenCV Mat
cv::Mat imageToMat(const Image& img) {
    cv::Mat mat(img.height, img.width, CV_8UC3);
    // Process pixels sequentially row by row
    for (int y = 0; y < img.height; y++) {
        for (int x = 0; x < img.width; x++) {
            for (int c = 0; c < img.channels; c++) {
                mat.at<cv::Vec3b>(y, x)[c] = img.at(y, x, c);
            }
        }
    }
    return mat;
}

// Sequential RGB to grayscale conversion
Image rgbToGray(const Image& input) {
    Image output(input.width, input.height, 1);
    // Process pixels sequentially row by row
    for (int y = 0; y < input.height; y++) {
        for (int x = 0; x < input.width; x++) {
            // Sequential calculation of grayscale value
            double gray = 0.299 * input.at(y, x, 0) +  // R
                         0.587 * input.at(y, x, 1) +  // G
                         0.114 * input.at(y, x, 2);   // B
            output.at(y, x, 0) = static_cast<unsigned char>(gray);
        }
    }
    return output;
}

// Sequential min/max finding
void findMinMax(const Image& img, double& minVal, double& maxVal) {
    minVal = std::numeric_limits<double>::max();
    maxVal = std::numeric_limits<double>::lowest();
    
    // Process pixels sequentially row by row
    for (int y = 0; y < img.height; y++) {
        for (int x = 0; x < img.width; x++) {
            double val = img.at(y, x, 0);
            if (val < minVal) minVal = val;
            if (val > maxVal) maxVal = val;
        }
    }
}

// Sequential image normalization
void normalizeImageBackward(Image& img) {
    double minVal, maxVal;
    findMinMax(img, minVal, maxVal);
    
    double range = maxVal - minVal;
    if (range == 0) range = 1; // Avoid division by zero
    
    // Process pixels sequentially row by row
    for (int y = 0; y < img.height; y++) {
        for (int x = 0; x < img.width; x++) {
            double normalized = (img.at(y, x, 0) - minVal) * (255.0 / range);
            img.at(y, x, 0) = static_cast<unsigned char>(normalized);
        }
    }
}

// Forward energy calculation based on pure energy costs
Image calculateForwardEnergy(const Image& gray) {
    int H = gray.height;
    int W = gray.width;

    Image energy(W, H, 1);
    
    // Use double for higher precision in cumulative energy
    std::vector<std::vector<double>> M(H, std::vector<double>(W, 0.0));
    
    // Initialize first row to 0
    for (int x = 0; x < W; x++) {
        M[0][x] = 0.0;
    }

    // Forward energy dynamic programming
    for (int y = 1; y < H; y++) {
        for (int x = 0; x < W; x++) {
            // Get neighboring pixel values safely with bounds
            double left = (x > 0) ? static_cast<double>(gray.at(y, x-1, 0)) : 
                                  static_cast<double>(gray.at(y, x, 0));
            double right = (x < W-1) ? static_cast<double>(gray.at(y, x+1, 0)) : 
                                     static_cast<double>(gray.at(y, x, 0));
            double up = static_cast<double>(gray.at(y-1, x, 0));
            double upLeft = (x > 0) ? static_cast<double>(gray.at(y-1, x-1, 0)) : up;
            double upRight = (x < W-1) ? static_cast<double>(gray.at(y-1, x+1, 0)) : up;

            // Compute directional costs
            // Cost for going straight up (horizontal gradient)
            double cU = std::abs(right - left);

            // Cost for going up-left
            double cL = cU + std::abs(up - left);

            // Cost for going up-right
            double cR = cU + std::abs(up - right);

            // Get minimum previous path cost
            double min_energy = M[y-1][x] + cU;  // Cost of going straight up
            if (x > 0) {
                min_energy = std::min(min_energy, M[y-1][x-1] + cL);  // Cost of going up-left
            }
            if (x < W-1) {
                min_energy = std::min(min_energy, M[y-1][x+1] + cR);  // Cost of going up-right
            }

            // Store the minimum energy
            M[y][x] = min_energy;
        }
    }

    // Find min and max for normalization
    double minVal = 1e9, maxVal = -1e9;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            minVal = std::min(minVal, M[y][x]);
            maxVal = std::max(maxVal, M[y][x]);
        }
    }

    // Normalize and store in energy image
    double range = maxVal - minVal;
    if (range == 0) range = 1; // Avoid division by zero
    
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            double normalized = (M[y][x] - minVal) * (255.0 / range);
            energy.at(y, x, 0) = static_cast<unsigned char>(normalized);
        }
    }

    return energy;
}

// Sobel filter for edge detection at a specific pixel
double sobelFilterAt(const Image& img, int cx, int cy) {
    // Sobel kernels
    static const double gx[3][3] = {
        {1.0, 0.0, -1.0},
        {2.0, 0.0, -2.0},
        {1.0, 0.0, -1.0}
    };

    static const double gy[3][3] = {
        {1.0, 2.0, 1.0},
        {0.0, 0.0, 0.0},
        {-1.0, -2.0, -1.0}
    };

    double sx = 0.0;
    double sy = 0.0;

    // Apply Sobel kernels
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int x = cx + dx;
            int y = cy + dy;
            
            // Handle boundary conditions
            double c = 0.0;
            if (x >= 0 && x < img.width && y >= 0 && y < img.height) {
                c = static_cast<double>(img.at(y, x, 0));
            }
            
            sx += c * gx[dy + 1][dx + 1];
            sy += c * gy[dy + 1][dx + 1];
        }
    }

    // Return gradient magnitude
    return std::sqrt(sx*sx + sy*sy);
}

// Backward energy calculation using Sobel filter
Image calculateBackwardEnergy(const Image& gray) {
    Image energy(gray.width, gray.height, 1);
    
    // Apply Sobel filter to compute energy
    for (int y = 0; y < gray.height; y++) {
        for (int x = 0; x < gray.width; x++) {
            energy.at(y, x, 0) = static_cast<unsigned char>(sobelFilterAt(gray, x, y));
        }
    }

    // Normalize energy values
    double minVal = 1e9, maxVal = -1e9;
    for (int y = 0; y < gray.height; y++) {
        for (int x = 0; x < gray.width; x++) {
            double val = static_cast<double>(energy.at(y, x, 0));
            minVal = std::min(minVal, val);
            maxVal = std::max(maxVal, val);
        }
    }

    double range = maxVal - minVal;
    if (range == 0) range = 1; // Avoid division by zero
    
    for (int y = 0; y < gray.height; y++) {
        for (int x = 0; x < gray.width; x++) {
            double normalized = (static_cast<double>(energy.at(y, x, 0)) - minVal) * (255.0 / range);
            energy.at(y, x, 0) = static_cast<unsigned char>(normalized);
        }
    }

    return energy;
}

// Sequential seam finding using dynamic programming
std::vector<int> findSeam(const Image& energy) {
    int H = energy.height;
    int W = energy.width;
    
    // Create and initialize DP table sequentially
    std::vector<std::vector<int>> dp(H, std::vector<int>(W));
    
    // Initialize first row sequentially
    for (int x = 0; x < W; x++) {
        dp[0][x] = energy.at(0, x, 0);
    }
    
    // Fill DP table sequentially row by row
    for (int y = 1; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int min_prev = dp[y-1][x];
            if (x > 0) min_prev = std::min(min_prev, dp[y-1][x-1]);
            if (x < W-1) min_prev = std::min(min_prev, dp[y-1][x+1]);
            dp[y][x] = energy.at(y, x, 0) + min_prev;
        }
    }
    
    // Find minimum path sequentially
    std::vector<int> seam(H);
    int min_x = 0;
    for (int x = 1; x < W; x++) {
        if (dp[H-1][x] < dp[H-1][min_x]) {
            min_x = x;
        }
    }
    seam[H-1] = min_x;
    
    // Backtrack to find the seam sequentially
    for (int y = H-2; y >= 0; y--) {
        int x = seam[y+1];
        int min_prev = dp[y][x];
        int best_x = x;
        
        if (x > 0 && dp[y][x-1] < min_prev) {
            min_prev = dp[y][x-1];
            best_x = x-1;
        }
        if (x < W-1 && dp[y][x+1] < min_prev) {
            best_x = x+1;
        }
        
        seam[y] = best_x;
    }
    
    return seam;
}

// Sequential seam removal
Image removeSeam(const Image& input, const std::vector<int>& seam) {
    Image output(input.width - 1, input.height, input.channels);
    
    // Process pixels sequentially row by row
    for (int y = 0; y < input.height; y++) {
        int seam_x = seam[y];
        for (int x = 0; x < input.width - 1; x++) {
            for (int c = 0; c < input.channels; c++) {
                if (x < seam_x) {
                    output.at(y, x, c) = input.at(y, x, c);
                } else {
                    output.at(y, x, c) = input.at(y, x + 1, c);
                }
            }
        }
    }
    
    return output;
}

// Hybrid energy calculation that combines both backward and forward energy
Image calculateHybridEnergy(const Image& gray) {
    int H = gray.height;
    int W = gray.width;

    // Calculate both energy types
    Image backwardEnergy = calculateBackwardEnergy(gray);
    Image forwardEnergy = calculateForwardEnergy(gray);

    // Create result image
    Image hybridEnergy(W, H, 1);

    // Normalize both energy maps to the same range
    double bMin = 1e9, bMax = -1e9;
    double fMin = 1e9, fMax = -1e9;

    // Find min/max for both energy maps
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            double bVal = static_cast<double>(backwardEnergy.at(y, x, 0));
            double fVal = static_cast<double>(forwardEnergy.at(y, x, 0));
            
            bMin = std::min(bMin, bVal);
            bMax = std::max(bMax, bVal);
            fMin = std::min(fMin, fVal);
            fMax = std::max(fMax, fVal);
        }
    }

    // Normalize ranges
    double bRange = bMax - bMin;
    double fRange = fMax - fMin;
    if (bRange == 0) bRange = 1;
    if (fRange == 0) fRange = 1;

    // Combine energies by choosing the maximum normalized value
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            // Normalize both energy values
            double bNorm = (static_cast<double>(backwardEnergy.at(y, x, 0)) - bMin) / bRange;
            double fNorm = (static_cast<double>(forwardEnergy.at(y, x, 0)) - fMin) / fRange;

            // Choose the higher energy value
            double hybridVal = std::max(bNorm, fNorm);

            // Scale back to 0-255 range
            hybridEnergy.at(y, x, 0) = static_cast<unsigned char>(hybridVal * 255.0);
        }
    }

    return hybridEnergy;
}

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: seamCarving <input_image> <output_image> <num_seams_to_remove> --energy [backward|forward|hybrid]" << std::endl;
        std::cerr << "Example: seamCarving input.jpg output.jpg 100 --energy hybrid" << std::endl;
        return 1;
    }

    // Parse energy type
    std::string energy_type = "backward"; // default
    if (std::string(argv[4]) == "--energy") {
        energy_type = argv[5];
        if (energy_type != "backward" && energy_type != "forward" && energy_type != "hybrid") {
            std::cerr << "Error: Energy type must be either 'backward', 'forward', or 'hybrid'" << std::endl;
            return 1;
        }
    } else {
        std::cerr << "Error: Missing --energy option" << std::endl;
        return 1;
    }

    // Timing variables
    auto start_time = std::chrono::high_resolution_clock::now();
    auto end_time = start_time;
    double total_dp_time = 0.0;
    double total_seam_comp_time = 0.0;
    double total_seam_removal_time = 0.0;
    double total_energy_time = 0.0;

    // Read input image
    start_time = std::chrono::high_resolution_clock::now();
    cv::Mat input_mat = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (!input_mat.data) {
        std::cerr << "Error: Could not open or find the input image at: " << argv[1] << std::endl;
        return -1;
    }
    end_time = std::chrono::high_resolution_clock::now();
    double image_load_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    std::cout << "Image loading time: " << image_load_time << " ms" << std::endl;

    // Convert to our Image structure
    Image image = matToImage(input_mat);
    int num_seams = std::atoi(argv[3]);

    // Initial grayscale conversion
    start_time = std::chrono::high_resolution_clock::now();
    Image gray = rgbToGray(image);
    end_time = std::chrono::high_resolution_clock::now();
    double luminance_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    std::cout << "Luminance computation time: " << luminance_time << " ms" << std::endl;

    // Initial energy computation
    start_time = std::chrono::high_resolution_clock::now();
    Image energy(gray.width, gray.height, 1);  // Initialize with proper dimensions
    if (energy_type == "forward") {
        std::cout << "Using forward energy calculation" << std::endl;
        energy = calculateForwardEnergy(gray);
    } else if (energy_type == "backward") {
        std::cout << "Using backward energy calculation" << std::endl;
        energy = calculateBackwardEnergy(gray);
    } else {
        std::cout << "Using hybrid energy calculation" << std::endl;
        energy = calculateHybridEnergy(gray);
    }
    end_time = std::chrono::high_resolution_clock::now();
    double initial_energy_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    std::cout << "Initial energy computation time: " << initial_energy_time << " ms" << std::endl;

    std::cout << "\nProcessing image sequentially..." << std::endl;
    for (int i = 0; i < num_seams; i++) {
        // Dynamic programming for seam finding
        start_time = std::chrono::high_resolution_clock::now();
        std::vector<int> seam = findSeam(energy);
        end_time = std::chrono::high_resolution_clock::now();
        double dp_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        total_dp_time += dp_time;

        // Seam computation
        start_time = std::chrono::high_resolution_clock::now();
        // Seam computation is part of findSeam, so we just record the time
        end_time = std::chrono::high_resolution_clock::now();
        double seam_comp_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        total_seam_comp_time += seam_comp_time;

        // Seam removal
        start_time = std::chrono::high_resolution_clock::now();
        image = removeSeam(image, seam);
        end_time = std::chrono::high_resolution_clock::now();
        double seam_removal_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        total_seam_removal_time += seam_removal_time;

        // Energy update
        start_time = std::chrono::high_resolution_clock::now();
        gray = rgbToGray(image);
        if (energy_type == "forward") {
            energy = calculateForwardEnergy(gray);
        } else if (energy_type == "backward") {
            energy = calculateBackwardEnergy(gray);
        } else {
            energy = calculateHybridEnergy(gray);
        }
        end_time = std::chrono::high_resolution_clock::now();
        double energy_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        total_energy_time += energy_time;

        if (i % 10 == 0) {
            std::cout << "Processing seam " << i << "..." << std::endl;
        }
    }

    // Save output image
    cv::Mat output_mat = imageToMat(image);
    if (!cv::imwrite(argv[2], output_mat)) {
        std::cerr << "Error: Failed to save the output image to: " << argv[2] << std::endl;
        return 1;
    }

    // Print timing breakdown
    std::cout << "\nSeam removal breakdown:" << std::endl;
    std::cout << "  Dynamic programming: " << total_dp_time << " ms" << std::endl;
    std::cout << "  Seam computation: " << total_seam_comp_time << " ms" << std::endl;
    std::cout << "  Seam removal: " << total_seam_removal_time << " ms" << std::endl;
    std::cout << "  Energy update: " << total_energy_time << " ms" << std::endl;
    std::cout << "  Total seam removal time: " << (total_dp_time + total_seam_comp_time + 
                                                 total_seam_removal_time + total_energy_time) << " ms" << std::endl;

    std::cout << "\nSuccess! Processed image saved as: " << argv[2] << std::endl;
    return 0;
}