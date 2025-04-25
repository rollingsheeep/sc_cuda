cmake_minimum_required(VERSION 3.18)
project(SeamCarving CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set runtime library to match CUDA's default
if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
endif()

# Find CUDA
find_package(CUDA REQUIRED)

# Create a library for common code
add_library(common_lib STATIC
    common.cpp
    common.h
)

# Sequential version
add_executable(seam_carving_seq
    sequential.cpp
)
target_link_libraries(seam_carving_seq common_lib)

# OpenMP version
add_executable(seam_carving_omp
    openmp.cpp
)
if(MSVC)
    target_compile_options(seam_carving_omp PRIVATE /openmp)
else()
    find_package(OpenMP REQUIRED)
    target_link_libraries(seam_carving_omp PRIVATE OpenMP::OpenMP_CXX)
endif()
target_link_libraries(seam_carving_omp common_lib)

# CUDA version
cuda_add_executable(seam_carving_cuda
    cuda.cu
)
target_link_libraries(seam_carving_cuda common_lib)

# Set optimization flags
if(MSVC)
    # Visual Studio
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")
else()
    # GCC/Clang
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")
endif()

# Installation
install(TARGETS seam_carving_seq seam_carving_omp seam_carving_cuda common_lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
) 