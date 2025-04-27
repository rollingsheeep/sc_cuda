# Seam Carving Implementation

This project implements seam carving algorithms using different parallel computing approaches: CUDA, OpenMP, and MPI.

## Requirements

Before building and running the program, ensure you have the following installed:

- CUDA Toolkit (for CUDA implementation)
- OpenMP (for OpenMP implementation)
- MPI (for MPI implementation)
- Python 3.x
- CMake (version 3.10 or higher)
- C++ compiler with C++11 support
- NVIDIA GPU (for CUDA implementation)

## Building the Program

1. Create a build directory and navigate to it:
```bash
mkdir build
cd build
```

2. Configure and build the project:
```bash
cmake ..
cmake --build . --config Release
```

3. After successful build, copy the generated executables to the root directory:
```bash
# On Windows
copy Release\*.exe ..\

# On Linux
cp *.exe ../
```

Source code:
- `sequential.cpp` (Sequential implementation)
- `openmp.cpp` (OpenMP implementation)
- `cuda.cu` (CUDA implementation)
- `mpi.cpp` (MPI implementation)

The following executables will be created:
- `seam_carving_seq.exe` (Sequential implementation)
- `seam_carving_omp.exe` (OpenMP implementation)
- `seam_carving_cuda.exe` (CUDA implementation)
- `seam_carving_mpi.exe` (MPI implementation)

## Running the Program

Use the Python script `run_program.py` to run the seam carving process:

```bash
python run_program.py <input_image> <output_prefix> <number_of_seams>
```

Example:
```bash
python run_program.py inputdata/3056.jpg detail 800
```

This will:
1. Process the input image using all implementations
2. Generate output images for each implementation
3. Display performance metrics for each implementation

## Performance Evaluation

Performance evaluation and analysis can be found in the following Jupyter notebooks:

### Single Image Analysis
- `detail.ipynb`: Contains detailed performance analysis for a single image including:
  - Individual component timing analysis
  - Throughput comparison
  - Speedup analysis
  - Image quality verification

### Multi-Image Analysis
The following notebooks analyze performance across 50 images of varying sizes:
- `sequential_performance.ipynb`: Analysis of sequential implementation performance
- `openmp_performance.ipynb`: Analysis of OpenMP implementation performance
- `cuda_performance.ipynb`: Analysis of CUDA implementation performance
- `mpi_performance.ipynb`: Analysis of MPI implementation performance

The notebooks provide comprehensive analysis of:
- Grayscale conversion time
- Energy calculation time (Backward, Forward, and Hybrid)
- Dynamic programming time
- Seam tracing and removal time
- Total processing time
- Throughput (seams per second)
- Speedup relative to sequential implementation

## Running Analysis Notebooks

### Prerequisites
1. Ensure you have input images in the `inputdata` directory
   - If you don't have images, you can download them from our [Google Drive](https://drive.google.com/drive/folders/1tQkhsvuiwpFwHxvHuTgq-ueyz29ZJC77?usp=sharing)
   - Images should be in JPG format

### Analysis Workflow

1. Convert input images to PNM format:
```bash
python convert_input.py
```
This will convert all JPG images in `inputdata` directory to PNM format, which is required for the analysis.

2. Run the analysis notebooks in the following order:
   - `sequential_performance.ipynb`
   - `openmp_performance.ipynb`
   - `cuda_performance.ipynb`
   - `mpi_performance.ipynb`

   Each notebook will:
   - Process the test image set
   - Generate performance metrics
   - Create visualizations
   - Save results in `outputdatatemp` directory

3. Convert output images back to JPG format:
```bash
python convert_output.py
```
This will convert all processed images in `outputdatatemp` directory from PNM format back to JPG format.

### Analysis Results
The notebooks will generate comprehensive performance analysis including:
- Processing time for each component
- Throughput comparison
- Speedup analysis
- Memory usage patterns
- Scaling characteristics

Results will be saved in both raw data format and visualizations for easy comparison between implementations.