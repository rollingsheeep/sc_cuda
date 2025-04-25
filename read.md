seamcarving.exe input/surfer.jpg surfer_cuda.jpg 1280

nvcc parallel.cu -o seamcarvinghybrid.exe 

python process_image.py input/surfer.jpg surfer_cuda_backward 1000

python process_image.py input/surfer.jpg surfer_cuda_hybrid 1000 --energy hybrid

cd build

rm -r *

cmake ..

cmake --build . --config Release