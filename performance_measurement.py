import os
import time
import subprocess
import matplotlib.pyplot as plt
import numpy as np

def measure_performance(input_image, num_seams, num_runs=3):
    """Measure performance of different seam carving implementations"""
    
    implementations = [
        "seam_carving_seq.exe",
        "seam_carving_omp.exe",
        "seam_carving_cuda.exe",
        "seam_carving_mpi.exe"
    ]
    
    results = []
    
    for impl in implementations:
        if not os.path.exists(impl):
            print(f"Warning: {impl} not found, skipping...")
            continue
            
        times = []
        for run in range(num_runs):
            start_time = time.time()
            
            cmd = ["python", "process_image.py", input_image, f"output_{impl}", str(num_seams)]
            subprocess.run(cmd, check=True)
            
            end_time = time.time()
            execution_time = end_time - start_time
            times.append(execution_time)
            
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        results.append({
            'Implementation': impl.replace('seam_carving_', '').replace('.exe', ''),
            'Average Time (s)': avg_time,
            'Std Dev (s)': std_time,
            'Speedup': results[0]['Average Time (s)'] / avg_time if results else 1.0
        })
    
    return results

def plot_results(results):
    """Plot the performance results"""
    plt.figure(figsize=(10, 5))

    implementations = [r['Implementation'] for r in results]
    times = [r['Average Time (s)'] for r in results]
    errors = [r['Std Dev (s)'] for r in results]

    plt.bar(implementations, times, yerr=errors, capsize=5)
    plt.title('Seam Carving Performance Comparison')
    plt.xlabel('Implementation')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test with a single image
    test_image = "test_images/small.jpg"
    num_seams = 100
    
    results = measure_performance(test_image, num_seams)
    print("Performance Results:")
    for result in results:
        print(f"{result['Implementation']}: {result['Average Time (s)']:.2f} ± {result['Std Dev (s)']:.2f} seconds (Speedup: {result['Speedup']:.2f}x)")
    
    plot_results(results) 