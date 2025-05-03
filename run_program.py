import os
import sys
from PIL import Image
import subprocess
import re
from collections import defaultdict

def convert_jpg_to_ppm(jpg_path, ppm_path):
    """Convert JPG image to PPM format"""
    try:
        # Open the JPG image
        img = Image.open(jpg_path)
        
        # Convert to RGB if not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Get image dimensions
        width, height = img.size
        
        # Save as PPM in P3 format (ASCII)
        with open(ppm_path, 'w') as f:
            # Write PPM header
            f.write(f"P3\n{width} {height}\n255\n")
            
            # Write pixel data
            for y in range(height):
                for x in range(width):
                    r, g, b = img.getpixel((x, y))
                    f.write(f"{r} {g} {b}\n")
        
        print(f"Successfully converted {jpg_path} to {ppm_path}")
        print(f"File exists: {os.path.exists(ppm_path)}")
        print(f"File size: {os.path.getsize(ppm_path)} bytes")
        return True
    except Exception as e:
        print(f"Error converting image: {e}")
        return False

def convert_ppm_to_jpg(ppm_path, jpg_path):
    """Convert PPM image to JPG format"""
    try:
        # Open the PPM image
        img = Image.open(ppm_path)
        
        # Save as JPG with high quality
        img.save(jpg_path, 'JPEG', quality=95)
        print(f"Successfully converted {ppm_path} to {jpg_path}")
        return True
    except Exception as e:
        print(f"Error converting PPM to JPG: {e}")
        return False

def extract_metrics(output):
    """Extract performance metrics from program output"""
    metrics = {}
    current_impl = None
    
    # Initialize metrics structure
    implementation_markers = {
        'seq': 'Seam Carving Performance Analysis:',
        'omp': 'OpenMP Implementation Performance Analysis:',
        'cuda': 'CUDA Implementation Performance Analysis:',
        'mpi': 'Seam Carving Performance Analysis (MPI):'
    }
    
    lines = output.split('\n')
    for i, line in enumerate(lines):
        # Detect implementation type from the analysis header
        for impl_key, marker in implementation_markers.items():
            if marker in line:
                current_impl = impl_key
                metrics[current_impl] = {}
                break
                
        # Skip if we haven't identified an implementation yet
        if not current_impl:
            continue
            
        # Extract timing metrics
        try:
            # Look for lines with timing information (containing "ms")
            if ': ' in line and 'ms' in line:
                parts = line.split(': ')
                if len(parts) == 2:
                    metric_name = parts[0].strip()
                    # Extract the number before "ms"
                    time_str = parts[1].strip().split()[0]
                    try:
                        time_value = float(time_str)
                        metrics[current_impl][metric_name] = time_value
                    except ValueError:
                        print(f"Warning: Could not parse time value from: {time_str}")
        except Exception as e:
            print(f"Warning: Could not parse metric from line: {line}")
            continue
    
    # Verify all implementations have required metrics
    required_metrics = [
        'Grayscale conversion',
        'Backward energy (Sobel)',
        'Forward energy',
        'Hybrid energy',
        'Dynamic programming',
        'Seam tracing and removal',
        'Total seam carving time'
    ]
    
    # Initialize missing metrics with None
    for impl in metrics:
        for metric in required_metrics:
            if metric not in metrics[impl]:
                print(f"Warning: Missing metric '{metric}' for implementation '{impl}'")
                metrics[impl][metric] = None
    
    return metrics

def calculate_performance_metrics(metrics, num_seams, image_size):
    """Calculate throughput and speedup metrics"""
    results = defaultdict(dict)
    
    # Get sequential time as baseline
    if 'seq' in metrics and metrics['seq'].get('Total seam carving time') is not None:
        seq_time = metrics['seq']['Total seam carving time']
    else:
        print("Warning: Sequential implementation metrics not found or invalid")
        return results
    
    # Calculate metrics for each implementation
    for impl, data in metrics.items():
        total_time = data.get('Total seam carving time')
        if total_time is None:
            print(f"Warning: Missing total time for {impl} implementation")
            continue
            
        # Calculate throughput (seams/second)
        throughput = (num_seams * 1000) / total_time  # multiply by 1000 because time is in milliseconds
        
        # Calculate speedup relative to sequential
        speedup = seq_time / total_time
        
        # Store results
        results[impl] = {
            'Total Time (ms)': total_time,
            'Throughput (seams/sec)': throughput,
            'Speedup': speedup
        }
    
    return results

def run_seam_carving(executable, input_file, output_base, num_seams):
    """Run seam carving with appropriate command based on executable type"""
    try:
        if "mpi" in executable.lower():
            # Use mpiexec for MPI version with 4 processes
            cmd = [
                "mpiexec",
                "-n", "16",  # Use 4 processes
                executable,
                input_file,
                output_base,
                num_seams
            ]
        else:
            # Regular execution for non-MPI versions
            cmd = [
                executable,
                input_file,
                output_base,
                num_seams
            ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, 
            check=True, 
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True
        )
        print(result.stdout)  # Print the output
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running seam carving: {e}")
        return False, None

def print_performance_summary(perf_metrics):
    """Print a formatted performance summary"""
    print("\nPerformance Summary:")
    print("-" * 80)
    print(f"{'Implementation':<15} {'Total Time (ms)':<15} {'Throughput (seams/s)':<20} {'Speedup':<10}")
    print("-" * 80)
    
    for impl, metrics in perf_metrics.items():
        print(f"{impl:<15} {metrics['Total Time (ms)']:<15.2f} {metrics['Throughput (seams/sec)']:<20.2f} {metrics['Speedup']:<10.2f}")
    print("-" * 80)

def main():
    if len(sys.argv) != 4:
        print("Usage: python process_image.py input.jpg output_base_name number_of_seams")
        print("Example: python process_image.py input.jpg output 100")
        sys.exit(1)

    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get input file path
    input_jpg = os.path.abspath(sys.argv[1])
    output_base = sys.argv[2]
    num_seams = int(sys.argv[3])  # Convert to int here

    # Create output and temp directories
    output_dir = os.path.join(current_dir, "output")
    temp_dir = os.path.join(current_dir, "temp")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # Convert JPG to PPM
    temp_ppm = os.path.join(temp_dir, "temp_input.pnm")
    print(f"\nConverting image...")
    print(f"Input JPG: {input_jpg}")
    print(f"Temp PPM: {temp_ppm}")
    
    if not convert_jpg_to_ppm(input_jpg, temp_ppm):
        sys.exit(1)

    # Get image dimensions
    with Image.open(input_jpg) as img:
        image_size = img.size[0] * img.size[1]  # width * height

    # Store all output for metric extraction
    all_output = ""
    
    # Process each implementation
    implementations = [
        'seq', 
        'omp', 
        'cuda', 
        'mpi'
        ]
    metrics = {}
    pnm_files = []  # Store paths of generated PNM files
    
    for impl in implementations:
        print(f"\nProcessing {impl} implementation...")
        exe_name = f"seam_carving_{impl}.exe"
        # The implementations already append their suffix (_seq, _omp, etc.)
        output_ppm = os.path.join(temp_dir, f"{output_base}.pnm")
        output_jpg = os.path.join(output_dir, f"{output_base}_{impl}.jpg")
        
        try:
            # Prepare the command based on implementation
            if impl == 'mpi':
                cmd = ["mpiexec", "-n", "16", os.path.join(".", exe_name), temp_ppm, output_ppm, str(num_seams)]
            elif impl == 'cuda':
                cmd = [os.path.join(".", exe_name), temp_ppm, output_ppm, str(num_seams), "4", "4"]
            else:
                cmd = [os.path.join(".", exe_name), temp_ppm, output_ppm, str(num_seams)]
                
            print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Print the output and any errors
            if result.stdout:
                print("Output:")
                print(result.stdout)
            if result.stderr:
                print("Errors:")
                print(result.stderr)
            
            # Extract metrics from the output
            impl_metrics = extract_metrics(result.stdout)
            if impl_metrics:
                metrics[impl] = impl_metrics[impl]
            
            # The actual output PNM will have the implementation suffix
            actual_output_ppm = os.path.join(temp_dir, f"{output_base}_{impl}.pnm")
            if os.path.exists(actual_output_ppm):
                pnm_files.append((actual_output_ppm, output_jpg))
            else:
                print(f"Warning: Output file {actual_output_ppm} not found")
                
        except Exception as e:
            print(f"Error running {impl} implementation: {str(e)}")
            continue
    
    # Convert all PNM files to JPG at the end
    print("\nConverting all PNM files to JPG...")
    for pnm_path, jpg_path in pnm_files:
        if convert_ppm_to_jpg(pnm_path, jpg_path):
            print(f"Created: {jpg_path}")
            # Clean up the temporary PPM file
            os.remove(pnm_path)
        else:
            print(f"Failed to convert {pnm_path} to JPG")
    
    # Calculate and display performance metrics
    if metrics:
        print("\nPerformance Metrics:")
        print("-" * 80)
        perf_metrics = calculate_performance_metrics(metrics, num_seams, image_size)
        
        # Display metrics in a table format
        headers = ['Implementation', 'Total Time (ms)', 'Throughput (seams/sec)', 'Speedup']
        row_format = "{:<15} {:<20} {:<25} {:<15}"
        
        print(row_format.format(*headers))
        print("-" * 80)
        
        for impl, data in perf_metrics.items():
            if data:  # Only print if we have valid data
                row = [
                    impl,
                    f"{data['Total Time (ms)']:.2f}",
                    f"{data['Throughput (seams/sec)']:.2f}",
                    f"{data['Speedup']:.2f}"
                ]
                print(row_format.format(*row))
    else:
        print("\nNo valid metrics were collected from any implementation.")
    
    # Cleanup temporary files
    try:
        os.remove(temp_ppm)
        print("\nTemporary files cleaned up successfully")
    except Exception as e:
        print(f"\nError cleaning up temporary files: {str(e)}")

if __name__ == "__main__":
    main() 