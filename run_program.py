import os
import sys
from PIL import Image
import subprocess

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

def run_seam_carving(executable, input_file, output_base, num_seams):
    """Run seam carving with appropriate command based on executable type"""
    try:
        if "mpi" in executable.lower():
            # Use mpiexec for MPI version with 4 processes
            cmd = [
                "mpiexec",
                "-n", "4",  # Use 4 processes
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
        subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running seam carving: {e}")
        return False

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
    num_seams = sys.argv[3]

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

    # Run seam carving for each version
    executables = [
        "seam_carving_seq.exe",
        "seam_carving_omp.exe",
        "seam_carving_cuda.exe",
        "seam_carving_mpi.exe"
    ]

    for exe in executables:
        if not os.path.exists(exe):
            print(f"Warning: {exe} not found, skipping...")
            continue

        print(f"\nRunning {exe}...")
        output_ppm_base = os.path.join(temp_dir, output_base)
        if not run_seam_carving(exe, temp_ppm, output_ppm_base, num_seams):
            print(f"Failed to run {exe}")
            continue

        # Convert output to JPG
        suffix = exe.replace("seam_carving_", "").replace(".exe", "")
        output_ppm = os.path.join(temp_dir, f"{output_base}_{suffix}.pnm")
        output_jpg = os.path.join(output_dir, f"{output_base}_{suffix}.jpg")
        
        if os.path.exists(output_ppm):
            if convert_ppm_to_jpg(output_ppm, output_jpg):
                print(f"Created: {output_jpg}")
            else:
                print(f"Failed to convert {output_ppm} to JPG")
        else:
            print(f"Warning: Output file {output_ppm} not found")

    # Clean up temporary files
    print("\nCleaning up temporary files...")
    for file in os.listdir(temp_dir):
        if file.endswith('.pnm'):
            temp_file = os.path.join(temp_dir, file)
            os.remove(temp_file)
            print(f"Removed: {temp_file}")

if __name__ == "__main__":
    main() 