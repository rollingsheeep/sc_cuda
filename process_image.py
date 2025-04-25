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

    # Run seam carving
    output_ppm_base = os.path.join(temp_dir, output_base)
    print(f"\nRunning seam carving...")
    print(f"Output PPM base: {output_ppm_base}")
    
    try:
        # Use relative paths for the executable and files
        seamcarving_exe = "seamcarvinghybrid2.exe"
        temp_ppm_rel = os.path.relpath(temp_ppm, current_dir)
        output_ppm_rel = os.path.relpath(output_ppm_base, current_dir)
        
        print(f"Seam carving executable: {seamcarving_exe}")
        print(f"Command: {seamcarving_exe} {temp_ppm_rel} {output_ppm_rel} {num_seams}")
        
        # Run the command with the current directory as working directory
        subprocess.run([
            seamcarving_exe,
            temp_ppm_rel,
            output_ppm_rel,
            num_seams
        ], check=True, cwd=current_dir)
        
        print("\nSeam carving completed successfully!")
        print(f"Output files created in temp directory:")
        print(f"- {output_base}_host.pnm")
        print(f"- {output_base}_device.pnm")

        # Convert PPM files to JPG and move to output directory
        print("\nConverting results to JPG format...")
        host_ppm = os.path.join(temp_dir, f"{output_base}_host.pnm")
        device_ppm = os.path.join(temp_dir, f"{output_base}_device.pnm")
        
        host_jpg = os.path.join(output_dir, f"{output_base}_host.jpg")
        device_jpg = os.path.join(output_dir, f"{output_base}_device.jpg")
        
        if convert_ppm_to_jpg(host_ppm, host_jpg):
            print(f"Created: {host_jpg}")
        if convert_ppm_to_jpg(device_ppm, device_jpg):
            print(f"Created: {device_jpg}")
            
    except subprocess.CalledProcessError as e:
        print(f"Error running seam carving: {e}")
        # Print the contents of the temp directory to help debug
        print("\nContents of temp directory:")
        for file in os.listdir(temp_dir):
            print(f"- {file}")
    finally:
        # Clean up temporary files
        for file in os.listdir(temp_dir):
            if file.endswith('.pnm'):
                temp_file = os.path.join(temp_dir, file)
                os.remove(temp_file)
                print(f"Cleaned up temporary file: {temp_file}")

if __name__ == "__main__":
    main() 