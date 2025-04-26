import os
import sys
from PIL import Image

def convert_pnm_to_jpg(pnm_path, jpg_path):
    """Convert PNM image to JPG format"""
    try:
        # Open the PNM image directly using PIL
        img = Image.open(pnm_path)
        
        # Save as JPG with high quality
        img.save(jpg_path, 'JPEG', quality=95)
        print(f"Successfully converted {pnm_path} to {jpg_path}")
        return True
    except Exception as e:
        print(f"Error converting image {pnm_path}: {e}")
        return False

def main():
    # Create outputdata directory if it doesn't exist
    input_dir = "outputdata_temp"
    output_dir = "outputdata"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all pnm images from input directory
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.pnm')]
    
    # Sort files by width and implementation
    def sort_key(filename):
        # Extract width from filename (e.g., output_100_cuda.pnm -> 100)
        width = int(filename.split('_')[1])
        return width
    
    image_files.sort(key=sort_key)
    
    print(f"Found {len(image_files)} images to convert")
    
    # Convert each image
    success_count = 0
    for image_file in image_files:
        input_path = os.path.join(input_dir, image_file)
        # Remove 'output_' prefix and change extension to .jpg
        output_filename = image_file.replace('output_', '').replace('.pnm', '.jpg')
        output_path = os.path.join(output_dir, output_filename)
        
        if convert_pnm_to_jpg(input_path, output_path):
            success_count += 1
    
    print(f"\nConversion complete!")
    print(f"Successfully converted {success_count} out of {len(image_files)} images")
    print(f"JPG files are stored in {output_dir} directory")

if __name__ == "__main__":
    main() 