import os
import sys
from PIL import Image

def convert_jpg_to_pnm(jpg_path, pnm_path):
    """Convert JPG image to PNM format"""
    try:
        # Open the JPG image
        img = Image.open(jpg_path)
        
        # Convert to RGB if not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Get image dimensions
        width, height = img.size
        
        # Save as PNM in P3 format (ASCII)
        with open(pnm_path, 'w') as f:
            # Write PNM header
            f.write(f"P3\n{width} {height}\n255\n")
            
            # Write pixel data
            for y in range(height):
                for x in range(width):
                    r, g, b = img.getpixel((x, y))
                    f.write(f"{r} {g} {b}\n")
        
        print(f"Successfully converted {jpg_path} to {pnm_path}")
        return True
    except Exception as e:
        print(f"Error converting image: {e}")
        return False

def main():
    # Create data_temp directory if it doesn't exist
    data_dir = "inputdata"
    temp_dir = "inputdata_temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Get all jpg images from data directory
    image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
    image_files.sort(key=lambda x: int(x.split('.')[0]))  # Sort by width
    
    print(f"Found {len(image_files)} images to convert")
    
    # Convert each image
    success_count = 0
    for image_file in image_files:
        input_path = os.path.join(data_dir, image_file)
        output_path = os.path.join(temp_dir, image_file.replace('.jpg', '.pnm'))
        
        if convert_jpg_to_pnm(input_path, output_path):
            success_count += 1
    
    print(f"\nConversion complete!")
    print(f"Successfully converted {success_count} out of {len(image_files)} images")
    print(f"PNM files are stored in {temp_dir} directory")

if __name__ == "__main__":
    main() 