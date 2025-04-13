#!/usr/bin/env python3
"""
Image Resizer Script for PCB Defect Detection

This script resizes all images in a source directory and saves them to a target directory.
Useful for preparing large PCB images for analysis in Colab.

Usage:
  python resize_images.py --source /path/to/original/images --target /path/to/resized/images --size 800
"""

import os
import cv2
import argparse
import glob
from pathlib import Path

def resize_image(image_path, output_path, max_size=800):
    """
    Resize an image while preserving aspect ratio
    
    Args:
        image_path: Path to the original image
        output_path: Path where the resized image will be saved
        max_size: Maximum width or height in pixels
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read {image_path}")
            return False
        
        # Get original dimensions
        h, w = img.shape[:2]
        original_size = f"{w}x{h}"
        
        # Calculate new dimensions while preserving aspect ratio
        if h > w:
            new_h, new_w = max_size, int(w * max_size / h)
        else:
            new_h, new_w = int(h * max_size / w), max_size
        
        # Resize the image
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the resized image
        cv2.imwrite(output_path, resized)
        
        print(f"Resized {os.path.basename(image_path)} from {original_size} to {new_w}x{new_h}")
        return True
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def process_directory(source_dir, target_dir, max_size=800, recursive=True):
    """
    Process all images in a directory and its subdirectories
    
    Args:
        source_dir: Source directory containing images
        target_dir: Target directory for resized images
        max_size: Maximum width or height in pixels
        recursive: Whether to process subdirectories
    
    Returns:
        int: Number of successfully processed images
    """
    # Ensure source directory exists
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory {source_dir} does not exist")
        return 0
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    image_paths = []
    
    for ext in image_extensions:
        if recursive:
            image_paths.extend(glob.glob(os.path.join(source_dir, '**', ext), recursive=True))
        else:
            image_paths.extend(glob.glob(os.path.join(source_dir, ext)))
    
    # Add uppercase extensions
    for ext in [e.upper() for e in image_extensions]:
        if recursive:
            image_paths.extend(glob.glob(os.path.join(source_dir, '**', ext), recursive=True))
        else:
            image_paths.extend(glob.glob(os.path.join(source_dir, ext)))
    
    print(f"Found {len(image_paths)} images in {source_dir}")
    
    # Process each image
    success_count = 0
    for image_path in image_paths:
        # Determine relative path to preserve directory structure
        rel_path = os.path.relpath(image_path, source_dir)
        output_path = os.path.join(target_dir, rel_path)
        
        # Resize the image
        if resize_image(image_path, output_path, max_size):
            success_count += 1
    
    print(f"Successfully processed {success_count} out of {len(image_paths)} images")
    return success_count

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Resize images for PCB defect detection')
    parser.add_argument('--source', required=True, help='Source directory containing images')
    parser.add_argument('--target', required=True, help='Target directory for resized images')
    parser.add_argument('--size', type=int, default=800, help='Maximum width or height in pixels (default: 800)')
    parser.add_argument('--no-recursive', action='store_true', help='Do not process subdirectories')
    
    args = parser.parse_args()
    
    # Process the directory
    process_directory(args.source, args.target, args.size, not args.no_recursive)

if __name__ == "__main__":
    main()
