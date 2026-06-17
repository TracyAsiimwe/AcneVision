"""
Step 1: Dataset Collection and Cleaning
========================================
Remove corrupted, blurry, duplicate, or unreadable images.
Verify correct folder labeling and dataset consistency.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import hashlib

# Configuration
DATASET_PATH = '../dataset'  # Adjust path as needed
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
MIN_IMAGE_SIZE = (50, 50)  # Minimum width, height
BLUR_THRESHOLD = 100.0     # Laplacian variance threshold (lower = blurrier)

def calculate_image_hash(image_path):
    """Calculate perceptual hash for duplicate detection."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    # Resize to 8x8 and compute average hash
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
    avg = img.mean()
    diff = img > avg
    # Convert to hash string
    hash_str = ''.join(str(int(b)) for b in diff.flatten())
    return hash_str

def check_blurriness(image_path):
    """Detect if image is blurry using Laplacian variance."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return True  # Treat unreadable as blurry
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    return laplacian_var < BLUR_THRESHOLD

def check_corrupted(image_path):
    """Verify image can be properly decoded."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return True
        if img.size == 0:
            return True
        # Check if image has valid dimensions
        if img.shape[0] < MIN_IMAGE_SIZE[1] or img.shape[1] < MIN_IMAGE_SIZE[0]:
            return True
        return False
    except Exception:
        return True

def remove_duplicates(image_paths):
    """Remove duplicate images based on perceptual hashing."""
    hashes = defaultdict(list)
    duplicates = []
    
    print("[INFO] Checking for duplicates...")
    for path in image_paths:
        img_hash = calculate_image_hash(path)
        if img_hash:
            hashes[img_hash].append(path)
    
    # Keep first occurrence, mark rest as duplicates
    for hash_val, paths in hashes.items():
        if len(paths) > 1:
            duplicates.extend(paths[1:])  # Skip first one
    
    return duplicates

def clean_dataset(dataset_path):
    """
    Main cleaning function.
    Returns statistics of cleaning operations.
    """
    stats = {
        'total_scanned': 0,
        'corrupted_removed': 0,
        'blurry_removed': 0,
        'duplicates_removed': 0,
        'invalid_ext_removed': 0,
        'final_count': 0
    }
    
    print("=" * 60)
    print("ACNEVISION DATASET CLEANING PIPELINE")
    print("=" * 60)
    
    all_images = []
    
    # Walk through dataset directory
    for root, dirs, files in os.walk(dataset_path):
        # Skip if not in train/val/test structure
        if 'train' not in root and 'validation' not in root and 'test' not in root:
            continue
            
        for file in files:
            file_path = os.path.join(root, file)
            ext = Path(file).suffix.lower()
            
            # Check valid extension
            if ext not in VALID_EXTENSIONS:
                print(f"[REMOVE] Invalid extension: {file_path}")
                os.remove(file_path)
                stats['invalid_ext_removed'] += 1
                continue
            
            stats['total_scanned'] += 1
            all_images.append(file_path)
    
    print(f"\n[INFO] Total images scanned: {stats['total_scanned']}")
    
    # Step 1: Remove corrupted images
    print("\n[STEP 1] Removing corrupted/unreadable images...")
    for img_path in all_images[:]:  # Copy list to allow removal during iteration
        if check_corrupted(img_path):
            print(f"[CORRUPTED] {img_path}")
            os.remove(img_path)
            all_images.remove(img_path)
            stats['corrupted_removed'] += 1
    
    # Step 2: Remove blurry images
    print("\n[STEP 2] Removing blurry images...")
    for img_path in all_images[:]:
        if check_blurriness(img_path):
            print(f"[BLURRY] {img_path} (Laplacian variance below threshold)")
            os.remove(img_path)
            all_images.remove(img_path)
            stats['blurry_removed'] += 1
    
    # Step 3: Remove duplicates
    print("\n[STEP 3] Removing duplicate images...")
    duplicates = remove_duplicates(all_images)
    for dup in duplicates:
        print(f"[DUPLICATE] {dup}")
        os.remove(dup)
        all_images.remove(dup)
        stats['duplicates_removed'] += len(duplicates)
    
    # Final count
    stats['final_count'] = len(all_images)
    
    # Print summary
    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)
    print(f"Total scanned:      {stats['total_scanned']}")
    print(f"Invalid ext removed: {stats['invalid_ext_removed']}")
    print(f"Corrupted removed:   {stats['corrupted_removed']}")
    print(f"Blurry removed:      {stats['blurry_removed']}")
    print(f"Duplicates removed:  {stats['duplicates_removed']}")
    print(f"Final clean count:   {stats['final_count']}")
    print(f"Total removed:       {stats['total_scanned'] - stats['final_count']}")
    print("=" * 60)
    
    return stats

def verify_folder_labels(dataset_path):
    """
    Verify that folder names match expected class labels.
    Expected: clear_skin, mild, moderate, severe
    """
    expected_classes = {'clear_skin', 'mild', 'moderate', 'severe'}
    
    print("\n[INFO] Verifying folder structure...")
    
    for split in ['train', 'validation', 'test']:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            print(f"[WARNING] Missing split folder: {split_path}")
            continue
        
        found_classes = set(os.listdir(split_path))
        
        # Check for unexpected folders
        unexpected = found_classes - expected_classes
        if unexpected:
            print(f"[WARNING] Unexpected folders in {split}: {unexpected}")
        
        # Check for missing folders
        missing = expected_classes - found_classes
        if missing:
            print(f"[WARNING] Missing class folders in {split}: {missing}")
        
        # Count images per class
        for cls in found_classes & expected_classes:
            cls_path = os.path.join(split_path, cls)
            if os.path.isdir(cls_path):
                count = len([f for f in os.listdir(cls_path) 
                           if Path(f).suffix.lower() in VALID_EXTENSIONS])
                print(f"  {split}/{cls}: {count} images")

if __name__ == '__main__':
    # Run cleaning
    stats = clean_dataset(DATASET_PATH)
    verify_folder_labels(DATASET_PATH)
    
    print("\n[SUCCESS] Dataset cleaning complete!")