"""
HD10K Dataset Reorganizer
Converts IROS2022_Dataset to YOLO-compatible structure

Original structure:
IROS2022_Dataset/
├── train/
│   ├── solid_dirts/
│   │   ├── images/scene_0/*.jpg
│   │   └── solid_dirts_bboxes/scene_0/*.txt
│   └── liquid_dirts/
└── test/

Target structure:
HD10K_yolo/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import random

def mask_to_bbox(mask_path, img_width, img_height):
    """
    Convert binary mask (.png) to YOLO bbox format

    Returns:
        List of bboxes in YOLO format: [class_id, x_center, y_center, width, height]
        All values normalized to [0, 1]
    """
    # Read mask as grayscale
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        return []

    # Find contours (each contour = one object)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for contour in contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Skip too small objects (likely noise)
        if w < 5 or h < 5:
            continue

        # Convert to YOLO format (normalized)
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width = w / img_width
        height = h / img_height

        # class_id = 1 for liquid_stain
        bboxes.append([1, x_center, y_center, width, height])

    return bboxes

def reorganize_hd10k():
    """Reorganize HD10K dataset to YOLO format"""

    # Paths
    base_dir = Path("/Users/inter4259/Projects/SE_G10/datasets")
    src_dir = base_dir / "IROS2022_Dataset"
    dst_dir = base_dir / "HD10K_yolo"

    # Create output directories
    (dst_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dst_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dst_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dst_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("HD10K Dataset Reorganization (80/20 split)")
    print("="*60)

    # Set random seed for reproducibility
    random.seed(42)

    # ===== Process solid_dirts (class_id = 0) =====
    print("\n[1/2] Processing solid_dirts...")

    solid_images_dir = src_dir / "train" / "solid_dirts" / "images"
    solid_labels_dir = src_dir / "train" / "solid_dirts" / "solid_dirts_bboxes"

    scene_dirs = sorted([d for d in solid_images_dir.iterdir() if d.is_dir()])

    # Collect all valid image-label pairs
    all_solid_pairs = []

    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        image_files = sorted(scene_dir.glob("*.jpg"))

        for img_file in image_files:
            label_file = solid_labels_dir / scene_name / img_file.with_suffix('.txt').name
            if label_file.exists():
                all_solid_pairs.append((img_file, label_file, scene_name))

    # Shuffle and split 80/20
    random.shuffle(all_solid_pairs)
    split_idx = int(len(all_solid_pairs) * 0.8)
    train_pairs = all_solid_pairs[:split_idx]
    val_pairs = all_solid_pairs[split_idx:]

    print(f"  Total: {len(all_solid_pairs)}, Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    train_count = 0
    val_count = 0

    # Copy train set
    for img_file, label_file, scene_name in tqdm(train_pairs, desc="  Train"):
        new_name = f"solid_{scene_name}_{img_file.stem}"
        shutil.copy(img_file, dst_dir / "images" / "train" / f"{new_name}.jpg")
        shutil.copy(label_file, dst_dir / "labels" / "train" / f"{new_name}.txt")
        train_count += 1

    # Copy val set
    for img_file, label_file, scene_name in tqdm(val_pairs, desc="  Val"):
        new_name = f"solid_{scene_name}_{img_file.stem}"
        shutil.copy(img_file, dst_dir / "images" / "val" / f"{new_name}.jpg")
        shutil.copy(label_file, dst_dir / "labels" / "val" / f"{new_name}.txt")
        val_count += 1

    print(f"  ✓ Solid dirts: {train_count} train, {val_count} val")

    # ===== Process liquid_dirts (class_id = 1) =====
    print("\n[2/2] Processing liquid_dirts...")

    liquid_dir = src_dir / "train" / "liquid_dirts"

    if (liquid_dir / "images").exists():
        liquid_images_dir = liquid_dir / "images"
        liquid_labels_dir = liquid_dir / "liquid_dirts_masks"

        scene_dirs = sorted([d for d in liquid_images_dir.iterdir() if d.is_dir()])

        # Collect all valid samples
        all_liquid_samples = []

        for scene_dir in scene_dirs:
            scene_name = scene_dir.name
            image_files = sorted(scene_dir.glob("*.jpg"))

            for img_file in tqdm(image_files, desc=f"  Scanning {scene_name}", leave=False):
                mask_file = liquid_labels_dir / scene_name / img_file.with_suffix('.png').name

                if not mask_file.exists():
                    continue

                # Get image dimensions
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                img_height, img_width = img.shape[:2]

                # Convert mask to bboxes
                bboxes = mask_to_bbox(mask_file, img_width, img_height)

                if len(bboxes) == 0:
                    continue

                all_liquid_samples.append((img_file, bboxes, scene_name))

        # Shuffle and split 80/20
        random.shuffle(all_liquid_samples)
        split_idx = int(len(all_liquid_samples) * 0.8)
        train_samples = all_liquid_samples[:split_idx]
        val_samples = all_liquid_samples[split_idx:]

        print(f"  Total: {len(all_liquid_samples)}, Train: {len(train_samples)}, Val: {len(val_samples)}")

        liquid_train = 0
        liquid_val = 0

        # Copy train set
        for img_file, bboxes, scene_name in tqdm(train_samples, desc="  Train"):
            new_name = f"liquid_{scene_name}_{img_file.stem}"
            shutil.copy(img_file, dst_dir / "images" / "train" / f"{new_name}.jpg")

            with open(dst_dir / "labels" / "train" / f"{new_name}.txt", 'w') as f_out:
                for bbox in bboxes:
                    f_out.write(f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")
            liquid_train += 1

        # Copy val set
        for img_file, bboxes, scene_name in tqdm(val_samples, desc="  Val"):
            new_name = f"liquid_{scene_name}_{img_file.stem}"
            shutil.copy(img_file, dst_dir / "images" / "val" / f"{new_name}.jpg")

            with open(dst_dir / "labels" / "val" / f"{new_name}.txt", 'w') as f_out:
                for bbox in bboxes:
                    f_out.write(f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")
            liquid_val += 1

        print(f"  ✓ Liquid stains: {liquid_train} train, {liquid_val} val")
    else:
        print(f"  ⚠ Liquid dirts directory structure differs, skipping for now")

    # ===== Summary =====
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    total_train = len(list((dst_dir / "images" / "train").glob("*.jpg")))
    total_val = len(list((dst_dir / "images" / "val").glob("*.jpg")))

    print(f"Train images: {total_train}")
    print(f"Val images: {total_val}")
    print(f"Total: {total_train + total_val}")
    print(f"\nOutput directory: {dst_dir}")
    print("="*60)

    return dst_dir

if __name__ == "__main__":
    output_dir = reorganize_hd10k()
    print(f"\n✓ Done! Dataset ready at: {output_dir}")
