"""
Merge HomeObjects-3K + HD10K into single dataset for YOLO fine-tuning

Input:
- HomeObjects-3K: 2,689 images (12 classes: 0-11)
- HD10K: 4,000 images (2 classes: 0-1)

Output:
- vacuum_cleaner_merged: 6,689 images (14 classes: 0-13)
  - HomeObjects classes 0-11 stay the same
  - HD10K class 0 (solid) → class 12 (solid_waste)
  - HD10K class 1 (liquid) → class 13 (liquid_stain)
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

def merge_datasets():
    """Merge HomeObjects-3K and HD10K into single dataset"""

    base_dir = Path("/Users/inter4259/Projects/SE_G10/datasets")

    # Input paths
    homeobjects_dir = base_dir / "homeobjects-3K"
    hd10k_dir = base_dir / "HD10K_yolo"

    # Output path
    output_dir = base_dir / "vacuum_cleaner_merged"

    # Create output directories
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Merging HomeObjects-3K + HD10K for Vacuum Cleaner Dataset")
    print("="*80)

    total_copied = {"train": 0, "val": 0}

    # ===== Part 1: Copy HomeObjects-3K (classes 0-11 unchanged) =====
    print("\n[1/2] Copying HomeObjects-3K...")

    for split in ["train", "val"]:
        src_images = homeobjects_dir / "images" / split
        src_labels = homeobjects_dir / "labels" / split

        dst_images = output_dir / "images" / split
        dst_labels = output_dir / "labels" / split

        image_files = list(src_images.glob("*.jpg"))

        for img_file in tqdm(image_files, desc=f"  HomeObjects {split}"):
            # Copy image
            shutil.copy(img_file, dst_images / img_file.name)

            # Copy label (no modification needed - classes 0-11 stay same)
            label_file = src_labels / img_file.with_suffix('.txt').name
            if label_file.exists():
                shutil.copy(label_file, dst_labels / label_file.name)

            total_copied[split] += 1

    print(f"  ✓ HomeObjects: {total_copied['train']} train, {total_copied['val']} val")

    # ===== Part 2: Copy HD10K with class remapping =====
    print("\n[2/2] Copying HD10K (remapping class IDs)...")

    hd10k_copied = {"train": 0, "val": 0}

    for split in ["train", "val"]:
        src_images = hd10k_dir / "images" / split
        src_labels = hd10k_dir / "labels" / split

        dst_images = output_dir / "images" / split
        dst_labels = output_dir / "labels" / split

        image_files = list(src_images.glob("*.jpg"))

        for img_file in tqdm(image_files, desc=f"  HD10K {split}"):
            # Copy image
            shutil.copy(img_file, dst_images / img_file.name)

            # Copy and remap label
            # HD10K: 0 (solid) → 12, 1 (liquid) → 13
            label_file = src_labels / img_file.with_suffix('.txt').name

            if label_file.exists():
                with open(label_file, 'r') as f_in:
                    lines = f_in.readlines()

                with open(dst_labels / label_file.name, 'w') as f_out:
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            # Remap: 0 → 12, 1 → 13
                            new_class_id = class_id + 12
                            parts[0] = str(new_class_id)
                            f_out.write(' '.join(parts) + '\n')

            hd10k_copied[split] += 1
            total_copied[split] += 1

    print(f"  ✓ HD10K: {hd10k_copied['train']} train, {hd10k_copied['val']} val")

    # ===== Summary =====
    print("\n" + "="*80)
    print("Summary")
    print("="*80)

    print(f"\nFinal dataset statistics:")
    print(f"  Train images: {total_copied['train']}")
    print(f"  Val images: {total_copied['val']}")
    print(f"  Total: {total_copied['train'] + total_copied['val']}")

    print(f"\nClass mapping:")
    print(f"  0-11: HomeObjects (12 classes)")
    print(f"  12: solid_waste (HD10K)")
    print(f"  13: liquid_stain (HD10K)")

    print(f"\nOutput directory: {output_dir}")
    print(f"YAML file: {base_dir / 'vacuum_cleaner.yaml'}")

    print("\n" + "="*80)
    print("✓ Dataset merge complete!")
    print("\nNext step: Train YOLO model")
    print("  yolo train data=vacuum_cleaner.yaml model=yolo11n.pt epochs=50 freeze=10")
    print("="*80)

    return output_dir

if __name__ == "__main__":
    merge_datasets()
