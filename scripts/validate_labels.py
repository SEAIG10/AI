"""
Validate YOLO label files
Find and fix problematic labels
"""

from pathlib import Path

def validate_labels():
    """Find problematic label files"""

    label_dir = Path("/Users/inter4259/Projects/SE_G10/datasets/vacuum_cleaner_merged/labels")

    problems = {
        'wrong_fields': [],
        'out_of_range': [],
        'empty': []
    }

    for split in ['train', 'val']:
        label_files = (label_dir / split).glob("*.txt")

        for label_file in label_files:
            with open(label_file, 'r') as f:
                lines = f.readlines()

            # Check empty
            if len(lines) == 0:
                problems['empty'].append(str(label_file))
                continue

            # Check each line
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                parts = line.split()

                # Check field count
                if len(parts) != 5:
                    problems['wrong_fields'].append(f"{label_file}:{line_num} ({len(parts)} fields)")
                    continue

                try:
                    class_id, x, y, w, h = map(float, parts)

                    # Check ranges
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        problems['out_of_range'].append(f"{label_file}:{line_num}")

                    if not (0 <= class_id <= 13):
                        problems['out_of_range'].append(f"{label_file}:{line_num} (class {class_id})")

                except ValueError:
                    problems['wrong_fields'].append(f"{label_file}:{line_num} (non-numeric)")

    # Report
    print("="*80)
    print("Label Validation Report")
    print("="*80)

    print(f"\n[Empty files]: {len(problems['empty'])}")
    for f in problems['empty'][:10]:
        print(f"  {f}")
    if len(problems['empty']) > 10:
        print(f"  ... and {len(problems['empty']) - 10} more")

    print(f"\n[Wrong field count]: {len(problems['wrong_fields'])}")
    for f in problems['wrong_fields'][:10]:
        print(f"  {f}")
    if len(problems['wrong_fields']) > 10:
        print(f"  ... and {len(problems['wrong_fields']) - 10} more")

    print(f"\n[Out of range]: {len(problems['out_of_range'])}")
    for f in problems['out_of_range'][:10]:
        print(f"  {f}")
    if len(problems['out_of_range']) > 10:
        print(f"  ... and {len(problems['out_of_range']) - 10} more")

    print("\n" + "="*80)

    total_problems = sum(len(v) for v in problems.values())
    if total_problems > 0:
        print(f"⚠️  Total problems found: {total_problems}")
        print("\nFix these issues before training!")
    else:
        print("✓ All labels are valid!")

    return problems

if __name__ == "__main__":
    validate_labels()
