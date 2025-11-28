import json
import os
from collections import defaultdict

def check_images_exist(json_path, image_base_dir):
    """
    Check if all images referenced in the JSON file exist in the specified directory.
    
    Args:
        json_path: Path to the JSON file containing image references
        image_base_dir: Base directory where images should be located
    """
    # Load the JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract all image paths
    image_paths = []
    for idx, entry in enumerate(data):
        if 'image' in entry:
            image_paths.append(entry['image'])
    
    print(f"Checking {len(image_paths)} images...\n")
    
    # Check which images exist
    missing_images = []
    existing_count = 0
    missing_by_subdir = defaultdict(list)
    
    for img_path in image_paths:
        # Extract the relative path after "data/"
        assert img_path.startswith("data/images/"), f"Unexpected path format: {img_path}"
        relative_path = img_path.replace("data/", "")
        actual_path = os.path.join(image_base_dir, relative_path)
        
        if os.path.exists(actual_path):
            existing_count += 1
        else:
            subdir = os.path.dirname(relative_path).split(os.sep)[0] if os.path.dirname(relative_path) else "root"
            missing_by_subdir[subdir].append({
                'original': img_path,
                'expected': actual_path
            })
            missing_images.append({
                'original': img_path,
                'expected': actual_path
            })
    
    # Print summary
    total = len(image_paths)
    missing_count = len(missing_images)
    
    print(f"{'='*60}")
    print(f"Summary: {existing_count}/{total} images found ({existing_count/total*100:.1f}%)")
    print(f"{'='*60}")
    
    if missing_images:
        print(f"\n⚠ Missing: {missing_count} images")
        print("\nMissing by category:")
        for subdir, imgs in sorted(missing_by_subdir.items()):
            print(f"  • {subdir}: {len(imgs)} images")
        
        # Save detailed report
        output_file = "missing_images.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Missing Images Report\n")
            f.write(f"{'='*60}\n")
            f.write(f"Total missing: {len(missing_images)}\n\n")
            for subdir, imgs in sorted(missing_by_subdir.items()):
                f.write(f"\n{subdir} ({len(imgs)} images):\n")
                f.write("-" * 60 + "\n")
                for img in imgs:
                    f.write(f"{img['original']}\n")
        print(f"\n→ Detailed report saved to: {output_file}")
    else:
        print("\n✓ All images found!")
    
    return {
        'total': total,
        'found': existing_count,
        'missing': missing_count
    }

def remove_missing_entries(json_path, image_base_dir, output_path):
    print(f"Loading data from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_count = len(data)
    print(f"Original entries: {original_count}")
    
    # Filter entries with existing images
    valid_entries = []
    missing_count = 0
    
    for entry in data:
        if 'image' not in entry:
            valid_entries.append(entry)
            continue
        
        img_path = entry['image']
        assert img_path.startswith("data/images/"), f"Unexpected path format: {img_path}"
        relative_path = img_path.replace("data/", "")
        actual_path = os.path.join(image_base_dir, relative_path)
        
        if os.path.exists(actual_path):
            valid_entries.append(entry)
        else:
            missing_count += 1
    
    # Save the filtered data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(valid_entries, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Filtering complete!")
    print(f"{'='*60}")
    print(f"Original entries: {original_count}")
    print(f"Removed entries:  {missing_count}")
    print(f"Remaining entries: {len(valid_entries)} ({len(valid_entries)/original_count*100:.1f}%)")
    print(f"\n→ Filtered data saved to: {output_path}")
    
    return {
        'original': original_count,
        'removed': missing_count,
        'remaining': len(valid_entries)
    }

if __name__ == "__main__":
    # Configuration
    json_file = "data/train_small.json"
    image_directory = "../visualPRM_data/images/"
    output_file = "data/train_small_cleaned.json"
    
    results = check_images_exist(json_file, image_directory)
    
    clean_results = remove_missing_entries(json_file, image_directory, output_file)

