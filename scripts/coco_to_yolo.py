import os
import shutil
import json
import numpy as np
from sklearn.model_selection import train_test_split
import yaml # For saving dataset.yaml
from collections import Counter # Import Counter for stratification check

def coco_bbox_to_yolo(bbox, img_width, img_height):
    """
    Converts COCO bbox format [x_top_left, y_top_left, width, height]
    to YOLO format [x_center, y_center, normalized_width, normalized_height].
    """
    x_tl, y_tl, bbox_width, bbox_height = bbox

    # Ensure bbox dimensions are not negative or zero (can happen with bad data)
    if bbox_width <= 0 or bbox_height <= 0 or img_width <= 0 or img_height <= 0:
        # Handle potential division by zero or invalid dimensions
        print(f"Warning: Invalid dimensions encountered in coco_bbox_to_yolo: bbox={bbox}, img_width={img_width}, img_height={img_height}. Returning zeros.")
        return [0.0, 0.0, 0.0, 0.0]


    x_center = (x_tl + bbox_width / 2) / img_width
    y_center = (y_tl + bbox_height / 2) / img_height
    
    normalized_width = bbox_width / img_width
    normalized_height = bbox_height / img_height
    
    # Ensure coordinates are within [0, 1] range after normalization
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    normalized_width = max(0.0, min(1.0, normalized_width))
    normalized_height = max(0.0, min(1.0, normalized_height))


    return [x_center, y_center, normalized_width, normalized_height]

def scan_dataset_from_coco_json(json_path, images_base_path):
    """
    Scans dataset information from a COCO-formatted JSON file.

    Args:
        json_path (str): Path to the COCO-formatted JSON annotation file.
        images_base_path (str): Base path where image files are located.

    Returns:
        list: A list of tuples, where each tuple is
              (full_image_path, list_of_yolo_annotations, tuple_of_classes_in_image).
              list_of_yolo_annotations is a list of [class_id, x_c, y_c, w, h].
        list: A sorted list of all unique class IDs found.
        dict: A dictionary mapping class_id to class_name.
    """
    paired_files = []
    all_class_ids = set()
    class_id_to_name = {}

    print(f"Loading annotations from {json_path}...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Annotation JSON file not found at {json_path}. Please check the path.")
        return [], [], {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}. Please check the file format.")
        return [], [], {}
    except Exception as e:
        print(f"An unexpected error occurred while reading {json_path}: {e}")
        return [], [], {}


    # Build a map for quick access to image info (width, height, file_name)
    images_info = {}
    if 'images' in coco_data:
        for img in coco_data['images']:
            # Added checks for required keys
            if all(k in img for k in ('id', 'file_name', 'width', 'height')):
                 images_info[img['id']] = img
            else:
                 print(f"Warning: Skipping image entry with missing required keys: {img}")
    else:
        print("Error: JSON does not contain an 'images' key.") # Changed to Error as images are essential
        return [], [], {} # Exit early if no images section

    # Build a map for annotations grouped by image
    annotations_by_image = {}
    if 'annotations' in coco_data:
        for ann in coco_data['annotations']:
             # Added checks for required keys
             if all(k in ann for k in ('image_id', 'category_id', 'bbox')):
                img_id = ann['image_id']
                if img_id not in annotations_by_image:
                    annotations_by_image[img_id] = []
                annotations_by_image[img_id].append(ann)
             else:
                 print(f"Warning: Skipping annotation entry with missing required keys: {ann}")
    else:
        print("Warning: JSON does not contain an 'annotations' key.") # Warning, can proceed if images exist

    # Extract class names and collect all category IDs found in categories section
    if 'categories' in coco_data:
        for cat in coco_data['categories']:
            if 'id' in cat and 'name' in cat:
                class_id_to_name[cat['id']] = cat['name']
                # Ensure all category IDs from categories section are collected
                all_class_ids.add(cat['id'])
            else:
                print(f"Warning: Skipping category entry with missing keys: {cat}")
    else:
         print("Warning: JSON does not contain a 'categories' key. Class names will use IDs.")


    print("Processing images and annotations...")
    processed_image_count = 0
    # Iterate through images that have annotations or are listed in the 'images' section
    # Only process images that are listed in the 'images' section
    image_ids_to_process = set(images_info.keys())

    if not image_ids_to_process:
         print("Warning: No image IDs found in the 'images' section. Nothing to process.")
         return [], [], {}


    for img_id in sorted(list(image_ids_to_process)): # Process in a consistent order
        img_info = images_info[img_id]
        file_name = img_info.get('file_name')
        img_width = img_info.get('width')
        img_height = img_info.get('height')

        if not file_name or img_width is None or img_height is None:
             print(f"Warning: Image info missing required fields (file_name, width, or height) for ID {img_id}. Skipping.")
             continue

        full_image_path = os.path.join(images_base_path, file_name)

        if not os.path.exists(full_image_path):
            print(f"Warning: Image file not found: {full_image_path}. Skipping.")
            continue
        
        current_image_yolo_annotations = []
        current_image_class_ids = set()

        has_annotations_in_json = img_id in annotations_by_image

        if has_annotations_in_json:
            for ann in annotations_by_image[img_id]:
                # Check required annotation keys again (already done when building map, but safety)
                if all(k in ann for k in ('image_id', 'category_id', 'bbox')):
                    category_id = ann['category_id']
                    bbox_coco = ann['bbox'] # [x_top_left, y_top_left, width, height]

                    # Add category ID to the set of all found IDs if not already added from categories
                    # Ensure category_id is an integer
                    try:
                        category_id = int(category_id)
                        all_class_ids.add(category_id)
                    except ValueError:
                        print(f"Warning: Invalid category_id {category_id} for annotation in image {file_name}. Skipping annotation.")
                        continue


                    try:
                        # Ensure bbox is list or tuple of 4 numbers
                        if not isinstance(bbox_coco, (list, tuple)) or len(bbox_coco) != 4:
                            print(f"Warning: Invalid bbox format {bbox_coco} for annotation in image {file_name}. Skipping annotation.")
                            continue
                        
                        yolo_bbox = coco_bbox_to_yolo(bbox_coco, img_width, img_height)
                        
                        # Check for valid YOLO coordinates (should be between 0 and 1 if conversion is good)
                        # Allow small epsilon tolerance for floating point inaccuracies
                        epsilon = 1e-6
                        if all(0.0 - epsilon <= coord <= 1.0 + epsilon for coord in yolo_bbox):
                             current_image_yolo_annotations.append([category_id] + yolo_bbox)
                             current_image_class_ids.add(category_id)
                        else:
                             print(f"Warning: Converted YOLO bbox {yolo_bbox} is out of [0, 1] range for image {file_name} (bbox={bbox_coco}). Skipping annotation.")


                    except ValueError as e:
                        print(f"Warning: Skipping annotation in image {file_name} due to conversion error: {e}")
                        continue
                    except Exception as e:
                        print(f"Warning: Unexpected error processing annotation in image {file_name}: {e}. Skipping annotation.")
                        continue
                else:
                    print(f"Warning: Skipping annotation with missing required keys for image ID {img_id}: {ann}")


        # Only add images that have *at least one* valid annotation after parsing
        if current_image_yolo_annotations:
            paired_files.append((full_image_path, current_image_yolo_annotations, tuple(sorted(list(current_image_class_ids)))))
            processed_image_count += 1
        else:
            # Print warning if the image was found but had no valid annotations after processing
            # Only warn if the image *exists* and has annotations listed for it, but none were valid.
            # Or if it exists but has no annotations at all listed for it.
            if os.path.exists(full_image_path):
                 if has_annotations_in_json and img_id in annotations_by_image and annotations_by_image[img_id]:
                     # This warning indicates annotations were listed but none were valid/convertible
                     print(f"Warning: Image {full_image_path} listed in JSON but found no *valid* annotations after processing. Skipping for split.")
                 elif not has_annotations_in_json or img_id not in annotations_by_image or not annotations_by_image[img_id]:
                     # This warning indicates the image exists but has no annotations listed or found
                      print(f"Warning: Image {full_image_path} found, but has no annotations listed in JSON. Skipping for split.")


    print(f"Successfully processed annotations for {processed_image_count} images that will be included in the split.")
    # Sort to ensure consistent order for reproducibility before splitting
    paired_files.sort()

    return paired_files, sorted(list(all_class_ids)), class_id_to_name

# --- MODIFIED FUNCTION START ---
def create_output_structure(output_path):
    """Creates the necessary output directories: train/, val/, images/, labels/."""
    # Define paths for train and val root directories within the output path
    train_split_path = os.path.join(output_path, 'train')
    val_split_path = os.path.join(output_path, 'val')

    # Create train and val root directories
    os.makedirs(train_split_path, exist_ok=True)
    os.makedirs(val_split_path, exist_ok=True)

    # Create images and labels subdirectories directly under train and val
    train_images_path = os.path.join(train_split_path, 'images')
    train_labels_path = os.path.join(train_split_path, 'labels')
    val_images_path = os.path.join(val_split_path, 'images')
    val_labels_path = os.path.join(val_split_path, 'labels')

    os.makedirs(train_images_path, exist_ok=True)
    os.makedirs(train_labels_path, exist_ok=True)
    os.makedirs(val_images_path, exist_ok=True)
    os.makedirs(val_labels_path, exist_ok=True)

    # Return the paths to the 'train' and 'val' directories
    return train_split_path, val_split_path
# --- MODIFIED FUNCTION END ---


# --- MODIFIED FUNCTION START ---
def copy_files_and_create_json(file_list, destination_split_path, output_json_path, original_images_base_path):
    """
    Copies image files, creates YOLO .txt annotation files in separate subfolders
    (images/ and labels/ directly under destination_split_path), and generates a JSON index.

    Args:
        file_list (list): List of (full_image_path, list_of_yolo_annotations, class_tuple) tuples for this split.
        destination_split_path (str): Path to the output split directory (e.g., .../output_root/train or .../output_root/val).
        output_json_path (str): Path where the JSON index file will be saved (e.g., .../output_root/train/train.json).
        original_images_base_path (str): Base path of the original images to make paths relative.
    """
    json_data = {"images": []}

    # Determine the paths to the images/ and labels/ subfolders directly under destination_split_path
    dest_images_folder = os.path.join(destination_split_path, 'images')
    dest_labels_folder = os.path.join(destination_split_path, 'labels')
    # These should already exist from create_output_structure

    print(f"Copying files and creating JSON/TXTs to {destination_split_path}...")
    for img_path, yolo_annotations, _ in file_list:
        # Get relative path from the original images base path for metadata
        rel_img_path = os.path.relpath(img_path, original_images_base_path)
        
        # Determine new destination paths (maintaining original filename but in subfolders)
        img_filename = os.path.basename(img_path)
        ann_filename = os.path.splitext(img_filename)[0] + '.txt'

        dest_img_path = os.path.join(dest_images_folder, img_filename)
        dest_ann_path = os.path.join(dest_labels_folder, ann_filename)

        # Copy image file
        try:
            # Use copy2 to preserve metadata like timestamp if needed
            shutil.copy2(img_path, dest_img_path)
        except Exception as e:
            print(f"Error copying image {img_path} to {dest_img_path}: {e}. Skipping this image's annotations and JSON entry.")
            continue # Skip annotation writing and JSON entry if image copy fails

        # Create and write YOLO .txt annotation file
        # Write the file even if no annotations, but it will be empty
        try:
            with open(dest_ann_path, 'w') as f:
                for ann in yolo_annotations:
                    # Format: class_id x_center y_center width height
                    # Added formatting to ensure consistent floating point representation
                    f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")
        except Exception as e:
            print(f"Error writing annotation file {dest_ann_path}: {e}. JSON entry will show annotations, but file might be missing/incomplete.")
            # Do NOT clear yolo_annotations here if you want JSON to reflect what was parsed

        json_data["images"].append({
            # Use path relative to the split root folder (train/ or val/) in JSON
            "file_name": os.path.join('images', img_filename).replace('\\', '/'),
            "label_file_name": os.path.join('labels', ann_filename).replace('\\', '/'), # Add label file path for reference
            "original_relative_path": rel_img_path.replace('\\', '/'), # Keep original relative path for reference
            "annotations": [{"class_id": a[0], "bbox_yolo": a[1:]} for a in yolo_annotations]
        })

    try:
        with open(output_json_path, 'w') as f:
            json.dump(json_data, f, indent=4)
    except Exception as e:
         print(f"Error writing JSON summary file {output_json_path}: {e}")


def create_split_config(output_path, train_split_relative, val_split_relative, class_names_map=None, all_class_ids_found=None):
    """
    Creates a configuration file listing the train and validation data locations.
    Uses a format similar to YOLOv5 dataset.yaml for compatibility.

    Args:
        output_path (str): Root path of the output split (where dataset.yaml will be).
        train_split_relative (str): Relative path from output_path to the train folder (should be 'train').
        val_split_relative (str): Relative path from output_path to the val folder (should be 'val').
        class_names_map (dict, optional): Dictionary mapping class_id to class_name.
        all_class_ids_found (list, optional): List of all unique class IDs found during scan.
    """
    config_path = os.path.join(output_path, 'dataset.yaml')
    config_data = {
        # Path is the directory containing train/ and val/ folders (output_split_root)
        'path': './',
        'train': train_split_relative.replace('\\', '/'), # Should be 'train'
        'val': val_split_relative.replace('\\', '/'),   # Should be 'val'
        'note': 'Images are expected in images/ and labels in labels/ relative to train/val paths'
    }
    
    names_list = []
    if class_names_map:
        sorted_class_ids_from_map = sorted(class_names_map.keys())
        # Filter map to include only IDs actually found in annotations if possible, or just use the map?
        # Let's use the map for names, but nc should match the actual found IDs
        
        # Filter map to include only IDs actually found in annotations during scan
        filtered_class_names_map = {cid: name for cid, name in class_names_map.items() if cid in all_class_ids_found}
        
        if filtered_class_names_map:
             sorted_filtered_ids = sorted(filtered_class_names_map.keys())
             names_list = [filtered_class_names_map[cid] for cid in sorted_filtered_ids]
             config_data['nc'] = len(names_list) # nc is number of classes
             config_data['names'] = names_list
        elif all_class_ids_found: # Fallback if no names map provided or filter resulted in empty
             config_data['nc'] = len(all_class_ids_found)
             config_data['names'] = [f'class_{i}' for i in all_class_ids_found] # Placeholder names ordered by ID
        else:
             config_data['nc'] = 0
             config_data['names'] = []

    elif all_class_ids_found:
        config_data['nc'] = len(all_class_ids_found)
        config_data['names'] = [f'class_{i}' for i in all_class_ids_found]
    else:
        config_data['nc'] = 0
        config_data['names'] = []

    try:
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, sort_keys=False)
    except Exception as e:
        print(f"Error writing dataset.yaml config file {config_path}: {e}")


# --- Main Execution ---

if __name__ == "__main__":
    # --- REQUIRED CHANGES: SET YOUR PATHS HERE ---
    # 1. Set the base path where your IMAGE FILES are located.
    #    This should be the directory that contains the images mentioned in your JSON file_name entries.
    #    Example: If your JSON has "file_name": "image1.jpg" and the actual image is at E:\data\my_project\image1.jpg,
    #             then original_images_base_path should be r"E:\data\my_project".
    #    Based on your previous path:
    original_images_base_path = r"C:\Users\Alumno.DESKTOP-GV16N45.000\Downloads\Alldata\dataset\data"

    # 2. Set the path to your single JSON annotation file.
    #    Based on your previous path:
    annotation_json_path = r"C:\Users\Alumno.DESKTOP-GV16N45.000\Downloads\Alldata\dataset\labels.json" # <--- CHECK THIS PATH AND FILENAME

    # 3. Set the root directory where the 'train' and 'val' output folders will be created.
    #    This is the directory that will contain the 'train' and 'val' subfolders directly.
    output_split_root = r"C:\Users\Alumno.DESKTOP-GV16N45.000\Downloads\Alldata\output"
    # --- END OF REQUIRED CHANGES ---

    train_size = 0.8 # 80% for training, 20% for validation
    random_seed = 42 # For reproducibility of the split

    # 1. Scan the dataset information from the COCO JSON
    print(f"Scanning dataset from COCO JSON at {annotation_json_path}...")
    # paired_files contains tuples: (full_image_path, list_of_yolo_annotations, tuple_of_classes_in_image)
    paired_files, all_class_ids_found_during_scan, class_id_to_name_map = scan_dataset_from_coco_json(
        annotation_json_path, original_images_base_path
    )

    # Calculate total instance counts for each class across all images with valid annotations
    print("\nCalculating total instance counts across all images that will be included in the split...")
    total_class_counts = {class_id: 0 for class_id in all_class_ids_found_during_scan}
    for _, yolo_anns, _ in paired_files: # Use paired_files as it only contains images with valid annotations
        for ann in yolo_anns:
            class_id = ann[0] # class_id is the first element in the yolo_annotation list
            if class_id in total_class_counts: # Ensure class_id is one we're tracking
                total_class_counts[class_id] += 1
            # Else: The class_id wasn't in all_class_ids_found_during_scan initially, which is odd but possible


    print(f"\nFound {len(paired_files)} images with valid annotations that will be included in the split.")
    print(f"Found {len(all_class_ids_found_during_scan)} unique classes: {all_class_ids_found_during_scan}")
    if class_id_to_name_map:
        print("Class ID to Name Mapping found in JSON categories:")
        # Print sorted by ID for categories that were actually found in annotations
        sorted_found_class_ids = sorted(cid for cid in all_class_ids_found_during_scan if cid in class_id_to_name_map)
        if sorted_found_class_ids:
            for cid in sorted_found_class_ids:
                print(f"  ID {cid}: {class_id_to_name_map[cid]}")
        else:
             print("  (None of the found class IDs have names in the JSON categories section)")


    if not paired_files:
        print("\nNo images with corresponding annotation data found in the JSON or no images had valid annotations after processing. Exiting.")
        exit()

    # Prepare data for sklearn's train_test_split
    # X will be a list of (full_image_path, list_of_yolo_annotations) tuples
    # y will be the stratification key (tuple of sorted class IDs)
    items_for_split = [(item[0], item[1]) for item in paired_files] # X: tuples of (image_path, yolo_annotations)
    stratify_keys = [item[2] for item in paired_files] # y: stratification key (tuple of class IDs)

    # 2. Perform stratified split using sklearn.model_selection.train_test_split
    print(f"\nPerforming {train_size*100:.0f}%/{ (1-train_size)*100:.0f}% split using sklearn.")
    print("Note: This method stratifies by unique combinations of classes present in images,")
    print("      which is a good heuristic but may not perfectly balance instance counts for each class.")

    train_items = []
    val_items = []

    try:
        # Check if stratification is possible (at least 2 samples per key for test_size > 0)
        can_stratify = True
        if 1 - train_size > 0: # Only need multiple samples if splitting into train AND val
             key_counts = Counter(stratify_keys)
             if any(count < 2 for count in key_counts.values()):
                 can_stratify = False

        if can_stratify:
             train_items, val_items, _, _ = train_test_split(
                 items_for_split,
                 stratify_keys,
                 test_size=1-train_size,
                 random_state=random_seed,
                 shuffle=True
             )
        elif len(items_for_split) > 0:
             # Fallback to simple shuffle split if stratification is not feasible
             print("Warning: Not enough samples per stratification key for effective stratification.")
             print("Using simple shuffle split instead. Class distribution may be less balanced.")
             split_index = int(len(items_for_split) * train_size)
             # Ensure shuffling is reproducible even in fallback
             rng = np.random.RandomState(random_seed)
             shuffled_indices = rng.permutation(len(items_for_split))
             train_indices = shuffled_indices[:split_index]
             val_indices = shuffled_indices[split_index:]
             train_items = [items_for_split[i] for i in train_indices]
             val_items = [items_for_split[i] for i in val_indices]
        else:
             print("No items to split.")


    except ValueError as e:
        print(f"\nError during train_test_split (stratified): {e}")
        print("This can happen if a stratification key (combination of classes) has only one sample.")
        print("Consider removing images with very rare class combinations or using simple random split.")

        # Fallback to simple random split if stratified split fails
        print("Attempting simple shuffle split fallback...")
        try:
            if len(items_for_split) > 0:
                 split_index = int(len(items_for_split) * train_size)
                 rng = np.random.RandomState(random_seed) # Ensure reproducibility even in fallback
                 shuffled_indices = rng.permutation(len(items_for_split))
                 train_indices = shuffled_indices[:split_index]
                 val_indices = shuffled_indices[split_index:]
                 train_items = [items_for_split[i] for i in train_indices]
                 val_items = [items_for_split[i] for i in val_indices]
            else:
                 print("No items to split even with simple split.")
                 train_items = []
                 val_items = []
        except Exception as simple_split_e:
             print(f"Error during simple shuffle split fallback: {simple_split_e}")
             train_items = []
             val_items = [] # Ensure lists are empty if both splits fail


    # Reconstruct paired_files format for copying/json generation
    # Use image path as the unique key for lookup
    original_tuple_lookup_by_path = {item[0]: item for item in paired_files}

    train_paired_files = [original_tuple_lookup_by_path[item[0]] for item in train_items]
    val_paired_files = [original_tuple_lookup_by_path[item[0]] for item in val_items]


    print(f"\nFinal Train set size: {len(train_paired_files)} images")
    print(f"Final Validation set size: {len(val_paired_files)} images")


    # Optional: Verify class distribution (instance count breakdown) - Use the final split files
    print("\nVerifying class distribution (instance count breakdown) in FINAL sets:")
    final_train_class_counts = {class_id: 0 for class_id in all_class_ids_found_during_scan}
    final_val_class_counts = {class_id: 0 for class_id in all_class_ids_found_during_scan}

    for _, yolo_anns, _ in train_paired_files:
        for ann in yolo_anns:
            class_id = ann[0]
            if class_id in final_train_class_counts:
                 final_train_class_counts[class_id] += 1

    for _, yolo_anns, _ in val_paired_files:
        for ann in yolo_anns:
            class_id = ann[0]
            if class_id in final_val_class_counts:
                 final_val_class_counts[class_id] += 1

    print("Class ID | Total | Train | Val | Train % | Val %")
    print("--------|-------|-------|-----|---------|-------")
    # Use total from the count calculated after scan_dataset
    for class_id in sorted(all_class_ids_found_during_scan):
        total = total_class_counts.get(class_id, 0)
        train = final_train_class_counts.get(class_id, 0)
        val = final_val_class_counts.get(class_id, 0)
        train_percent = (train / total * 100) if total > 0 else 0
        val_percent = (val / total * 100) if total > 0 else 0
        # Check if total train+val equals original total (should be if no errors occurred and all images were included)
        status = "OK" if (train + val) == total else "MISMATCH" # Mismatch could indicate images skipped during scan
        print(f"{class_id:<8}| {total:<6}| {train:<6}| {val:<4}| {train_percent:<8.2f}| {val_percent:.2f} | {status}")


    # 3. Create output directory structure (train/, val/, images/, labels/)
    print(f"\nCreating output directory structure at {output_split_root}...")
    # These return paths to the 'train' and 'val' folders directly
    train_split_path, val_split_path = create_output_structure(output_split_root)

    # 4. Copy files and create JSONs (and .txt files in labels/)
    # Pass the paths to the 'train' and 'val' folders
    print("\nCopying train files to images/, creating YOLO .txts in labels/, and train.json...")
    train_json_path = os.path.join(train_split_path, 'train.json') # JSON path is now inside train/ or val/
    copy_files_and_create_json(train_paired_files, train_split_path, train_json_path, original_images_base_path)

    print("\nCopying validation files to images/, creating YOLO .txts in labels/, and val.json...")
    val_json_path = os.path.join(val_split_path, 'val.json') # JSON path is now inside train/ or val/
    copy_files_and_create_json(val_paired_files, val_split_path, val_json_path, original_images_base_path)

    # 5. Generate top-level config file (dataset.yaml)
    print("\nGenerating dataset config file (dataset.yaml)...")
    # These relative paths point to the 'train' and 'val' folders
    train_split_relative = 'train/images'
    val_split_relative = 'val/images'

    create_split_config(output_split_root, train_split_relative, val_split_relative,
                        class_names_map=class_id_to_name_map, all_class_ids_found=all_class_ids_found_during_scan)

    print("\nDataset splitting complete!")
    print(f"Train images: {os.path.join(train_split_path, 'images')}")
    print(f"Train labels: {os.path.join(train_split_path, 'labels')}")
    print(f"Validation images: {os.path.join(val_split_path, 'images')}")
    print(f"Validation labels: {os.path.join(val_split_path, 'labels')}")
    print(f"Train JSON summary: {train_json_path}")
    print(f"Validation JSON summary: {val_json_path}")
    print(f"Overall dataset config: {os.path.join(output_split_root, 'dataset.yaml')}")