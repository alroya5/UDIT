import os
import shutil
import json
import numpy as np
from sklearn.model_selection import train_test_split
import yaml # For saving dataset_split.yaml

def coco_bbox_to_yolo(bbox, img_width, img_height):
    """
    Converts COCO bbox format [x_top_left, y_top_left, width, height]
    to YOLO format [x_center, y_center, normalized_width, normalized_height].
    """
    x_tl, y_tl, bbox_width, bbox_height = bbox

    x_center = (x_tl + bbox_width / 2) / img_width
    y_center = (y_tl + bbox_height / 2) / img_height
    
    normalized_width = bbox_width / img_width
    normalized_height = bbox_height / img_height
    
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
    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # Build a map for quick access to image info (width, height, file_name)
    images_info = {img['id']: img for img in coco_data['images']}

    # Build a map for annotations grouped by image
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    # Extract class names
    for cat in coco_data['categories']:
        class_id_to_name[cat['id']] = cat['name']
        all_class_ids.add(cat['id']) # Ensure all categories are collected

    print("Processing images and annotations...")
    for img_id, img_info in images_info.items():
        file_name = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        full_image_path = os.path.join(images_base_path, file_name)

        if not os.path.exists(full_image_path):
            print(f"Warning: Image file not found: {full_image_path}. Skipping.")
            continue
        
        current_image_yolo_annotations = []
        current_image_class_ids = set()

        if img_id in annotations_by_image:
            for ann in annotations_by_image[img_id]:
                category_id = ann['category_id']
                bbox_coco = ann['bbox'] # [x_top_left, y_top_left, width, height]

                try:
                    yolo_bbox = coco_bbox_to_yolo(bbox_coco, img_width, img_height)
                    current_image_yolo_annotations.append([category_id] + yolo_bbox)
                    current_image_class_ids.add(category_id)
                except Exception as e:
                    print(f"Warning: Error converting bbox {bbox_coco} for image {file_name}: {e}. Skipping annotation.")
                    continue

        if current_image_yolo_annotations: # Only add images that have annotations
            paired_files.append((full_image_path, current_image_yolo_annotations, tuple(sorted(list(current_image_class_ids)))))
        else:
            print(f"Warning: Image {full_image_path} has no valid annotations. Skipping for split.")

    # Sort to ensure consistent order for reproducibility before splitting
    paired_files.sort() 
    
    return paired_files, sorted(list(all_class_ids)), class_id_to_name

def create_output_structure(output_path):
    """Creates the necessary output directories."""
    train_data_path = os.path.join(output_path, 'train', 'Data')
    val_data_path = os.path.join(output_path, 'val', 'Data')

    os.makedirs(train_data_path, exist_ok=True)
    os.makedirs(val_data_path, exist_ok=True)

    return train_data_path, val_data_path

def copy_files_and_create_json(file_list, destination_data_path, output_json_path, original_images_base_path):
    """
    Copies image files, creates YOLO .txt annotation files, and generates a JSON index.

    Args:
        file_list (list): List of (full_image_path, list_of_yolo_annotations, class_tuple) tuples for this split.
        destination_data_path (str): Path to the Data folder in the output split directory.
        output_json_path (str): Path where the JSON index file will be saved.
        original_images_base_path (str): Base path of the original images to make paths relative.
    """
    json_data = {"images": []}

    for img_path, yolo_annotations, _ in file_list:
        # Get relative path from the original images base path for metadata
        rel_img_path = os.path.relpath(img_path, original_images_base_path)
        
        # Determine new destination paths (maintaining original filename)
        img_filename = os.path.basename(img_path)
        ann_filename = os.path.splitext(img_filename)[0] + '.txt'

        dest_img_path = os.path.join(destination_data_path, img_filename)
        dest_ann_path = os.path.join(destination_data_path, ann_filename)

        # Copy image file
        try:
            shutil.copy2(img_path, dest_img_path)
        except Exception as e:
            print(f"Error copying image {img_path}: {e}. Skipping this image.")
            continue

        # Create and write YOLO .txt annotation file
        try:
            with open(dest_ann_path, 'w') as f:
                for ann in yolo_annotations:
                    # Format: class_id x_center y_center width height
                    f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")
        except Exception as e:
            print(f"Error writing annotation file {dest_ann_path}: {e}. Skipping annotation for JSON.")
            yolo_annotations = [] # Clear annotations if write failed for JSON

        json_data["images"].append({
            "file_name": img_filename, 
            "original_relative_path": rel_img_path,
            "annotations": [{"class_id": a[0], "bbox_yolo": a[1:]} for a in yolo_annotations]
        })

    with open(output_json_path, 'w') as f:
        json.dump(json_data, f, indent=4)

def create_split_config(output_path, train_data_relative, val_data_relative, class_names_map=None, all_class_ids_found=None):
    """
    Creates a configuration file listing the train and validation data locations.
    Uses a format similar to YOLOv5 dataset.yaml for compatibility.

    Args:
        output_path (str): Root path of the output split.
        train_data_relative (str): Relative path from output_path to train data.
        val_data_relative (str): Relative path from output_path to val data.
        class_names_map (dict, optional): Dictionary mapping class_id to class_name.
        all_class_ids_found (list, optional): List of all unique class IDs found during scan.
    """
    config_path = os.path.join(output_path, 'dataset_split.yaml') 
    config_data = {
        'path': './', 
        'train': train_data_relative,
        'val': val_data_relative,
    }
    
    names_list = []
    if class_names_map:
        # Ensure names are ordered by ID
        for cid in sorted(class_names_map.keys()):
            names_list.append(class_names_map[cid])
        config_data['nc'] = len(names_list)
        config_data['names'] = names_list
    elif all_class_ids_found: # Fallback if no names map is provided
        config_data['nc'] = len(all_class_ids_found)
        config_data['names'] = [f'class_{i}' for i in all_class_ids_found] # Placeholder names
    else:
        config_data['nc'] = 0 
        config_data['names'] = []

    with open(config_path, 'w') as f:
        yaml.dump(config_data, f, sort_keys=False)

# --- Main Execution ---

if __name__ == "__main__":
    # --- REQUIRED CHANGES ---
    # 1. Set the base path where your IMAGE FILES are located.
    #    If your JSON refers to "image1.jpg" and the file is in E:\Uni\MaterIA\dataset\images\image1.jpg,
    #    then this path would be r"E:\Uni\MaterIA\dataset\images".
    original_images_base_path = r"E:\Uni\MaterIA\Asignaturas\Computer vision\dataset\train\data" 
    
    # 2. Set the path to your single JSON annotation file.
    annotation_json_path = r"E:\Uni\MaterIA\Asignaturas\Computer vision\dataset\train\labels.json" # <--- CHANGE THIS
    
    # 3. Set the root directory where the 'train' and 'val' output folders will be created.
    output_split_root = r"E:\Uni\MaterIA\Asignaturas\Computer vision\dataset\Val" 
    # --- END OF REQUIRED CHANGES ---

    train_size = 0.8 # 80% for training, 20% for validation
    random_seed = 42 # For reproducibility of the split

    # 1. Scan the dataset information from the COCO JSON
    print(f"Scanning dataset from COCO JSON at {annotation_json_path}...")
    paired_files, all_class_ids_found_during_scan, class_id_to_name_map = scan_dataset_from_coco_json(
        annotation_json_path, original_images_base_path
    )
    print(f"Found {len(paired_files)} images with valid annotations for splitting.")
    print(f"Found {len(all_class_ids_found_during_scan)} unique classes: {all_class_ids_found_during_scan}")

    if not paired_files:
        print("No images with corresponding annotation data found in the JSON. Exiting.")
        exit()

    # Prepare data for sklearn's train_test_split
    # X will be a list of (full_image_path, list_of_yolo_annotations) tuples
    # y will be the stratification key (tuple of sorted class IDs)
    items_for_split = [(item[0], item[1]) for item in paired_files] # Exclude class_tuple from X
    stratify_keys = [item[2] for item in paired_files] # Use the tuple of sorted class IDs for stratification

    # 2. Perform stratified split using sklearn.model_selection.train_test_split
    print(f"\nPerforming {train_size*100:.0f}%/{ (1-train_size)*100:.0f}% split using sklearn.")
    print("Note: This method stratifies by unique combinations of classes present in images,")
    print("      which is a good heuristic but may not perfectly balance instance counts for each class.")
    
    train_items, val_items, _, _ = train_test_split(
        items_for_split, # X: tuples of (image_path, yolo_annotations)
        stratify_keys,   # y: stratification key (tuple of class IDs)
        test_size=1-train_size, 
        random_state=random_seed, 
        shuffle=True 
    )
    
    # Reconstruct paired_files format for copying/json generation
    train_paired_files = [(img_path, yolo_anns, class_tuple) for (img_path, yolo_anns), class_tuple in zip(train_items, stratify_keys)]
    val_paired_files = [(img_path, yolo_anns, class_tuple) for (img_path, yolo_anns), class_tuple in zip(val_items, stratify_keys)]

    print(f"\nTrain set size: {len(train_paired_files)} images")
    print(f"Validation set size: {len(val_paired_files)} images")

    # Optional: Verify class distribution (instance count breakdown)
    print("\nVerifying class distribution (instance count breakdown):")
    train_class_counts = {class_id: 0 for class_id in all_class_ids_found_during_scan}
    val_class_counts = {class_id: 0 for class_id in all_class_ids_found_during_scan}
    total_class_counts = {class_id: 0 for class_id in all_class_ids_found_during_scan}

    # Count for all images in original dataset
    for _, yolo_anns, _ in paired_files:
        for ann in yolo_anns:
            class_id = ann[0] # class_id is the first element in the yolo_annotation list
            total_class_counts[class_id] = total_class_counts.get(class_id, 0) + 1

    # Count for train images
    for _, yolo_anns, _ in train_paired_files:
        for ann in yolo_anns:
            class_id = ann[0]
            train_class_counts[class_id] = train_class_counts.get(class_id, 0) + 1

    # Count for validation images
    for _, yolo_anns, _ in val_paired_files:
        for ann in yolo_anns:
            class_id = ann[0]
            val_class_counts[class_id] = val_class_counts.get(class_id, 0) + 1

    print("Class ID | Total | Train | Val | Train % | Val %")
    print("--------|-------|-------|-----|---------|-------")
    for class_id in sorted(all_class_ids_found_during_scan):
        total = total_class_counts.get(class_id, 0)
        train = train_class_counts.get(class_id, 0)
        val = val_class_counts.get(class_id, 0)
        train_percent = (train / total * 100) if total > 0 else 0
        val_percent = (val / total * 100) if total > 0 else 0
        print(f"{class_id:<8}| {total:<6}| {train:<6}| {val:<4}| {train_percent:<8.2f}| {val_percent:.2f}")


    # 3. Create output directory structure
    print(f"\nCreating output directory structure at {output_split_root}...")
    train_data_path, val_data_path = create_output_structure(output_split_root)

    # 4. Copy files and create JSONs (and .txt files)
    print("Copying train files, creating YOLO .txts, and train.json...")
    train_json_path = os.path.join(output_split_root, 'train', 'train.json')
    copy_files_and_create_json(train_paired_files, train_data_path, train_json_path, original_images_base_path)

    print("Copying validation files, creating YOLO .txts, and val.json...")
    val_json_path = os.path.join(output_split_root, 'val', 'val.json')
    copy_files_and_create_json(val_paired_files, val_data_path, val_json_path, original_images_base_path)

    # 5. Generate top-level config file (dataset_split.yaml)
    print("Generating dataset split config file (dataset_split.yaml)...")
    train_data_relative = os.path.join('train', 'Data')
    val_data_relative = os.path.join('val', 'Data')

    create_split_config(output_split_root, train_data_relative, val_data_relative, 
                        class_names_map=class_id_to_name_map, all_class_ids_found=all_class_ids_found_during_scan)

    print("\nDataset splitting complete!")
    print(f"Train images & YOLO .txts: {train_data_path}")
    print(f"Validation images & YOLO .txts: {val_data_path}")
    print(f"Train JSON summary: {train_json_path}")
    print(f"Validation JSON summary: {val_json_path}")
    print(f"Overall split config: {os.path.join(output_split_root, 'dataset_split.yaml')}")