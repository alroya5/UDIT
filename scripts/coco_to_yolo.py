import os
import json
import shutil
import random
from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path


def coco_to_yolo(coco_dir, output_dir, test_ratio=0.15, val_ratio=0.1):
    # Crear las carpetas necesarias para YOLOv5
    images_dir = Path(output_dir) / 'data'  # Actualizado para la carpeta 'data'
    labels_dir = Path(output_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Obtener los archivos JSON de las anotaciones
    annotation_file = os.path.join(coco_dir, 'labels.json')

    with open(annotation_file) as f:
        coco_data = json.load(f)

    # Obtener las clases del dataset COCO
    classes = {category['id']: category['name'] for category in coco_data['categories']}

    # Obtener las imágenes del dataset COCO
    images = {image['id']: image for image in coco_data['images']}

    # Preparar la lista de imágenes
    image_ids = [image['id'] for image in coco_data['images']]
    random.shuffle(image_ids)

    # Dividir en entrenamiento, validación y prueba
    num_images = len(image_ids)
    test_size = int(num_images * test_ratio)
    val_size = int(num_images * val_ratio)
    train_size = num_images - test_size - val_size

    # Crear las listas de imágenes
    train_images = image_ids[:train_size]
    val_images = image_ids[train_size:train_size+val_size]
    test_images = image_ids[train_size+val_size:]

    # Función para procesar y guardar las imágenes y etiquetas
    def save_image_and_labels(image_id, split):
        image_info = images[image_id]
        image_filename = image_info['file_name']

        # Modificar la ruta de las imágenes para que apunten a 'train/train/data'
        image_path = os.path.join(coco_dir, 'data', image_filename)  # Actualizado aquí
        
        if not os.path.exists(image_path):
            print(f"¡Advertencia! La imagen {image_filename} no existe en el directorio especificado.")
            return
        
        # Copiar imagen a la carpeta correspondiente
        output_image_dir = os.path.join(images_dir, split)
        os.makedirs(output_image_dir, exist_ok=True)
        shutil.copy(image_path, os.path.join(output_image_dir, image_filename))

        # Crear archivo de etiquetas en formato YOLO
        annotations = [
            ann for ann in coco_data['annotations'] if ann['image_id'] == image_id
        ]
        
        label_filename = image_filename.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(labels_dir, split, label_filename)
        os.makedirs(os.path.dirname(label_path), exist_ok=True)

        with open(label_path, 'w') as label_file:
            for annotation in annotations:
                category_id = annotation['category_id']
                bbox = annotation['bbox']
                x_center = (bbox[0] + bbox[2] / 2) / image_info['width']
                y_center = (bbox[1] + bbox[3] / 2) / image_info['height']
                width = bbox[2] / image_info['width']
                height = bbox[3] / image_info['height']
                
                # Escribir en formato YOLO (class_id, x_center, y_center, width, height)
                label_file.write(f"{category_id} {x_center} {y_center} {width} {height}\n")

    # Guardar imágenes y etiquetas para cada conjunto de datos (train, val, test)
    for split, image_ids_split in zip(['train', 'val', 'test'], [train_images, val_images, test_images]):
        for image_id in tqdm(image_ids_split, desc=f"Processing {split} set"):
            save_image_and_labels(image_id, split)

    print(f"Dataset convertido a YOLOv5 y guardado en: {output_dir}")


def main():
    parser = ArgumentParser(description="Convertir dataset COCO a YOLOv5")
    parser.add_argument("coco_dir", type=str, help="Ruta al directorio del dataset COCO")
    parser.add_argument("output_dir", type=str, help="Ruta al directorio de salida")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Proporción de datos para el conjunto de prueba")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Proporción de datos para el conjunto de validación")
    args = parser.parse_args()

    coco_to_yolo(args.coco_dir, args.output_dir, test_ratio=args.test_ratio, val_ratio=args.val_ratio)


if __name__ == "__main__":
    main()

