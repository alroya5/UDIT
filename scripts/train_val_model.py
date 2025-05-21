# from ultralytics import YOLO

# # Load the YOLOv8n model (pre-trained or from scratch)
# model = YOLO(r"C:\Users\Alumno.DESKTOP-GV16N45.000\AppData\UDIT\cfg\model\yolov8.yaml")  # Path to your pre-trained or custom YOLOv8n model

# model.train(data = r"C:\Users\Alumno.DESKTOP-GV16N45.000\AppData\output\dataset.yaml", epochs = 100, imgsz = 640)

from ultralytics import YOLO

def main():
    # Load the YOLOv8n model
    model = YOLO(r"C:\Users\Alumno.DESKTOP-GV16N45.000\AppData\UDIT\cfg\model\yolov8.yaml")

    # Train the model
    model.train(
        data=r"C:\Users\Alumno.DESKTOP-GV16N45.000\AppData\output\dataset.yaml",
        epochs=100,
        imgsz=640,
        name="yolov8n_custom_train"
    )

if __name__ == "__main__":
    # import multiprocessing
    # multiprocessing.freeze_support()  # Safe for frozen executables
    main()
