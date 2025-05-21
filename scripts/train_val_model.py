from ultralytics import YOLO, settings
import mlflow

def main():
    # Set valid MLflow tracking URI
    mlflow.set_tracking_uri("file:///C:/Users/Alumno.DESKTOP-GV16N45.000/AppData/UDIT/runs/mlflow")

    # Enable logging
    settings.update({"tensorboard": True, "mlflow": True})

    model = YOLO("yolov8n.pt")

    model.train(
        data=r"C:\Users\Alumno.DESKTOP-GV16N45.000\AppData\output\dataset.yaml",
        cfg=r"C:\Users\Alumno.DESKTOP-GV16N45.000\AppData\UDIT\cfg\default.yaml",
        epochs=100,
        imgsz=640,
        name="yolov8n_custom_train"
    )

if __name__ == "__main__":
    main()