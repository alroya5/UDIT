from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolov8m.yaml")  # build a new model from scratch
    # Use the model
    model.train(data = r"E:\UDIT\SDC_dataset\dataset.yaml",cfg=r"E:\UDIT\UDIT_SDC\cfg\default.yaml")  # train the model
