from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO(r"E:\UDIT\UDIT_SDC\UDIT_test\UDIT_test2\weights\best.pt")  # build a new model from scratch
    # Use the model
    model.val(data = r"E:\UDIT\SDC_dataset\dataset.yaml",cfg=r"E:\UDIT\UDIT_SDC\cfg\default.yaml")  # train the model