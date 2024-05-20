import fiftyone as fo
from argparse import ArgumentParser
from ultralytics import YOLO
import fiftyone.utils.ultralytics as fou

def load_yolo_dataset(dataset_dir,split):
    dataset = fo.Dataset.from_dir(dataset_type=fo.types.YOLOv5Dataset,dataset_dir= dataset_dir,yaml_path=f"{dataset_dir}/dataset.yaml",split=split)
    dataset.compute_metadata()
    return dataset


def arguments():
    parser=ArgumentParser(description="Visualize UDIT NN/CV project database")
    parser.add_argument("--path","--p",type=str,help="Path to the root folder of the database")
    parser.add_argument("--weights","--w",type=str,help="Path to the model weights file(.pt)")
    args=parser.parse_args()
    return args

if __name__ == "__main__":

    args = arguments()

    dataset = load_yolo_dataset(args.path,"val")
    # Load a model
    model = YOLO(args.weights)  # build a new model from scratch
    
    for sample in dataset.iter_samples(progress=True):
        result = model.predict(sample.filepath)[0]
        sample["predictions"] = fou.to_detections(result)
        sample.save()             
    session = fo.launch_app(dataset)
    session.wait()
