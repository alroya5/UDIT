from argparse import ArgumentParser
import fiftyone as fo

def  load_coco_dataset(dataset_dir):
    # Load COCO formatted dataset
    coco_dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path= f"{dataset_dir}/data",
        labels_path= f"{dataset_dir}/labels.json",
        include_id=True,
    )
    coco_dataset.compute_metadata()
    return coco_dataset

def load_yolo_dataset(dataset_dir,split):
    dataset = fo.Dataset.from_dir(dataset_type=fo.types.YOLOv5Dataset,dataset_dir= dataset_dir,yaml_path=f"{dataset_dir}/dataset.yaml",split=split)
    return dataset

def arguments():
    parser=ArgumentParser(description="Visualize UDIT NN/CV project dataset")
    parser.add_argument("--path","--p",type=str,help="Path to the root folder of the dataset")
    parser.add_argument("--format","--f",type=str,choices=["yolo","coco"],help="Format of the dataset")
    parser.add_argument("--split","--s",required=False,type=str,choices=["train","val","test"],help="Split of the dataset (only for YOLO format)")
    args=parser.parse_args()
    return args

if __name__ == "__main__":

    args = arguments()

    if args.format =="coco":
     dataset = load_coco_dataset(args.path)
    elif args.format =="yolo":
        dataset = load_yolo_dataset(args.path,args.split)

    session = fo.launch_app(dataset)
    session.wait()
