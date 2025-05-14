from argparse import ArgumentParser
from ultralytics import YOLO


def arguments():
    parser = ArgumentParser(
        description="Validation of YOLO model")
    parser.add_argument("--model", type=str,
                        help="Path to model .pt file")
    parser.add_argument("--cfg", type=str,
                        help="Path to the cfg definining train options)")
    parser.add_argument("--dataset", type=str,
                        help="Path to the root folder of the database")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    
    args = arguments()
    # Load a model
    model = YOLO(args.model)
    # Use the model
    model.val(data= args.dataset,
              cfg=args.cfg)  # val the model
