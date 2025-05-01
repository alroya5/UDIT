from argparse import ArgumentParser

from ultralytics import YOLO

def arguments():
    parser = ArgumentParser(
        description="Train YOLO model")
    parser.add_argument("--model", type=str,
                        help="Path to model yaml or .pt file")
    parser.add_argument("--cfg", type=str,
                        help="Path to the cfg definining train options)")
    parser.add_argument("--dataset", type=str,
                        help="Path to the root folder of the database")
    args = parser.parse_args()

    return args
if __name__ == '__main__':
    
    args = arguments()
    # Load a model
    model = YOLO(args.model)  # build a new model from scratch
    # Use the model
    model.train(data = args.dataset ,cfg=args.cfg)  # train the model
