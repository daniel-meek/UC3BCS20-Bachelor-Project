import os
import time
import argparse

from utils import info, graph_acc_size_time_vs_features
from data_loader import load_csv, create_binary_packet_df, create_multiclass_8_packet_df, create_multiclass_28_packet_df, create_binary_flow_df, create_multiclass_8_flow_df, create_multiclass_28_flow_df
from models import model_dtc, model_rfc, model_lrc, model_fnn
from features import rfe

# === Main Section ===

def get_args():
    
    parser = argparse.ArgumentParser()
    
    # Dataset settings
    parser.add_argument("--csv", type=str, help="The root Dataset directory")
    parser.add_argument("--path", type=str, default="Dataset", help="The root Dataset directory")

    parser.add_argument("-d", "--dataset", choices=["packet", "flow"], default="packet", help="Select the dataset type <flow or packet>")
    parser.add_argument("-c", "--classification", choices=["binary", "multiclass8", "multiclass28"], default="binary", help="Select the classification type")

    parser.add_argument("-n", "--rows", type=int, help="Total rows per dataset. Omitting this creates ballanced datasets")

    # Global random state
    parser.add_argument("--random_state", type=int, default=0, help="Sets the random state for reproducibility")

    # Model settings
    parser.add_argument("--save_model", action="store_true", help="Saves the trained models")
    parser.add_argument("--save_report", action="store_true", help="Saves the model report")
    parser.add_argument("--save_dataset", action="store_true", help="Saves the dataset")
    parser.add_argument("--verbose_results", action="store_true", help="Adds aditional model spesific data to the results")

    parser.add_argument("--dtc", action="store_true", help="Use Decision Tree Classifier")
    parser.add_argument("--rfc", action="store_true", help="Use Random Forest Classifier")
    parser.add_argument("--lrc", action="store_true", help="Use Logistic Regression Classifier")
    parser.add_argument("--fnn", action="store_true", help="Use Feedforward Neural Network")

    # Feature selection settings
    parser.add_argument("-r", "--rfe", action="store_true", help="Perform Recursive Feature Elimination")
    parser.add_argument("--min_features", type=int, default=1, help="Sets the cut-off point for RFE")
    parser.add_argument("--cut_features", type=int, default=1, help="Sets the number of features to cut")
    parser.add_argument("--rfe_proxy", choices=["dtc", "rfc", "lrc"], default="dtc", help="Selects the proxy model for models that dont support RFE")

    # Graphing settings
    parser.add_argument("--graph", action="store_true", help="Graps the results to .pdf")

    args = parser.parse_args()

    # arg validation
    
    # Check if .csv file is present
    if args.csv != None:     
        if not os.path.exists(args.csv):
            parser.error(f"[!]{args.csv} not found! Exiting...")

    return args

def main():
    # Load args and global variables
    args = get_args()
    root = f"models_{time.strftime('%y%m%d%H%M')}/"
    dtg = time.strftime("%y%m%d%H%M")

    # Make the root folder
    if not os.path.exists(root):
        os.makedirs(root)

    # Print run info to screen
    info(args)

    # === Dataset Section: ===
    if args.csv:
        # Load the .csv file if selected
        df = load_csv(args.csv)

    else:
        # Create appropriate DataFrame from path="./Dataset"
        dtg = time.strftime("%y%m%d%H%M")

        if args.dataset == "packet":        
            if args.classification == "binary":
                df = create_binary_packet_df(args, root)
            elif args.classification == "multiclass8":
                df = create_multiclass_8_packet_df(args, root)
            elif args.classification == "multiclass28":
                df = create_multiclass_28_packet_df(args, root)

        elif args.dataset == "flow":
            if args.classification == "binary":
                df = create_binary_flow_df(args, root)
            elif args.classification == "multiclass8":
                df = create_multiclass_8_flow_df(args, root)
            elif args.classification == "multiclass28":
                df = create_multiclass_28_flow_df(args, root)
    
    # === Model Section: ===

    if args.dtc:
        print(f"\n[*] === DecisionTreeClassifier ===")
        if args.rfe:
            rfe(df.copy(), "DecisionTreeClassifier", args, root)
        else:
            # Perfrom model training if not RFE
            folder = f"{root}dt_{dtg}/"
            if not os.path.exists(folder):
                os.makedirs(folder)
            model_dtc(df.copy(), folder, args)

    if args.rfc:
        print(f"\n[*] === RandomForestClassifier ===")
        if args.rfe:
            rfe(df.copy(), "RandomForestClassifier", args, root)
        else:
            # Perfrom model training if not RFE
            folder = f"{root}rf_{dtg}/"
            if not os.path.exists(folder):
                os.makedirs(folder)
            model_rfc(df.copy(), folder, args)

    if args.lrc:
        print(f"\n[*] === LogisticRegressionClassifier ===")
        if args.rfe:
            rfe(df.copy(), "LogisticRegressionClassifier", args, root)
        else:
            # Perfrom model training if not RFE
            folder = f"{root}lr_{dtg}/"
            if not os.path.exists(folder):
                os.makedirs(folder)
            model_lrc(df.copy(), folder, args)
    
    if args.fnn:
        print(f"\n[*] === FeedforwardNeuralNetwork ===")
        if args.rfe:
            rfe(df.copy(), "FeedforwardNeuralNetwork", args, root)
        else:
            # Perfrom model training if not RFE
            folder = f"{root}fnn_{dtg}/"
            if not os.path.exists(folder):
                os.makedirs(folder)
            model_fnn(df.copy(), folder, args)
    
    # === Graph Section ===
    if args.graph:
        # Graphing can only be done with rfe or sfs results
        if args.rfe or args.sfs:
            graph_acc_size_time_vs_features(root, args)
        else:
            print(f"[!] '--graph' set, but there are no results to graph!")

    return

if __name__ == "__main__":
    main()