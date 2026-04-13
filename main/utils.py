import os
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import StrMethodFormatter

# === Util Section ==

def info(args):
    use_csv = args.csv
    
    data_type = args.dataset
    classification = args.classification
    rows = args.rows
    random_state = args.random_state

    save_model = args.save_model
    save_report = args.save_report
    save_dataset = args.save_dataset

    use_dtc = args.dtc
    use_rfc = args.rfc
    use_lrc = args.lrc
    use_fnn = args.fnn

    use_rfe = args.rfe
    min_features = args.min_features
    cut_features = args.cut_features

    use_graph = args.graph

    print(f"=== tmp_main.py info() ===")
    
    print(f"--- Dataset Info: ---")
    if use_csv:
        print(f"Using dataset {use_csv}")
    else:
        print(f"Creating dataset from ./{args.path}")
        print(f"   Datatype:   {data_type}")
        print(f"   Classification: {classification}")
        print(f"   Total Rows: {rows}")
        print(f"   Random State: {random_state}")
    
    print(f"\n--- Model Info: ---")
    print(f"DecisionTreeClassifier:        {use_dtc}")
    print(f"RandomForestClassifier:        {use_rfc}")
    print(f"LogisticRegressionClassifier:  {use_lrc}")
    print(f"FeedforwardNeuralNetwork:      {use_fnn}")
    print(f"Recursive Feature Elimination: {use_rfe}")
    if use_rfe:
        print(f"  - RFE Proxy Model: {args.rfe_proxy}")
        print(f"  - Min Features:    {min_features}")
        print(f"  - Cut Features:    {cut_features}")
    
    print(f"\n--- Save Info: ---")
    print(f"Save Model:   {save_model}")
    print(f"Save Report:  {save_report}")
    if not args.csv:
        print(f"Save Dataset: {save_dataset}")
    print(f"Save Graph:   {use_graph}")

    print(f"{"="*26}\n")

# === Graphing Section ===

def graph_acc_size_time_vs_features(root_folder, args):

    # Dataset labeling
    dataset = args.dataset.capitalize()
    
    if args.classification == "binary":
        classification = "Binary"
    elif args.classification == "multiclass8":
        classification = "8 Classes"
    elif args.classification == "multiclass28":
        classification = "28 Classes"
    
    rows = args.rows

    # Graphing
    plt.figure(figsize=(12, 7))

    metrics = ['accuracy', 'training_time', 'size_in_KB']
    titles = ['Model Accuracy', 'Training Time (s) (log)', 'Model Size (KB) (log)']

    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    for root, dirs, files in os.walk(root_folder):
        for filename in files:
            if filename.endswith('_results.csv'):
                path = os.path.join(root, filename)
                df = pd.read_csv(path).sort_values(by='num_features')
                model_label = filename.split('_')[0].upper()

                # Plot each metric in its own dedicated subplot
                for i, metric in enumerate(metrics):
                    axes[i].plot(df['num_features'], df[metric], 
                                marker='', label=f'{model_label}', 
                                linewidth=2)
                    axes[i].set_ylabel(titles[i])
                    axes[i].grid(True, linestyle='--', alpha=0.5)

    # Formatting for all subplots
    for ax in axes:
        ax.invert_xaxis() 
        ax.legend(loc='best', fontsize='small', ncol=2)

    axes[0].set_ylim(0.85, 1.0) # Zoom
    axes[1].set_yscale("log")
    axes[1].yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    axes[2].set_yscale("log")
    axes[2].yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    axes[2].set_xlabel('Number of Features after RFE')
    plt.suptitle(f'Model Performance Metrics during RFE\n({dataset} Based - {classification} - {rows})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{root_folder}results.pdf")
