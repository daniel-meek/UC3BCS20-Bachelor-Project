import sys
import time
import pandas as pd
import numpy as np

import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# === Model Section ===

def model_dtc(df, folder, args):

    # Set main dtc variables
    now = time.localtime()
    dtg = time.strftime("%y%m%d%H%M", now)
    random_state = args.random_state
    save_model = args.save_model
    save_report = args.save_report

    # --- Dataset processing ---
    print(f"\r[*] Processing dataset...".ljust(100), end="", flush=True)

    X = df.drop(['label'], axis=1)
    y = df['label']

    (X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=0.8, random_state=random_state)

    # --- DecisionTreeClassifier creation ---
    print(f"\r[*] Training DecisionTreeClassifier...".ljust(100), end="", flush=True)

    # Best performer after ranndom search with cross validation
    dtc = DecisionTreeClassifier(
        random_state=random_state,
        splitter='random',
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        max_depth=30,
        criterion='entropy',
        ccp_alpha=0
    )

    start_perf = time.perf_counter()
    start_cpu = time.process_time()
    dtc.fit(X_train, y_train)
    cpu_time = time.process_time() - start_cpu
    perf_time = time.perf_counter() - start_perf

    # --- Save the model ---
    num_features = dtc.n_features_in_
    num_classes = dtc.n_classes_
    dataset_rows = len(df)
    model_name = f"dtc_c{num_classes}_f{num_features:03d}_n{dataset_rows}_{dtg}"

    print(f"\r[+] Done! Model {model_name} trained in {cpu_time:.6f} s ({perf_time:.2f} s)".ljust(100), flush=True)

    if save_model:
        print(f"\r[*] Saving the model as {model_name}.pkl".ljust(100), end="", flush=True)
        with open(f"{folder}{model_name}.pkl", "wb") as file:
            pickle.dump(dtc, file)
    
    # --- Model report ---
    if save_report:
        print(f"\r[*] Saving the report as {model_name}_report.txt".ljust(100), end="", flush=True)

        # Get core statistics
        y_pred = dtc.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_text = classification_report(y_test, y_pred, digits=4)
        size_kb = sys.getsizeof(pickle.dumps(dtc)) / 1024
        param = dtc.get_params()
        
        # Get DT staisitcs
        depth = dtc.get_depth()
        leaves = dtc.get_n_leaves()
        
        # Process Feature Importances
        importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': dtc.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        # Write to _report.txt 
        report_filename = f"{folder}{model_name}_report.txt"
        with open(report_filename, 'w') as f:
            f.write(f"{'='*70}\n")
            f.write(f"Model Report: {model_name}\n")
            f.write(f"Model Creation: {time.strftime("%Y-%m-%d %H:%M", now)}\n")
            f.write(f"{'='*70}\n")

            f.write(f"\n--- CORE STATS ---\n")
            f.write(f"Features:      {num_features}\n")
            f.write(f"Accuracy:      {report["accuracy"]:.4f}\n")
            f.write(f"Precision:     {round(report['macro avg']['precision'], 4)}\n")
            f.write(f"Recall:        {round(report['macro avg']['recall'], 4)}\n")
            f.write(f"F1-score:      {round(report['macro avg']['f1-score'], 4)}\n")
            f.write(f"Training Time: {cpu_time:.6f} s\n")
            f.write(f"Model Size:    {size_kb:.6f} KB\n")

            f.write(f"\n--- DECISION TREE STATS ---\n")
            f.write(f"Depth:         {depth}\n")
            f.write(f"Leaves:        {leaves}\n")

            f.write(f"\n--- CLASSIFICATION REPORT ---\n")
            f.write(report_text)

            f.write(f"\n--- FEATURE IMPORTANCE  ---\n")
            f.write(f"{importances.to_string(index=False)}\n")
            unused = importances[importances['Importance'] == 0]['Feature'].tolist()
            f.write(f"\n--- UNUSED FEATURES ({len(unused)}) ---\n")
            f.write(", ".join(unused) if unused else "None")

            f.write(f"\n--- PARAMETERS ---\n")
            for p, v in param.items(): 
                f.write(f"{p:30}{v}\n")
        
    return dtc, cpu_time, model_name

def model_rfc(df, folder, args):
    
    # Set main dtc variables
    now = time.localtime()
    dtg = time.strftime("%y%m%d%H%M", now)
    random_state = args.random_state
    save_model = args.save_model
    save_report = args.save_report

    # --- Dataset processing ---
    print(f"\r[*] Processing dataset...".ljust(100), end="", flush=True)

    X = df.drop(['label'], axis=1)
    y = df['label']

    (X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=0.8, random_state=random_state)

    # --- RandomForestClassifier creation ---
    print(f"\r[*] Training RandomForestClassifier...".ljust(100), end="", flush=True)
    rfc = RandomForestClassifier(
        random_state=0,
        warm_start=True,
        n_estimators=100,
        min_samples_leaf=2,
        max_samples=0.5,
        max_leaf_nodes=None,
        max_features=0.1,
        max_depth=None,
        criterion='entropy',
        bootstrap=True,
        n_jobs=-1
    )
    
    start_perf = time.perf_counter()
    start_cpu = time.process_time()
    rfc.fit(X_train, y_train)
    cpu_time = time.process_time() - start_cpu
    perf_time = time.perf_counter() - start_perf

    # --- Save the model ---
    num_features = rfc.n_features_in_
    num_classes = rfc.n_classes_
    dataset_rows = len(df)
    model_name = f"rfc_c{num_classes}_f{num_features:03d}_n{dataset_rows}_{dtg}"

    print(f"\r[+] Done! Model {model_name} trained in {cpu_time:.6f} s ({perf_time:.2f} s)".ljust(100), flush=True)

    if save_model:
        print(f"\r[*] Saving the model as {model_name}.pkl".ljust(100), end="", flush=True)
        with open(f"{folder}{model_name}.pkl", "wb") as file:
            pickle.dump(rfc, file)

    # --- Model report ---
    if save_report:
        print(f"\r[*] Saving the report as {model_name}_report.txt".ljust(100), end="", flush=True)

        # Get core statistics
        y_pred = rfc.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_text = classification_report(y_test, y_pred, digits=4)
        size_kb = sys.getsizeof(pickle.dumps(rfc)) / 1024
        param = rfc.get_params()
        
        # Get model attributes
        param = rfc.get_params()
        depths = [tree.get_depth() for tree in rfc.estimators_]
        leaves = [tree.get_n_leaves() for tree in rfc.estimators_]
        trees = rfc.n_estimators
        avg_depth = np.average(depths)
        max_depth = np.max(depths)
        min_depth = np.min(depths)
        avg_leaves = np.average(leaves)
        max_leaves = np.max(leaves)
        min_leaves = np.min(leaves)
        
        # Process Feature Importances
        importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rfc.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        # Write to _report.txt 
        report_filename = f"{folder}{model_name}_report.txt"
        with open(report_filename, 'w') as f:
            f.write(f"{'='*70}\n")
            f.write(f"Model Report: {model_name}\n")
            f.write(f"Model Creation: {time.strftime("%Y-%m-%d %H:%M", now)}\n")
            f.write(f"{'='*70}\n")

            f.write(f"\n--- CORE STATS ---\n")
            f.write(f"Features:      {num_features}\n")
            f.write(f"Accuracy:      {report["accuracy"]:.4f}\n")
            f.write(f"Precision:     {round(report['macro avg']['precision'], 4)}\n")
            f.write(f"Recall:        {round(report['macro avg']['recall'], 4)}\n")
            f.write(f"F1-score:      {round(report['macro avg']['f1-score'], 4)}\n")
            f.write(f"Training Time: {cpu_time:.6f} s\n")
            f.write(f"Model Size:    {size_kb:.6f} KB\n")

            f.write(f"\n--- RANDOM FOREST STATS ---\n")
            f.write(f"Trees:         {trees}\n")
            f.write(f"Avg Depth:     {avg_depth:.2f}\n")
            f.write(f"Max Depth:     {max_depth}\n")
            f.write(f"Min Depth:     {min_depth}\n")
            f.write(f"Avg Leaves:    {avg_leaves:.2f}\n")
            f.write(f"Max Leaves:    {max_leaves}\n")
            f.write(f"Min Leaves:    {min_leaves}\n")

            f.write(f"\n--- CLASSIFICATION REPORT ---\n")
            f.write(report_text)

            f.write(f"\n--- FEATURE IMPORTANCE  ---\n")
            f.write(f"{importances.to_string(index=False)}\n")
            unused = importances[importances['Importance'] == 0]['Feature'].tolist()
            f.write(f"\n--- UNUSED FEATURES ({len(unused)}) ---\n")
            f.write(", ".join(unused) if unused else "None")

            f.write(f"\n--- PARAMETERS ---\n")
            for p, v in param.items(): 
                f.write(f"{p:30}{v}\n")
        
    return rfc, cpu_time, model_name

def model_lrc(df, folder, args):
    
    # Set main dtc variables
    now = time.localtime()
    dtg = time.strftime("%y%m%d%H%M", now)
    random_state = args.random_state
    save_model = args.save_model
    save_report = args.save_report

    # --- Dataset processing ---
    print(f"\r[*] Processing dataset...".ljust(100), end="", flush=True)

    X = df.drop(['label'], axis=1)
    y = df['label']

    (X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=0.8, random_state=random_state)

    # --- LogisticRegressionClasifier (LRC) creation ---
    print(f"\r[*] Training LogisticRegression...".ljust(100), end="", flush=True)
    lrc = LogisticRegression(
        random_state=random_state,
        n_jobs=-1,
        C=947259726504655,
        class_weight="balanced",
        max_iter=None,
        solver="saga"
    )

    start_perf = time.perf_counter()
    start_cpu = time.process_time()
    lrc.fit(X_train, y_train)
    cpu_time = time.process_time() - start_cpu
    perf_time = time.perf_counter() - start_perf

    # --- Save the model ---
    num_features = lrc.n_features_in_
    num_classes = len(lrc.classes_)
    dataset_rows = len(df)
    model_name = f"lrc_c{num_classes}_f{num_features:03d}_n{dataset_rows}_{dtg}"

    print(f"\r[+] Done! Model {model_name} trained in {cpu_time:.6f} s ({perf_time:.2f} s)".ljust(100), flush=True)

    if save_model:
        print(f"\r[*] Saving the model as {model_name}.pkl".ljust(100), end="", flush=True)
        with open(f"{folder}{model_name}.pkl", "wb") as file:
            pickle.dump(lrc, file)

    # --- Model report ---
    if save_report:
        print(f"\r[*] Saving the report as {model_name}_report.txt".ljust(100), end="", flush=True)

        # Get core statistics
        y_pred = lrc.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_text = classification_report(y_test, y_pred, digits=4)
        size_kb = sys.getsizeof(pickle.dumps(lrc)) / 1024
        param = lrc.get_params()

        # Get LR statistics

        # Process Feature Importances
        if lrc.coef_.ndim > 1:
            # Multiclass: average absolute value across all classes
            importance_values = np.mean(np.abs(lrc.coef_), axis=0).flatten()
        else:
            # Binary: just the absolute value
            importance_values = np.abs(lrc.coef_[0]).flatten()

        importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importance_values
        }).sort_values(by='Importance', ascending=False)

        report_filename = f"{folder}{model_name}_report.txt"
        with open(report_filename, 'w') as f:
            f.write(f"{'='*70}\n")
            f.write(f"Model Report: {model_name}\n")
            f.write(f"Model Creation: {time.strftime("%Y-%m-%d %H:%M", now)}\n")
            f.write(f"{'='*70}\n")

            f.write(f"\n--- CORE STATS ---\n")
            f.write(f"Features:      {num_features}\n")
            f.write(f"Accuracy:      {report["accuracy"]:.4f}\n")
            f.write(f"Precision:     {round(report['macro avg']['precision'], 4)}\n")
            f.write(f"Recall:        {round(report['macro avg']['recall'], 4)}\n")
            f.write(f"F1-score:      {round(report['macro avg']['f1-score'], 4)}\n")
            f.write(f"Training Time: {cpu_time:.6f} s\n")
            f.write(f"Model Size:    {size_kb:.6f} KB\n")

            f.write(f"\n--- LOGISTIC REGRESSION STATS ---\n")

            f.write(f"\n--- CLASSIFICATION REPORT ---\n")
            f.write(report_text)

            f.write(f"\n--- FEATURE IMPORTANCE  ---\n")
            f.write(f"{importances.to_string(index=False)}\n")
            unused = importances[importances['Importance'] == 0]['Feature'].tolist()
            f.write(f"\n--- UNUSED FEATURES ({len(unused)}) ---\n")
            f.write(", ".join(unused) if unused else "None")

            f.write(f"\n--- PARAMETERS ---\n")
            for p, v in param.items(): 
                f.write(f"{p:30}{v}\n")

    return lrc, cpu_time, model_name

def model_fnn(df, folder, args):

    # Set main variables
    now = time.localtime()
    dtg = time.strftime("%y%m%d%H%M", now)
    random_state = args.random_state
    save_model = args.save_model
    save_report = args.save_report

    # --- Dataset processing ---
    print(f"\r[*] Processing dataset...".ljust(100), end="", flush=True)

    X = df.drop(['label'], axis=1)
    y = df['label']

    (X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=0.8, random_state=random_state)

    # --- FeedforwardNeuralNetwork (FNN) creation ---
    print(f"\r[*] Training Feedforward Neural Network...".ljust(100), end="", flush=True)
    
    fnn = MLPClassifier(
        random_state=random_state,
        hidden_layer_sizes=(64, 64, 64), 
        learning_rate_init=0.001,
        alpha=0.0001,
        max_iter=500, 
        early_stopping=True,
        validation_fraction=0.1,
        activation='tanh',
        solver='adam'
    )
    
    start_perf = time.perf_counter()
    start_cpu = time.process_time()
    fnn.fit(X_train, y_train) # .fit() takes ~2h for the 1M dataset...
    cpu_time = time.process_time() - start_cpu
    perf_time = time.perf_counter() - start_perf

    # --- Save the model ---
    num_features = fnn.n_features_in_
    num_classes = len(fnn.classes_)
    dataset_rows = len(df)
    model_name = f"fnn_c{num_classes}_f{num_features:03d}_n{dataset_rows}_{dtg}"

    print(f"\r[+] Done! Model {model_name} trained in {cpu_time:.6f} s ({perf_time:.2f} s)".ljust(100), flush=True)

    if save_model:
        print(f"\r[*] Saving the model as {model_name}.pkl".ljust(100), end="", flush=True)
        with open(f"{folder}{model_name}.pkl", "wb") as file:
            pickle.dump(fnn, file)

    # --- Model report ---
    if save_report:
        print(f"\r[*] Saving the report as {model_name}_report.txt".ljust(100), end="", flush=True)

        # Get core statistics
        y_pred = fnn.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_text = classification_report(y_test, y_pred, digits=4)
        size_kb = sys.getsizeof(pickle.dumps(fnn)) / 1024
        param = fnn.get_params()
        
        # Get FNN staisitcs
        hidden_layers = fnn.hidden_layer_sizes
        iter = fnn.n_iter_
        activation = fnn.activation    

        # Write to _report.txt 
        report_filename = f"{folder}{model_name}_report.txt"
        with open(report_filename, 'w') as f:
            f.write(f"{'='*70}\n")
            f.write(f"Model Report: {model_name}\n")
            f.write(f"Model Creation: {time.strftime("%Y-%m-%d %H:%M", now)}\n")
            f.write(f"{'='*70}\n")

            f.write(f"\n--- CORE STATS ---\n")
            f.write(f"Features:      {num_features}\n")
            f.write(f"Accuracy:      {report["accuracy"]:.4f}\n")
            f.write(f"Precision:     {round(report['macro avg']['precision'], 4)}\n")
            f.write(f"Recall:        {round(report['macro avg']['recall'], 4)}\n")
            f.write(f"F1-score:      {round(report['macro avg']['f1-score'], 4)}\n")
            f.write(f"Training Time: {cpu_time:.6f} s\n")
            f.write(f"Model Size:    {size_kb:.6f} KB\n")

            f.write(f"\n--- FEEDFORWARD NEURAL NETWORK STATS ---\n")
            f.write(f"Hidden Layers: {hidden_layers}\n")
            f.write(f"Iterations:    {iter}\n")
            f.write(f"Activation:    {activation}\n")

            f.write(f"\n--- CLASSIFICATION REPORT ---\n")
            f.write(report_text)

            f.write(f"\n--- PARAMETERS ---\n")
            for p, v in param.items(): 
                f.write(f"{p:30}{v}\n")
            
    return fnn, cpu_time, model_name
