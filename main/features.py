import os
import sys
import time
import pandas as pd
import numpy as np

import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


from sklearn.metrics import classification_report
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFE as RecursiveFeatureElimination

from models import model_dtc, model_rfc, model_lrc, model_fnn

# === Feature Selection Section ===

def rfe(df, model_type, args, root):

    # Folder creation
    name_map = {
        "DecisionTreeClassifier": "dtc",
        "RandomForestClassifier": "rfc",
        "LogisticRegressionClassifier": "lrc",
        "FeedforwardNeuralNetwork": "fnn",
    }

    dtg = time.strftime("%y%m%d%H%M")
    prefix = name_map.get(model_type, "NaN")
    folder = f"{root}{prefix}_{dtg}/"
    os.makedirs(folder, exist_ok=True)

    results_list = []
    dropped_features_history = []
    iteration = 1
    min_features = args.min_features
    current_feature_count = len(df.columns) - 1

    start_time = time.perf_counter()
    
    while (len(df.columns) - 1 >= min_features):
        print(f"[*] Training model iteration #{iteration} ({current_feature_count} features)".ljust(100), end="", flush=True)

        # Train appropriate model
        match model_type:
            case "DecisionTreeClassifier":
                model, training_time, model_name = model_dtc(df, folder, args)
            case "RandomForestClassifier":
                model, training_time, model_name = model_rfc(df, folder, args)
            case "LogisticRegressionClassifier":
                model, training_time, model_name = model_lrc(df, folder, args)
            case "FeedforwardNeuralNetwork":
                model, training_time, model_name = model_fnn(df, folder, args)
            case _:
                raise ValueError(f"[!] Unknown model type: {model_type}")

        # Get stats for results 
        results = {}
            
        X = df.drop(columns=['label'])
        y = df['label']
        y_pred = model.predict(X)
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        class_ids = [k for k in report.keys() if k.isdigit() or isinstance(k, int)]

        results['precision_list'] = [report[c]['precision'] for c in class_ids]
        results['recall_list']    = [report[c]['recall'] for c in class_ids]
        results['f1_list']        = [report[c]['f1-score'] for c in class_ids]
        
        size_kb = sys.getsizeof(pickle.dumps(model)) / 1024

        match model_type:
            case "DecisionTreeClassifier":
                results_list.append({
                    'model_name': f"{model_name}",
                    'num_features': current_feature_count,
                    'accuracy': round(report['accuracy'], 4),
                    'precision': round(report['macro avg']['precision'], 4),
                    'recall': round(report['macro avg']['recall'], 4),
                    'f1-score': round(report['macro avg']['f1-score'], 4),
                    'training_time': training_time,
                    'size_in_KB': round(size_kb, 4),
                })
                if args.verbose_results:
                    results_list.append({
                        'depth': results['depth'],
                        'leaves': results['leaves'],
                        'precision_per_class': results['precision_list'],
                        'recall_per_class': results['recall_list'],
                        'f1-score_per_class': results['f1_list']
                    })
            case "RandomForestClassifier":
                results_list.append({
                    'model_name': f"{model_name}",
                    'num_features': current_feature_count,
                    'accuracy': round(report['accuracy'], 4),
                    'precision': round(report['macro avg']['precision'], 4),
                    'recall': round(report['macro avg']['recall'], 4),
                    'f1-score': round(report['macro avg']['f1-score'], 4),
                    'training_time': training_time,
                    'size_in_KB': round(size_kb, 4),
                })
                if args.verbose_results:
                    results_list.append({
                        'trees': results['trees'],
                        'avg_depth': results['avg_depth'],
                        'max_depth': results['max_depth'],
                        'min_depth': results['min_depth'],
                        'avg_leaves': results['avg_leaves'],
                        'max_leaves': results['max_leaves'],
                        'min_leaves': results['min_leaves'],
                        'precision_per_class': results['precision_list'],
                        'recall_per_class': results['recall_list'],
                        'f1-score_per_class': results['f1_list']
                    })
                    
            case "LogisticRegressionClassifier":
                results_list.append({
                    'model_name': f"{model_name}",
                    'num_features': current_feature_count,
                    'accuracy': round(report['accuracy'], 4),
                    'precision': round(report['macro avg']['precision'], 4),
                    'recall': round(report['macro avg']['recall'], 4),
                    'f1-score': round(report['macro avg']['f1-score'], 4),
                    'training_time': training_time,
                    'size_in_KB': round(size_kb, 4),
                })
                # if args.verbose_results:
                    
            case "FeedforwardNeuralNetwork":
                results_list.append({
                    'model_name': f"{model_name}",
                    'num_features': current_feature_count,
                    'accuracy': round(report['accuracy'], 4),
                    'precision': round(report['macro avg']['precision'], 4),
                    'recall': round(report['macro avg']['recall'], 4),
                    'f1-score': round(report['macro avg']['f1-score'], 4),
                    'training_time': training_time,
                    'size_in_KB': round(size_kb, 4),
                })
                # if args.verbose_results:
                    
        # Perform RFE with model feedback
        cut = args.cut_features
        df_no_label = df.drop(columns=['label'])

        # For models with no native feature importances
        if not hasattr(model, 'feature_importances_') and not hasattr(model, 'coef_'):
            # Select the proxy model
            if args.rfe_proxy == "dtc":
                proxy = DecisionTreeClassifier(
                    random_state=args.random_state
                )

            elif args.rfe_proxy == "rfc":
                proxy = RandomForestClassifier(
                    random_state=args.random_state, 
                    max_samples=0.1, 
                    n_jobs=-1
                )
                
            elif args.rfe_proxy == "lrc":
                proxy = LogisticRegression(
                    random_state=args.random_state
                    #n_jobs=-1 # n_jobs will be removed in 1.10 > remove
                )
            
            # Train the proxy model to the whole dataset
            df_label = df['label']            
            proxy.fit(df_no_label, df_label)
            
            # If proxy has 'feature_importances_' (dtc, rfc, xgb)
            if hasattr(proxy, 'feature_importances_'):
                importances = pd.Series(proxy.feature_importances_, index=df_no_label.columns)
            
            # If proxy has 'coef_' (lrc)
            else:
                if proxy.coef_.ndim > 1:
                    weights = np.mean(np.abs(proxy.coef_), axis=0).flatten()
                else:
                    weights = np.abs(proxy.coef_).flatten()
                importances = pd.Series(weights, index=df_no_label.columns)


        # For models with .feature_importances_ (dtc, rfc, xgb)
        elif hasattr(model, 'feature_importances_'):
            importances = pd.Series(model.feature_importances_, index=df_no_label.columns)

        # For models with .coef_ (lrc)
        elif hasattr(model, 'coef_'):
            if model.coef_.ndim > 1:
                weights = np.mean(np.abs(model.coef_), axis=0)
            else:
                weights = np.abs(model.coef_)
            importances = pd.Series(weights, index=df_no_label.columns)

        sorted_importances = importances.sort_values()
        max_allowable_cut = current_feature_count - min_features
        actual_cut_amount = min(cut, max_allowable_cut)

        if actual_cut_amount <= 0:
            print(f"\r[!] Reached minimum feature threshold ({min_features}). Stopping.")
            break
        
        cols_to_drop = sorted_importances.head(actual_cut_amount).index.tolist()
        
        for col in cols_to_drop:
            dropped_features_history.append({
                'feature': col,
                'importance_at_drop': importances[col],
                'iteration_dropped': iteration,
                'rank': current_feature_count  # Higher rank = less important
            })
        
        zero_importance_cols = [c for c in cols_to_drop if importances[c] == 0]
        lowest_scoring_cols = [c for c in cols_to_drop if importances[c] > 0]

        # Drop the least important columns
        df.drop(columns=cols_to_drop, inplace=True)

        print(f"\r[+] Recursive Feature Elimination Done!".ljust(100), flush=True)
        if zero_importance_cols:
            print(f"  - Dropped {len(zero_importance_cols)} unused features (0.0)")
        if lowest_scoring_cols:
            pairs = [f"{col} ({importances[col]:.6f})" for col in lowest_scoring_cols]
            print(f"  - Dropped {len(lowest_scoring_cols)} weak features: {', '.join(pairs)}")
            
        # Iterate the counters
        current_feature_count = len(df.columns) - 1
        iteration += 1
    
    surving_features = df.drop(columns=['label']).columns
    for i, col in enumerate(surving_features):
        dropped_features_history.append({
            'feature': col,
            'importance_at_drop': "Survived",
            'iteration_dropped': "None",
            'rank': i + 1 
        })

    processing_time = time.perf_counter() - start_time

    # Save results.csv
    print(f"[+] Recursive Feature Elimination complete! Time: {processing_time:.2f}s".ljust(100))
    y = df['label']
    num_class = len(df['label'].unique())
    num_rows = len(df)
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(f"{folder}{prefix}_c{num_class}_n{num_rows}_results.csv", index=False)
    print(f"[+] Summary report saved to {folder}{prefix}_c{num_class}_n{num_rows}_results.csv")

    # Save feature_results.csv
    feature_report_df = pd.DataFrame(dropped_features_history)
    feature_report_df = feature_report_df.sort_values(by='rank')
    feature_report_df.to_csv(f"{folder}{prefix}_c{num_class}_n{num_rows}_cut{cut}_features.csv", index=False)

    return

