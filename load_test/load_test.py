import time
import psutil
import numpy as np
import pandas as pd
import pickle
import argparse
import sys
import warnings
import os
import matplotlib.pyplot as plt

from matplotlib.ticker import StrMethodFormatter

"""
When running on the target system, use:
    taskset -c 0 python3 load_test.py --all --data dataset_packet_multiclass8_n10000.csv 
    OR
    taskset -c 0 python3 load_test.py --model dtc_load_tester.pkl --data dataset_packet_multiclass8_n10000.csv --name dtc --graph
"""

# Fix for the "Feature Names" Warning...
warnings.filterwarnings("ignore", category=UserWarning)

# === Global config ===
PRE_TEST_DURATION = 10
POST_TEST_DURATION = 10
TEST_DURATION = 40
TARGET_RATES = [5, 5000, 5000000]
MONITOR_INTERVAL = 0.1  # Record stats every X seconds

class StatsContainer:
    # Container to hold the counter for simulation.
    def __init__(self):
        self.rows_processed_cumulative = 0

class GlobalStats:
    # Container to hold summary stats for all models.
    def __init__(self):
        self.records = []

    def add_record(self, model_name, target_rate, avg_cpu, avg_rows, num_features):
        self.records.append({
            "Model": model_name,
            "Num_Features": num_features,
            "Target_Rate": target_rate,
            "Avg_CPU": avg_cpu,
            "Avg_Rows_Per_Sec": avg_rows
        })

    def save_summary(self, filename="all_models_summary.csv"):
        if not self.records:
            print("No records to save.")
            return
        df = pd.DataFrame(self.records)
        df.to_csv(filename, index=False)
        print(f"\n[+] Global summary saved to {filename}")

class SystemMonitor:
    # 
    def __init__(self, output_prefix, stats_container):
        self.stats = []
        self.output_prefix = output_prefix
        self.stats_container = stats_container
        self.process = psutil.Process()
        self.process.cpu_percent() # Prime the counter

    def capture_snapshot(self, status):
        # Records a single data point
        # interval=None is non-blocking
        cpu = self.process.cpu_percent(interval=None) 
        current_rows = self.stats_container.rows_processed_cumulative
        
        self.stats.append({
            "timestamp": time.time(),
            "cpu_usage": cpu,
            "cumulative_rows": current_rows,
            "status": status  # Fourth column: 'buffer' or 'load_test'
        })

    def _get_dataframe_with_rates(self):
        df = pd.DataFrame(self.stats)
        # Calculate instant rows per second
        df['rows_per_sec_instant'] = df['cumulative_rows'].diff() / df['timestamp'].diff()
        df['rows_per_sec_instant'] = df['rows_per_sec_instant'].fillna(0)
        return df

    def save_to_csv(self, rate_label):
        filename = f"{self.output_prefix}_rate_{rate_label}.csv"
        df = self._get_dataframe_with_rates()
        df.to_csv(filename, index=False)
        print(f"\n[+] Stats saved to {filename}".ljust(100))

    def get_avg_stats(self):
        """Returns (avg_cpu, avg_rows_per_sec) for the 'load_test' phase only."""
        df = self._get_dataframe_with_rates()
        
        # Filter for only the actual load test portion
        mask = df['status'] == 'load_test'
        load_df = df[mask]
        
        if load_df.empty:
            return 0.0, 0.0
            
        avg_cpu = load_df['cpu_usage'].mean()
        avg_rows = load_df['rows_per_sec_instant'].mean()
        
        return avg_cpu, avg_rows

def load_full_dataframe(data_path):
    # Load data once to keep in memory
    print(f"Loading full dataset from: {data_path}")
    try:
        df = pd.read_csv(data_path, engine='pyarrow', index_col=0)
        
        # Drop "label" if exists, generic cleanup
        if "label" in df.columns:
            df.drop(columns=["label"], inplace=True)
            
        return df

    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)

def get_model_specific_data(model, full_df):
    required_features = model.feature_names_in_

    # Subset the dataframe
    df_subset = full_df[required_features]
    return df_subset.to_numpy(), len(required_features)

def prepare_batch(data, batch_size):
    # Make sure there is enough data for the batch
    if len(data) >= batch_size:
        return data[:batch_size]
    else:
        repeat_factor = (batch_size // len(data)) + 1
        large_data = np.tile(data, (repeat_factor, 1))
        return large_data[:batch_size]

def run_buffer_period(duration, monitor):
    # Monitor CPU before and after the load test
    start = time.time()
    last_tick = start
    while (time.time() - start) < duration:
        now = time.time()
        if now - last_tick >= MONITOR_INTERVAL:
            monitor.capture_snapshot(status="buffer") # Pass status
            last_tick = now
        time.sleep(0.01) # Small sleep to prevent busy waiting during buffer

def run_load_test(model, source_data, target_rate, output_prefix):
    print(f"\n[*] Starting Test: Target {target_rate} rows/s...")
    
    stats = StatsContainer()
    monitor = SystemMonitor(output_prefix, stats)

    # Batch size    
    desired_latency = 0.05 
    calculated_batch = int(target_rate * desired_latency)
    
    batch_size = max(1, min(5000, calculated_batch))

    input_batch = prepare_batch(source_data, batch_size)

    # Pre-Test Buffer
    print(f"  > Pre-test buffer ({PRE_TEST_DURATION}s)...".ljust(100), end=f"\r")
    run_buffer_period(PRE_TEST_DURATION, monitor)
    
    # Load test
    print(f"\r  > Running Load ({TEST_DURATION}s)...".ljust(100), end=f"\r")
    start_time = time.time()
    last_monitor_time = start_time
    
    local_row_count = 0
    
    while (time.time() - start_time) < TEST_DURATION:
        model.predict(input_batch)
        
        local_row_count += batch_size
        stats.rows_processed_cumulative = local_row_count
        
        current_time = time.time()
        elapsed = current_time - start_time

        if (current_time - last_monitor_time) >= MONITOR_INTERVAL:
            monitor.capture_snapshot(status="load_test") # Pass status
            last_monitor_time = current_time
        
        expected_rows = elapsed * target_rate
        
        if local_row_count > expected_rows:
            sleep_needed = (local_row_count - expected_rows) / target_rate
            
            if sleep_needed > 0:
                time.sleep(sleep_needed)

    # Post-test buffer
    print(f"\r  > Post-test buffer ({POST_TEST_DURATION}s)...".ljust(100), end=f"")
    run_buffer_period(POST_TEST_DURATION, monitor)
    
    monitor.save_to_csv(target_rate)
    
    real_speed = local_row_count / TEST_DURATION
    print(f"[+] Finished. Avg Speed: {real_speed:.2f} rows/s (Target: {target_rate})")

    # Return the monitor so we can extract averages later
    return monitor

def plot_results(prefix):
    # Plot the graph if --graph is true
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15), sharex=False)
    fig.suptitle(f'Load Test Visualization: {prefix}', fontsize=16)
    
    found_files = False
    
    for i, rate in enumerate(TARGET_RATES):
        filename = f"{prefix}_rate_{rate}.csv"
        ax1 = axes[i]
        
        if not os.path.exists(filename):
            ax1.text(0.5, 0.5, f"File not found: {filename}", 
                     ha='center', va='center', transform=ax1.transAxes)
            continue
            
        found_files = True
        print(f"Plotting {filename}...".ljust(100), end=f"\r")
        
        df = pd.read_csv(filename)
        if df.empty:
            continue

        start_time = df['timestamp'].iloc[0]
        df['relative_time'] = df['timestamp'] - start_time
        
        # --- Plot CPU ---
        color_cpu = 'tab:red'
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('CPU Usage (%)', color=color_cpu, fontweight='bold')
        l1 = ax1.plot(df['relative_time'], df['cpu_usage'], color=color_cpu, label='CPU', linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color_cpu)
        ax1.set_ylim(0, 105)
        ax1.grid(True, alpha=0.3)
        
        # --- Plot Rows/Sec ---
        ax2 = ax1.twinx()
        color_rows = 'tab:blue'
        ax2.set_ylabel('Rows / Sec', color=color_rows, fontweight='bold')
        ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
        l2 = ax2.plot(df['relative_time'], df['rows_per_sec_instant'], color=color_rows, label='Rows/Sec', linewidth=2, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color_rows)
        
        ax1.set_title(f"Target Rate: {rate} rows/s", fontsize=12, pad=10, loc='left')
        
        # Highlight active test section
        if 'status' in df.columns:
            active_df = df[df['status'] == 'load_test']
            if not active_df.empty:
                start_active = active_df['relative_time'].min()
                end_active = active_df['relative_time'].max()
                ax1.axvspan(start_active, end_active, color='gray', alpha=0.1, label='Active Test Phase')
        else:
            ax1.axvspan(PRE_TEST_DURATION, PRE_TEST_DURATION + TEST_DURATION, color='gray', alpha=0.1, label='Active Test Phase')

        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')

    if found_files:
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        output_filename = f"{prefix}_visualization.png"
        plt.savefig(output_filename)
        print(f"\rSuccess! Graph saved to: {output_filename}")
    else:
        print("\nNo CSV files found :(.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Search for all .pkl files")
    parser.add_argument("--model", help="Path to .pkl model file")
    parser.add_argument("--data", required=True, help="Path to input CSV data")
    parser.add_argument("--name", default="test_results", help="Prefix for output CSV files")
    parser.add_argument("--graph", action="store_true", help="Save the results to a .png file")
    
    args = parser.parse_args()

    if args.all and (args.graph or args.model):
        print("--all cannot be used with --graph or --model. Exiting...")
        sys.exit(1)

    # Load full dataset once to memory
    full_df = load_full_dataframe(args.data)

    # run load test for all models in the folder
    if args.all:
        
        global_stats = GlobalStats() # Initialize stats collector

        for root, dirs, files in os.walk("."):
            for file in sorted(files): # makes the list of files sorted
                if file.endswith(".pkl"):
                    model_path = os.path.join(root, file)
                    model_name = os.path.splitext(file)[0]
                    
                    try:
                        print(f"Found: {model_path}")
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                    
                    except Exception as e:
                        print(f"Error reading {file}: {e}")
                        continue 

                    # Dynamically subset data for this specific model
                    data_subset, num_features = get_model_specific_data(model, full_df)

                    try:
                        for rate in TARGET_RATES:
                            monitor_result = run_load_test(model, data_subset, rate, f"{model_name}")
                            
                            # Calculate and store averages
                            avg_cpu, avg_rows = monitor_result.get_avg_stats()
                            global_stats.add_record(model_name, rate, avg_cpu, avg_rows, num_features)
                
                    except KeyboardInterrupt:
                        print("\nTest stopped by user.")
                        # Save what we have so far
                        global_stats.save_summary() # Save the data aquired if aborted by the user
                        sys.exit(0)

                    except Exception as e:
                        print(f"An error occurred during test: {e}")
        
        # Save the combined summary after all models
        global_stats.save_summary()
        sys.exit(0)

    # Run test for selected model    
    else:
        # Load single model
        print(f"Loading model from: {args.model}")
        with open(args.model, 'rb') as f:
            model = pickle.load(f)

        # Get data subset for this model
        data_subset, _ = get_model_specific_data(model, full_df)

        if data_subset is not None:
            try:
                for rate in TARGET_RATES:
                    monitor = run_load_test(model, data_subset, rate, args.name)
                    
                if args.graph:
                    plot_results(args.name)
                
            except KeyboardInterrupt:
                print("\nTest stopped by user.")
                sys.exit(0)

            except Exception as e:
                print(f"An error occurred: {e}")
                sys.exit(1)

        sys.exit(0)
