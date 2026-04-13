import os
import sys
import time
import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler

# === Dataset Section ===

def load_csv(path):

    print(f"\r[*] Loading {path}...".ljust(100), end="", flush=True)
    df = pd.read_csv(path, index_col=0)
    print(f"\r[+] Sucessfully loaded {path}!".ljust(100), flush=True)

    return df

def create_class_df(label, file):
    
    dfs = []
    start_perf = time.perf_counter()
    start_cpu = time.process_time()

    data_name = file.rsplit("/", 1)[1]

    first_file_path = f"{file}.csv"
    if os.path.exists(first_file_path):
        sample = pd.read_csv(first_file_path, nrows=0)
        all_cols = sample.columns.tolist()

        # Columns to drop up front:
        # Packet Based features: 
        #   - steram, mac, and ip are unrealistic
        #   - http_host is only 'none'
        #   - handshake_cipersuites is only '-1' 
        #   - device_mac is an identifier
        #   - eth_src and eth_dst are identifiers
        # Flow Based features:
        #   - Flow ID, Src IP, Dst IP are unrealistic
        #   - Label is added later in the dataset processing
        #   - Timestamp cannot be encoded?
        
        drop_cols = [
            "stream", "src_mac", "dst_mac", "src_ip", "dst_ip", "http_host", "handshake_ciphersuites", "device_mac", "eth_src_oui", "eth_dst_oui",
            "Flow ID", "Src IP", "Dst IP", "Label", "NeedManualLabel", "Timestamp"
        ]
        keep_cols = [col for col in all_cols if col not in drop_cols]
    else:
        print(f"[!] Error: {file} file not found.")
        sys.exit(1)

    i = 0
    while True:
        # Handle .csv naming, first file has no integer
        suffix = "" if i == 0 else str(i)
        file_path = f"{file}{suffix}.csv"

        print(f"\r[*] Processing {file_path.rsplit("/", 1)[-1]}....".ljust(100), end="", flush=True)
        
        # Check if the file exists before trying to read it
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, usecols=keep_cols, engine='pyarrow')
            dfs.append(df)
            i += 1
    
        else:
            # No more files for this subtype, break the while loop
            break

    if not dfs:
        print(f"No files found for path: {file}")
        return None

    # Combine everything into one dataframe
    if len(dfs) > 1:
        print(f"\r[*] Concatenating {len(dfs)} dataframes...".ljust(100), end="", flush=True)

        df = pd.concat(dfs, ignore_index=True)
    else:
        df = dfs[0]
    df["label"] = label

    # Memory optimization: Downcast numeric types, turns 64-bit numbers into 32-bit where possible
    fcols = df.select_dtypes('float').columns
    icols = df.select_dtypes('integer').columns
    df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
    df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')

    # Measure and print how long df creation took
    cpu_time = time.process_time() - start_cpu
    perf_time = time.perf_counter() - start_perf

    print(f"\r[+] {data_name} Done! Processing took: {cpu_time:.6f} s ({perf_time:.2f} s)".ljust(100))

    return df

def combine_class_dfs(dfs_list, args, root, save_dataset):

    entries = args.rows
    random_state = args.random_state
    
    if entries is None:
        # A "balanced" approach where the dataset is created using the maximum number of rows availabe to any subset.
        entries_per_df = min(len(df) for df in dfs_list)
        entries = entries_per_df * len(dfs_list)
    elif entries < len(dfs_list):
        # Ensure there is always 1 entry per class
        entries_per_df = len(dfs_list)
        entries = entries_per_df * len(dfs_list)
    else:
        entries_per_df = round(entries / len(dfs_list))
    
    main_df = pd.DataFrame()
    processed_list = []

    # Start time of the main function to measure how long the concat + preprocessing takes
    start_perf = time.perf_counter()
    start_cpu = time.process_time()

    # Main concat function
    for i, df in enumerate(dfs_list):
        print(f"\r[*] Processing dataframe #{i}...".ljust(100), end="", flush=True)

        # min() ensures we dont crash if a DF has fewer rows than requested
        n_samples = min(len(df), entries_per_df)
        if (n_samples < entries_per_df):
            print(f"\r[!] Not enough rows in class {df["label"].iloc[0]}, using all {n_samples} rows.".ljust(100), flush=True)

        processed_list.append(df.sample(n=n_samples, random_state=random_state))

    print(f"\r[*] Concatenating dataframes...".ljust(100), end="", flush=True)
    main_df = pd.concat(processed_list, ignore_index=True)

    # Preprocessing: Encoding and Normalization
    target_col = "label"
    features = main_df.columns.drop(target_col) if target_col in main_df.columns else main_df.columns

    # Replace all NaN values with 0
    print(f"\r[*] Filling NaN values with 0...".ljust(100), end="", flush=True)
    main_df = main_df.fillna(0)

    # Encoding using OrdinalEncoder()
    object_cols = main_df[features].select_dtypes(include=["object", "category"]).columns
    if not object_cols.empty:
        print(f"\r[*] Encoding non-integer values...".ljust(100), end="", flush=True)

        try:
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            main_df[object_cols] = encoder.fit_transform(main_df[object_cols])
        except Exception as e:
            print(f"\r[!] Error: {e}.ljust(100)")

    # Normalization using MinMaxScaler()
    try:
        print(f"\r[*] Normalizing values to range [0, 1]...".ljust(100), end="", flush=True)

        # Replace all Inf values with NaN so that the MinMaxScaler can fit all values to [0,1]
        main_df[features] = main_df[features].replace([np.inf, -np.inf], np.nan)
        
        scaler = MinMaxScaler()
        main_df[features] = scaler.fit_transform(main_df[features])

        # Fill all NaN values so that Inf becomes the highest value in column
        main_df[features] = main_df[features].fillna(1.0)
    except Exception as e:
        print(f"\r[!] Error: {e}".ljust(100))

    # Save to .csv if True
    if save_dataset:
        dataset_name = f"{root}{save_dataset}{entries}"

        print(f"\r[*] Saving dataframe to {dataset_name}.csv...".ljust(100), end="", flush=True)
        main_df.to_csv(f"{dataset_name}.csv")
        print(f"\r[+] Dataframe successfully saved to {dataset_name}.csv!".ljust(100), flush=True)

    # Measure and print how long concat + preprocessing took
    cpu_time = time.process_time() - start_cpu
    perf_time = time.perf_counter() - start_perf
    print(f"\r[+] Done! Processing dataset took: {cpu_time:.6f} s ({perf_time:.2f} s)".ljust(100))

    return main_df

# Packet datasets

def create_binary_packet_df(args, root):

    # Check if ./Dataset is present
    if not os.path.exists("../CIC IoT-IDAD 2024 Dataset"):
        print(f"[!]./Dataset not found! Exiting...")
        sys.exit(1)

    path = "../CIC IoT-IDAD 2024 Dataset/Device Identification_Anomaly Detection - Packet Based Features/"

    # BenignTraffic
    df_benign = create_class_df(0, f"{path}BenignTraffic/BenignTraffic")
    
    # BruteForce
    df_bruteforce = create_class_df(1, f"{path}BruteForce/DictionaryBruteForce/DictionaryBruteForce")
    
    # DDoS
    df_ddos_ack = create_class_df(1, f"{path}DDoS/DDoS-ACK_Fragmentation/DDoS-ACK_Fragmentation")
    df_ddos_http = create_class_df(1, f"{path}DDoS/DDoS-HTTP_Flood/DDoS-HTTP_Flood")
    df_ddos_icmp = create_class_df(1, f"{path}DDoS/DDoS-ICMP_Fragmentation/DDoS-ICMP_Fragmentation")
    df_ddos_slow = create_class_df(1, f"{path}DDoS/DDoS-SlowLoris/DDoS-SlowLoris")
    df_ddos_syn_ip = create_class_df(1, f"{path}DDoS/DDoS-SynonymousIP_Flood/DDoS-SynonymousIP_Flood")
    df_ddos_tcp = create_class_df(1, f"{path}DDoS/DDoS-TCP_Flood/DDoS-TCP_Flood")
    df_ddos_udp = create_class_df(1, f"{path}DDoS/DDoS-UDP_Flood/DDoS-UDP_Flood")
    df_ddos_udp_frag = create_class_df(1, f"{path}DDoS/DDoS-UDP_Fragmentation/DDoS-UDP_Fragmentation")

    start_perf = time.perf_counter()
    start_cpu = time.process_time()
    df_ddos = pd.concat([df_ddos_ack, df_ddos_http, df_ddos_icmp, df_ddos_slow, df_ddos_syn_ip, df_ddos_tcp, df_ddos_udp, df_ddos_udp_frag], ignore_index=True)
    cpu_time = time.process_time() - start_cpu
    perf_time = time.perf_counter() - start_perf
    print(f"\r[+] DDoS Done! Processing took: {cpu_time:.6f} s ({perf_time:.2f} s)".ljust(100))

    # DoS
    df_dos_http = create_class_df(1, f"{path}DoS/DoS-HTTP_Flood/DoS-HTTP_Flood")
    df_dos_syn = create_class_df(1, f"{path}DoS/DoS-SYN_Flood/DoS-SYN_Flood")
    df_dos_tcp = create_class_df(1, f"{path}DoS/DoS-TCP_Flood/DoS-TCP_Flood")
    df_dos_udp = create_class_df(1, f"{path}DoS/DoS-UDP_Flood/DoS-UDP_Flood")

    start_perf = time.perf_counter()
    start_cpu = time.process_time()
    df_dos = pd.concat([df_dos_http, df_dos_syn, df_dos_tcp, df_dos_udp], ignore_index=True)
    cpu_time = time.process_time() - start_cpu
    perf_time = time.perf_counter() - start_perf
    print(f"\r[+] DoS Done! Processing took: {cpu_time:.6f} s ({perf_time:.2f} s)".ljust(100))

    # Botnet (Mirai)
    df_botnet = create_class_df(1, f"{path}Mirai/Mirai-greip_flood/Mirai-greip_flood")

    # Recon
    df_recon_host = create_class_df(1, f"{path}Recon/Recon-HostDiscovery/Recon-HostDiscovery")
    df_recon_os = create_class_df(1, f"{path}Recon/Recon-OSScan/Recon-OSScan")
    df_recon_ping = create_class_df(1, f"{path}Recon/Recon-PingSweep/Recon-PingSweep")
    df_recon_port = create_class_df(1, f"{path}Recon/Recon-PortScan/Recon-PortScan")
    df_recon_vuln = create_class_df(1, f"{path}Recon/VulnerabilityScan/VulnerabilityScan")

    print(f"\r[*] Processing Recon...".ljust(100), end="", flush=True)
    start_perf = time.perf_counter()
    start_cpu = time.process_time()
    df_recon = pd.concat([df_recon_host, df_recon_os, df_recon_ping, df_recon_port, df_recon_vuln], ignore_index=True)
    cpu_time = time.process_time() - start_cpu
    perf_time = time.perf_counter() - start_perf
    print(f"\r[+] Recon Done! Processing took: {cpu_time:.6f} s ({perf_time:.2f} s)".ljust(100))

    # Spoofing 
    df_spoof_dns = create_class_df(1, f"{path}Spoofing/DNS_Spoofing/DNS_Spoofing")
    df_spoof_mitm = create_class_df(1, f"{path}Spoofing/MITM-ArpSpoofing/MITM-ArpSpoofing")
    
    print(f"\r[*] Processing Spoofing...".ljust(100), end="", flush=True)
    start_perf = time.perf_counter()
    start_cpu = time.process_time()
    df_spoof = pd.concat([df_spoof_dns, df_spoof_mitm], ignore_index=True)
    cpu_time = time.process_time() - start_cpu
    perf_time = time.perf_counter() - start_perf
    print(f"\r[+] Spoofing Done! Processing took: {cpu_time:.6f} s ({perf_time:.2f} s)".ljust(100))

    # Web
    df_web_backdoor = create_class_df(1, f"{path}Web-Based/Backdoor_Malware/Backdoor_Malware")
    df_web_browser = create_class_df(1, f"{path}Web-Based/BrowserHijacking/BrowserHijacking")
    df_web_command = create_class_df(1, f"{path}Web-Based/CommandInjection/CommandInjection")
    df_web_sql = create_class_df(1, f"{path}Web-Based/SqlInjection/SqlInjection")
    df_web_uploading = create_class_df(1, f"{path}Web-Based/Uploading_Attack/Uploading_Attack")
    df_web_xss = create_class_df(1, f"{path}Web-Based/XSS/XSS")

    print(f"\r[*] Processing Web...".ljust(100), end="", flush=True)
    start_perf = time.perf_counter()
    start_cpu = time.process_time()
    df_web = pd.concat([df_web_backdoor, df_web_browser, df_web_command, df_web_sql, df_web_uploading, df_web_xss], ignore_index=True)
    cpu_time = time.process_time() - start_cpu
    perf_time = time.perf_counter() - start_perf
    print(f"\r[+] Web Done! Processing took: {cpu_time:.6f} s ({perf_time:.2f} s)".ljust(100))

    # Concat anomalous data
    print(f"\r[*] Concatting anomalous data...".ljust(100), end="", flush=True)
    df_anomalous = pd.concat([ df_bruteforce, df_ddos, df_dos, df_botnet, df_recon, df_spoof, df_web], ignore_index=True)

    save_dataset = f"dataset_packet_binary_n" if args.save_dataset else False

    return combine_class_dfs([df_benign, df_anomalous],args, root, save_dataset)

def create_multiclass_8_packet_df(args, root):

    # Check if ./Dataset is present
    if not os.path.exists("../CIC IoT-IDAD 2024 Dataset"):
        print(f"[!]./Dataset not found! Exiting...")
        sys.exit(1)

    path = "../CIC IoT-IDAD 2024 Dataset/Device Identification_Anomaly Detection - Packet Based Features/"

    # BenignTraffic
    df_benign = create_class_df(0, f"{path}BenignTraffic/BenignTraffic")
    
    # BruteForce
    df_bruteforce = create_class_df(1, f"{path}BruteForce/DictionaryBruteForce/DictionaryBruteForce")
    
    # DDoS
    df_ddos_ack = create_class_df(2, f"{path}DDoS/DDoS-ACK_Fragmentation/DDoS-ACK_Fragmentation")
    df_ddos_http = create_class_df(2, f"{path}DDoS/DDoS-HTTP_Flood/DDoS-HTTP_Flood")
    df_ddos_icmp = create_class_df(2, f"{path}DDoS/DDoS-ICMP_Fragmentation/DDoS-ICMP_Fragmentation")
    df_ddos_slow = create_class_df(2, f"{path}DDoS/DDoS-SlowLoris/DDoS-SlowLoris")
    df_ddos_syn_ip = create_class_df(2, f"{path}DDoS/DDoS-SynonymousIP_Flood/DDoS-SynonymousIP_Flood")
    df_ddos_tcp = create_class_df(2, f"{path}DDoS/DDoS-TCP_Flood/DDoS-TCP_Flood")
    df_ddos_udp = create_class_df(2, f"{path}DDoS/DDoS-UDP_Flood/DDoS-UDP_Flood")
    df_ddos_udp_frag = create_class_df(2, f"{path}DDoS/DDoS-UDP_Fragmentation/DDoS-UDP_Fragmentation")

    print(f"\r[*] Processing DDoS...".ljust(100), end="", flush=True)
    start_perf = time.perf_counter()
    start_cpu = time.process_time()
    df_ddos = pd.concat([df_ddos_ack, df_ddos_http, df_ddos_icmp, df_ddos_slow, df_ddos_syn_ip, df_ddos_tcp, df_ddos_udp, df_ddos_udp_frag], ignore_index=True)
    cpu_time = time.process_time() - start_cpu
    perf_time = time.perf_counter() - start_perf
    print(f"\r[+] DDoS Done! Processing took: {cpu_time:.6f} s ({perf_time:.2f} s)".ljust(100))

    # DoS
    df_dos_http = create_class_df(3, f"{path}DoS/DoS-HTTP_Flood/DoS-HTTP_Flood")
    df_dos_syn = create_class_df(3, f"{path}DoS/DoS-SYN_Flood/DoS-SYN_Flood")
    df_dos_tcp = create_class_df(3, f"{path}DoS/DoS-TCP_Flood/DoS-TCP_Flood")
    df_dos_udp = create_class_df(3, f"{path}DoS/DoS-UDP_Flood/DoS-UDP_Flood")

    print(f"\r[*] Processing DoS...".ljust(100), end="", flush=True)
    start_perf = time.perf_counter()
    start_cpu = time.process_time()
    df_dos = pd.concat([df_dos_http, df_dos_syn, df_dos_tcp, df_dos_udp], ignore_index=True)
    cpu_time = time.process_time() - start_cpu
    perf_time = time.perf_counter() - start_perf
    print(f"\r[+] DoS Done! Processing took: {cpu_time:.6f} s ({perf_time:.2f} s)".ljust(100))

    # Botnet (Mirai)
    df_botnet = create_class_df(4, f"{path}Mirai/Mirai-greip_flood/Mirai-greip_flood")

    # Recon
    df_recon_host = create_class_df(5, f"{path}Recon/Recon-HostDiscovery/Recon-HostDiscovery")
    df_recon_os = create_class_df(5, f"{path}Recon/Recon-OSScan/Recon-OSScan")
    df_recon_ping = create_class_df(5, f"{path}Recon/Recon-PingSweep/Recon-PingSweep")
    df_recon_port = create_class_df(5, f"{path}Recon/Recon-PortScan/Recon-PortScan")
    df_recon_vuln = create_class_df(5, f"{path}Recon/VulnerabilityScan/VulnerabilityScan")

    print(f"\r[*] Processing Recon...".ljust(100), end="", flush=True)
    start_perf = time.perf_counter()
    start_cpu = time.process_time()
    df_recon = pd.concat([df_recon_host, df_recon_os, df_recon_ping, df_recon_port, df_recon_vuln], ignore_index=True)
    cpu_time = time.process_time() - start_cpu
    perf_time = time.perf_counter() - start_perf
    print(f"\r[+] Recon Done! Processing took: {cpu_time:.6f} s ({perf_time:.2f} s)".ljust(100))

    # Spoofing 
    df_spoof_dns = create_class_df(6, f"{path}Spoofing/DNS_Spoofing/DNS_Spoofing")
    df_spoof_mitm = create_class_df(6, f"{path}Spoofing/MITM-ArpSpoofing/MITM-ArpSpoofing")
    
    print(f"[*] Processing Spoofing...".ljust(100), end="", flush=True)
    start_perf = time.perf_counter()
    start_cpu = time.process_time()
    df_spoof = pd.concat([df_spoof_dns, df_spoof_mitm], ignore_index=True)
    cpu_time = time.process_time() - start_cpu
    perf_time = time.perf_counter() - start_perf
    print(f"\r[+] Spoofing Done! Processing took: {cpu_time:.6f} s ({perf_time:.2f} s)".ljust(100))

    # Web
    df_web_backdoor = create_class_df(7, f"{path}Web-Based/Backdoor_Malware/Backdoor_Malware")
    df_web_browser = create_class_df(7, f"{path}Web-Based/BrowserHijacking/BrowserHijacking")
    df_web_command = create_class_df(7, f"{path}Web-Based/CommandInjection/CommandInjection")
    df_web_sql = create_class_df(7, f"{path}Web-Based/SqlInjection/SqlInjection")
    df_web_uploading = create_class_df(7, f"{path}Web-Based/Uploading_Attack/Uploading_Attack")
    df_web_xss = create_class_df(7, f"{path}Web-Based/XSS/XSS")

    print(f"\r[*] Processing Web...".ljust(100), end="", flush=True)
    start_perf = time.perf_counter()
    start_cpu = time.process_time()
    df_web = pd.concat([df_web_backdoor, df_web_browser, df_web_command, df_web_sql, df_web_uploading, df_web_xss], ignore_index=True)
    cpu_time = time.process_time() - start_cpu
    perf_time = time.perf_counter() - start_perf
    print(f"\r[+] Web Done! Processing took: {cpu_time:.6f} s ({perf_time:.2f} s)".ljust(100))

    save_dataset = f"dataset_packet_multiclass8_n" if args.save_dataset else False

    return combine_class_dfs([df_benign, df_bruteforce, df_ddos, df_dos, df_botnet, df_recon, df_spoof, df_web],args, root, save_dataset)

def create_multiclass_28_packet_df(args, root):

    # Check if ./Dataset is present
    if not os.path.exists("../CIC IoT-IDAD 2024 Dataset"):
        print(f"[!]./Dataset not found! Exiting...")
        sys.exit(1)

    path = "../CIC IoT-IDAD 2024 Dataset/Device Identification_Anomaly Detection - Packet Based Features/"

    # BenignTraffic
    df_benign = create_class_df(0, f"{path}BenignTraffic/BenignTraffic")
    
    # BruteForce
    df_bruteforce = create_class_df(1, f"{path}BruteForce/DictionaryBruteForce/DictionaryBruteForce")
    
    # DDoS
    df_ddos_ack = create_class_df(2, f"{path}DDoS/DDoS-ACK_Fragmentation/DDoS-ACK_Fragmentation")
    df_ddos_http = create_class_df(3, f"{path}DDoS/DDoS-HTTP_Flood/DDoS-HTTP_Flood")
    df_ddos_icmp = create_class_df(4, f"{path}DDoS/DDoS-ICMP_Fragmentation/DDoS-ICMP_Fragmentation")
    df_ddos_slow = create_class_df(5, f"{path}DDoS/DDoS-SlowLoris/DDoS-SlowLoris")
    df_ddos_syn_ip = create_class_df(6, f"{path}DDoS/DDoS-SynonymousIP_Flood/DDoS-SynonymousIP_Flood")
    df_ddos_tcp = create_class_df(7, f"{path}DDoS/DDoS-TCP_Flood/DDoS-TCP_Flood")
    df_ddos_udp = create_class_df(8, f"{path}DDoS/DDoS-UDP_Flood/DDoS-UDP_Flood")
    df_ddos_udp_frag = create_class_df(9, f"{path}DDoS/DDoS-UDP_Fragmentation/DDoS-UDP_Fragmentation")

    # DoS
    df_dos_http = create_class_df(10, f"{path}DoS/DoS-HTTP_Flood/DoS-HTTP_Flood")
    df_dos_syn = create_class_df(11, f"{path}DoS/DoS-SYN_Flood/DoS-SYN_Flood")
    df_dos_tcp = create_class_df(12, f"{path}DoS/DoS-TCP_Flood/DoS-TCP_Flood")
    df_dos_udp = create_class_df(13, f"{path}DoS/DoS-UDP_Flood/DoS-UDP_Flood")

    # Botnet (Mirai)
    df_botnet = create_class_df(14, f"{path}Mirai/Mirai-greip_flood/Mirai-greip_flood")

    # Recon
    df_recon_host = create_class_df(15, f"{path}Recon/Recon-HostDiscovery/Recon-HostDiscovery")
    df_recon_os = create_class_df(16, f"{path}Recon/Recon-OSScan/Recon-OSScan")
    df_recon_ping = create_class_df(17, f"{path}Recon/Recon-PingSweep/Recon-PingSweep")
    df_recon_port = create_class_df(18, f"{path}Recon/Recon-PortScan/Recon-PortScan")
    df_recon_vuln = create_class_df(19, f"{path}Recon/VulnerabilityScan/VulnerabilityScan")

    # Spoofing 
    df_spoof_dns = create_class_df(20, f"{path}Spoofing/DNS_Spoofing/DNS_Spoofing")
    df_spoof_mitm = create_class_df(21, f"{path}Spoofing/MITM-ArpSpoofing/MITM-ArpSpoofing")

    # Web
    df_web_backdoor = create_class_df(22, f"{path}Web-Based/Backdoor_Malware/Backdoor_Malware")
    df_web_browser = create_class_df(23, f"{path}Web-Based/BrowserHijacking/BrowserHijacking")
    df_web_command = create_class_df(24, f"{path}Web-Based/CommandInjection/CommandInjection")
    df_web_sql = create_class_df(25, f"{path}Web-Based/SqlInjection/SqlInjection")
    df_web_uploading = create_class_df(26, f"{path}Web-Based/Uploading_Attack/Uploading_Attack")
    df_web_xss = create_class_df(27, f"{path}Web-Based/XSS/XSS")

    save_dataset = f"dataset_packet_multiclass28_n" if args.save_dataset else False

    return combine_class_dfs([df_benign, df_bruteforce, df_ddos_ack, df_ddos_http, df_ddos_icmp, df_ddos_slow, df_ddos_syn_ip, df_ddos_tcp, df_ddos_udp, df_ddos_udp_frag, df_dos_http, df_dos_syn, df_dos_tcp, df_dos_udp, df_botnet, df_recon_host, df_recon_os, df_recon_ping, df_recon_port, df_recon_vuln, df_spoof_dns, df_spoof_mitm, df_web_backdoor, df_web_browser, df_web_command, df_web_sql, df_web_uploading, df_web_xss],args, root, save_dataset)

# Flow dataset

def create_binary_flow_df(args, root):

    # Check if ./Dataset is present
    if not os.path.exists("../CIC IoT-IDAD 2024 Dataset"):
        print(f"[!]./Dataset not found! Exiting...")
        sys.exit(1)

    path = "../CIC IoT-IDAD 2024 Dataset/Anomaly Detection - Flow Based features/"

    # BenignTraffic
    df_benign = create_class_df(0, f"{path}Benign/BenignTraffic.pcap_Flow")
    
    # BruteForce
    df_bruteforce = create_class_df(1, f"{path}BruteForce/DictionaryBruteForce.pcap_Flow")
    
    # DDoS
    df_ddos_ack = create_class_df(1, f"{path}DDoS/DDoS ACK Fragmentation/DDoS-ACK_Fragmentation.pcap_Flow")
    df_ddos_icmp = create_class_df(1, f"{path}DDoS/DDoS ICMP Flood/DDoS-ICMP_Flood.pcap_Flow")
    df_ddos_http = create_class_df(1, f"{path}DDoS/DDoS-HTTP Flood/DDoS-HTTP_Flood.pcap_Flow")
    df_ddos_icmp_frag = create_class_df(1, f"{path}DDoS/DDoS-ICMP_Fragmentation/DDoS-ICMP_Fragmentation.pcap_Flow")
    
    # Missing DDoS data...
    # df_ddos_pshack = create_class_df(1, f"{path}DDoS/DDoS-PSHACK_Flood/DDoS-PSHACK.pcap_Flow")
    # df_ddos_rstfin = create_class_df(1, f"{path}DDoS/DDoS-RSTFINFlood/DDoS-RSTFIN_Flood.pcap_Flow")
    # df_ddos_slowloris = create_class_df(1, f"{path}DDoS/DDoS-SlowLoris/DDoS-SlowLoris.pcap_Flow")
    # df_ddos_syn = create_class_df(1, f"{path}DDoS/DDoS-SYN_Flood/DDoS-SYN_Flood.pcap_Flow")
    # df_ddos_synip = create_class_df(1, f"{path}DDoS/DDoS-SynonymousIP_Flood/DDoS-SynonymousIP_Flood.pcap_Flow")
    # df_ddos_tcp = create_class_df(1, f"{path}DDoS/DDoS-TCP_Flood/DDoS-TCP_Flood.pcap_Flow")
    # df_ddos_udp = create_class_df(1, f"{path}DDoS/DDoS-UDP_Flood/DDoS-UDP_Flood.pcap_Flow")
    # df_ddos_udp_frag = create_class_df(1, f"{path}DDoS/DDoS-UDP_Fragmentation/DDoS-UDP_Fragmentation.pcap_Flow")

    print(f"\r[*] Processing DDoS...".ljust(100), end="", flush=True)
    start_perf = time.perf_counter()
    start_cpu = time.process_time()
    df_ddos = pd.concat([df_ddos_ack, df_ddos_icmp, df_ddos_http, df_ddos_icmp_frag], ignore_index=True)
    cpu_time = time.process_time() - start_cpu
    perf_time = time.perf_counter() - start_perf
    print(f"\r[+] DDoS Done! Processing took: {cpu_time:.6f} s ({perf_time:.2f} s)".ljust(100))

    # DoS
    df_dos_http = create_class_df(1, f"{path}DoS/DoS-HTTP_Flood/DoS-HTTP_Flood.pcap_Flow")
    df_dos_syn = create_class_df(1, f"{path}DoS/DoS SYN Flood/DoS-SYN_Flood.pcap_Flow")
    # df_dos_tcp = create_class_df(1, f"{path}DoS/DoS-TCP_Flood/DoS-TCP_Flood.pcap_Flow") # Broken...
    df_dos_udp = create_class_df(1, f"{path}DoS/DoS-UDP_Flood/DoS-UDP_Flood.pcap_Flow")

    print(f"\r[*] Processing DoS...".ljust(100), end="", flush=True)
    start_perf = time.perf_counter()
    start_cpu = time.process_time()
    df_dos = pd.concat([df_dos_http, df_dos_syn, df_dos_udp], ignore_index=True)
    cpu_time = time.process_time() - start_cpu
    perf_time = time.perf_counter() - start_perf
    print(f"\r[+] DoS Done! Processing took: {cpu_time:.6f} s ({perf_time:.2f} s)".ljust(100))

    # Botnet (Mirai)
    df_botnet = create_class_df(1, f"{path}Mirai/Mirai-greeth_flood.pcap_Flow")

    # Recon
    df_recon = create_class_df(1, f"{path}Recon/VulnerabilityScan/VulnerabilityScan.pcap_Flow")

    # Spoofing 
    df_spoof_arp = create_class_df(1, f"{path}Spoofing/ARP Spoofing/MITM-ArpSpoofing.pcap_Flow")
    df_spoof_dns = create_class_df(1, f"{path}Spoofing/DNS Spoofing/DNS_Spoofing.pcap_Flow")
    
    print(f"\r[*] Processing Spoofing...".ljust(100), end="", flush=True)
    start_perf = time.perf_counter()
    start_cpu = time.process_time()
    df_spoof = pd.concat([df_spoof_arp, df_spoof_dns], ignore_index=True)
    cpu_time = time.process_time() - start_cpu
    perf_time = time.perf_counter() - start_perf
    print(f"\r[+] Spoofing Done! Processing took: {cpu_time:.6f} s ({perf_time:.2f} s)".ljust(100))

    # Web
    df_web_sql = create_class_df(1, f"{path}Web-Based/sqlinjection/SqlInjection.pcap_Flow")
    df_web_uploading = create_class_df(1, f"{path}Web-Based/Uploading_Attack/Uploading_Attack.pcap_Flow")
    df_web_xss = create_class_df(1, f"{path}Web-Based/XSS/XSS.pcap_Flow")

    print(f"\r[*] Processing Web...".ljust(100), end="", flush=True)
    start_perf = time.perf_counter()
    start_cpu = time.process_time()
    df_web = pd.concat([df_web_sql, df_web_uploading, df_web_xss], ignore_index=True)
    cpu_time = time.process_time() - start_cpu
    perf_time = time.perf_counter() - start_perf
    print(f"\r[+] Web Done! Processing took: {cpu_time:.6f} s ({perf_time:.2f} s)".ljust(100))

    # Concat anomalous data
    print(f"\r[*] Concatting anomalous data...".ljust(100), end="", flush=True)
    df_anomalous = pd.concat([df_bruteforce, df_ddos, df_dos, df_botnet, df_recon, df_spoof, df_web], ignore_index=True)

    save_dataset = f"dataset_flow_binary_n" if args.save_dataset else False

    return combine_class_dfs([df_benign, df_anomalous], args, root, save_dataset)

def create_multiclass_8_flow_df(args, root):

    # Check if ./Dataset is present
    if not os.path.exists("../CIC IoT-IDAD 2024 Dataset"):
        print(f"[!]./Dataset not found! Exiting...")
        sys.exit(1)

    path = "../CIC IoT-IDAD 2024 Dataset/Anomaly Detection - Flow Based features/"

    # BenignTraffic
    df_benign = create_class_df(0, f"{path}Benign/BenignTraffic.pcap_Flow")
    
    # BruteForce
    df_bruteforce = create_class_df(1, f"{path}BruteForce/DictionaryBruteForce.pcap_Flow")
    
    # DDoS
    df_ddos_ack = create_class_df(2, f"{path}DDoS/DDoS ACK Fragmentation/DDoS-ACK_Fragmentation.pcap_Flow")
    df_ddos_icmp = create_class_df(2, f"{path}DDoS/DDoS ICMP Flood/DDoS-ICMP_Flood.pcap_Flow")
    df_ddos_http = create_class_df(2, f"{path}DDoS/DDoS-HTTP Flood/DDoS-HTTP_Flood.pcap_Flow")
    df_ddos_icmp_frag = create_class_df(2, f"{path}DDoS/DDoS-ICMP_Fragmentation/DDoS-ICMP_Fragmentation.pcap_Flow")
    
    # Missing DDoS data...
    # df_ddos_pshack = create_class_df(1, f"{path}DDoS/DDoS-PSHACK_Flood/DDoS-PSHACK.pcap_Flow")
    # df_ddos_rstfin = create_class_df(1, f"{path}DDoS/DDoS-RSTFINFlood/DDoS-RSTFIN_Flood.pcap_Flow")
    # df_ddos_slowloris = create_class_df(1, f"{path}DDoS/DDoS-SlowLoris/DDoS-SlowLoris.pcap_Flow")
    # df_ddos_syn = create_class_df(1, f"{path}DDoS/DDoS-SYN_Flood/DDoS-SYN_Flood.pcap_Flow")
    # df_ddos_synip = create_class_df(1, f"{path}DDoS/DDoS-SynonymousIP_Flood/DDoS-SynonymousIP_Flood.pcap_Flow")
    # df_ddos_tcp = create_class_df(1, f"{path}DDoS/DDoS-TCP_Flood/DDoS-TCP_Flood.pcap_Flow")
    # df_ddos_udp = create_class_df(1, f"{path}DDoS/DDoS-UDP_Flood/DDoS-UDP_Flood.pcap_Flow")
    # df_ddos_udp_frag = create_class_df(1, f"{path}DDoS/DDoS-UDP_Fragmentation/DDoS-UDP_Fragmentation.pcap_Flow")

    start_perf = time.perf_counter()
    start_cpu = time.process_time()
    df_ddos = pd.concat([df_ddos_ack, df_ddos_icmp, df_ddos_http, df_ddos_icmp_frag], ignore_index=True)
    cpu_time = time.process_time() - start_cpu
    perf_time = time.perf_counter() - start_perf
    print(f"\r[+] DDoS Done! Processing took: {cpu_time:.6f} s ({perf_time:.2f} s)".ljust(100))

    # DoS
    df_dos_http = create_class_df(3, f"{path}DoS/DoS-HTTP_Flood/DoS-HTTP_Flood.pcap_Flow")
    df_dos_syn = create_class_df(3, f"{path}DoS/DoS SYN Flood/DoS-SYN_Flood.pcap_Flow")
    # df_dos_tcp = create_class_df(3, f"{path}DoS/DoS-TCP_Flood/DoS-TCP_Flood.pcap_Flow") # Broken...
    df_dos_udp = create_class_df(3, f"{path}DoS/DoS-UDP_Flood/DoS-UDP_Flood.pcap_Flow")

    start_perf = time.perf_counter()
    start_cpu = time.process_time()
    df_dos = pd.concat([df_dos_http, df_dos_syn, df_dos_udp], ignore_index=True)
    cpu_time = time.process_time() - start_cpu
    perf_time = time.perf_counter() - start_perf
    print(f"\r[+] DoS Done! Processing took: {cpu_time:.6f} s ({perf_time:.2f} s)".ljust(100))

    # Botnet (Mirai)
    df_botnet = create_class_df(4, f"{path}Mirai/Mirai-greeth_flood.pcap_Flow")

    # Recon
    df_recon = create_class_df(5, f"{path}Recon/VulnerabilityScan/VulnerabilityScan.pcap_Flow")

    # Spoofing 
    df_spoof_arp = create_class_df(1, f"{path}Spoofing/ARP Spoofing/MITM-ArpSpoofing.pcap_Flow")
    df_spoof_dns = create_class_df(6, f"{path}Spoofing/DNS Spoofing/DNS_Spoofing.pcap_Flow")
    
    start_perf = time.perf_counter()
    start_cpu = time.process_time()
    df_spoof = pd.concat([df_spoof_arp, df_spoof_dns], ignore_index=True)
    cpu_time = time.process_time() - start_cpu
    perf_time = time.perf_counter() - start_perf
    print(f"\r[+] Recon Done! Processing took: {cpu_time:.6f} s ({perf_time:.2f} s)".ljust(100))

    # Web
    df_web_sql = create_class_df(7, f"{path}Web-Based/sqlinjection/SqlInjection.pcap_Flow")
    df_web_uploading = create_class_df(7, f"{path}Web-Based/Uploading_Attack/Uploading_Attack.pcap_Flow")
    df_web_xss = create_class_df(7, f"{path}Web-Based/XSS/XSS.pcap_Flow")

    start_perf = time.perf_counter()
    start_cpu = time.process_time()
    df_web = pd.concat([df_web_sql, df_web_uploading, df_web_xss], ignore_index=True)
    cpu_time = time.process_time() - start_cpu
    perf_time = time.perf_counter() - start_perf
    print(f"\r[+] Recon Done! Processing took: {cpu_time:.6f} s ({perf_time:.2f} s)".ljust(100))

    save_dataset = f"dataset_flow_multiclass8_n" if args.save_dataset else False

    return combine_class_dfs([df_benign, df_bruteforce, df_ddos, df_dos, df_botnet, df_recon, df_spoof, df_web],args, root, save_dataset)

def create_multiclass_28_flow_df(args, root):

    # Check if ./Dataset is present
    if not os.path.exists("../CIC IoT-IDAD 2024 Dataset"):
        print(f"[!]./Dataset not found! Exiting...")
        sys.exit(1)

    path = "../CIC IoT-IDAD 2024 Dataset/Anomaly Detection - Flow Based features/"

    # BenignTraffic
    df_benign = create_class_df(0, f"{path}Benign/BenignTraffic.pcap_Flow")
    
    # BruteForce
    df_bruteforce = create_class_df(1, f"{path}BruteForce/DictionaryBruteForce.pcap_Flow")
    
    # DDoS
    df_ddos_ack = create_class_df(2, f"{path}DDoS/DDoS ACK Fragmentation/DDoS-ACK_Fragmentation.pcap_Flow")
    df_ddos_icmp = create_class_df(3, f"{path}DDoS/DDoS ICMP Flood/DDoS-ICMP_Flood.pcap_Flow")
    df_ddos_http = create_class_df(4, f"{path}DDoS/DDoS-HTTP Flood/DDoS-HTTP_Flood.pcap_Flow")
    df_ddos_icmp_frag = create_class_df(5, f"{path}DDoS/DDoS-ICMP_Fragmentation/DDoS-ICMP_Fragmentation.pcap_Flow")
    
    # Missing DDoS data...
    # df_ddos_pshack = create_class_df(6, f"{path}DDoS/DDoS-PSHACK_Flood/DDoS-PSHACK.pcap_Flow")
    # df_ddos_rstfin = create_class_df(7, f"{path}DDoS/DDoS-RSTFINFlood/DDoS-RSTFIN_Flood.pcap_Flow")
    # df_ddos_slowloris = create_class_df(8, f"{path}DDoS/DDoS-SlowLoris/DDoS-SlowLoris.pcap_Flow")
    # df_ddos_syn = create_class_df(9, f"{path}DDoS/DDoS-SYN_Flood/DDoS-SYN_Flood.pcap_Flow")
    # df_ddos_synip = create_class_df(10, f"{path}DDoS/DDoS-SynonymousIP_Flood/DDoS-SynonymousIP_Flood.pcap_Flow")
    # df_ddos_tcp = create_class_df(11, f"{path}DDoS/DDoS-TCP_Flood/DDoS-TCP_Flood.pcap_Flow")
    # df_ddos_udp = create_class_df(12, f"{path}DDoS/DDoS-UDP_Flood/DDoS-UDP_Flood.pcap_Flow")
    # df_ddos_udp_frag = create_class_df(13, f"{path}DDoS/DDoS-UDP_Fragmentation/DDoS-UDP_Fragmentation.pcap_Flow")

    # DoS
    df_dos_http = create_class_df(14, f"{path}DoS/DoS-HTTP_Flood/DoS-HTTP_Flood.pcap_Flow")
    df_dos_syn = create_class_df(15, f"{path}DoS/DoS SYN Flood/DoS-SYN_Flood.pcap_Flow")
    # df_dos_tcp = create_class_df(16, f"{path}DoS/DoS-TCP_Flood/DoS-TCP_Flood.pcap_Flow") # Broken...
    df_dos_udp = create_class_df(17, f"{path}DoS/DoS-UDP_Flood/DoS-UDP_Flood.pcap_Flow")

    # Botnet (Mirai)
    df_botnet = create_class_df(18, f"{path}Mirai/Mirai-greeth_flood.pcap_Flow")

    # Recon
    df_recon = create_class_df(19, f"{path}Recon/VulnerabilityScan/VulnerabilityScan.pcap_Flow")

    # Spoofing 
    df_spoof_arp = create_class_df(1, f"{path}Spoofing/ARP Spoofing/MITM-ArpSpoofing.pcap_Flow")
    df_spoof_dns = create_class_df(21, f"{path}Spoofing/DNS Spoofing/DNS_Spoofing.pcap_Flow")

    # Web
    df_web_sql = create_class_df(22, f"{path}Web-Based/sqlinjection/SqlInjection.pcap_Flow")
    df_web_uploading = create_class_df(23, f"{path}Web-Based/Uploading_Attack/Uploading_Attack.pcap_Flow")
    df_web_xss = create_class_df(24, f"{path}Web-Based/XSS/XSS.pcap_Flow")

    save_dataset = f"dataset_flow_multiclass24_n" if args.save_dataset else False

    return combine_class_dfs([df_benign, df_bruteforce, df_ddos_ack, df_ddos_icmp, df_ddos_http, df_ddos_icmp_frag, df_dos_http, df_dos_syn, df_dos_udp, df_botnet, df_recon, df_spoof_arp, df_spoof_dns, df_web_sql, df_web_uploading, df_web_xss],args, root, save_dataset)
    