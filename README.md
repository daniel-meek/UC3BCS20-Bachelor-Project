# UC3BCS20 Bachelor Project

## About

This repository contains the code and tools developed for the bachelor project: A Framework for On-Device Anomaly Detection in IoT Networks.

The code facilitates the training, optimization, and evaluation of machine learning models for anomaly detection on resource-constrained Internet of Things (IoT) hardware. It utilizes the CICIoT2024 dataset to evaluate four model architectures (Decision Tree, Random Forest, Logistic Regression, and Feedforward Neural Network). The project leverages Recursive Feature Elimination (RFE) to optimize feature subsets and includes a load-testing suite designed for a Raspberry Pi 3B+ to measure real-world performance metrics like CPU utilization and packet throughput.

## Report

This repository serves as Appendix A to the thesis submitted in partial fulfillment of the requirements for the Bachelor in Cyber Security at Noroff University College (2025/2026).

- **Title**: A Framework for On-Device Anomaly Detection in IoT Networks
- **Institution**: Noroff University College
- **Supervisor**: Bertram Haskins

## Appendix A - Tools

During the thesis, two programs and a handfull of scripts were developed to implement the methodology. These were `main.py` and `load_test.py`, allong with the scripts in `/helper_scripts`

These scripts come with minimal error handling, as they were buildt to only be used by the author. They can, and will break.

### `main`

`main.py` is the main tool used for dataset pre-processing, model training and RFE. 

#### Dependencies

**Third-party Libraries**:

- pandas
- scikit-learn
- numpy
- matplotlib
- psutil
- pyarrow

To quickly install the required libareis use:
```
pip install pandas scikit-learn numpy matplotlib psutil pyarrow
```

**Python Standard Libaries**

- os
- sys
- time
- argparse
- warnings
- pickle

**Dataset**

The script relies on the CICIoT2024 dataset found at [https://www.unb.ca/cic/datasets/iot-diad-2024.html](https://www.unb.ca/cic/datasets/iot-diad-2024.html)

#### Usage

**Arguments**

**Dataset Settings:**
- `--csv` : `str` - Path to a pre-processed dataset `.csv` file. Skips new dataset creation.
- `--path` : `str` - The root directory of the CICIoT2024 dataset files (default: `Dataset`).
- `-d`, `--dataset` : `[packet, flow]` - Select the data-subset type to use (default: `packet`).
- `-c`, `--classification` : `[binary, multiclass8, multiclass28]` - Select the classification (default: `binary`).
- `-n`, `--rows` : `int` - Total rows to sample for the dataset. (default: generates a maximum sized balanced dataset depending on the available rows for each class).

**Global Settings:**
- `--random_state` : `int` - Sets all random_state variable for reproducibility (default: `0`).

**Output & Saving Settings:**
- `--save_model` - Saves all trained models as `.pkl` files.
- `--save_report` - Saves model reports as `.txt` files.
- `--save_dataset` - Saves the generated dataset as a `.csv` file.
- `--verbose_results` - Adds additional model-specific data (like tree depth, leaves) to the result.txt files.

**Model Selection:**
- `--dtc` - Use a Decision Tree Classifier.
- `--rfc` - Use a Random Forest Classifier.
- `--lrc` - Use a Logistic Regression Classifier.
- `--fnn` - Use a Feedforward Neural Network.

**Feature Selection (RFE) Settings:**
- `-r`, `--rfe` - Perform Recursive Feature Elimination (RFE) on the selected models.
- `--min_features` : `int` - The feature threshold to stop RFE (default: `1`).
- `--cut_features` : `int` - The number of features to drop per RFE iteration (default: `1`).
- `--rfe_proxy` : `[dtc, rfc, lrc]` - Selects the proxy model used to calculate feature importance for models that do not natively support it, mainly the FNN (default: `dtc`).

**Graphing Settings:**
- `--graph` - Generates and saves a `.pdf` graph of model accuracy, training time, and size versus the number of features after RFE.

The arguments used to get the results seen in the thesis was:

```
python .\main.py --path Dataset --dataset packet --classificaton multiclass8 --rows 100000 --random_state 0 --save_model --save_report --save_dataset --dtc --rfc --lrc --fnn --rfe --min_features 1 --cut_features 1 --rfe_proxy dtc --graph
```

### `load_test`

`load_test` is the tool used for load testing the models created by `main`. Written for Linux running on a Raspberry Pi 3b+.

#### Dependencies

**Third-party Libraries:**:
- `psutil`
- `numpy`
- `pandas`
- `matplotlib`

To quickly install the required third-party libraries, use:

```bash
pip install psutil numpy pandas matplotlib
```

**Python Standard Libraries:**
- time
- pickle
- argparse
- sys
- warnings
- os

#### Usage

**Arguments**

**Target Selection:**
- `--all` - Automatically searches for and tests all `.pkl` model files in the current directory and subdirectories.
- `--model` : `str` - Path to a single `.pkl` model file to test.

**Data & Output Settings:**
- `--data` : `str` - Path to the input `.csv` dataset.
- `--name` : `str` - Prefix for the output `.csv` results files (default: `test_results`).
- `--graph` - Generates and saves a `.png` visualization of the CPU usage and Rows/Sec throughput. *Note: Cannot be used with `--all`.*

`taskset` was used to bind the program to a single core for the results seen in the thesis:

```bash
taskset -c 0 python3 load_test.py --model dtc_load_tester.pkl --data dataset_packet_multiclass8_n10000.csv --name dtc --graph
```


### `helper_scripts`

`helper_scripts` is a collection of `.ipynb` and `.py` helper scripts used to create graphs, optimize models, and explore the CICIoT2024 dataset.
- `model_dtc` - 
- `model_rfc` - 
- `model_lrc` - 
- `model_fnn` - 
- `graphing` - 
- `dataset` - 

## Changelog

### v1.0 - April 2026
- Initial release accompanying the thesis submission. Includes full dataset preprocessing, model training, RFE implementation, and Raspberry Pi load testing scripts used during the thesis.

## Author

Daniel Larsen Meek
Bachelor in Cyber Security, Noroff University College

## Licese

All Rights Reserved