# On the Privacy Risks of Platform-Wide Telemetry in Virtual Reality Systems  

This repository contains the inference framework (called **VRScanner**) introduced in our submission. It is designed to fingerprint encrypted traffic generated entirely by the platform OS (i.e., Meta Horzion OS). By analyzing protocol-level metadata from packet sequences, it identifies app usage (i.e., VR application) and in-app behavior (i.e., virtual world) without accessing payload contents.

The framework extracts protocol header fields from encrypted packets and applies normalization or tokenization techniques. A modular deep learning classifier then selects optimal input encodings and sequential models (e.g., GRU, LSTM) to enable accurate classification. This design avoids manual feature engineering and supports fine-grained analysis.

## Datasets

We provide small sample datasets to support replication. The full datasets are not released due to privacy and storage constraints.

### VR App Dataset

- **Collection Period**: April 25 – September 4, 2024 (public lab), and June 28 – August 27, 2024 (campus network)
- **Size**: ~600 GB  
- **Files**: 84,492 `.pcap` files (851–966 per app)
- **Details**: All free 91 apps from official Meta store at the time of collection (February 15, 2024).
- **Metadata**: Application name, device ID (1–8), and headset model (Quest 2, 3, or Pro)  

### Virtual World Dataset

- **Collection Period**: March 12 – May 8, 2025  
- **Size**: ~120 GB  
- **Files**: 15,228 `.pcap` files (300–320 per world)  
- **Details**: Captured from the top 50 most-visited VRChat worlds using five VR headsets.  
- **Note**: Only private worlds were launched to avoid interfering with other users.

## Getting Started

### Prerequisites

Before running VRScanner, please make sure your environment satisfies the following requirements.

- **Python version**: 3.9 or higher
- **CUDA-compatible GPU** (recommended for faster training)

### Python Dependencies

You can install the necessary Python packages via:

- pip install -r requirements.txt

## Example: Inference Using Only IP Length field

You can run VR app inference directly using the following command:
Note: Due to storage constraints, we could not upload the full preprocessed datasets. Instead, we provide representative sample pcap files.


```bash
# For VR application inference
./default-fingerprinting.sh meta meta-free-apps


## Project Structure

```text
VRScanner/
├── vrscanner.py                      # Entry point to run the full VRScanner pipeline
├── vrscanner/
│   ├── __init__.py                   # Package initializer
│   ├── config.py                     # Argument parser and configuration handler
│   ├── utils.py                      # Logging and leaderboard helpers
│   ├── model.py                      # Model definition (VRScannerModel)
│   ├── core/                         # Core machine learning components
│   │   ├── __init__.py
│   │   ├── model_selection.py        # Model builder (get_model)
│   │   ├── trainer.py                # Training procedures (train, train_kfold)
│   │   └── evaluator.py              # Evaluation functions (evaluate, evaluate_loss, etc.)
│   ├── experiment/                   # High-level experiment logic
│   │   ├── __init__.py
│   │   ├── general.py                # Unified handler for step1, step2, step3, training
│   │   ├── longitudinal.py           # Longitudinal evaluation across time windows
│   │   └── openworld.py              # Open-world classification and detection
│   ├── loader/                       # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── data_loader.py            # Load CSVs, extract metadata and labels
│   │   └── batch_loader.py           # Normalize/tokenize metadata and build batches
│
├── preprocessor/
│   ├── preprocessing-pcap.py               # PCAP preprocessing for Meta Free Apps
│   ├── preprocessing-pcap-for-vrchat.py    # PCAP preprocessing for VRChat Worlds
│   ├── extracting-metadata.py              # Extract protocol header metadata (IP/TCP fields) from PCAPs
│   ├── loading-metadata.py                 # Load extracted metadata into training-ready formats
│   ├── load-metadata.sh                    # Shell wrapper to automate metadata loading
│   └── misc/
│       └── meta-ip-prefix.list       # Known IP prefix list for Meta servers (AS32934)
├── scripts/
│   ├── step1-find-optimal-normalization.sh      # Run step1: explore normalization methods
│   ├── step2-find-optimal-recurrent-layer.sh    # Run step2: explore RNN architectures
│   ├── step3-train-on-all-metadata.sh           # Run step3: evaluate feature and temperal attention scores
│   ├── default-fingerprinting.sh                # Train for VR app fingerprinting
│   ├── default-fingerprinting-for-world.sh      # Train for Virtual world fingerprinting 
│   ├── longitudinal-evaluation.sh               # Perform longitudinal evaluation
│   └── openworld-evaluation.sh                  # Perform open-world evaluation
│
├── pcap-samples/
│   ├── meta-free-apps/           # Sample pcap files for VR application dataset
│   └── virtual-worlds/           # Sample pcap files for Virtual world dataset
├── output/
│   ├── *.debug           # Training logs (stdout + debug info)
│   └── *.csv             # Leaderboard metrics (accuracy, F1, etc.)

