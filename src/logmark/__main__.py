"""
Logmark Framework - End-to-End Pipeline Example
===============================================
This script demonstrates how to use the Logmark framework to build a complete 
log processing pipeline. It covers:
1. Downloading and loading a dataset (Apache, HDFS, or BGL)
2. Parsing raw logs into structured events using Drain
3. Applying sliding windows (Count-based) over the events using Polars
4. Collating the windowed events into PyTorch tensors
5. Passing the tensors through deep learning embedders (Mean Pooling & Count/BoW)
6. Detecting anomalies with a PyTorch-wrapped Isolation Forest
7. Evaluating performance and rendering a live Dash dashboard
"""

import sys
import torch
import polars as pl
from logmark.datasets import LoghubDataset
from logmark.parsers import get_parser
from logmark.windowing import LogWindowing, WindowType
from logmark.collators.WindowCollator import WindowCollator
from logmark.embedders import build_encoder
from logmark.models import build_model
from logmark.evaluation import compute_anomaly_metrics, launch_dashboard

def run_pipeline(dataset_name: str = "Apache", num_lines: int = 1000):
    print(f"\n{'='*60}")
    print(f" LOGMARK PIPELINE TUTORIAL: {dataset_name} Dataset")
    print(f"{'='*60}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Framework dynamically resolving hardware. Running on: {device.type.upper()}\n")

    # Load Dataset
    print(f"[1/7] Initializing Loghub Dataset: {dataset_name}")
    dataset = LoghubDataset(dataset_name)
    dataset.download(debug=False)

    # Parsing Logs
    print("\n[2/7] Parsing raw logs using Drain Parser...")
    parser = get_parser("drain", sim_th=0.5, depth=4)
    
    parsed_data = []
    vocab_map = {}
    current_vocab_id = 1  # 0 is reserved for padding

    for i, line in enumerate(dataset.get_log_iterator()):
        if i >= num_lines:
            break
            
        cluster_id = parser.get_cluster_id(line)
        
        if cluster_id not in vocab_map:
            vocab_map[cluster_id] = current_vocab_id
            current_vocab_id += 1

        parsed_data.append({
            "timestamp": i,
            "event_id": vocab_map[cluster_id],
            "raw_cluster_id": cluster_id
        })

    if not parsed_data:
        print("      No data parsed. Exiting.")
        return

    # Windowing
    print("\n[3/7] Applying Windowing Logic using Polars...")
    df_raw = pl.DataFrame(parsed_data).lazy()

    window_manager = LogWindowing(
        window_type=WindowType.COUNT,
        window_size=50,
        step_size=25
    )

    df_windowed = window_manager.transform(df_raw).collect()

    # Data Collation for PyTorch
    print(f"\n[4/7] Collating Polars Lists into PyTorch Tensors on {device.type.upper()}...")
    collator = WindowCollator(padding_value=0)
    padded_input, mask = collator.collate(df_windowed, token_col="event_id")
    
    padded_input = padded_input.to(device)
    mask = mask.to(device)

    # Embedding Strategies
    print("\n[5/7] Passing tensors through Deep Learning Embedders...")
    vocab_size = current_vocab_id

    print("      > Extracting features using Mean Pooling Embedder")
    embed_dim = 32
    embedder_a = build_encoder("mean_pool", vocab_size=vocab_size, dim=embed_dim)
    embedder_a = embedder_a.to(device)
    
    tensor_features = embedder_a(padded_input, mask)

    # Anomaly Detection Models
    print("\n[6/7] Training Baseline Model: Isolation Forest...")
    print("      > Utilizing PyTorch wrapper to offload tensor to CPU tree algorithms seamlessly")
    model = build_model("isolation_forest", n_estimators=50, contamination=0.1)
    
    model.fit(tensor_features)
    y_pred = model.predict(tensor_features)
    
    print(f"      > Model predictions shape: {y_pred.shape} (Predicted Anomalies: {y_pred.sum().item()})")

    # Evaluation & Dashboard
    print("\n[7/7] Generating evaluation metrics and launching Dashboard...")
    
    # Mocking Ground Truths for Demonstration
    torch.manual_seed(42)
    y_true_mock = torch.randint(0, 2, (df_windowed.height,), device=device)
    
    # Calculate Metrics
    metrics = compute_anomaly_metrics(y_true_mock, y_pred)
    
    # Compile Results
    results = [
        {
            "dataset": dataset_name,
            "model": "Isolation Forest (MeanPool)",
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"]
        }
    ]
    
    print(f"\n{'='*60}")
    print(" Pipeline Execution Complete! Starting Dash Server...")
    print(f"{'='*60}\n")
    
    # Launch Dashboard
    launch_dashboard(results, port=8050)


if __name__ == "__main__":
    # Change the dataset here
    dataset_to_run = "Apache"
    
    if len(sys.argv) > 1:
        dataset_to_run = sys.argv[1]
        
    run_pipeline(dataset_name=dataset_to_run, num_lines=1000)
