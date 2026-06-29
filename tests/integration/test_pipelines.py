import pytest
import polars as pl
import torch
from pathlib import Path

from logmark.datasets import LoghubDataset
from logmark.parsers import get_parser
from logmark.windowing import LogWindowing, WindowType
from logmark.collators.WindowCollator import WindowCollator
from logmark.embedders import build_encoder
from logmark.models import build_model
from logmark.evaluation import compute_anomaly_metrics

@pytest.mark.parametrize("dataset_name", ["Apache", "HDFS", "BGL"])
def test_end_to_end_pipeline(dataset_name, tmp_path):
    """End-to-End integration test across the entire Logmark framework"""
    # Define primary device (CUDA if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset setup
    dataset = LoghubDataset(dataset_name, data_dir=str(tmp_path / "data"))
    dataset.download()
    
    # Parsing (using Drain)
    parser = get_parser("drain", sim_th=0.5, depth=4)
    
    parsed_data = []
    vocab_map = {}
    current_vocab_id = 1
    
    # Process the first 200 lines
    for i, line in enumerate(dataset.get_log_iterator()):
        if i >= 200:
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
        
    assert len(parsed_data) > 0, "No data parsed from the dataset"
    
    # Windowing (Count based)
    df_raw = pl.DataFrame(parsed_data).lazy()
    window_manager = LogWindowing(
        window_type=WindowType.COUNT,
        window_size=50,
        step_size=25
    )
    df_windowed = window_manager.transform(df_raw).collect()
    assert df_windowed.height > 0
    
    # Collation
    collator = WindowCollator(padding_value=0)
    padded_input, mask = collator.collate(df_windowed, token_col="event_id")
    padded_input = padded_input.to(device)
    mask = mask.to(device)
    
    # Embedding
    vocab_size = current_vocab_id
    embed_dim = 16
    embedder = build_encoder("mean_pool", vocab_size=vocab_size, dim=embed_dim)
    embedder = embedder.to(device)
    
    tensor_output = embedder(padded_input, mask)
    assert tensor_output.device.type == device.type
    assert tensor_output.shape == (df_windowed.height, embed_dim)
    
    # ML Model (Isolation Forest Baseline)
    model = build_model("isolation_forest", n_estimators=10)
    
    # Fit the model
    model.fit(tensor_output)
    assert model.is_fitted
    
    # Predict
    y_pred = model.predict(tensor_output)
    assert y_pred.device.type == tensor_output.device.type
    assert y_pred.shape == (df_windowed.height,)
    
    # Evaluation
    # TODO: add true labels (train/test)
    torch.manual_seed(42)
    y_true_mock = torch.randint(0, 2, (df_windowed.height,), device=device)
    
    metrics = compute_anomaly_metrics(y_true_mock, y_pred)
    
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    
    for k, v in metrics.items():
        assert 0.0 <= v <= 1.0, f"Metric {k} is out of bounds: {v}"
