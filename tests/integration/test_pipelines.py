import pytest
import polars as pl
from pathlib import Path

from logmark.datasets import LoghubDataset
from logmark.parsers import get_parser
from logmark.windowing import LogWindowing, WindowType
from logmark.collators.WindowCollator import WindowCollator
from logmark.embedders import build_encoder

@pytest.mark.parametrize("dataset_name", ["Apache", "HDFS", "BGL"])
def test_end_to_end_pipeline(dataset_name, tmp_path):
    """
    Integration test including the steps:
    1. Download and load a dataset
    2. Parse logs
    3. Window the events
    4. Collate into tensors
    5. Feed through an embedder
    """
    dataset = LoghubDataset(dataset_name, data_dir=str(tmp_path / "data"))
    dataset.download()
    
    # Using Drain
    parser = get_parser("drain", sim_th=0.5, depth=4)
    
    parsed_data = []
    vocab_map = {}
    current_vocab_id = 1
    
    # Process only first 200 lines
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
    
    # Count based windowing
    df_raw = pl.DataFrame(parsed_data).lazy()
    window_manager = LogWindowing(
        window_type=WindowType.COUNT,
        window_size=50,
        step_size=25
    )
    df_windowed = window_manager.transform(df_raw).collect()
    
    assert df_windowed.height > 0, "No windows generated from the parsed logs"
    
    # Collation
    collator = WindowCollator(padding_value=0)
    padded_input, mask = collator.collate(df_windowed, token_col="event_id")
    
    # Embedding
    vocab_size = current_vocab_id + 1
    embed_dim = 16
    embedder = build_encoder("mean_pool", vocab_size=vocab_size, dim=embed_dim)
    tensor_output = embedder(padded_input, mask)
    
    # Verification
    assert tensor_output.shape == (df_windowed.height, embed_dim)
