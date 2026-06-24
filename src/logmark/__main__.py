import sys
import polars as pl
from logmark.datasets import LoghubDataset
from logmark.parsers import get_parser
from logmark.windowing import LogWindowing, WindowType
from logmark.collators.WindowCollator import WindowCollator
from logmark.embedders import build_encoder


def run_pipeline(dataset_name: str = "Apache", num_lines: int = 1000):
    print(f"\n{'='*60}")
    print(f" LOGMARK PIPELINE TUTORIAL: {dataset_name} Dataset")
    print(f"{'='*60}\n")

    # Load Dataset
    print(f"[1/5] Initializing Loghub Dataset: {dataset_name}")
    dataset = LoghubDataset(dataset_name)
    
    # Download the dataset if it's not already cached locally
    print(f"Downloading/Extracting dataset (if needed)...")
    dataset.download(debug=False)
    print(f"Dataset is ready!")

    # Parsing Logs
    print("\n[2/5] Parsing raw logs using Drain Parser...")
    # Initialize Drain parser
    parser = get_parser("drain", sim_th=0.5, depth=4)
    
    parsed_data = []
    vocab_map = {}
    current_vocab_id = 1  # 0 is reserved for padding

    # Iterate over the raw text log stream
    for i, line in enumerate(dataset.get_log_iterator()):
        if i >= num_lines:
            break
            
        # Parse the raw line into a cluster ID
        cluster_id = parser.get_cluster_id(line)
        
        # Map the string cluster ID to an integer
        if cluster_id not in vocab_map:
            vocab_map[cluster_id] = current_vocab_id
            current_vocab_id += 1

        parsed_data.append({
            "timestamp": i, # Mocking timestamp
            "event_id": vocab_map[cluster_id],
            "raw_cluster_id": cluster_id
        })

    print(f"Successfully parsed {len(parsed_data)} lines.")
    print(f"Discovered {len(vocab_map)} unique log templates (Event IDs).")

    if not parsed_data:
        print("No data parsed. Exiting.")
        return

    # Windowing
    print("\n[3/5] Applying Windowing Logic using Polars...")
    # Convert parsed dictionaries to Polars LazyFrame
    df_raw = pl.DataFrame(parsed_data).lazy()

    # Count-based sliding window
    window_manager = LogWindowing(
        window_type=WindowType.COUNT,
        window_size=50,
        step_size=25
    )

    df_windowed = window_manager.transform(df_raw).collect()
    print(f"Generated {df_windowed.height} sequential sliding windows.")
    print("Preview of the first window (event_ids):")
    print(df_windowed.select(["event_id"]).head(1).item())

    # Data Collation for PyTorch
    print("\n[4/5] Collating Polars Lists into PyTorch Tensors...")
    collator = WindowCollator(padding_value=0)
    padded_input, mask = collator.collate(df_windowed, token_col="event_id")
    
    print(f"Input Tensor Shape: {padded_input.shape} (batch_size, max_seq_len)")
    print(f"Mask Tensor Shape:  {mask.shape}")

    # Embedding
    print("\n[5/5] Passing tensors through Deep Learning Embedders...")
    vocab_size = current_vocab_id + 1  # Account for the 0 padding index

    # Mean Pool
    print("\n> Strategy A: Mean Pooling Embedder")
    print("(Embeds each event to a dense vector, then averages across the window)")
    
    embed_dim = 32
    embedder_a = build_encoder("mean_pool", vocab_size=vocab_size, dim=embed_dim)
    
    tensor_a = embedder_a(padded_input, mask)
    print(f"Output Shape: {tensor_a.shape} -> [Num Windows, Embedding Dim]")
    print(f"Sample features (first 5 dims): {tensor_a[0, :5].detach().numpy().round(4)}")

    # Count Vector
    print("\n> Strategy B: Count Vector Embedder")
    print("(Creates a sparse frequency distribution over the vocabulary)")
    
    embedder_b = build_encoder("count", vocab_size=vocab_size)
    
    tensor_b = embedder_b(padded_input, mask)
    print(f"Output Shape: {tensor_b.shape} -> [Num Windows, Vocab Size]")
    print(f"Sample frequencies (first 10 vocab IDs): {tensor_b[0, :10].detach().numpy()}")
    
    print(f"\n{'='*60}")
    print(" Pipeline Execution Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # You can change the dataset here
    dataset_to_run = "Apache"
    
    if len(sys.argv) > 1:
        dataset_to_run = sys.argv[1]
        
    run_pipeline(dataset_name=dataset_to_run, num_lines=1000)
