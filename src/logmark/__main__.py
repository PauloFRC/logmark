import polars as pl
from logmark.datasets import LoghubDataset
from logmark.parsers import get_parser

from logmark.windowing import LogWindowing, WindowType
from logmark.embedders import build_encoder
from logmark.collators.WindowCollator import WindowCollator

def main():
    print("=== Pipeline Demo ===")

    dataset_name = "HDFS"
    dataset = LoghubDataset(dataset_name)
    dataset.download()
    log_path = dataset.get_log_path()

    drain = get_parser("drain", sim_th=0.5, depth=4)

    print("\n[1/3] Parsing logs with Drain...")
    parsed_data = []
    vocab_map = {}
    current_vocab_id = 1

    with open(log_path, "r") as f:
        for i, line in enumerate(f):
            if i >= 1000: break

            cluster_id = drain.get_cluster_id(line.strip())

            if cluster_id not in vocab_map:
                vocab_map[cluster_id] = current_vocab_id
                current_vocab_id += 1

            parsed_data.append({
                "timestamp": i,
                "event_id": vocab_map[cluster_id],
                "raw_cluster_id": cluster_id
            })

    print(f"Parsed {len(parsed_data)} lines. Unique Event IDs found: {len(vocab_map)}")

    print("\n[2/3] Applying Count-based Windowing via Polars...")
    df_raw = pl.DataFrame(parsed_data).lazy()

    window_manager = LogWindowing(
        window_type=WindowType.COUNT,
        window_size=50,
        step_size=25
    )

    df_windowed = window_manager.transform(df_raw).collect()
    print(f"Generated {df_windowed.height} sliding windows.")
    print(df_windowed.select(["event_id"]).head(2))

    print("\n[3/3] Testing PyTorch Embedders...")
    vocab_size = current_vocab_id + 1  # Include padding index 0

    collator = WindowCollator()
    padded_input, mask = collator.collate(df_windowed, token_col="event_id")

    print("\n--- Flow A: Pre-Window (Semantic Pooling) ---")
    embedder_a = build_encoder("mean_pool", vocab_size=vocab_size, dim=64)

    tensor_a = embedder_a(padded_input, mask)
    print(f"Flow A Output Shape: {tensor_a.shape}")
    print(f"Sample vector (first 5 dims): {tensor_a[0, :5].detach().numpy()}")

    print("\n--- Flow B: Post-Window (Frequency Distribution) ---")
    embedder_b = build_encoder("count", vocab_size=vocab_size)

    tensor_b = embedder_b(padded_input, mask)
    print(f"Flow B Output Shape: {tensor_b.shape}")
    print(f"Sample frequency counts (first 10 vocab IDs): {tensor_b[0, :10].detach().numpy()}")

if __name__ == "__main__":
    main()
