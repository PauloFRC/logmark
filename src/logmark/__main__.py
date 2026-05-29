from logmark.datasets import LoghubDataset
from logmark.parsers import DrainParser

def main():
    print("=== Demo ===")
    
    dataset_name = "HDFS"
    dataset = LoghubDataset(dataset_name)
    
    print(f"Checking for {dataset_name} dataset...")
    dataset.download()
    
    log_path = dataset.get_log_path()
    print(f"Log file found at: {log_path}")
    
    parser = DrainParser()
    
    print("Parsing first 10 lines...")
    with open(log_path, "r") as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            cluster_id = parser.get_cluster_id(line.strip())
            print(f"Line {i+1}: Cluster ID {cluster_id}")
            
    print("\nTemplates found:")
    for i, template in enumerate(parser.get_templates()):
        print(f"{i+1}: {template}")

if __name__ == "__main__":
    main()
