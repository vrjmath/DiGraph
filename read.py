import torch
import numpy as np
from collections import Counter

dataset_file = "devmap.pth"
dataset = torch.load(dataset_file, weights_only=False)  # allow full pickling
print(f"Loaded {len(dataset)} graphs")


# 1️⃣ Node count statistics
node_counts = np.array([g.num_nodes for g in dataset])
print("\nNode count stats:")
print(f"  min: {node_counts.min()}")
print(f"  max: {node_counts.max()}")
print(f"  median: {np.median(node_counts)}")
print(f"  25th percentile: {np.percentile(node_counts, 25)}")
print(f"  75th percentile: {np.percentile(node_counts, 75)}")

# 2️⃣ Edge count statistics
edge_counts = np.array([g.num_edges for g in dataset])
print("\nEdge count stats:")
print(f"  min: {edge_counts.min()}")
print(f"  max: {edge_counts.max()}")
print(f"  median: {np.median(edge_counts)}")
print(f"  25th percentile: {np.percentile(edge_counts, 25)}")
print(f"  75th percentile: {np.percentile(edge_counts, 75)}")

# 3️⃣ Label distribution
labels = np.array([g.y.item() for g in dataset])
label_counter = Counter(labels)
print("\nLabel distribution:")
for lbl, count in label_counter.items():
    print(f"  Label {lbl}: {count} graphs ({count/len(dataset)*100:.2f}%)")

# 4️⃣ Node feature frequency
all_node_ids = torch.cat([g.x.squeeze(1) for g in dataset], dim=0).tolist()
node_counter = Counter(all_node_ids)
print("\nTop 20 most frequent node feature indices:")
for node_id, count in node_counter.most_common(20):
    print(f"  Node {node_id}: {count} occurrences")
