import torch
from torch_geometric.loader import DataLoader
from model import GCN  # or GraphSAGE, GAT, MLP
from utils import compute_metrics
from torch_geometric.utils import to_scipy_sparse_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
dataset = torch.load("devmap.pth")  # list of PyG data objects

# Collect all unique node IDs across dataset
all_node_ids = torch.cat([data.x for data in dataset]).unique()
# Map original node ID -> contiguous ID starting from 0
id_map = {old.item(): new for new, old in enumerate(all_node_ids)}
num_node_ids = len(id_map)

# Create embedding table
EMB_DIM = 32
node_embedding = torch.nn.Embedding(num_node_ids, EMB_DIM).to(device)

# Replace node IDs with mapped IDs
for data in dataset:
    mapped_x = torch.tensor([id_map[i.item()] for i in data.x], device=device)
    data.x = node_embedding(mapped_x)  # no detach



# Split dataset
n = len(dataset)
train_dataset = dataset[:int(0.7 * n)]
val_dataset = dataset[int(0.7 * n):int(0.85 * n)]
test_dataset = dataset[int(0.85 * n):]

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Initialize model
model = GCN(in_dim=EMB_DIM, hidden_dim=64, out_dim=1).to(device)
optimizer = torch.optim.Adam(list(model.parameters()) + list(node_embedding.parameters()), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

# Training loop
for epoch in range(1, 500):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        adj_matrix = to_scipy_sparse_matrix(data.edge_index)   # get adjacency
        adj_tensor = torch.from_numpy(adj_matrix.toarray()).float().to(device)  # dense tensor

        out = model(data.x, data.edge_index, data.batch)
  # GraphSAGE should now accept dense adj
        loss = criterion(out, data.y.float().unsqueeze(1))

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Evaluate
    val_metrics = compute_metrics(model, val_loader, device)
    print(
        f"Epoch {epoch}: Train Loss {total_loss:.4f}, "
        f"Val F1 {val_metrics['f1']:.4f}, ROC-AUC {val_metrics['roc_auc']:.4f}"
    )
