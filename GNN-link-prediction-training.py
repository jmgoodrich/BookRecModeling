import networkx as nx
import pandas as pd
import random
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

# === Load or Build Graph ===
csv_file_path = "C:\\Users\\fahim\\Documents\\Courses\\AI Modeling\\Books_rating_file.csv"
graph_path = "C:\\Users\\fahim\\Documents\\Courses\\AI Modeling\\book_graph.pkl"
chunk_size = 100000

G = nx.Graph()

if os.path.exists(graph_path):
    print(f"Loading existing graph from {graph_path}...")
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
else:
    print(f"No existing graph found. Building graph from CSV: {csv_file_path}...")
    total_rows = 0

    for chunk in pd.read_csv(csv_file_path, chunksize=chunk_size):
        chunk = chunk.dropna(subset=['User_id', 'Id', 'review/score'])
        chunk['review/score'] = pd.to_numeric(chunk['review/score'], errors='coerce')
        chunk = chunk.dropna(subset=['review/score'])

        for _, row in chunk.iterrows():
            user = f"user_{row['User_id']}"
            item = f"item_{row['Id']}"
            score = float(row['review/score'])
            G.add_edge(user, item, weight=score)

        total_rows += len(chunk)
        print(f"Processed {total_rows:,} rows...")

    print(f"Finished building graph with {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges.")
    with open(graph_path, 'wb') as f:
        pickle.dump(G, f)
    print(f"Graph saved to {graph_path}.")

# === Prepare nodes and edges ===
all_nodes = sorted(G.nodes())
users = [node for node in all_nodes if node.startswith('user_')]
items = [node for node in all_nodes if node.startswith('item_')]

edges = [(u, v, d['weight']) for u, v, d in G.edges(data=True)]

# === Split edges ===
train_edges, test_edges = train_test_split(edges, test_size=0.2, random_state=14)

# === Map node IDs to indices ===
node_id_map = {node: i for i, node in enumerate(all_nodes)}
num_nodes = len(all_nodes)

# === Convert edges to PyG format ===
def edge_list_to_index(edges):
    return [(node_id_map[u], node_id_map[v]) for u, v, _ in edges]

train_edge_index = torch.tensor(edge_list_to_index(train_edges), dtype=torch.long).t().contiguous()

def get_negative_edges(num_samples):
    neg_edges = set()
    while len(neg_edges) < num_samples:
        u = random.choice(users)
        v = random.choice(items)
        if not G.has_edge(u, v):
            neg_edges.add((node_id_map[u], node_id_map[v]))
    return torch.tensor(list(neg_edges), dtype=torch.long).t().contiguous()

pos_edge_index = torch.tensor(edge_list_to_index(test_edges), dtype=torch.long).t().contiguous()
neg_edge_index = get_negative_edges(len(test_edges))

# === Define GCN model with embeddings ===
class GCNLinkPredictor(torch.nn.Module):
    def __init__(self, num_nodes, embed_dim, hidden_channels):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_nodes, embed_dim)
        self.conv1 = GCNConv(embed_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def encode(self, edge_index):
        x = self.embedding.weight
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.3, training=self.training)  # ← Dropout after first conv
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

    def forward(self, edge_index, edge_pairs):
        z = self.encode(edge_index)
        return self.decode(z, edge_pairs)

# === Setup for training ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCNLinkPredictor(num_nodes=num_nodes, embed_dim=64, hidden_channels=128).to(device)

train_edge_index = train_edge_index.to(device)
pos_edge_index = pos_edge_index.to(device)
neg_edge_index = neg_edge_index.to(device)

# === Fix negative samples for training (new) ===
fixed_neg_train_edges = get_negative_edges(train_edge_index.size(1)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# === Training loop ===
def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_edge_index)

    pos_score = model.decode(z, train_edge_index)
    pos_label = torch.ones(pos_score.size(0), device=device)

    neg_score = model.decode(z, fixed_neg_train_edges)
    neg_label = torch.zeros(neg_score.size(0), device=device)

    scores = torch.cat([pos_score, neg_score], dim=0)
    labels = torch.cat([pos_label, neg_label], dim=0)

    loss = F.binary_cross_entropy_with_logits(scores, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test():
    model.eval()
    z = model.encode(train_edge_index)

    pos_pred = model.decode(z, pos_edge_index).sigmoid()
    neg_pred = model.decode(z, neg_edge_index).sigmoid()

    preds = torch.cat([pos_pred, neg_pred]).cpu()
    labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))])

    auc = roc_auc_score(labels, preds)
    ap = average_precision_score(labels, preds)
    return auc, ap

# === Run training ===
print(f"Training on {num_nodes} nodes and {train_edge_index.size(1)} edges...")

train_losses = []
aucs = []
aps = []

for epoch in range(1, 101):
    loss = train()
    auc, ap = test()

    train_losses.append(loss)
    aucs.append(auc)
    aps.append(ap)

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}")

# === Save model after training ===
save_path = "C:\\Users\\fahim\\Documents\\Courses\\AI Modeling\\gcn_link_predictor.pth"

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'num_nodes': num_nodes,
    'node_id_map': node_id_map
}, save_path)

print(f"Model saved to {save_path}")

# Save your training history
history = {
    'epochs': list(range(1, 101)),  # every epoch
    'losses': train_losses,
    'aucs': aucs,
    'aps': aps
}

with open("C:\\Users\\fahim\\Documents\\Courses\\AI Modeling\\training_history.pkl", "wb") as f:
    pickle.dump(history, f)

print("Training history saved!")