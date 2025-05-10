import json
import os
import random
from typing import Dict, List, Any

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv

##############################################################################
# 1) Data utilities: parse and embed literals
##############################################################################
def parse_literal(literal: str):
    sign = +1
    core = literal.strip()
    if core.startswith("¬"):
        sign = -1
        core = core[1:].strip()
    pred = core.split("(", 1)[0]
    inside = core[len(pred) + 1 : -1].strip()
    if inside == "":
        args = []
    else:
        args = [x.strip() for x in inside.split(",")]
    return sign, pred, args

def embed_literal(literal: str, predicate_to_idx: Dict[str, int], max_args: int = 3) -> torch.Tensor:
    sign, pred, args = parse_literal(literal)
    sign_feature = 0 if sign > 0 else 1
    pred_idx = predicate_to_idx.get(pred, 0)  # default to 0 if unknown
    arg_types = []
    for arg in args[:max_args]:
        if "(" in arg and arg.endswith(")"):
            arg_types.append(2)
        elif arg and arg[0].isupper():
            arg_types.append(0)
        else:
            arg_types.append(1)
    while len(arg_types) < max_args:
        arg_types.append(-1)
    feat = [sign_feature, float(pred_idx)] + [float(x) for x in arg_types]
    return torch.tensor(feat, dtype=torch.float)

def build_graph_from_example(example: Dict[str, Any],
                             predicate_to_idx: Dict[str, int],
                             max_args: int = 3) -> Data:
    clauses = example["clauses"]
    resolvable_pairs = example["resolvable_pairs"]
    best_pair = example["best_pair"]
    node_map = {}
    node_features = []
    global_node_idx = 0
    for ci, clause in enumerate(clauses):
        for li, lit_str in enumerate(clause):
            node_map[(ci, li)] = global_node_idx
            feat = embed_literal(lit_str, predicate_to_idx, max_args)
            node_features.append(feat)
            global_node_idx += 1
    x = torch.stack(node_features, dim=0)
    edge_src = []
    edge_dst = []
    edge_labels = []
    for pair in resolvable_pairs:
        i, idxA = pair["clauseA_index"], pair["literalA_index"]
        j, idxB = pair["clauseB_index"], pair["literalB_index"]
        nA = node_map[(i, idxA)]
        nB = node_map[(j, idxB)]
        is_best = False
        if best_pair is not None:
            if (i == best_pair["clauseA_index"] and
                idxA == best_pair["literalA_index"] and
                j == best_pair["clauseB_index"] and
                idxB == best_pair["literalB_index"]):
                is_best = True
        edge_src.append(nA)
        edge_dst.append(nB)
        edge_labels.append(1 if is_best else 0)
    if len(edge_src) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        y = torch.empty(0, dtype=torch.long)
    else:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        y = torch.tensor(edge_labels, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    return data

##############################################################################
# 2) PyTorch Geometric Dataset class
##############################################################################
class ClauseResolutionDataset(Dataset):
    def __init__(self, jsonl_file: str, predicate_list: List[str], max_args: int = 3):
        super().__init__(root=None)
        self.jsonl_file = jsonl_file
        self.max_args = max_args
        self.predicate_to_idx = {pred: i+1 for i, pred in enumerate(predicate_list)}
        with open(jsonl_file, "r") as f:
            lines = f.readlines()
        self.examples = [json.loads(line) for line in lines]

    def len(self):
        return len(self.examples)

    def get(self, idx):
        example = self.examples[idx]
        data = build_graph_from_example(example, self.predicate_to_idx, max_args=self.max_args)
        return data

##############################################################################
# 3) Define a GNN model for edge classification
##############################################################################
class EdgeClassifierGNN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=2):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(2*hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        src, dst = edge_index
        edge_reps = torch.cat([h[src], h[dst]], dim=1)
        logits = self.edge_mlp(edge_reps)
        return logits

##############################################################################
# 4) Train/Test split and training functions
##############################################################################
def split_dataset(dataset: Dataset, train_ratio=0.8):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split_point = int(len(dataset) * train_ratio)
    train_idx = indices[:split_point]
    test_idx  = indices[split_point:]
    train_set = torch.utils.data.Subset(dataset, train_idx)
    test_set = torch.utils.data.Subset(dataset, test_idx)
    return train_set, test_set

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index)
        if batch.y.size(0) == 0:
            continue
        loss = F.cross_entropy(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader) if len(loader) > 0 else 0.0

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            logits = model(batch.x, batch.edge_index)
            if batch.y.size(0) == 0:
                continue
            pred = logits.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    return correct / total if total > 0 else 0.0

##############################################################################
# 5) Fine-tuning: Load pre-trained model (or train from scratch) and fine-tune once
##############################################################################
def main():
    # Configuration
    jsonl_file = "toy_gnn_dataset.jsonl"   # Your dataset file
    known_predicates = ["Pred1", "Pred2", "Pred3"]  # As used during training/fine-tuning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    finetune_epochs = 10                    # Fine-tune for a fixed number of epochs
    finetune_lr = 1e-4                      # Use a lower learning rate for fine-tuning
    pretrained_checkpoint = "Models/gnn_model_00.pt"  # Pre-trained model checkpoint path

    # Load dataset and split it
    dataset = ClauseResolutionDataset(jsonl_file=jsonl_file, predicate_list=known_predicates, max_args=3)
    train_set, test_set = split_dataset(dataset, train_ratio=0.8)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    test_loader  = DataLoader(test_set, batch_size=8, shuffle=False)

    # Define model. The node feature vector is of size 5 (sign, predicate, 3 arg types)
    in_dim = 5
    hidden_dim = 64
    model = EdgeClassifierGNN(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=2).to(device)

    # Attempt to load pre-trained weights.
    if os.path.exists(pretrained_checkpoint):
        model.load_state_dict(torch.load(pretrained_checkpoint, map_location=device))
        print(f"Loaded pre-trained model from {pretrained_checkpoint}")
    else:
        # If no pre-trained checkpoint exists, we proceed to train the model from scratch.
        print(f"Pre-trained checkpoint {pretrained_checkpoint} not found. Training model from scratch.")

    # Set up optimizer for (fine-)tuning.
    optimizer = torch.optim.Adam(model.parameters(), lr=finetune_lr)

    # Fine-tuning (or initial training) loop.
    for epoch in range(finetune_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        train_acc = evaluate(model, train_loader, device)
        test_acc  = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1:02d} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

    # Save the updated model checkpoint.
    updated_checkpoint = "Models/gnn_model_00.pt"
    torch.save(model.state_dict(), updated_checkpoint)
    print(f"Model saved to {updated_checkpoint}")

if __name__ == "__main__":
    main()


"""
• Loads your dataset and splits it into training and test sets.
• Instantiates the model and loads the pre-trained weights from the checkpoint.
• Defines an optimizer (using a typically lower learning rate for fine tuning).
• Runs a single fine tuning phase (for a set number of epochs) on the training data.
• Finally, saves the fine-tuned model to disk.
"""
