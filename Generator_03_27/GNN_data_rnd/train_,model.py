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
    """
    Parse a literal of the form:
       "Pred(...)"
       "¬Pred(...)"

    Returns:
      sign: +1 or -1
      predicate: e.g. "Pred1"
      args: list of arguments [arg1, arg2, ...]
    """
    sign = +1
    core = literal.strip()
    if core.startswith("¬"):
        sign = -1
        core = core[1:].strip()

    # Now core should be something like "Pred(...)" or "Pred(...)"
    pred = core.split("(", 1)[0]
    # Pull out the argument list from inside parentheses
    inside = core[len(pred) + 1 : -1].strip()  # remove "Pred(" and the trailing ")"
    if inside == "":
        args = []
    else:
        args = [x.strip() for x in inside.split(",")]
    return sign, pred, args


def embed_literal(
    literal: str,
    predicate_to_idx: Dict[str, int],
    max_args: int = 3
) -> torch.Tensor:
    """
    Convert a literal into a numeric feature vector.

    For demonstration:
      - sign: +1 => 0, -1 => 1
      - predicate index: embed as integer
      - argument types:
          0 = variable (heuristic if it starts with uppercase letter),
          1 = constant,
          2 = function (if it looks like "func(...)")
      - we store up to max_args of these types. If fewer, pad with -1.

    This is a minimal approach. You can make it more sophisticated with 
    trainable embeddings, sub-graph representations, etc.
    """
    sign, pred, args = parse_literal(literal)

    sign_feature = 0 if sign > 0 else 1
    pred_idx = predicate_to_idx.get(pred, 0)  # 0 if unknown

    # encode arguments by "type"
    arg_types = []
    for arg in args[:max_args]:
        if "(" in arg and arg.endswith(")"):
            # treat it as a function call
            arg_types.append(2)
        elif arg and arg[0].isupper():
            # treat it as a variable
            arg_types.append(0)
        else:
            # else treat as constant
            arg_types.append(1)

    # pad if fewer than max_args
    while len(arg_types) < max_args:
        arg_types.append(-1)

    # shape: [ sign_feature, pred_idx, arg_types... ]
    feat = [sign_feature, float(pred_idx)] + [float(at) for at in arg_types]
    return torch.tensor(feat, dtype=torch.float)


def build_graph_from_example(
    example: Dict[str, Any],
    predicate_to_idx: Dict[str, int],
    max_args: int = 3
) -> Data:
    """
    Convert one dictionary from toy_gnn_dataset.jsonl into a PyG Data object.

    We'll treat each literal as a node; for each resolvable pair, we create an edge.
    The label y for each edge is 1 if it corresponds to the "best_pair", else 0.
    """
    clauses = example["clauses"]
    resolvable_pairs = example["resolvable_pairs"]
    best_pair = example["best_pair"]

    # 1) Gather all literals from all clauses, store node features
    node_map = {}  # maps (clause_idx, literal_idx) => global_node_idx
    node_features = []
    global_node_idx = 0

    for ci, clause in enumerate(clauses):
        for li, lit_str in enumerate(clause):
            node_map[(ci, li)] = global_node_idx
            feat = embed_literal(lit_str, predicate_to_idx, max_args)
            node_features.append(feat)
            global_node_idx += 1

    x = torch.stack(node_features, dim=0)  # shape: [num_nodes, feature_dim]

    # 2) Create edges for each resolvable pair, label=1 for best pair, else 0
    edge_src = []
    edge_dst = []
    edge_labels = []

    for pair in resolvable_pairs:
        i, idxA = pair["clauseA_index"], pair["literalA_index"]
        j, idxB = pair["clauseB_index"], pair["literalB_index"]

        nA = node_map[(i, idxA)]
        nB = node_map[(j, idxB)]

        # Check if this is the best pair
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

    # Build the edge_index tensor
    if len(edge_src) == 0:
        # No resolvable edges
        edge_index = torch.empty((2, 0), dtype=torch.long)
        y = torch.empty(0, dtype=torch.long)
    else:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        y = torch.tensor(edge_labels, dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y  # edge-level labels
    )
    return data


##############################################################################
# 2) PyTorch Geometric Dataset class
##############################################################################

class ClauseResolutionDataset(Dataset):
    """
    A dataset wrapper that reads the entire toy_gnn_dataset.jsonl
    and converts each example to a PyG Data object.
    """
    def __init__(
        self,
        jsonl_file: str,
        predicate_list: List[str],
        max_args: int = 3
    ):
        super().__init__(root=None)  # we won't use root
        self.jsonl_file = jsonl_file
        self.max_args = max_args

        # build a simple mapping from predicate -> index
        self.predicate_to_idx = {}
        for i, pred in enumerate(predicate_list):
            self.predicate_to_idx[pred] = i + 1  # +1 so we reserve 0 for "unknown"

        # read in all lines
        with open(jsonl_file, "r") as f:
            lines = f.readlines()
        self.examples = [json.loads(line) for line in lines]

    def len(self):
        return len(self.examples)

    def get(self, idx):
        example = self.examples[idx]
        data = build_graph_from_example(
            example,
            self.predicate_to_idx,
            max_args=self.max_args
        )
        return data


##############################################################################
# 3) Define a GNN model for edge classification
##############################################################################

class EdgeClassifierGNN(torch.nn.Module):
    """
    A GNN that produces node embeddings, then classifies edges.
    We do:
       h = GNN(x, edge_index) -> node embeddings [num_nodes, hidden_dim]
       For each edge (u, v): concat h[u], h[v] -> pass through MLP => class logits
    """
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
        # x: [num_nodes, in_dim]
        # edge_index: [2, num_edges]
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        h = F.relu(h)

        # for each edge, gather (h[u], h[v]) -> [2*hidden_dim]
        src, dst = edge_index
        edge_reps = torch.cat([h[src], h[dst]], dim=1)

        # classify edges
        logits = self.edge_mlp(edge_reps)  # [num_edges, 2]
        return logits


##############################################################################
# 4) Train / Test Split and Training Loop
##############################################################################

def split_dataset(dataset: Dataset, train_ratio=0.8):
    """
    Shuffle and split the dataset into train & test subsets.
    """
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split_point = int(len(dataset) * train_ratio)
    train_idx = indices[:split_point]
    test_idx  = indices[split_point:]

    # Subset objects
    train_set = torch.utils.data.Subset(dataset, train_idx)
    test_set  = torch.utils.data.Subset(dataset, test_idx)
    return train_set, test_set

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch.x, batch.edge_index)
        # batch.y are the edge labels
        # shape: logits [num_edges, 2], y [num_edges]
        if len(batch.y) == 0:
            # No edges to classify => skip
            continue

        loss = F.cross_entropy(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            logits = model(batch.x, batch.edge_index)
            if len(batch.y) == 0:
                continue
            pred = logits.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    acc = correct / total if total > 0 else 0.0
    return acc


def main():
    ########################################
    # Config
    ########################################
    jsonl_file = "toy_gnn_dataset.jsonl"  # path to your dataset
    known_predicates = ["Pred1", "Pred2", "Pred3"]  # from your random generator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ########################################
    # Dataset
    ########################################
    dataset = ClauseResolutionDataset(
        jsonl_file=jsonl_file,
        predicate_list=known_predicates,
        max_args=3  # up to 3 arguments per literal
    )

    train_set, test_set = split_dataset(dataset, train_ratio=0.8)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=8, shuffle=False)

    ########################################
    # Model & Optim
    ########################################
    # Input dim for node features: 
    #   sign_feature => 1
    #   predicate_index => 1
    #   arg_types => max_args (3)
    # => total = 1 + 1 + 3 = 5
    in_dim = 5
    hidden_dim = 64

    model = EdgeClassifierGNN(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    ########################################
    # Training Loop
    ########################################
    num_epochs = 20
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        train_acc = evaluate(model, train_loader, device)
        test_acc  = evaluate(model, test_loader, device)

        print(f"Epoch {epoch+1:02d} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
