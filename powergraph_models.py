"""GINe graph-classification model for the WM9B7 PowerGraph submission.

Trimmed from the full architecture-comparison module (which also contained
GCN, TransformerConv and GPS variants) to only the components used by the
notebook's live demo:

- ``GraphPooling`` — multi-aggregation readout (mean/max/sum or concatenations).
- ``GINe_Graph`` — GIN with edge-feature conditioning (Hu et al., 2020).

The removed architectures are referenced indirectly via
``architecture_comparison_summary.json`` in the bundle; their training code
lives in the project's full repository.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, aggr


class GraphPooling(nn.Module):
    """Configurable graph-level readout.

    Supported pooling modes:
        - ``"max"``: element-wise max over nodes (out_dim = hidden_dim).
        - ``"sum"``: element-wise sum over nodes (out_dim = hidden_dim).
        - ``"mean_max"``: concatenation of mean and max (out_dim = 2 * hidden_dim).
        - ``"mean_max_sum"``: concatenation of mean, max and sum (out_dim = 3 * hidden_dim).
    """

    def __init__(self, pooling: str, hidden_dim: int) -> None:
        super().__init__()
        self.pooling = pooling
        self.hidden_dim = hidden_dim
        if pooling == "max":
            self.pool = aggr.MaxAggregation()
        elif pooling == "sum":
            self.pool = aggr.SumAggregation()
        elif pooling == "mean_max":
            self.pool = aggr.MultiAggregation(aggrs=["mean", "max"], mode="cat")
        elif pooling == "mean_max_sum":
            self.pool = aggr.MultiAggregation(aggrs=["mean", "max", "sum"], mode="cat")
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        return self.pool(x, batch)

    @property
    def out_dim(self) -> int:
        if self.pooling in {"max", "sum"}:
            return self.hidden_dim
        if self.pooling == "mean_max":
            return self.hidden_dim * 2
        if self.pooling == "mean_max_sum":
            return self.hidden_dim * 3
        raise ValueError(f"Unknown pooling: {self.pooling}")


class GINe_Graph(nn.Module):
    """GIN with edge-feature conditioning for graph classification.

    Stack of ``num_layers`` ``GINEConv`` blocks, each followed by batch norm,
    ReLU and dropout. A configurable readout produces a graph-level embedding,
    which is passed through a two-layer MLP head to produce class logits.
    """

    def __init__(
        self,
        num_node_features: int = 3,
        num_edge_features: int = 5,
        hidden_dim: int = 32,
        num_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.3,
        pooling: str = "max",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        first_mlp = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.convs.append(GINEConv(first_mlp, edge_dim=num_edge_features))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=num_edge_features))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.pool = GraphPooling(pooling, hidden_dim)
        self.lin1 = nn.Linear(self.pool.out_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch,
        )
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)
