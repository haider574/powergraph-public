"""Graph neural network architectures for PowerGraph.

Implements the guide's Section 5 baselines:
- GCN_Graph
- GINe_Graph
- Transformer_Graph
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINEConv, TransformerConv, GPSConv
from torch_geometric.nn import aggr


class GraphPooling(nn.Module):
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


class GCN_Graph(nn.Module):
    """GCN baseline that intentionally ignores edge features."""

    def __init__(
        self,
        num_node_features: int = 3,
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
        self.convs.append(GCNConv(num_node_features, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.pool = GraphPooling(pooling, hidden_dim)
        self.lin1 = nn.Linear(self.pool.out_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)


class GINe_Graph(nn.Module):
    """GIN with edge features."""

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
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)


class Transformer_Graph(nn.Module):
    """TransformerConv graph classifier with edge features."""

    def __init__(
        self,
        num_node_features: int = 3,
        num_edge_features: int = 5,
        hidden_dim: int = 32,
        num_layers: int = 3,
        num_classes: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
        pooling: str = "max",
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("Transformer_Graph requires num_layers >= 2")
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(
            TransformerConv(
                num_node_features,
                hidden_dim // heads,
                heads=heads,
                edge_dim=num_edge_features,
                dropout=dropout,
            )
        )
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(
                TransformerConv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    edge_dim=num_edge_features,
                    dropout=dropout,
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.convs.append(
            TransformerConv(
                hidden_dim,
                hidden_dim,
                heads=1,
                edge_dim=num_edge_features,
                dropout=dropout,
            )
        )
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.pool = GraphPooling(pooling, hidden_dim)
        self.lin1 = nn.Linear(self.pool.out_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)


class GPS_Graph(nn.Module):
    """GPS (General Powerful Scalable) Graph Transformer.

    Rampasek et al., NeurIPS 2022. Combines local GINEConv message passing
    with global multi-head self-attention per layer, plus Laplacian
    positional encodings. First application to any power grid task.

    PowerGraph grids have fixed topology (all graphs share the same
    edge_index), so LapPE is computed once and cached.
    """

    def __init__(
        self,
        num_node_features: int = 3,
        num_edge_features: int = 5,
        hidden_dim: int = 32,
        num_layers: int = 3,
        num_classes: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
        pooling: str = "max",
        pe_dim: int = 8,
    ) -> None:
        super().__init__()
        self.pe_dim = pe_dim

        # Input projection: node features + LapPE -> hidden_dim
        self.node_encoder = nn.Linear(num_node_features + pe_dim, hidden_dim)

        # GPS layers (local GINEConv + global multi-head attention)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            local_nn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            local_conv = GINEConv(local_nn, edge_dim=num_edge_features)
            self.convs.append(
                GPSConv(
                    channels=hidden_dim,
                    conv=local_conv,
                    heads=heads,
                    attn_type="multihead",
                    dropout=dropout,
                )
            )

        self.pool = GraphPooling(pooling, hidden_dim)
        self.lin1 = nn.Linear(self.pool.out_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

        # LapPE cache (same topology => same eigenvectors for every graph)
        self._pe_cache: torch.Tensor | None = None
        self._pe_num_nodes: int | None = None

    @staticmethod
    def _compute_laplacian_pe(
        edge_index: torch.Tensor, num_nodes: int, pe_dim: int
    ) -> torch.Tensor:
        from torch_geometric.utils import get_laplacian, to_dense_adj

        ei, ew = get_laplacian(edge_index, normalization="sym", num_nodes=num_nodes)
        laplacian = to_dense_adj(ei, edge_attr=ew, max_num_nodes=num_nodes)[0]
        _, eigvecs = torch.linalg.eigh(laplacian)

        k = min(pe_dim, num_nodes - 1)
        pe = eigvecs[:, 1 : k + 1]  # skip constant eigenvector
        if pe.shape[1] < pe_dim:
            pe = F.pad(pe, (0, pe_dim - pe.shape[1]))

        # Fix sign ambiguity: first nonzero entry per eigenvector is positive
        for i in range(pe.shape[1]):
            col = pe[:, i]
            nz = col[col.abs() > 1e-6]
            if len(nz) > 0 and nz[0] < 0:
                pe[:, i] = -col
        return pe

    def _get_pe_for_batch(
        self, edge_index: torch.Tensor, batch: torch.Tensor, num_nodes_total: int
    ) -> torch.Tensor:
        num_graphs = int(batch.max().item()) + 1
        n = num_nodes_total // num_graphs

        if self._pe_cache is None or self._pe_num_nodes != n:
            mask = (edge_index[0] < n) & (edge_index[1] < n)
            single_ei = edge_index[:, mask]
            self._pe_cache = self._compute_laplacian_pe(single_ei, n, self.pe_dim)
            self._pe_num_nodes = n

        return self._pe_cache.to(batch.device).repeat(num_graphs, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch,
        )

        pe = self._get_pe_for_batch(edge_index, batch, x.shape[0])
        x = self.node_encoder(torch.cat([x, pe], dim=-1))

        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)

        x = self.pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)
