"""PowerGraph data pipeline utilities.

Implements the post-EDA steps from the implementation guide:
- PyG InMemoryDataset conversion from raw MAT files
- Stratified 85/5/10 split helpers
- Pandera-based dataset validation for graph-level binary tasks
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
import h5py

try:
    import pandera.pandas as pa
    from pandera.pandas import Check, Column, DataFrameSchema

    HAS_PANDERA = True
except Exception:
    pa = None
    Check = None
    Column = None
    DataFrameSchema = None
    HAS_PANDERA = False


def _loadmat_first(path: str, keys: Sequence[str]) -> np.ndarray:
    mat = loadmat(path)
    for key in keys:
        if key in mat:
            return np.asarray(mat[key])
    raise KeyError(f"None of keys {keys} found in {path}")


def _h5_load_numeric(path: str, keys: Sequence[str]) -> np.ndarray:
    with h5py.File(path, "r") as f:
        for key in keys:
            if key in f:
                return np.asarray(f[key])
    raise KeyError(f"None of keys {keys} found in {path}")


def _h5_load_cell_arrays(path: str, key: str) -> List[np.ndarray]:
    with h5py.File(path, "r") as f:
        refs = np.asarray(f[key]).ravel()
        return [np.asarray(f[r]) for r in refs]


def _load_blist(path: str) -> np.ndarray:
    try:
        return _normalize_edge_list(_loadmat_first(path, ["blist", "bList"]))
    except NotImplementedError:
        return _normalize_edge_list(_h5_load_numeric(path, ["blist", "bList"]))


def _load_Bf(path: str) -> np.ndarray:
    try:
        return _normalize_graph_tensor(_loadmat_first(path, ["Bf", "B_f_tot"]), 3)
    except NotImplementedError:
        items = _h5_load_cell_arrays(path, "B_f_tot")
        return np.stack([x.T if x.ndim == 2 and x.shape[0] == 3 else x for x in items], axis=0)


def _load_Ef(path: str) -> np.ndarray:
    try:
        return _normalize_graph_tensor(_loadmat_first(path, ["Ef", "E_f_post"]), 4)
    except NotImplementedError:
        items = _h5_load_cell_arrays(path, "E_f_post")
        return np.stack([x.T if x.ndim == 2 and x.shape[0] == 4 else x for x in items], axis=0)


def _load_of_bi(path: str) -> np.ndarray:
    try:
        return _normalize_binary_labels(_loadmat_first(path, ["of_bi", "output_features"]))
    except NotImplementedError:
        items = _h5_load_cell_arrays(path, "output_features")
        vals = [float(np.asarray(x).reshape(-1)[0]) for x in items]
        return np.asarray(vals, dtype=np.int64)


def _load_of_mc(path: str) -> np.ndarray:
    try:
        return _normalize_multiclass_labels(_loadmat_first(path, ["of_mc", "category"]))
    except NotImplementedError:
        items = _h5_load_cell_arrays(path, "category")
        labels = []
        for x in items:
            v = np.asarray(x).reshape(-1)
            labels.append(int(np.argmax(v) + 1 if v.size > 1 else v[0]))
        return np.asarray(labels, dtype=np.int64)


def _load_of_reg(path: str) -> np.ndarray:
    try:
        return np.asarray(_loadmat_first(path, ["of_reg", "dns_MW"])).squeeze().astype(np.float32)
    except NotImplementedError:
        return np.asarray(_h5_load_numeric(path, ["of_reg", "dns_MW"])).squeeze().astype(np.float32)


def _load_exp(path: str, num_edges: int) -> np.ndarray:
    try:
        return _normalize_explanations(_loadmat_first(path, ["exp", "explainations"]), num_edges=num_edges)
    except NotImplementedError:
        items = _h5_load_cell_arrays(path, "explainations")
        mask = np.zeros((len(items), num_edges), dtype=np.int64)
        for i, x in enumerate(items):
            idx = np.asarray(x).reshape(-1).astype(np.int64)
            idx = idx[(idx > 0) & (idx <= num_edges)]
            if idx.size:
                mask[i, idx - 1] = 1
        return mask


def _normalize_edge_list(blist: np.ndarray) -> np.ndarray:
    blist = np.asarray(blist)
    if blist.ndim != 2:
        raise ValueError(f"Expected 2D edge list, got shape={blist.shape}")
    if blist.shape[0] == 2 and blist.shape[1] != 2:
        blist = blist.T
    if blist.shape[1] != 2:
        raise ValueError(f"Expected edge list with 2 columns, got shape={blist.shape}")
    return blist.astype(np.int64)


def _normalize_graph_tensor(arr: np.ndarray, expected_feature_dim: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == object:
        items = [np.asarray(x) for x in arr.ravel()]
        norm_items = []
        for x in items:
            if x.ndim == 2 and x.shape[0] == expected_feature_dim and x.shape[1] != expected_feature_dim:
                x = x.T
            norm_items.append(x)
        return np.stack(norm_items, axis=0)
    return arr


def _normalize_binary_labels(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr).squeeze()
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr.astype(np.int64)


def _normalize_multiclass_labels(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr).squeeze()
    if arr.ndim == 2:
        # Handle one-hot matrices that may be [N, C] or [C, N].
        if arr.shape[0] in (3, 4) and arr.shape[1] > arr.shape[0]:
            arr = arr.T
        if arr.shape[1] > 1:
            return np.argmax(arr, axis=1).astype(np.int64) + 1
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr.astype(np.int64)


def _normalize_explanations(arr: np.ndarray, num_edges: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == object:
        mask = np.zeros((arr.size, num_edges), dtype=np.int64)
        for i, x in enumerate(arr.ravel()):
            idx = np.asarray(x).reshape(-1).astype(np.int64)
            idx = idx[(idx > 0) & (idx <= num_edges)]
            if idx.size:
                mask[i, idx - 1] = 1
        return mask
    if arr.ndim == 2 and arr.shape[1] == num_edges:
        return arr.astype(np.int64)
    if arr.ndim == 2 and arr.shape[0] == num_edges:
        return arr.T.astype(np.int64)
    raise ValueError(f"Unexpected explanation mask shape={arr.shape}, num_edges={num_edges}")


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.85
    val_ratio: float = 0.05
    test_ratio: float = 0.10
    random_state: int = 23


class PowerGraphDataset(InMemoryDataset):
    """PowerGraph cascading failure dataset for graph-level tasks."""

    def __init__(
        self,
        root: str,
        grid_name: str = "ieee24",
        task: str = "binary",
        transform=None,
        pre_transform=None,
    ) -> None:
        self.grid_name = grid_name
        self.task = task
        if self.task not in {"binary", "multiclass", "regression"}:
            raise ValueError("task must be one of: binary, multiclass, regression")
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [
            "blist.mat",
            "Bf.mat",
            "Ef.mat",
            "of_bi.mat",
            "of_mc.mat",
            "of_reg.mat",
            "exp.mat",
        ]

    @property
    def processed_file_names(self) -> List[str]:
        return [f"{self.grid_name}_{self.task}.pt"]

    def _resolve_raw_source_dir(self) -> str:
        direct = self.raw_dir
        nested = os.path.join(self.raw_dir, self.grid_name)
        has_all_direct = all(os.path.exists(os.path.join(direct, f)) for f in self.raw_file_names)
        has_all_nested = all(os.path.exists(os.path.join(nested, f)) for f in self.raw_file_names)
        if has_all_direct:
            return direct
        if has_all_nested:
            return nested
        raise FileNotFoundError(
            "Could not locate PowerGraph MAT files. Expected either "
            f"{direct} or {nested} to contain: {self.raw_file_names}"
        )

    def process(self) -> None:
        raw_src = self._resolve_raw_source_dir()

        blist = _load_blist(os.path.join(raw_src, "blist.mat"))
        Bf = _load_Bf(os.path.join(raw_src, "Bf.mat"))
        Ef = _load_Ef(os.path.join(raw_src, "Ef.mat"))
        exp = _load_exp(os.path.join(raw_src, "exp.mat"), num_edges=Ef.shape[1])

        if self.task == "binary":
            labels = _load_of_bi(os.path.join(raw_src, "of_bi.mat"))
        elif self.task == "multiclass":
            labels = _load_of_mc(os.path.join(raw_src, "of_mc.mat"))
        else:
            labels = _load_of_reg(os.path.join(raw_src, "of_reg.mat"))

        edge_index_undirected = blist - 1
        edge_index_bidir = np.concatenate(
            [edge_index_undirected, edge_index_undirected[:, [1, 0]]],
            axis=0,
        )

        if not (Bf.shape[0] == Ef.shape[0] == labels.shape[0] == exp.shape[0]):
            raise ValueError(
                "Graph count mismatch: "
                f"Bf={Bf.shape[0]}, Ef={Ef.shape[0]}, labels={labels.shape[0]}, exp={exp.shape[0]}"
            )

        data_list: List[Data] = []
        for i in range(Bf.shape[0]):
            x = torch.tensor(Bf[i], dtype=torch.float32)

            ef_forward = np.asarray(Ef[i], dtype=np.float32)
            ef_reverse = ef_forward.copy()
            ef_reverse[:, 0] = -ef_reverse[:, 0]
            ef_reverse[:, 1] = -ef_reverse[:, 1]

            dir_forward = np.zeros((ef_forward.shape[0], 1), dtype=np.float32)
            dir_reverse = np.ones((ef_forward.shape[0], 1), dtype=np.float32)
            ef_fwd_aug = np.concatenate([ef_forward, dir_forward], axis=1)
            ef_rev_aug = np.concatenate([ef_reverse, dir_reverse], axis=1)

            edge_attr = torch.tensor(np.concatenate([ef_fwd_aug, ef_rev_aug], axis=0), dtype=torch.float32)
            edge_index = torch.tensor(edge_index_bidir.T, dtype=torch.long)

            if self.task == "regression":
                y = torch.tensor([labels[i]], dtype=torch.float32)
            else:
                y = torch.tensor([labels[i]], dtype=torch.long)

            exp_mask = np.asarray(exp[i]).reshape(-1)
            explanation_mask = torch.tensor(np.concatenate([exp_mask, exp_mask]), dtype=torch.float32)

            data_list.append(
                Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y,
                    explanation_mask=explanation_mask,
                )
            )

        self.save(data_list, self.processed_paths[0])


def split_dataset_indices(dataset: PowerGraphDataset, cfg: SplitConfig = SplitConfig()) -> Tuple[List[int], List[int], List[int]]:
    if not np.isclose(cfg.train_ratio + cfg.val_ratio + cfg.test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    labels = [int(dataset[i].y.item()) for i in range(len(dataset))]
    indices = list(range(len(dataset)))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(1.0 - cfg.train_ratio),
        stratify=labels,
        random_state=cfg.random_state,
    )

    temp_labels = [labels[i] for i in temp_idx]
    test_fraction_within_temp = cfg.test_ratio / (cfg.val_ratio + cfg.test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_fraction_within_temp,
        stratify=temp_labels,
        random_state=cfg.random_state,
    )

    return train_idx, val_idx, test_idx


def make_dataloaders(
    dataset: PowerGraphDataset,
    batch_size: int = 16,
    split_cfg: SplitConfig = SplitConfig(),
) -> Tuple[Subset, Subset, Subset, DataLoader, DataLoader, DataLoader]:
    train_idx, val_idx, test_idx = split_dataset_indices(dataset, split_cfg)

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def validate_powergraph_dataset(dataset: PowerGraphDataset, grid_name: str) -> Dict[str, object]:
    """Run lightweight Pandera checks over graph-level summary features."""
    records = []
    for i in range(len(dataset)):
        data = dataset[i]
        nf = data.x.detach().cpu().numpy()
        ef = data.edge_attr.detach().cpu().numpy()
        records.append(
            {
                "graph_idx": i,
                "num_nodes": int(data.x.shape[0]),
                "num_edges": int(data.edge_index.shape[1]),
                "label": int(data.y.item()),
                "P_net_mean": float(nf[:, 0].mean()),
                "P_net_std": float(nf[:, 0].std()),
                "V_min": float(nf[:, 2].min()),
                "V_max": float(nf[:, 2].max()),
                "P_flow_max": float(np.abs(ef[:, 0]).max()),
                "line_rating_min": float(ef[:, 3].min()),
                "node_abs_max": float(np.abs(nf).max()),
                "edge_abs_max": float(np.abs(ef).max()),
                "has_explanation": float(data.explanation_mask.sum().item() > 0),
            }
        )

    df = pd.DataFrame(records)
    expected_nodes = {"ieee24": 24, "ieee39": 39, "ieee118": 118, "uk": 29}

    report: Dict[str, object] = {
        "grid": grid_name,
        "total_graphs": len(dataset),
    }
    if HAS_PANDERA:
        schema = DataFrameSchema(
            {
                "num_nodes": Column(int, Check.eq(expected_nodes.get(grid_name, 24))),
                "num_edges": Column(int, Check.gt(0)),
                "label": Column(int, Check.isin([0, 1])),
                "P_net_std": Column(float, Check.ge(0.0)),
                "V_min": Column(float, Check.gt(-10.0)),
                "V_max": Column(float, Check.lt(2.0)),
                "line_rating_min": Column(float, Check.ge(0.0)),
                "node_abs_max": Column(float, Check.lt(10.0)),
                "edge_abs_max": Column(float, Check.lt(10.0)),
            }
        )
        try:
            schema.validate(df, lazy=True)
            report["status"] = "PASSED"
            report["errors"] = 0
        except pa.errors.SchemaErrors as err:
            report["status"] = "FAILED"
            report["errors"] = int(len(err.failure_cases))
            report["failure_cases"] = err.failure_cases.head(20).to_dict(orient="records")
    else:
        expected = expected_nodes.get(grid_name, 24)
        checks = [
            (df["num_nodes"] == expected),
            (df["num_edges"] > 0),
            (df["label"].isin([0, 1])),
            (df["P_net_std"] >= 0.0),
            (df["V_min"] > -10.0),
            (df["V_max"] < 10.0),
            (df["line_rating_min"] > -10.0),
            (df["node_abs_max"] < 10.0),
            (df["edge_abs_max"] < 10.0),
        ]
        valid_rows = checks[0]
        for chk in checks[1:]:
            valid_rows = valid_rows & chk
        failed = int((~valid_rows).sum())
        report["status"] = "PASSED" if failed == 0 else "FAILED"
        report["errors"] = failed
        report["validator"] = "manual_fallback"

    report["positive_rate"] = float(df["label"].mean())
    report["class_0_count"] = int((df["label"] == 0).sum())
    report["class_1_count"] = int((df["label"] == 1).sum())
    return report


def save_validation_report(report: Dict[str, object], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
