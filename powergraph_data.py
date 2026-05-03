"""PowerGraph data pipeline — binary task only, trimmed for the WM9B7 notebook.

Trimmed from the project's full data module. The original supported binary,
multiclass and regression tasks, both v5 and v7.3 (HDF5) ``.mat`` files via an
``h5py`` fallback, and Pandera-based dataset validation. None of those paths
are exercised by the notebook, so this version keeps only what is needed to
load a pre-processed binary IEEE-24 dataset from a ``.pt`` file and produce
deterministic train / val / test splits.

Public surface:
    - ``SplitConfig`` — dataclass holding split ratios and seed.
    - ``PowerGraphDataset`` — PyG ``InMemoryDataset`` subclass for the
      cascade-vs-no-cascade graph-classification task.
    - ``PowerGraphDataset.from_processed`` — classmethod that loads a
      pre-built ``.pt`` directly, bypassing the raw-``.mat`` pipeline.
    - ``split_dataset_indices`` — stratified 85/5/10 split helper.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, InMemoryDataset


# ---------------------------------------------------------------------------
# Raw-file helpers (loadmat only; v7.3 / HDF5 files are not supported here).
# ---------------------------------------------------------------------------

def _loadmat_first(path: str, keys: Sequence[str]) -> np.ndarray:
    mat = loadmat(path)
    for key in keys:
        if key in mat:
            return np.asarray(mat[key])
    raise KeyError(f"None of keys {keys} found in {path}")


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
        norm_items = []
        for x in arr.ravel():
            x = np.asarray(x)
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


# ---------------------------------------------------------------------------
# Split configuration.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.85
    val_ratio: float = 0.05
    test_ratio: float = 0.10
    random_state: int = 23


# ---------------------------------------------------------------------------
# Dataset.
# ---------------------------------------------------------------------------

class PowerGraphDataset(InMemoryDataset):
    """PowerGraph cascading-failure dataset (binary task only).

    Two construction paths:

    1. ``PowerGraphDataset(root, grid_name, task)`` — PyG-standard path that
       reads raw ``.mat`` files from ``<root>/raw/`` and writes a processed
       ``.pt`` to ``<root>/processed/``. Used at pre-build time on Azure.
    2. ``PowerGraphDataset.from_processed(processed_path, ...)`` — direct
       load from a pre-built ``.pt``. Used by the notebook.
    """

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
        if self.task != "binary":
            raise ValueError("Trimmed module supports only task='binary'.")
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ["blist.mat", "Bf.mat", "Ef.mat", "of_bi.mat"]

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

        blist = _normalize_edge_list(
            _loadmat_first(os.path.join(raw_src, "blist.mat"), ["blist", "bList"])
        )
        Bf = _normalize_graph_tensor(
            _loadmat_first(os.path.join(raw_src, "Bf.mat"), ["Bf", "B_f_tot"]), 3
        )
        Ef = _normalize_graph_tensor(
            _loadmat_first(os.path.join(raw_src, "Ef.mat"), ["Ef", "E_f_post"]), 4
        )
        labels = _normalize_binary_labels(
            _loadmat_first(os.path.join(raw_src, "of_bi.mat"), ["of_bi", "output_features"])
        )

        edge_index_undirected = blist - 1  # MATLAB → 0-indexed
        edge_index_bidir = np.concatenate(
            [edge_index_undirected, edge_index_undirected[:, [1, 0]]], axis=0
        )

        if not (Bf.shape[0] == Ef.shape[0] == labels.shape[0]):
            raise ValueError(
                "Graph count mismatch: "
                f"Bf={Bf.shape[0]}, Ef={Ef.shape[0]}, labels={labels.shape[0]}"
            )

        data_list: List[Data] = []
        edge_index_t = torch.tensor(edge_index_bidir.T, dtype=torch.long)
        for i in range(Bf.shape[0]):
            x = torch.tensor(Bf[i], dtype=torch.float32)

            ef_forward = np.asarray(Ef[i], dtype=np.float32)
            ef_reverse = ef_forward.copy()
            ef_reverse[:, 0] = -ef_reverse[:, 0]  # flip P_flow on reverse edge
            ef_reverse[:, 1] = -ef_reverse[:, 1]  # flip Q_flow on reverse edge

            dir_forward = np.zeros((ef_forward.shape[0], 1), dtype=np.float32)
            dir_reverse = np.ones((ef_forward.shape[0], 1), dtype=np.float32)
            ef_fwd_aug = np.concatenate([ef_forward, dir_forward], axis=1)
            ef_rev_aug = np.concatenate([ef_reverse, dir_reverse], axis=1)
            edge_attr = torch.tensor(
                np.concatenate([ef_fwd_aug, ef_rev_aug], axis=0), dtype=torch.float32
            )

            y = torch.tensor([labels[i]], dtype=torch.long)
            data_list.append(Data(x=x, edge_index=edge_index_t, edge_attr=edge_attr, y=y))

        self.save(data_list, self.processed_paths[0])

    @classmethod
    def from_processed(
        cls,
        processed_path: str,
        grid_name: str = "ieee24",
        task: str = "binary",
    ) -> "PowerGraphDataset":
        """Construct a dataset from a pre-built ``.pt`` file.

        Skips ``InMemoryDataset.__init__`` (which would invoke ``_download``
        and ``_process``, both of which need raw ``.mat`` files) and instead
        sets the minimal attributes required by ``load()`` and the
        ``__getitem__`` / ``__len__`` machinery, then loads the tensors
        directly from ``processed_path``.

        Tested against ``torch-geometric==2.6.1``.
        """
        if task != "binary":
            raise ValueError("Trimmed module supports only task='binary'.")
        obj = cls.__new__(cls)
        obj.grid_name = grid_name
        obj.task = task
        # Attributes that InMemoryDataset.load() and downstream get / __len__
        # / __getitem__ access without going through Dataset.__init__.
        obj.transform = None
        obj.pre_transform = None
        obj.pre_filter = None
        obj.log = True
        obj.force_reload = False
        obj._indices = None
        obj._data_list = None
        obj.root = os.path.dirname(os.path.abspath(processed_path)) or "."
        obj.load(processed_path)
        return obj


# ---------------------------------------------------------------------------
# Split helper.
# ---------------------------------------------------------------------------

def split_dataset_indices(
    dataset: PowerGraphDataset, cfg: SplitConfig = SplitConfig()
) -> Tuple[List[int], List[int], List[int]]:
    """Stratified 85/5/10 split on graph-level binary labels."""
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
