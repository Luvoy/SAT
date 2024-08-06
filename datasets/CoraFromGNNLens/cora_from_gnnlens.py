import json
import os
import os.path as osp

import torch
import torch.nn.functional as F
from torch import Tensor
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected
from torch_geometric.utils import subgraph

import networkx as nx
from typing import Callable, Optional


def read_json(path):
    ret = None
    print("in read_json, path", path)
    with open(path, "r", encoding="utf-8") as f:
        ret = json.load(f)
        f.close()
    return ret


class CoraFromGNNLens(InMemoryDataset):
    def __init__(
        self,
        root: str,
        name: str = "CoraFromGNNLens",
        # split: str = "public",
        num_train_per_class: int = 20,
        num_val: int = 500,
        num_test: int = 1000,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ) -> None:
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def process(self) -> None:
        graph = read_json(os.path.join(self.root, self.name, "raw", "graph.json"))
        masks = read_json(os.path.join(self.root, self.name, "raw", "masks.json"))
        node_sparse_features = read_json(
            os.path.join(self.root, self.name, "raw", "node-sparse-features.json")
        )

        num_nodes = len(graph["nodes"])
        num_node_features = node_sparse_features["numNodeFeatureDims"]

        x = torch.zeros(num_nodes, num_node_features)
        for i, indexes in enumerate(node_sparse_features["nodeFeatureIndexes"]):
            values = node_sparse_features["nodeFeatureValues"][i]
            for j, idx in enumerate(indexes):
                x[i, idx] = values[j]

        y = torch.tensor([node["label"] for node in graph["nodes"]], dtype=torch.long)

        edge_index = torch.tensor(
            [
                [edge["source"] for edge in graph["edges"]],
                [edge["target"] for edge in graph["edges"]],
            ],
            dtype=torch.long,
        )
        G = nx.DiGraph() if is_undirected(edge_index) else nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edge_index.T.tolist())

        edge_index_from_G = torch.tensor(
            list(G.edges()),
            dtype=torch.long,
        )  # E x 2

        data = Data(x=x, edge_index=edge_index_from_G.T, y=y)
        # data = Data(x=x, edge_index=edge_index, y=y)

        G2 = nx.DiGraph() if is_undirected(edge_index) else nx.Graph()
        G2.add_nodes_from(range(num_nodes))
        G2.add_edges_from(data.edge_index.T.tolist())
        print(
            "in data process, check edge: nx ==  pyg",
            torch.all(data.edge_index.T == torch.tensor(list(G2.edges()))),
        )

        data.train_idx = torch.tensor(masks["train"], dtype=torch.long)
        data.test_idx = torch.tensor(masks["test"], dtype=torch.long)
        data.val_idx = torch.tensor(masks["valid"], dtype=torch.long)

        data.train_mask = torch.tensor(
            [i in masks["train"] for i in range(num_nodes)], dtype=torch.bool
        )
        data.val_mask = torch.tensor(
            [i in masks["valid"] for i in range(num_nodes)], dtype=torch.bool
        )
        data.test_mask = torch.tensor(
            [i in masks["test"] for i in range(num_nodes)], dtype=torch.bool
        )

        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])


class SplitCoraDataset(InMemoryDataset):
    def __init__(
        self,
        dataset: CoraFromGNNLens,
        split: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.dataset = dataset
        self.split = split
        super().__init__(dataset.root, transform, pre_transform)

        # 根据split参数选择相应的数据
        self.data, self.slices = self.process()

    @property
    def raw_file_names(self):
        return []  # 这里我们不需要raw文件，因为我们直接使用已有的数据集

    @property
    def processed_file_names(self):
        return [f"{self.split}_data.pt"]

    def process(self):
        data = self.dataset[0]  # 假设 dataset 只有一个图
        if self.split == "train":
            mask = data.train_mask
        elif self.split == "val":
            mask = data.val_mask
        elif self.split == "test":
            mask = data.test_mask
        else:
            raise ValueError(f"Unknown split: {self.split}")

        # 提取子图并重新标记节点索引
        edge_index, _ = subgraph(mask, data.edge_index, relabel_nodes=True)

        sub_data = Data(
            x=data.x[mask],
            y=data.y[mask],
            edge_index=edge_index,
            train_mask=None,
            val_mask=None,
            test_mask=None,
        )

        # 将数据存储为InMemoryDataset需要的格式
        data_list = [sub_data]
        return self.collate(data_list)

    def len(self):
        return 1  # 数据集包含一个图

    def get(self, idx):
        return self.data
