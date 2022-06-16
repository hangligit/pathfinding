#!/usr/bin/env python
# coding: utf-8

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data, Batch
import random
import itertools
import tqdm
import pickle as pkl
from collections import Counter
import matplotlib.pyplot as plt
import os
import sys


"""
Dataset
"""


# a data object for a torch loadable graph 
class PathData(Data):
    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, start_node=0, goal_node=0, **kwargs):
        super().__init__()
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.start_v=start_node
        self.goal_v=goal_node
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __inc__(self, key, value, *args, **kwargs):
        if key=='start_node':
            return self.x.size(0)
        elif key=='goal_node':
            return self.x.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)
        
    @classmethod
    def from_nx_graph(cls, G, path, length):
        """construct a torch loadable graph from a networkx format and path"""
        # feats
        indicators=np.zeros(len(G))
        indicators[path[0]]=1
        indicators[path[-1]]=2

        feats=torch.zeros((len(G), 3), dtype=torch.float)
        feats[np.arange(len(G)),indicators]=1

        # edge weights
        e_=nx.get_edge_attributes(G, 'weight').items()
        rows = [x[0][0] for x in e_]
        cols = [x[0][1] for x in e_]
        e_w = [x[1] for x in e_]
        edges_attr=torch.tensor(e_w+e_w, dtype=torch.float)
        edges = torch.tensor([rows+cols, cols+rows])

        # labels
        ys = torch.zeros((len(G))).round().long()
        ys[path]=1

        # edge labels
        on_edges = set(list(zip(path[:-1],path[1:]))+list(zip(path[1:],path[:-1])))
        ye = torch.tensor([1 if (r,c) in on_edges or (c,r) in on_edges else 0 for r,c in zip(rows+cols, cols+rows)], dtype=torch.long)

        return cls(x=feats, edge_index=edges, edge_attr=edges_attr, y=ys, ye=ye, route=tuple(path), route_len=length, start_node=path[0], goal_node=path[-1])


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        super().__init__()
        self.filepaths=self._list_dir(root)
    
    def _list_dir(self, root):
        return sorted([os.path.join(root, x) for x in os.listdir(root)
                       if os.path.isfile(os.path.join(root, x))
                       and not x.startswith('.')])
    def __getitem__(self, idx):
        return pkl.load(open(self.filepaths[idx],'rb'))
    def __len__(self,):
        return len(self.filepaths)
