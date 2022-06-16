#!/usr/bin/env python
# coding: utf-8

import networkx as nx
import numpy as np
import random
import itertools
import tqdm
import pickle as pkl
import argparse
import os

from data import PathData


"""
Generate Graph Dataset
generate graphs with various number of nodes and structures
"""


parser=argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default='dataset/samples')
parser.add_argument('--num_graphs', type=int, default=100)
args=parser.parse_args()

np.random.seed(9999)

num_graphs=args.num_graphs
num_weights_per_structure=10
num_target_tasks_per_weights=20


output_dir=args.output_dir
os.makedirs(output_dir, exist_ok=False)
for folder in ['train', 'test', 'val']:
    os.makedirs(os.path.join(output_dir,folder), exist_ok=True)


graphs=[]
for n_nodes in np.random.choice([20,30], num_graphs):
    G=nx.waxman_graph(n_nodes,beta=0.4, alpha=0.2)
    Gc=G.subgraph(max(nx.connected_components(G),key=len))
    Gc=nx.convert_node_labels_to_integers(Gc)
    graphs.append(Gc)


gids = list(range(num_graphs))
random.shuffle(gids)
split_indices=['train']*int(len(gids)*0.7)+['val']*int(len(gids)*0.15)+['test']*(len(gids)-int(len(gids)*0.7)-int(len(gids)*0.15))
gids_split={k:v for k,v in zip(gids, split_indices)}


global_count=0
for gid, G in tqdm.tqdm(enumerate(graphs), total=len(graphs)):
    ws = np.random.randint(1, 10, (min(num_weights_per_structure, len(G.edges)), len(G.edges)))
    # check for duplicates
    wslist=[tuple(x) for x in ws]
    if not len(wslist)==len(set(wslist)):
        continue

    for _, w_ in enumerate(ws):
        w = [(e[0],e[1], w_[i]) for i,e in enumerate(G.edges)]
        G.add_weighted_edges_from(w)
        dropout=num_target_tasks_per_weights/(len(G.nodes)*(len(G.nodes)-1))
        for (src, tgt) in itertools.permutations(G.nodes, 2):
            if np.random.rand()>dropout: #sample tasks per graph
                continue
            p = nx.all_shortest_paths(G, src, tgt, weight='weight')
            p = list(p)
            if len(p)>1:
                continue
            p_len = nx.dijkstra_path_length(G, src, tgt)
            split_dir = gids_split[gid]


            d_i = PathData.from_nx_graph(G, p[0], p_len)

            pkl.dump(d_i, open(os.path.join(output_dir, split_dir, f'{global_count:09d}'),'wb'))
            global_count+=1
