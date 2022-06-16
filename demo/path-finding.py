import networkx as nx
import torch
import torch_sparse
import numpy as np
import pickle as pkl
import argparse
import matplotlib.pyplot as plt


"""
Script to test GNN model on a test example. 
Given a source node and a destination node as arguments, 
the model finds the lowest cost path between the two nodes.
It saves two figures to disk, showing the optimal path, please
check file names input_graph.png and prediction.png

In addition, you can specify the remove_edge argument. This gives
a corrupted graph with dropped edges. The GNN will find
the path between two nodes on this graph.

e.g.,
python path-finding.py --source_node=0 --destination_node=5
or
python path-finding.py --source_node=0 --destination_node=5 --remove_edge
"""


parser=argparse.ArgumentParser()
parser.add_argument('--source_node', type=int, default=0)
parser.add_argument('--destination_node', type=int, default=5)
parser.add_argument('--remove_edge',action='store_true')
args=parser.parse_args()

source_node=args.source_node
destination_node=args.destination_node
remove_edge=args.remove_edge


def draw(G, node_list=[], edge_list=[]):
    """
    Function for visualizing the
    graph and the path given by 
    the node_list and edge_list
    """
    plt.figure(figsize=(10,10), dpi=150)
    pos=nx.get_node_attributes(G,'pos')
    nx.draw_networkx(G, pos)
    nx.draw_networkx_nodes(G, pos, node_list, node_color='g')
    nx.draw_networkx_edges(G, pos, edge_list, edge_color='g', width=6, style='solid', alpha=0.4)
    nx.draw_networkx_edge_labels(G, pos, nx.get_edge_attributes(G,'weight'))


def find_shortest_path(G, start_node, target_node):
    """
    Uses Graphic Neural Network to 
    determine the shortest path between 
    the source and the destination node
    
    G: nx graph
    start_node: source node where the path starts
    target_node: destination node where the path ends
    """
    
    x=torch.zeros(len(G.nodes))
    x[start_node]=1
    x[target_node]=2
    
    edge_index=torch.tensor(list(G.edges)).T
    edge_index=torch.cat([edge_index, torch.flip(edge_index, (0,))],1)
    
    weights=nx.get_edge_attributes(G,'weight')
    weights.update({(x[1],x[0]):w for x,w in weights.items()})
    
    edge_weight = [weights[(int(u),int(v))] for u,v in zip(*edge_index)]
    edge_weight=torch.tensor(edge_weight)
    
    solution_node, solution_edge=model(x, edge_index, edge_weight)
    draw(G, solution_node.cpu().tolist(), solution_edge.cpu().tolist())


# Load a Test Graph
G = pkl.load(open('sample_test_graph.pkl','rb'))
draw(G)
plt.savefig('input_graph.png')


# If remove_edge, Corrupt the Graph
if remove_edge:
    G.remove_edge(7, 10)


# Load the Pretrained GNN Model
model = torch.jit.load('model_scripted.pt')
model = model.eval()


# Find Shortest Path
find_shortest_path(G, source_node, destination_node)
plt.savefig('prediction.png')
