#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn import GATConv, GATv2Conv


"""
Model and Loss
"""


class SymmetryLoss(nn.Module):
    def __init__(self, loss_weight=1.):
        super().__init__()
        self.loss_weight=loss_weight
        self.num_edges_per_graph=24 #hack
        self.half=self.num_edges_per_graph//2

    def forward_mse(self, out, data):
        bs=out['edge'].size(0)//self.num_edges_per_graph
        conf=out['edge'].reshape(bs,-1,2)
        return F.mse_loss(conf[:,:self.half,:], conf[:,self.half:,:]) * self.loss_weight
        
    def forward(self, out, data):
        bs=out['edge'].size(0)//self.num_edges_per_graph
        conf=out['edge'].reshape(bs,-1,2)
        P = conf[:,:self.half,:].reshape(-1,2)
        Q = conf[:,self.half:,:].reshape(-1,2)
        return F.kl_div(Q, P, reduction='batchmean', log_target=True) * self.loss_weight


class MatchingLoss(nn.Module):
    def __init__(self, loss_weight=1.):
        super().__init__()
        self.loss=nn.KLDivLoss(reduction='none', log_target=True)            
        self.loss_weight=loss_weight

    def forward(self, out, data):
        tails = data.edge_index[0]
        heads = data.edge_index[1]
        
        p_tails = out['node'][tails]
        p_heads = out['node'][heads]
        p_edges = out['edge']
        
        y_tails = p_tails.argmax(1)
        y_heads = p_heads.argmax(1)
        y_edges = p_edges.argmax(1)

        nequal_mask = (y_edges==0) & (y_tails==1)
        
        eq = self.loss(p_tails[~nequal_mask], p_edges[~nequal_mask])
        neq = self.loss(p_tails[nequal_mask], torch.flip(p_edges[nequal_mask], (1,)))
        
        return torch.cat([eq, neq]).mean(0).sum() * self.loss_weight

class LengthLoss(nn.Module):
    def __init__(self, loss_weight=1.):
        super().__init__()
        self.loss=nn.MSELoss()
        self.loss_weight=loss_weight

    def forward(self, out, data):
        edge_pred=torch.argmax(out['edge'],1)==1
        pred_edge_lengths=torch.where(edge_pred, data.edge_attr, torch.tensor(0., device=edge_pred.device))
        per_graph_len=scatter(pred_edge_lengths, data.edge_index_batch, reduce='sum')
        
        return self.loss_weight * self.loss(per_graph_len, torch.tensor(data.route_len,device=edge_pred.device))

# Model

class NodeHead(nn.Module):
    def __init__(self, in_channels, dropout=0.5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, 2))
    
    def forward(self, x):
        return self.mlp(x)


class EdgeHead(nn.Module):
    def __init__(self, in_channels, dropout=0.5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, 2))
    
    def forward(self, x, edge_index):
        heads, tails=edge_index
        edges = x[heads]+x[tails]
        return self.mlp(edges)


def _pool_prediction(pred, targ, cluster):
    err = scatter((pred!=targ).float(), cluster, dim=0, reduce='sum').data.cpu().numpy()
    return err<=0


class GNN(nn.Module):
    def __init__(self, cfg):
        super(GNN, self).__init__()
        self.hid = cfg['num_hidden']
        self.in_heads = 8
        self.in_channels = 3
        self.out_channels = cfg['num_hidden']
        self.dropout=nn.Dropout(p=cfg['dropout'])
        self.num_layers = cfg['num_layers']
        
        n_ins = [self.in_channels] + [self.hid*self.in_heads]*(self.num_layers-1)
        n_outs = [self.hid]*(self.num_layers-1) + [self.out_channels]
        concats = [True]*(self.num_layers-1) + [False]
        
        self.convs = nn.ModuleList(
            [GATv2Conv(in_ch, out_ch, self.in_heads, edge_dim=1, concat=ct) for in_ch,out_ch,ct in zip(n_ins,n_outs,concats)]
        )
        
        self.node_head=self.edge_head=None
        if cfg['node_head']['type']:
            self.node_head = NodeHead(self.out_channels)
        if cfg['edge_head']['type']:
            self.edge_head = EdgeHead(self.out_channels)

        self.node_loss = cfg['loss_cls']['node_loss']
        self.edge_loss = cfg['loss_cls']['edge_loss']
        self.matching_loss = MatchingLoss() if cfg['loss_cls']['matching_loss']['type'] else None
        self.symmetry_loss = LengthLoss() if cfg['loss_cls']['length_loss']['type'] else None
        
        self.node_acc=cfg['metric_cls']['node_acc']
        self.edge_acc=cfg['metric_cls']['edge_acc']
        self.graph_node=cfg['metric_cls']['graph_node']
        self.graph_edge=cfg['metric_cls']['graph_edge']
        
        self.weights=torch.tensor([1.,20], device='cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = self.dropout(F.relu(x))

        out = dict()
        if self.node_head:
            out['node']=F.log_softmax(self.node_head(x),1)
        if self.edge_head:
            out['edge']=F.log_softmax(self.edge_head(x, edge_index),1)
        return out
    
    def loss(self, data, out):
        loss=dict()
        if self.node_loss:
            loss['node']=F.nll_loss(out['node'], data.y, weight=self.weights)
        if self.edge_loss:
            loss['edge']=F.nll_loss(out['edge'], data.ye, weight=self.weights)
        if self.symmetry_loss:
            loss['symmetry']=self.symmetry_loss(out, data)
        if self.matching_loss:
            loss['matching']=self.matching_loss(out, data)
        return sum(loss.values())
    
    def metric(self, data, out):
        metric=dict()
        if self.node_acc:
            metric['node'] = (torch.argmax(out['node'],1)==data.y).float().mean()
        if self.edge_acc:
            metric['edge'] = (torch.argmax(out['edge'],1)==data.ye).float().mean()
        return (sum(metric.values())/len(metric)).item(), metric
    
    def inference(self, data, out):
        result=dict()
        if self.node_acc:
            node_score = torch.exp(out['node'])
            result['node_score'] = node_score[:,1]
            result['node'] = torch.argmax(node_score, 1)
        if self.edge_acc:
            edge_score = torch.exp(out['edge'])
            result['edge_score'] = edge_score[:,1]
            result['edge'] = torch.argmax(edge_score, 1)
        if self.graph_node:
            result['graph_node'] = list(_pool_prediction(result['node'], data.y, data.batch))
        if self.graph_edge:
            result['graph_edge'] = list(_pool_prediction(result['edge'], data.ye, data.edge_index_batch))   
        return result

    def split_inference(self, data, out, store=None):
        result = self.inference(data, out)
        data = data.cpu()
        if store is None:
            store=dict(
                graph=[],
                node_path=[],
                node_score=[],
                edge_path=[],
                edge_score=[]
            )
        
        curr_node=curr_edge=0
        for g in data.to_data_list():
            store['graph'].append(g)
            if 'node' in out:
                node_score = result['node_score'][curr_node:curr_node+g.num_nodes]
                node_path = result['node'][curr_node:curr_node+g.num_nodes]
                node_path = (torch.where(node_path)[0]).data.cpu().numpy()
                store['node_path'].append(node_path)
                store['node_score'].append(node_score)
                curr_node+=g.num_nodes
            if 'edge' in out:
                edge_score = result['edge_score'][curr_edge:curr_edge+g.num_edges]
                edge_path = result['edge'][curr_edge:curr_edge+g.num_edges]
                edge_path = edge_path.data.cpu().numpy()
                store['edge_path'].append(edge_path)
                store['edge_score'].append(edge_score)
                curr_edge+=g.num_edges
        return store
