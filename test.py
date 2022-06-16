#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_scatter import scatter
import json
import os

from model import GNN
from data import PathData, GraphDataset

import argparse
import sys
import logging

"""
Test and Evaluation
"""


logger = logging.getLogger('eval')
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


parser=argparse.ArgumentParser()
parser.add_argument('--exp_dir', type=str, default='output/gnn_samples/')
args=parser.parse_args()

outdir = args.exp_dir
cfg = json.load(open(os.path.join(outdir, 'cfg.json')))


logfile = logging.FileHandler(os.path.join(outdir, "log.txt"))
logger.addHandler(logfile)
logger.info("==================== New Eval ==================")


def _pool_prediction(pred, targ, cluster):
    err = scatter((pred!=targ).float(), cluster, dim=0, reduce='sum').data.cpu().numpy()
    return err<=0


def test(epoch, loader):
    model.eval()
    test_loss = 0
    predictions_n=[]
    predictions_e=[]
    predictions_graph_n=[]
    predictions_graph_e=[]
    predictions_graph_weights=[]
    with torch.no_grad():
        for batch in loader:
            batch=batch.to(device)
            out = model(batch)
            test_loss += model.loss(batch, out).item()
            results = model.inference(batch, out)
            predictions_graph_weights += [len(r_i)-1 for r_i in batch.route]
            if 'node' in out:
                predictions_n += list((results['node']==batch.y).float().data.cpu().numpy())
                predictions_graph_n += results['graph_node']
            if 'edge' in out:
                predictions_e += list((results['edge']==batch.ye).float().data.cpu().numpy())
                predictions_graph_e += results['graph_edge']
    # for validation
    test_losses.append(test_loss)
    test_counter.append(epoch*len(train_loader.dataset))
    test_acc = np.mean(predictions_n+predictions_e)
    test_accs.append(test_acc)
    
    # logging
    result=dict(micro_node=0,
                macro_node=0,
                weighted_node=0,
                micro_edge=0,
                macro_edge=0,
                weighted_edge=0)
    
    if predictions_n:
        result['micro_node']=np.mean(predictions_n)
        result['macro_node']=np.mean(predictions_graph_n)
        result['weighted_node']=scatter(torch.from_numpy(np.array(predictions_graph_n).astype('float32')), torch.tensor(predictions_graph_weights), reduce='mean')
    if predictions_e:
        result['micro_edge']=np.mean(predictions_e)
        result['macro_edge']=np.mean(predictions_graph_e)
        result['weighted_edge']=scatter(torch.from_numpy(np.array(predictions_graph_e).astype('float32')), torch.tensor(predictions_graph_weights), reduce='mean')
    if predictions_n and predictions_e:
        result['path_accuracy']=np.mean(np.array(predictions_graph_n)*np.array(predictions_graph_e))

    return test_loss, test_acc, result


# ## Data Utils

batch_size_train=cfg['batch_size_train']
batch_size_test=cfg['batch_size_test']


train_loader=DataLoader(GraphDataset(cfg['data_root']+'/train'), batch_size=batch_size_train, shuffle=True, follow_batch=['edge_index'], num_workers=cfg.get("num_workers",4))
valid_loader=DataLoader(GraphDataset(cfg['data_root']+'/val'), batch_size=batch_size_test, follow_batch=['edge_index'], num_workers=cfg.get("num_workers",4))
test_loader=DataLoader(GraphDataset(cfg['data_root']+'/test'), batch_size=batch_size_test, follow_batch=['edge_index'], num_workers=cfg.get("num_workers",4))


# ## Load Model


model = GNN(cfg['model_cfg'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


best_path = os.path.join(outdir, 'model_best.pth')
model.load_state_dict(torch.load(best_path))


# ## Testing

test_counter = []
test_losses = []
test_accs = []


ts = test(0, train_loader)
logger.info('\n{} loss: {:.4f} overall acc: {:.4f}'.format('Training', ts[0], ts[1]))
logger.info(f'path_accuracy: {ts[2]["path_accuracy"]:.4f}')


ts=test(0, test_loader)
logger.info('\n{} loss: {:.4f} overall acc: {:.4f}'.format('Test', ts[0], ts[1]))
logger.info(f'path_accuracy: {ts[2]["path_accuracy"]:.4f}')

