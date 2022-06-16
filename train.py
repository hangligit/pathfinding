#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_scatter import scatter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import json
import time
import os
import argparse

from model import GNN
from data import PathData, GraphDataset

import sys
import logging
logger = logging.getLogger('train')
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


"""
Training Script
"""


parser=argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='./dataset/samples/')
parser.add_argument('--output_dir', type=str, default='./output/gnn_samples')
parser.add_argument('--num_workers', type=int, default=0,  help='optional, default=0')
parser.add_argument('--batch_size', type=int, default=128, help='optional, default=128')
args=parser.parse_args()


# Config
cfg=dict(
    learning_rate=0.0001,
    n_epochs=30,
    data_root=args.data_root,
    batch_size_train=args.batch_size,
    batch_size_test=args.batch_size,
    num_workers=args.num_workers,
    model_cfg=dict(
        type='GAT',
        num_layers=12,
        num_hidden=200,
        dropout=0.1,
        node_head=dict(
            type=True,
            num_layers=2,
        ),
        edge_head=dict(
            type=True,
            num_layers=2,
        ),
        loss_cls=dict(
            node_loss=True,
            edge_loss=True,
            matching_loss=dict(
                type=False,
            ),
            length_loss=dict(
                type=False,
            ),
        ),
        metric_cls=dict(
            node_acc=True,
            graph_node=True,
            edge_acc=True,
            graph_edge=True,
        ),
    ),
    outdir=args.output_dir
)

outdir=cfg['outdir']
os.makedirs(outdir, exist_ok=True)


json.dump(cfg, open(os.path.join(outdir, 'cfg.json'), 'w'))

handlers = logger.handlers[:]
for handler in handlers:
    handler.close()
    logger.removeHandler(handler)

logfile = logging.FileHandler(os.path.join(outdir, "log.txt"))
logger.addHandler(logfile)
logger.info("==================== New Run ==================")

for k,v in cfg.items():
    logger.info(k + ': %s', v)

try:
    from shutil import copyfile
    copyfile(os.path.abspath(__file__), os.path.join(outdir, 'main.py'))
except Exception:
    pass


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

    test_acc_graph=np.mean(np.array(predictions_graph_e)*np.array(predictions_graph_n))
    logger.info('Test %4d: loss: %.4f test acc: %.4f path acc: %.4f'%(epoch, test_loss, test_acc, test_acc_graph))

    return test_loss, test_acc, result


def train(epoch):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = model.loss(batch, out)
        loss.backward()
        optimizer.step()

        if epoch % log_epoch_interval == 0 and batch_idx % log_interval == 0:
            train_acc,_ = model.metric(batch, out)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\t Acc: {:.4f}'.format(
                epoch, batch_idx * len(batch), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), train_acc))
            train_losses.append(loss.item())
            train_accs.append(train_acc)
            train_counter.append((batch_idx*batch_size_train) + ((epoch-1)*len(train_loader.dataset)))


batch_size_train=cfg['batch_size_train']
batch_size_test=cfg['batch_size_test']

train_loader=DataLoader(GraphDataset(cfg['data_root']+'/train'), batch_size=batch_size_train, shuffle=True, follow_batch=['edge_index'], num_workers=cfg["num_workers"])
valid_loader=DataLoader(GraphDataset(cfg['data_root']+'/val'), batch_size=batch_size_test, follow_batch=['edge_index'], num_workers=cfg["num_workers"])
test_loader=DataLoader(GraphDataset(cfg['data_root']+'/test'), batch_size=batch_size_test, follow_batch=['edge_index'], num_workers=cfg["num_workers"])

n_epochs = cfg['n_epochs']
log_epoch_interval = 1
log_interval = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GNN(cfg["model_cfg"]).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])


train_losses = []
train_accs = []
train_counter = []
test_losses = []
test_accs = []
test_counter = []

tic=time.time()

best_loss = np.inf
test(0, test_loader)
for epoch in range(1, n_epochs+1):
    train(epoch)
    if epoch % log_epoch_interval == 0:
        test(epoch, valid_loader)
        torch.save(model.state_dict(), os.path.join(outdir,'model.pth'))
        if test_losses[-1]<=best_loss:
            best_loss = test_losses[-1]
            torch.save(model.state_dict(), os.path.join(outdir,'model_best.pth'))

toc=time.time()
print('Time(s):', round((toc-tic)))

plt.figure(figsize=(8,8))
fig = plt.subplot(211)
plt.plot(train_counter, train_losses, color='blue')
plt.plot(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Val Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')

fig = plt.subplot(212)
plt.plot(train_counter, train_accs, color='blue')
plt.plot(test_counter, test_accs, color='red')
plt.legend(['Train Acc', 'Val Acc'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('accuracy')

plt.savefig(os.path.join(outdir, 'loss.png'))
