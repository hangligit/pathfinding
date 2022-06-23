# Path Finding

This is the official implementation for the paper 
[Biologically Inspired Neural Path Finding](https://arxiv.org/abs/2206.05971). 
We propose a GNN framework to find the optimal low cost path between 
a source node and a destination node on a weighted graph.

## Installation
Requirements: python=3.9, cuda=11.3
```shell
conda create -n ENVNAME python=3.9
conda activate ENVNAME
pip install -r requirements.txt
# install pytorch_geometric
pip install torch-scatter==2.0.9 torch-sparse==0.6.13 torch-cluster==1.6.0 torch-spline-conv==1.2.1 torch-geometric==2.0.4 -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
```
Note: Installation of pytorch_geometric is sensitive to the cuda version. 
For different cuda versions (e.g., cuda-10.2) please refer to [pytorch-geometric Installation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) 
to manually install pytorch_geometric.

## Demo
You can find a demo script for inference on an example graph with a pre-trained GNN 
model under the folder _./demo/path-finding.py_

## Data Generation
The following script creates a dataset of simulated graphs and ground truth paths
split into train/val/test sets.
```shell
python simulation.py --output_dir dataset/samples --num_graphs 10000
```

## Train
To train our gnn model, run the following code. It saves the experiment 
result in the _output_ directory defined in the config.
```shell
python train.py --data_root dataset/samples --output_dir output/gnn_samples --num_workers 8
```

## Evaluation
Use the command to evaluate a trained model. You need to specify the experiment
directory where the model is stored.
```shell
python test.py --exp_dir output/gnn_samples
```

## Citation
If you find our work helpful in your research, please consider citing us
```latex
Li H, Khan Q, Tresp V, Cremers D. Biologically Inspired Neural Path Finding. In Brain Informatics. BI 2022. Springer
```
