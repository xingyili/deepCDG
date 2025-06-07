import warnings
import copy
import numpy as np
import time
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops
from utils import get_ppi, k_folds, set_seed
from model import Net
import argparse
from tqdm import tqdm
from sklearn import metrics
from plot import plot
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CPDB',
                    choices=['CPDB', 'STRINGdb', 'MULTINET', 'PCNet', 'IRefIndex', 'IRefIndex_2015'],
                    help="The dataset to be used.")
parser.add_argument('--device', type=str, default='cuda:0',
                    choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
parser.add_argument('--in_channels', type=int, default=16)
parser.add_argument('--hidden_channel_1', type=int, default=48)
parser.add_argument('--hidden_channel_2', type=int, default=200)
parser.add_argument('--epochs', type=int, default=1200)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=0.)
parser.add_argument('--alpha', type=float, default=0.15)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--use_5_CV_pkl', type=bool, default=False,
                    help='whether to use existed 5_CV.pkl')
parser.add_argument('--logs', default=False, help='Save the results to a log file')
parser.add_argument('--times', type=int, default=1, help='Times of 5_CV')
parser.add_argument('--method_name', type=str, default='Mymodel')
args = parser.parse_args()

set_seed(args.seed)

# load data
device = torch.device(args.device)
data = get_ppi(args.dataset, PATH='./PPI_data/')
data.x = data.x[:, :48]
data = data.to(device)

# 5 fold
if args.use_5_CV_pkl:
    with open(f'../data/{args.dataset}_data.pkl', 'rb') as file:
        k_sets = pickle.load(file)
else:
    k_sets = k_folds(data)

# plot list
precision_list = []
recall_list = []
fpr_list = []
tpr_list = []

# auc prc
AUC = np.zeros(shape=(10, 5))
AUPR = np.zeros(shape=(10, 5))

@torch.no_grad()
def test(data, mask):
    model.eval()
    x = model(data.x, data.edge_index)
    pred = torch.sigmoid(x[mask])
    precision, recall, _ = metrics.precision_recall_curve(data.y[mask].cpu().numpy(),
                                                                    pred.cpu().detach().numpy())
    fpr, tpr, _ = metrics.roc_curve(data.y[mask].cpu().numpy(), pred.cpu().detach().numpy())
    precision_list.append(precision)
    recall_list.append(recall)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    area = metrics.auc(recall, precision)
    return metrics.roc_auc_score(data.y[mask].cpu().numpy(), pred.cpu().detach().numpy()), area

# train
auprc = 0
cnt = 0
for i in range(args.times):
    for cv_run in range(5):
        tr_mask, te_mask = k_sets[i][cv_run]
        model = Net(args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for epoch in tqdm(range(args.epochs)):
            model.train()
            optimizer.zero_grad()
            pred = model(data.x, data.edge_index)
            cls_loss = F.binary_cross_entropy_with_logits(pred[tr_mask], data.y[tr_mask].float())
            loss = cls_loss 
            loss.backward()
            optimizer.step()
        AUC[i][cv_run], AUPR[i][cv_run] = test(data, te_mask)
        auprc += AUPR[i][cv_run]
        cnt += 1
        print(f'AUC: {AUC[i][cv_run]}, AUPR: {AUPR[i][cv_run]}, cv_run: {cv_run}, mean: {auprc / cnt}')
