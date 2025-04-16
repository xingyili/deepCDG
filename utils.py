import torch

from torch_geometric.data import Data
import numpy as np
import os
import h5py
import random
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_ppi(dataset: str = 'CPDB', essential_gene=False, health_gene=False, PATH='PPI_data/'):
    print(dataset)
    ppi_PATH = os.path.join(PATH, f'{dataset}_multiomics.h5')
    ppi_essential_PATH = os.path.join(PATH, f'{dataset}_essential_test01_multiomics.h5')
    if health_gene:
        return get_health(dataset, PATH)  # get_health没实现
    elif essential_gene:
        f = h5py.File(ppi_essential_PATH, 'r')
    else:
        f = h5py.File(ppi_PATH, 'r')
    src, dst = np.nonzero(f['network'][:])
    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)
    x = torch.from_numpy(f['features'][:]).float()
    y = torch.from_numpy(
        np.logical_or(np.logical_or(f['y_test'][:], f['y_val'][:]), f['y_train'][:])).int()
    train_mask = torch.from_numpy(f['mask_train'][:])
    val_mask = torch.from_numpy(f['mask_val'][:])
    test_mask = torch.from_numpy(f['mask_test'][:])
    name = f['gene_names'][:]
    name = np.array([x.decode("utf-8") for x in name[:, 1]], dtype=str)

    g = Data(x=x, edge_index=edge_index, y=y)
    g.train_mask = train_mask
    g.val_mask = val_mask
    g.test_mask = test_mask
    # 通过train_mask test_mask val_mask得到idx_train, idx_test, idx_val
    g.idx_train = torch.nonzero(g.train_mask).squeeze()
    g.idx_test = torch.nonzero(g.test_mask).squeeze()
    g.idx_val = torch.nonzero(g.val_mask).squeeze()
    g.name = name
    edge_weight = torch.ones(edge_index.shape[1])
    g.adj = torch.sparse_coo_tensor(edge_index, edge_weight)
    g.num_classes = f['y_test'][:].shape[1]

    return g

def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def compute_auc(y_true, y_score):
    # 计算fpr, tpr
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # 计算auc
    auc_score = auc(fpr, tpr)
    return auc_score


def compute_auprc(y_true, y_score):
    # 计算precision, recall
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    # 计算auprc
    auprc_score = auc(recall, precision)
    return auprc_score

def k_folds(data, random_state=42, kfold=5):
    k_sets = []
    all_mask = (data.train_mask | data.val_mask | data.test_mask).cpu().numpy()
    y = data.y.squeeze()[all_mask.squeeze()].cpu().numpy()
    idx_list = np.arange(all_mask.shape[0])[all_mask.squeeze()]
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=random_state)
    # 根据五折交叉验证的划分结果生成train_mask和test_mask
    for _ in range(10):
        k_set = []
        for train_index, test_index in skf.split(idx_list, y):  # 划分训练集和测试集
            train_mask_set = []
            test_masks_set = []
            train_mask = np.full_like(all_mask, False)  # 初始化与all_mask相同大小的train_mask
            test_mask = np.full_like(all_mask, False)  # 初始化与all_mask相同大小的test_mask

            # 将训练集索引位置设置为True
            train_mask[idx_list[train_index]] = True
            # 将测试集索引位置设置为True
            test_mask[idx_list[test_index]] = True

            train_mask_set.append(train_mask)
            test_masks_set.append(test_mask)
            k_set.append([train_mask_set, test_masks_set])
        k_sets.append(k_set)

    return k_sets