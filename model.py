import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_adj


class Net(Module):

    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.dim_in = self.args.in_channels

        self.dim_hidden = self.args.hidden_channel_1
        self.dim_hidden2 = self.args.hidden_channel_2
        self.dropout = self.args.dropout
        self.act = torch.relu

        # project
        #self.project = GCNConv(3 * self.dim_in, self.dim_in, add_self_loops=False)
        self.encoder_omics12 = Encoder(self.dim_in, self.dim_hidden, self.dropout, self.act)
        self.encoder_omics23 = Encoder(self.dim_in, self.dim_hidden, self.dropout, self.act)
        self.project = Encoder(self.dim_hidden, 100, self.dropout, self.act)
        # attention
        self.mlp = MLP(self.dim_hidden, self.dim_hidden, self.dropout)
        self.fc1 = nn.Linear(self.dim_hidden + self.dim_hidden, self.dim_hidden)
        self.fc2 = nn.Linear(self.dim_hidden, self.dim_hidden)
        self.fc3 = nn.Linear(self.dim_hidden, self.dim_hidden)
        self.atten_strc_omic1 = AttentionLayer(self.dim_hidden, self.dim_hidden)
        self.atten_strc_omic2 = AttentionLayer(self.dim_hidden, self.dim_hidden)
        self.atten_strc_omic3 = AttentionLayer(self.dim_hidden, self.dim_hidden)
        self.atten_omics = AttentionLayer(self.dim_hidden, self.dim_hidden)
        # classifier
        self.classifier = Classifier(100, self.dim_hidden2, self.dropout, self.act)

    def forward(self, x, edge_index):
        edge_index, _ = dropout_adj(edge_index, p=self.dropout,
                                    force_undirected=True,
                                    num_nodes=x.shape[0],
                                    training=self.training)
        x = F.dropout(x, p=self.dropout, training=self.training)

        features_omics1 = x[:, 16:32]
        features_omics2 = x[:, 32:48]
        features_omics3 = x[:, 0:16]

        # GCN encoder
        emb_latent_omics1 = self.encoder_omics12(features_omics1, edge_index)
        emb_latent_omics21 = self.encoder_omics12(features_omics2, edge_index)
        emb_latent_omics23 = self.encoder_omics23(features_omics2, edge_index)
        emb_latent_omics3 = self.encoder_omics23(features_omics3, edge_index)
        emb_latent_omics21, emb_latent_omics23 = self.mlp(emb_latent_omics21, emb_latent_omics23)
        emb_latent_omics2 = self.fc1(torch.cat([emb_latent_omics21, emb_latent_omics23], dim=1))

        # Cross-omics attention
        emb_latent_combined = torch.stack([emb_latent_omics1, emb_latent_omics2, emb_latent_omics3])
        emb_latent_combined, alpha_omics = self.atten_omics(emb_latent_combined)
        
        # classifier
        emb_latent_combined = F.dropout(self.project(emb_latent_combined, edge_index), p=self.dropout, training=self.training)
        pred = self.classifier(emb_latent_combined, edge_index)

        return pred

class Omics_label_Predictor(Module):
    def __init__(self, z_emb_size1):
        super(Omics_label_Predictor, self).__init__()

        # input to first hidden layer
        self.hidden1 = nn.Linear(z_emb_size1, 5)

        # second hidden layer and output
        self.hidden2 = nn.Linear(5, 3)

    def forward(self, X):
        X = torch.sigmoid(self.hidden1(X))
        y_pre = self.hidden2(X)
        #y_pre = F.sigmoid(self.hidden2(X))
        return y_pre


class MLP(Module):
    def __init__(self, in_feat, out_feat, dropout):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_feat, out_feat),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, x, y):
        q_x = self.mlp(x)
        q_y = self.mlp(y)
        return q_x, q_y


class AttentionLayer(Module):
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)

    def forward(self, emb):
        self.emb = emb.transpose(0, 1)  # gene * omics * hidden
        self.v = torch.tanh(torch.matmul(self.emb, self.w_omega))  # g * o * h
        self.vu = torch.matmul(self.v, self.u_omega)  # g * o * 1
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6, dim=1)  # o
        emb_combined = torch.matmul(torch.transpose(self.emb, 1, 2), torch.unsqueeze(self.alpha, -1))
        return torch.squeeze(emb_combined), self.alpha


class Encoder(Module):
    def __init__(self, in_feat, out_feat, dropout, act):
        super(Encoder, self).__init__()
        self.add_self_loops = False
        self.conv1 = GCNConv(in_feat, out_feat, add_self_loops=self.add_self_loops)
        self.fc = nn.Linear(in_feat, out_feat)
        self.dropout = dropout
        self.act = act

    def forward(self, x, edge_index):
        x0 = self.act(self.fc(x))
        x = self.conv1(x, edge_index)
        return x0 + x


class Decoder(Module):
    def __init__(self, in_feat, out_feat):
        super(Decoder, self).__init__()
        self.add_self_loops = False
        self.conv1 = GCNConv(in_feat, out_feat, add_self_loops=self.add_self_loops)
        self.fc = nn.Linear(in_feat, out_feat)

    def forward(self, x, edge_index):
        x0 = self.fc(x)
        x = self.conv1(x, edge_index)
        return x + x0


class Classifier(Module):
    def __init__(self, in_feat, in_hidden, dropout, act):
        super(Classifier, self).__init__()
        self.add_self_loops = False
        self.conv1 = GCNConv(in_feat, in_hidden, add_self_loops=self.add_self_loops)
        self.conv2 = GCNConv(in_hidden, 1, add_self_loops=self.add_self_loops)
        self.fc1 = nn.Linear(in_feat, in_hidden)
        self.fc2 = nn.Linear(in_hidden, 1)
        self.act = act
        self.dropout = dropout

    def forward(self, x, edge_index):
        #x = F.dropout(x, p=self.dropout, training=self.training)
        x0 = self.act(self.fc1(x))
        x = self.act(self.conv1(x, edge_index))
        x = F.dropout(x0 + x, p=self.dropout, training=self.training)
        x0 = self.fc2(x)
        x = self.conv2(x, edge_index)
        return x + x0