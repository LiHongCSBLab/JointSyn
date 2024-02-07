import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim

import dgl
from dgl.nn.pytorch import GATConv
import os

import config

class Net_View1(nn.Module):
    def __init__(self, model_config, joint_graph):
        super(Net_View1, self).__init__()
        h_dim = model_config['h_dim']
        in_dim_drug = model_config['in_dim_drug']
        num_heads = model_config['num_heads']
        dropout = model_config['dropout']
        gpu = model_config['gpu']
        if gpu >= 0 and torch.cuda.is_available():
            self.device = torch.device(gpu)
        else:
            self.device = torch.device('cpu')
        
        self.joint_graph = joint_graph

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.drug_gcn1 = GATConv(in_dim_drug, h_dim, num_heads=num_heads, 
                                 feat_drop=dropout, allow_zero_in_degree=True)
        self.drug_gcn2 = GATConv(h_dim*num_heads, h_dim, num_heads=5, 
                                 feat_drop=dropout, allow_zero_in_degree=True)
        self.drug_gcn3 = GATConv(h_dim*5, h_dim, num_heads=1, 
                                 feat_drop=dropout, allow_zero_in_degree=True)
        self.drug_fc = nn.Linear(h_dim, h_dim*2)
        nn.init.kaiming_normal_(self.drug_fc.weight)
        
    def forward(self, triplets):
        triplets = triplets.long()
        drug_pair_g = self.joint_graph[triplets[:,3]]
        drug_pair_g = dgl.batch(drug_pair_g)
        drug_pair_g = drug_pair_g.to(self.device)
        
        drug_pair_h = drug_pair_g.ndata['feat'].to(torch.float32)
        drug_pair_h = F.elu(self.drug_gcn1(drug_pair_g,drug_pair_h)).flatten(1)
        drug_pair_h = F.elu(self.drug_gcn2(drug_pair_g,drug_pair_h)).flatten(1)
        drug_pair_h = F.elu(self.drug_gcn3(drug_pair_g,drug_pair_h)).flatten(1)
        drug_pair_g.ndata['h'] = drug_pair_h
        drug_pair_hg = dgl.max_nodes(drug_pair_g, 'h')
        del drug_pair_h, drug_pair_g
        drug_pair_hg = self.drug_fc(drug_pair_hg)
        drug_pair_hg = self.relu(drug_pair_hg)
        
        return drug_pair_hg


class Net_View2(nn.Module):
    def __init__(self, model_config, drug_fp, cell):
        super(Net_View2, self).__init__()
        h_dim = model_config['h_dim']
        dropout = model_config['dropout']
        gpu = model_config['gpu']
        in_dim_cell = model_config['in_dim_cell']
        if gpu >= 0 and torch.cuda.is_available():
            self.device = torch.device(gpu)
        else:
            self.device = torch.device('cpu')

        self.drug_fp = torch.Tensor(np.array(drug_fp))
        self.drug_fp = self.drug_fp.to(self.device)
        self.cell = torch.Tensor(np.array(cell))
        self.cell = self.cell.to(self.device)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.cell_fc = nn.Linear(in_dim_cell, h_dim)
        nn.init.kaiming_normal_(self.cell_fc.weight)

        self.fp_cell_fc1 = nn.Linear(1309+h_dim, 2048)
        nn.init.kaiming_normal_(self.fp_cell_fc1.weight)
        self.fp_cell_fc2 = nn.Linear(2048, 1024)
        nn.init.kaiming_normal_(self.fp_cell_fc2.weight)


    def forward(self, drug_cell):
        drug_cell = drug_cell.long()
        c = self.cell[drug_cell[:,1]]
        c = self.cell_fc(c)
        c = self.relu(c)

        drug_fp = self.drug_fp[drug_cell[:,0]].to(torch.float32)
        x = torch.cat((drug_fp, c), 1)
        del drug_fp, c
        x = self.relu(self.fp_cell_fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fp_cell_fc2(x))
        x = self.dropout(x)

        return x


class Net(nn.Module):
    def __init__(self, model_config, joint_graph, drug_fp, cell, model_View1, model_View2):
        super(Net, self).__init__()
        h_dim = model_config['h_dim']
        dropout = model_config['dropout']
        in_dim_cell = model_config['in_dim_cell']
        gpu = model_config['gpu']
        if gpu >= 0 and torch.cuda.is_available():
            self.device = torch.device(gpu)
        else:
            self.device = torch.device('cpu')
        
        self.cell = torch.Tensor(np.array(cell))
        self.cell = self.cell.to(self.device)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.model_View1 = model_View1
        self.cell_fc = nn.Linear(in_dim_cell, h_dim)
        nn.init.kaiming_normal_(self.cell_fc.weight)
        
        self.view1_fc1 = nn.Linear(h_dim*2+h_dim, 2048)
        nn.init.kaiming_normal_(self.view1_fc1.weight)
        self.view1_fc2 = nn.Linear(2048, 1024)
        nn.init.kaiming_normal_(self.view1_fc2.weight)
        
        self.model_View2_A = model_View2
        self.model_View2_B = model_View2

        self.prediction1 = nn.Linear(1024*3, 1024)
        nn.init.kaiming_normal_(self.prediction1.weight)
        self.prediction2 = nn.Linear(1024, 512)
        nn.init.kaiming_normal_(self.prediction2.weight)
        self.prediction3 = nn.Linear(512, 128)
        nn.init.kaiming_normal_(self.prediction3.weight)
        self.out = nn.Linear(128, 2)
        nn.init.kaiming_normal_(self.out.weight)
        
        
    def forward(self, triplets):
        triplets = triplets.long()
        c = self.cell[triplets[:,2]]
        
        drug_pair_hg = self.model_View1(triplets)
        c = self.cell_fc(c)
        c = self.relu(c)
        
        x_view1 = torch.cat((drug_pair_hg, c), 1)
        del drug_pair_hg
        x_view1 = self.relu(self.view1_fc1(x_view1))
        x_view1 = self.dropout(x_view1)
        x_view1 = self.relu(self.view1_fc2(x_view1))
        x_view1 = self.dropout(x_view1)

        xA = self.model_View2_A(triplets[:,[0,2]])
        xB = self.model_View2_B(triplets[:,[1,2]])
        x_view2 = torch.cat((xA, xB), 1)
        
        x = torch.cat((x_view1, x_view2), 1)
        x = self.relu(self.prediction1(x))
        x = self.dropout(x)
        x = self.relu(self.prediction2(x))
        x = self.dropout(x)
        x = self.relu(self.prediction3(x))
        x = self.dropout(x)
        x = self.out(x)
        return x