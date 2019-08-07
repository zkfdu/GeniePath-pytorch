import argparse
import os.path as osp

import torch
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv,AGNNConv
from sklearn.metrics import f1_score
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
import torch_geometric.transforms as T

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GeniePath')
args = parser.parse_args()
assert args.model in ['GeniePath', 'GeniePathLazy']

# path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'PPI')
# train_dataset = PPI(path, split='train')
# val_dataset = PPI(path, split='val')
# test_dataset = PPI(path, split='test')
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]

# dim = dataset.num_features
# lstm_hidden = dataset.num_features
dim = 128
lstm_hidden = 128
layer_num = 2


class Breadth(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Breadth, self).__init__()
        # self.gatconv = AGNNConv(requires_grad=True)
        self.gatconv = GATConv(in_dim, out_dim,dropout=0.4, heads=1)#这里in_dim和out_dim都=dim=256

    def forward(self, x, edge_index):
        x=F.dropout(x, p=0.4, training=self.training)
        x = torch.tanh(self.gatconv(x, edge_index))
        return x


class Depth(torch.nn.Module):
    def __init__(self, in_dim, hidden):
        super(Depth, self).__init__()
        self.lstm = torch.nn.LSTM(in_dim, hidden, 1, bias=False)

    def forward(self, x, h, c):
        x=F.dropout(x, p=0.4, training=self.training)

        x, (h, c) = self.lstm(x, (h, c))
        return x, (h, c)


class GeniePathLayer(torch.nn.Module):
    def __init__(self, in_dim):
        super(GeniePathLayer, self).__init__()
        self.breadth_func = Breadth(in_dim, dim)
        self.depth_func = Depth(dim, lstm_hidden)

    def forward(self, x, edge_index, h, c):
        x = self.breadth_func(x, edge_index)
        x = x[None, :]
        x, (h, c) = self.depth_func(x, h, c)
        x = x[0]
        return x, (h, c)


class GeniePath(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GeniePath, self).__init__()
        self.lin1 = torch.nn.Linear(in_dim, dim)
        self.gplayers = torch.nn.ModuleList(
            [GeniePathLayer(dim) for i in range(layer_num)])
        self.lin2 = torch.nn.Linear(dim, out_dim)

    def forward(self, x, edge_index):
        x=F.dropout(x, p=0.4, training=self.training)

        x = self.lin1(x)
        h = torch.zeros(1, x.shape[0], lstm_hidden, device=x.device)
        c = torch.zeros(1, x.shape[0], lstm_hidden, device=x.device)
        for i, l in enumerate(self.gplayers):
            x, (h, c) = self.gplayers[i](x, edge_index, h, c)
        x = self.lin2(x)
        # return x
        return F.log_softmax(x, dim=1)



class GeniePathLazy(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GeniePathLazy, self).__init__()
        self.lin1 = torch.nn.Linear(in_dim, dim)
        self.breadths = torch.nn.ModuleList(
            [Breadth(dim, dim) for i in range(layer_num)])
        self.depths = torch.nn.ModuleList(
            [Depth(dim * 2, lstm_hidden) for i in range(layer_num)])
        self.lin2 = torch.nn.Linear(dim, out_dim)

    def forward(self, x, edge_index):
        x = self.lin1(x)
        h = torch.zeros(1, x.shape[0], lstm_hidden, device=x.device)
        c = torch.zeros(1, x.shape[0], lstm_hidden, device=x.device)
        h_tmps = []
        for i, l in enumerate(self.breadths):
            h_tmps.append(self.breadths[i](x, edge_index))
        x = x[None, :]
        for i, l in enumerate(self.depths):
            in_cat = torch.cat((h_tmps[i][None, :], x), -1)
            x, (h, c) = self.depths[i](in_cat, h, c)
        x = self.lin2(x[0])
        return x


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
kwargs = {'GeniePath': GeniePath, 'GeniePathLazy': GeniePathLazy}
# model = kwargs[args.model](train_dataset.num_features,train_dataset.num_classes).to(device)
loss_op = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model, data = kwargs[args.model](dataset.num_features,dataset.num_classes).to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


def train():
    model.train()
    optimizer.zero_grad()
    # loss = loss_op(model(data.x, data.edge_index), data.y)
    loss = F.nll_loss(model(data.x, data.edge_index)[data.train_mask], data.y[data.train_mask])
    # loss = loss_op(model(data.x, data.edge_index)[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss 


def test():
    model.eval()
    logits, accs = model(data.x, data.edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

losslist=[]
for epoch in range(1, 1001):
    loss = train()
    losslist.append(loss)
    # val_f1 = test(val_loader)
    # test_f1 = test(test_loader)
    # print('Epoch: {:02d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(
    #     epoch, loss, val_f1, test_f1))
    log = 'Epoch: {:03d},train_loss:{:.7f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, loss,*test()))
# from matplotlib import pyplot as plt 
# # %matplotlib inline

# plt.plot(losslist)
# plt.show()












