import argparse
import os.path as osp

import torch
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv
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

dataset = 'Pubmed'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]

dim = dataset.num_features
lstm_hidden = dataset.num_features
layer_num = 2


class Breadth(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Breadth, self).__init__()
        self.gatconv = GATConv(in_dim, out_dim, heads=1)#这里in_dim和out_dim都=dim=256
        # self.gatconv = GATConv(256, 256, heads=1)

    def forward(self, x, edge_index):
        x = torch.tanh(self.gatconv(x, edge_index))
        return x


class Depth(torch.nn.Module):
    def __init__(self, in_dim, hidden):
        super(Depth, self).__init__()
        self.lstm = torch.nn.LSTM(in_dim, hidden, 1, bias=False)

    def forward(self, x, h, c):
        x, (h, c) = self.lstm(x, (h, c))
        return x, (h, c)


class GeniePathLayer(torch.nn.Module):
    def __init__(self, in_dim):
        super(GeniePathLayer, self).__init__()
        self.breadth_func = Breadth(in_dim, in_dim)
        self.depth_func = Depth(in_dim, lstm_hidden)

    def forward(self, x, edge_index, h, c):
        x = self.breadth_func(x, edge_index)
        x = x[None, :]
        x, (h, c) = self.depth_func(x, h, c)
        x = x[0]
        return x, (h, c)


class GeniePath(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GeniePath, self).__init__()
        # self.lin1 = torch.nn.Linear(in_dim, dim)
        self.gplayers = torch.nn.ModuleList(
            [GeniePathLayer(in_dim) for i in range(layer_num)])
        self.lin2 = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        # x = self.lin1(x)
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

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model, data = kwargs[args.model](dataset.num_features,dataset.num_classes).to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


def train():
    model.train()

    total_loss = 0
    # for data in train_loader:
    # num_graphs = data.num_graphs
    # data.batch = None
    # data = data.to(device)
    optimizer.zero_grad()
    # loss = loss_op(model(data.x, data.edge_index), data.y)
    loss = F.nll_loss(model(data.x, data.edge_index)[data.train_mask], data.y[data.train_mask])
    # loss = F.cross_entropy(model(data.x, data.edge_index)[data.train_mask], data.y[data.train_mask])
    # total_loss += loss.item() * num_graphs
    total_loss += loss.item() 
    loss.backward()
    optimizer.step()
    return total_loss / len(data)


# def test(loader):
#     model.eval()

#     ys, preds = [], []
#     for data in loader:
#         ys.append(data.y)
#         with torch.no_grad():
#             out = model(data.x.to(device), data.edge_index.to(device))
#         preds.append((out > 0).float().cpu())

#     y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
#     return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0

def test():
    model.eval()
    logits, accs = model(data.x, data.edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

losslist=[]
for epoch in range(1, 500):
    loss = train()
    losslist.append(loss)
    # val_f1 = test(val_loader)
    # test_f1 = test(test_loader)
    # print('Epoch: {:02d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(
    #     epoch, loss, val_f1, test_f1))
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, *test()))
# from matplotlib import pyplot as plt 
# # %matplotlib inline

# plt.plot(losslist)
# plt.show()

















"""
(aliatte)  ✘ zk@E211  /disk4/zk/charmsftp/ali_attention  cd /disk4/zk/charmsftp/ali_attention ; env PYTHONIOENCODING=UTF-8 PYTHONUNBUFFERED=1 /disk2/zk/sw/Anaconda2/envs/aliatte/bin/python /disk2/zk/.vscode-server-insiders/extensions/ms-python.python-2019.6.24221/pythonFiles/ptvsd_launcher.py --default --client --host localhost --port 45739 /disk4/zk/charmsftp/ali_attention/GeniePath-pytorch/cora_geniepath.py 
Epoch: 001, Train: 0.1429, Val: 0.1560, Test: 0.1440
Epoch: 002, Train: 0.3071, Val: 0.2160, Test: 0.2150
Epoch: 003, Train: 0.3643, Val: 0.2140, Test: 0.2230
Epoch: 004, Train: 0.4786, Val: 0.2940, Test: 0.2920
Epoch: 005, Train: 0.6429, Val: 0.5480, Test: 0.5250
Epoch: 006, Train: 0.6429, Val: 0.4360, Test: 0.4040
Epoch: 007, Train: 0.5214, Val: 0.2660, Test: 0.2660
Epoch: 008, Train: 0.3714, Val: 0.2720, Test: 0.2420
Epoch: 009, Train: 0.3214, Val: 0.2700, Test: 0.2590
Epoch: 010, Train: 0.2643, Val: 0.1860, Test: 0.1690
Epoch: 011, Train: 0.3857, Val: 0.2800, Test: 0.2710
Epoch: 012, Train: 0.4643, Val: 0.3500, Test: 0.3240
Epoch: 013, Train: 0.5929, Val: 0.4280, Test: 0.4190
Epoch: 014, Train: 0.4571, Val: 0.4660, Test: 0.4210
Epoch: 015, Train: 0.4786, Val: 0.3260, Test: 0.3040
Epoch: 016, Train: 0.6571, Val: 0.4380, Test: 0.4510
Epoch: 017, Train: 0.5500, Val: 0.4700, Test: 0.4790
Epoch: 018, Train: 0.6429, Val: 0.5740, Test: 0.5740
Epoch: 019, Train: 0.7286, Val: 0.5840, Test: 0.5570
Epoch: 020, Train: 0.7143, Val: 0.5660, Test: 0.5480
Epoch: 021, Train: 0.8929, Val: 0.7000, Test: 0.7060
Epoch: 022, Train: 0.9071, Val: 0.6760, Test: 0.6850
Epoch: 023, Train: 0.8214, Val: 0.6740, Test: 0.6880
Epoch: 024, Train: 0.9571, Val: 0.6660, Test: 0.6710
Epoch: 025, Train: 0.9071, Val: 0.6620, Test: 0.6610
Epoch: 026, Train: 0.8643, Val: 0.6320, Test: 0.6630
Epoch: 027, Train: 0.9857, Val: 0.6880, Test: 0.6760
Epoch: 028, Train: 0.8500, Val: 0.5180, Test: 0.5380
Epoch: 029, Train: 0.7714, Val: 0.4700, Test: 0.4930
Epoch: 030, Train: 0.8857, Val: 0.5300, Test: 0.5720
Epoch: 031, Train: 0.8143, Val: 0.5760, Test: 0.5790
Epoch: 032, Train: 0.8500, Val: 0.6180, Test: 0.6140
Epoch: 033, Train: 0.9357, Val: 0.6960, Test: 0.7090
Epoch: 034, Train: 0.8286, Val: 0.6240, Test: 0.6150
Epoch: 035, Train: 0.9286, Val: 0.6660, Test: 0.6570
Epoch: 036, Train: 0.9786, Val: 0.7060, Test: 0.7270
Epoch: 037, Train: 1.0000, Val: 0.7180, Test: 0.7330
Epoch: 038, Train: 1.0000, Val: 0.7080, Test: 0.7140
Epoch: 039, Train: 0.9929, Val: 0.6800, Test: 0.6960
Epoch: 040, Train: 0.9929, Val: 0.6420, Test: 0.6740
Epoch: 041, Train: 1.0000, Val: 0.6200, Test: 0.6500
Epoch: 042, Train: 1.0000, Val: 0.6200, Test: 0.6240
Epoch: 043, Train: 1.0000, Val: 0.6280, Test: 0.6480
Epoch: 044, Train: 1.0000, Val: 0.6660, Test: 0.6910
Epoch: 045, Train: 1.0000, Val: 0.7100, Test: 0.7280
Epoch: 046, Train: 1.0000, Val: 0.7360, Test: 0.7510
Epoch: 047, Train: 1.0000, Val: 0.7300, Test: 0.7470
Epoch: 048, Train: 1.0000, Val: 0.7220, Test: 0.7350
Epoch: 049, Train: 1.0000, Val: 0.7200, Test: 0.7210
Epoch: 050, Train: 1.0000, Val: 0.7000, Test: 0.7090
Epoch: 051, Train: 1.0000, Val: 0.6980, Test: 0.7060
Epoch: 052, Train: 1.0000, Val: 0.7060, Test: 0.7170
Epoch: 053, Train: 1.0000, Val: 0.7140, Test: 0.7250
Epoch: 054, Train: 1.0000, Val: 0.7240, Test: 0.7280
Epoch: 055, Train: 1.0000, Val: 0.7260, Test: 0.7310
Epoch: 056, Train: 1.0000, Val: 0.7320, Test: 0.7370
Epoch: 057, Train: 1.0000, Val: 0.7260, Test: 0.7360
Epoch: 058, Train: 1.0000, Val: 0.7240, Test: 0.7380
Epoch: 059, Train: 1.0000, Val: 0.7120, Test: 0.7380
Epoch: 060, Train: 1.0000, Val: 0.7220, Test: 0.7340
Epoch: 061, Train: 1.0000, Val: 0.7200, Test: 0.7280
Epoch: 062, Train: 1.0000, Val: 0.7260, Test: 0.7330
Epoch: 063, Train: 1.0000, Val: 0.7240, Test: 0.7400
Epoch: 064, Train: 1.0000, Val: 0.7220, Test: 0.7440
Epoch: 065, Train: 1.0000, Val: 0.7220, Test: 0.7440
Epoch: 066, Train: 1.0000, Val: 0.7260, Test: 0.7460
Epoch: 067, Train: 1.0000, Val: 0.7320, Test: 0.7460
Epoch: 068, Train: 1.0000, Val: 0.7320, Test: 0.7420
Epoch: 069, Train: 1.0000, Val: 0.7340, Test: 0.7420
Epoch: 070, Train: 1.0000, Val: 0.7380, Test: 0.7420
Epoch: 071, Train: 1.0000, Val: 0.7420, Test: 0.7400
Epoch: 072, Train: 1.0000, Val: 0.7440, Test: 0.7400
Epoch: 073, Train: 1.0000, Val: 0.7420, Test: 0.7410
Epoch: 074, Train: 1.0000, Val: 0.7400, Test: 0.7420
Epoch: 075, Train: 1.0000, Val: 0.7440, Test: 0.7430
Epoch: 076, Train: 1.0000, Val: 0.7400, Test: 0.7430
Epoch: 077, Train: 1.0000, Val: 0.7380, Test: 0.7430
Epoch: 078, Train: 1.0000, Val: 0.7420, Test: 0.7470
Epoch: 079, Train: 1.0000, Val: 0.7400, Test: 0.7450
Epoch: 080, Train: 1.0000, Val: 0.7420, Test: 0.7450
Epoch: 081, Train: 1.0000, Val: 0.7440, Test: 0.7450
Epoch: 082, Train: 1.0000, Val: 0.7400, Test: 0.7450
Epoch: 083, Train: 1.0000, Val: 0.7380, Test: 0.7470
Epoch: 084, Train: 1.0000, Val: 0.7380, Test: 0.7470
Epoch: 085, Train: 1.0000, Val: 0.7400, Test: 0.7480
Epoch: 086, Train: 1.0000, Val: 0.7380, Test: 0.7490
Epoch: 087, Train: 1.0000, Val: 0.7360, Test: 0.7480
Epoch: 088, Train: 1.0000, Val: 0.7400, Test: 0.7470
Epoch: 089, Train: 1.0000, Val: 0.7420, Test: 0.7470
Epoch: 090, Train: 1.0000, Val: 0.7440, Test: 0.7470
Epoch: 091, Train: 1.0000, Val: 0.7420, Test: 0.7450
Epoch: 092, Train: 1.0000, Val: 0.7400, Test: 0.7460
Epoch: 093, Train: 1.0000, Val: 0.7400, Test: 0.7460
Epoch: 094, Train: 1.0000, Val: 0.7400, Test: 0.7450
Epoch: 095, Train: 1.0000, Val: 0.7380, Test: 0.7450
Epoch: 096, Train: 1.0000, Val: 0.7380, Test: 0.7450
Epoch: 097, Train: 1.0000, Val: 0.7380, Test: 0.7460
Epoch: 098, Train: 1.0000, Val: 0.7380, Test: 0.7460
Epoch: 099, Train: 1.0000, Val: 0.7380, Test: 0.7480
Epoch: 100, Train: 1.0000, Val: 0.7380, Test: 0.7490
[1]    4693 terminated  env PYTHONIOENCODING=UTF-8 PYTHONUNBUFFERED=1   --default --client --host 



(aliatte)  ✘ zk@E211  /disk4/zk/charmsftp/ali_attention  cd /disk4/zk/charmsftp/ali_attention ; env PYTHONIOENCODING=UTF-8 PYTHONUNBUFFERED=1 /disk2/zk/sw/Anaconda2/envs/aliatte/bin/python /disk2/zk/.vscode-server-insiders/extensions/ms-python.python-2019.6.24221/pythonFiles/ptvsd_launcher.py --default --client --host localhost --port 32791 /disk4/zk/charmsftp/ali_attention/GeniePath-pytorch/cora_geniepath_v1.py 
Epoch: 001, Train: 0.1571, Val: 0.1140, Test: 0.1050
Epoch: 002, Train: 0.1929, Val: 0.1660, Test: 0.1530
Epoch: 003, Train: 0.3857, Val: 0.2200, Test: 0.2100
Epoch: 004, Train: 0.6786, Val: 0.4800, Test: 0.4770
Epoch: 005, Train: 0.6857, Val: 0.5080, Test: 0.4810
Epoch: 006, Train: 0.6357, Val: 0.4640, Test: 0.4350
Epoch: 007, Train: 0.4857, Val: 0.2800, Test: 0.2720
Epoch: 008, Train: 0.3071, Val: 0.2380, Test: 0.2070
Epoch: 009, Train: 0.3000, Val: 0.2540, Test: 0.2450
Epoch: 010, Train: 0.2357, Val: 0.1640, Test: 0.1570
Epoch: 011, Train: 0.2786, Val: 0.2200, Test: 0.2230
Epoch: 012, Train: 0.2857, Val: 0.2060, Test: 0.1980
Epoch: 013, Train: 0.4571, Val: 0.3640, Test: 0.3530
Epoch: 014, Train: 0.4071, Val: 0.3540, Test: 0.3270
Epoch: 015, Train: 0.5857, Val: 0.3800, Test: 0.3550
Epoch: 016, Train: 0.2857, Val: 0.2400, Test: 0.2530
Epoch: 017, Train: 0.5286, Val: 0.2980, Test: 0.3000
Epoch: 018, Train: 0.6000, Val: 0.5620, Test: 0.6070
Epoch: 019, Train: 0.5714, Val: 0.5340, Test: 0.5450
Epoch: 020, Train: 0.7000, Val: 0.5360, Test: 0.5220
Epoch: 021, Train: 0.8071, Val: 0.5380, Test: 0.5470
Epoch: 022, Train: 0.9286, Val: 0.6560, Test: 0.6240
Epoch: 023, Train: 0.9143, Val: 0.5960, Test: 0.6010
Epoch: 024, Train: 0.8571, Val: 0.5920, Test: 0.6050
Epoch: 025, Train: 0.9643, Val: 0.7140, Test: 0.7090
Epoch: 026, Train: 0.9571, Val: 0.6560, Test: 0.6810
Epoch: 027, Train: 0.9714, Val: 0.7100, Test: 0.7020
Epoch: 028, Train: 0.9786, Val: 0.7180, Test: 0.7200
Epoch: 029, Train: 1.0000, Val: 0.7040, Test: 0.7070
Epoch: 030, Train: 0.9929, Val: 0.6840, Test: 0.7010
Epoch: 031, Train: 1.0000, Val: 0.7240, Test: 0.7360
Epoch: 032, Train: 1.0000, Val: 0.7540, Test: 0.7680
Epoch: 033, Train: 1.0000, Val: 0.7540, Test: 0.7690
Epoch: 034, Train: 1.0000, Val: 0.7580, Test: 0.7690
Epoch: 035, Train: 1.0000, Val: 0.7540, Test: 0.7660
Epoch: 036, Train: 1.0000, Val: 0.7460, Test: 0.7520
Epoch: 037, Train: 1.0000, Val: 0.7460, Test: 0.7530
Epoch: 038, Train: 1.0000, Val: 0.7460, Test: 0.7670
Epoch: 039, Train: 1.0000, Val: 0.7500, Test: 0.7630
Epoch: 040, Train: 1.0000, Val: 0.7520, Test: 0.7690
Epoch: 041, Train: 1.0000, Val: 0.7600, Test: 0.7800
Epoch: 042, Train: 1.0000, Val: 0.7680, Test: 0.7700
Epoch: 043, Train: 1.0000, Val: 0.7700, Test: 0.7710
Epoch: 044, Train: 1.0000, Val: 0.7760, Test: 0.7810
Epoch: 045, Train: 1.0000, Val: 0.7700, Test: 0.7760
Epoch: 046, Train: 1.0000, Val: 0.7660, Test: 0.7780
Epoch: 047, Train: 1.0000, Val: 0.7600, Test: 0.7720
Epoch: 048, Train: 1.0000, Val: 0.7560, Test: 0.7710
Epoch: 049, Train: 1.0000, Val: 0.7560, Test: 0.7710
Epoch: 050, Train: 1.0000, Val: 0.7600, Test: 0.7710
Epoch: 051, Train: 1.0000, Val: 0.7600, Test: 0.7650
Epoch: 052, Train: 1.0000, Val: 0.7560, Test: 0.7660
Epoch: 053, Train: 1.0000, Val: 0.7560, Test: 0.7680
Epoch: 054, Train: 1.0000, Val: 0.7680, Test: 0.7750
Epoch: 055, Train: 1.0000, Val: 0.7720, Test: 0.7700
Epoch: 056, Train: 1.0000, Val: 0.7700, Test: 0.7680
Epoch: 057, Train: 1.0000, Val: 0.7740, Test: 0.7730
Epoch: 058, Train: 1.0000, Val: 0.7700, Test: 0.7710
Epoch: 059, Train: 1.0000, Val: 0.7660, Test: 0.7700
Epoch: 060, Train: 1.0000, Val: 0.7680, Test: 0.7710
Epoch: 061, Train: 1.0000, Val: 0.7640, Test: 0.7700
Epoch: 062, Train: 1.0000, Val: 0.7560, Test: 0.7670
Epoch: 063, Train: 1.0000, Val: 0.7560, Test: 0.7580
Epoch: 064, Train: 1.0000, Val: 0.7540, Test: 0.7560
Epoch: 065, Train: 1.0000, Val: 0.7620, Test: 0.7680
Epoch: 066, Train: 1.0000, Val: 0.7580, Test: 0.7700
Epoch: 067, Train: 1.0000, Val: 0.7640, Test: 0.7750
Epoch: 068, Train: 1.0000, Val: 0.7600, Test: 0.7780
Epoch: 069, Train: 1.0000, Val: 0.7620, Test: 0.7780
Epoch: 070, Train: 1.0000, Val: 0.7600, Test: 0.7690
Epoch: 071, Train: 1.0000, Val: 0.7560, Test: 0.7660
Epoch: 072, Train: 1.0000, Val: 0.7500, Test: 0.7650
Epoch: 073, Train: 1.0000, Val: 0.7480, Test: 0.7640
Epoch: 074, Train: 1.0000, Val: 0.7500, Test: 0.7690
Epoch: 075, Train: 1.0000, Val: 0.7500, Test: 0.7700
Epoch: 076, Train: 1.0000, Val: 0.7440, Test: 0.7700
Epoch: 077, Train: 1.0000, Val: 0.7460, Test: 0.7710
Epoch: 078, Train: 1.0000, Val: 0.7440, Test: 0.7660
Epoch: 079, Train: 1.0000, Val: 0.7460, Test: 0.7670
Epoch: 080, Train: 1.0000, Val: 0.7480, Test: 0.7650
Epoch: 081, Train: 1.0000, Val: 0.7460, Test: 0.7640
Epoch: 082, Train: 1.0000, Val: 0.7440, Test: 0.7650
Epoch: 083, Train: 1.0000, Val: 0.7400, Test: 0.7670
Epoch: 084, Train: 1.0000, Val: 0.7400, Test: 0.7690
Epoch: 085, Train: 1.0000, Val: 0.7400, Test: 0.7660
Epoch: 086, Train: 1.0000, Val: 0.7380, Test: 0.7640
Epoch: 087, Train: 1.0000, Val: 0.7420, Test: 0.7640
Epoch: 088, Train: 1.0000, Val: 0.7400, Test: 0.7680
Epoch: 089, Train: 1.0000, Val: 0.7400, Test: 0.7690
Epoch: 090, Train: 1.0000, Val: 0.7400, Test: 0.7660
Epoch: 091, Train: 1.0000, Val: 0.7400, Test: 0.7640
Epoch: 092, Train: 1.0000, Val: 0.7420, Test: 0.7640
Epoch: 093, Train: 1.0000, Val: 0.7400, Test: 0.7630
Epoch: 094, Train: 1.0000, Val: 0.7400, Test: 0.7640
Epoch: 095, Train: 1.0000, Val: 0.7400, Test: 0.7650
Epoch: 096, Train: 1.0000, Val: 0.7380, Test: 0.7640
Epoch: 097, Train: 1.0000, Val: 0.7360, Test: 0.7640
Epoch: 098, Train: 1.0000, Val: 0.7360, Test: 0.7620
Epoch: 099, Train: 1.0000, Val: 0.7380, Test: 0.7620
Epoch: 100, Train: 1.0000, Val: 0.7380, Test: 0.7620
[1]    6591 terminated  env PYTHONIOENCODING=UTF-8 PYTHONUNBUFFERED=1   --default --client --host 
"""