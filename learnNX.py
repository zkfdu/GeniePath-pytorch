import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
import os
import os.path as osp
import json

import torch
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip)
from torch_geometric.utils import remove_self_loops
# G = nx.Graph()                 #建立一个空的无向图G
# G.add_node('a')                  #添加一个节点1
# G.add_nodes_from(['b','c','d','e'])    #加点集合
# G.add_cycle(['f','g','h','j'])         #加环
# H = nx.path_graph(10)          #返回由10个节点挨个连接的无向图，所以有9条边
# G.add_nodes_from(H)            #创建一个子图H加入G
# G.add_node(H)                  #直接将图作为节点

# nx.draw(G, with_labels=True)
# plt.show()



for s, split in enumerate(['train']):
    path = osp.join("/disk4/zk/charmsftp/ali_attention/GeniePath-pytorch/data/PPI/raw", '{}_graph.json').format(split)
    with open(path, 'r') as f:
        print("***")

        G = nx.DiGraph(json_graph.node_link_graph(json.load(f)))
        # print("***")
        # nx.draw(G, with_labels=True)
        # print("***")
        # plt.show()
    x = np.load(osp.join("/disk4/zk/charmsftp/ali_attention/GeniePath-pytorch/data/PPI/raw", '{}_feats.npy').format(split))
    x = torch.from_numpy(x).to(torch.float)

    y = np.load(osp.join("/disk4/zk/charmsftp/ali_attention/GeniePath-pytorch/data/PPI/raw", '{}_labels.npy').format(split))
    y = torch.from_numpy(y).to(torch.float)

    data_list = []
    path = osp.join("/disk4/zk/charmsftp/ali_attention/GeniePath-pytorch/data/PPI/raw", '{}_graph_id.npy').format(split)
    idx = torch.from_numpy(np.load(path)).to(torch.long)
    idx = idx - idx.min()

    for i in range(idx.max().item() + 1):
        mask = idx == i#训练集一共20张图，只让第i张的mask为1  

        G_s = G.subgraph(mask.nonzero().view(-1).tolist())
        edge_index = torch.tensor(list(G_s.edges)).t().contiguous()
        edge_index = edge_index - edge_index.min()
        edge_index, _ = remove_self_loops(edge_index)

        data = Data(edge_index=edge_index, x=x[mask], y=y[mask])

        if self.pre_filter is not None and not self.pre_filter(data):
            continue

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data_list.append(data)
    torch.save(self.collate(data_list), self.processed_paths[s])
