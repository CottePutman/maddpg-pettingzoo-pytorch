import torch
import numpy as np
import cProfile
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from tpg.GCN import GCNwithAttention
from simple_env.data_gen_sim import DataGenerator
from tpg.heat_kernel import heat_kernel
from utils.common import get_history_and_abb

import networkx as nx
import matplotlib.pyplot as plt


# TODO 上GPU
# .to(device)应该只在进行大规模矩阵运算的时候使用
class TemporalPortfolioGraph:
    """
    构建一个资产之间的时序相似度图，并且依据时间戳进行更新
    
    每一个TPG应当唯一对应一个DataGenerator
    """
    # 初始化TPG不需要提供信息
    def __init__(self,
                 threshold=0.65,
                 output_dim=128, 
                 hidden_dim=128,
                 device='cpu'):
        """
        Args:
        - input_dim: [state=4, action=1, reward=1, next_state=4]
        - output_dim: the dimension of e_i
        """
        self.threshold = threshold
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.device = device

    # TODO 或许可以考虑更多可变参数
    # 需要注意，默认构建TPG是需要先执行reset的
    def reset(self, 
              num_asset: int, 
              price_feature: np.array):
        """
        重置图相关数据，但不包括邻接矩阵的具体数据

        Args:
        - num_asset: 资产数量，可变
        - price_feature: 用于初始化整张图的当前资产价格特征，[obs, next_obs]
        """
        self.num_asset = num_asset
        # 初始化边邻接矩阵
        self.weighted_adj = torch.zeros((self.num_asset, self.num_asset), dtype=torch.float32)

        # 初始化节点信息，最终的self.x形状应为(num_assets, node_features)，例如(4, 5+1+1+5=12)
        # TODO Volume值过大的问题，暂时抛弃Volume
        next_obs = torch.from_numpy(price_feature[0].reshape(self.num_asset, -1))   # (4,1,4) -> (4,4)
        obs = torch.from_numpy(price_feature[1].reshape(self.num_asset, -1))        # (4,4)
        act = np.random.rand(self.num_asset, 1)                                     # (4,1)
        e_x = np.exp(act - np.max(act))                                             # 对act进行softmax
        act = torch.from_numpy(e_x / e_x.sum(axis=0))
        rwd = torch.from_numpy(np.random.rand(self.num_asset, 1))                   # (4,1)
        x = torch.cat([obs, act, rwd, next_obs], dim=1)                             # (4,10)，即在第二个维度上合并
        # PyTorch官方的建议写法：
        self.x = x.clone().detach().type(torch.float32)

        self.gcn = GCNwithAttention(num_assets=self.num_asset, 
                                    input_dim=self.x.shape[1], 
                                    hidden_dim=self.hidden_dim, 
                                    output_dim=self.output_dim,
                                    device=self.device)

        self._update_similarity(init=True)

    # TODO 大规模矩阵处理的问题，此处暂时以初始化全部边进行计算，此处复杂度为o(n^2)
    # NOTE 憋瞎几把优化了，越优化越他妈慢。他循环就在update_similarity内部干得好好地你动他干什么，还用np.where和.copy_增加了新的内存开销
    # NOTE 在_update_similarity循环相比于在heat_kernel内循环10次的成绩，从67ms提升到了58ms
    def _update_similarity(self, init=False, embedding:torch.tensor=None):
        for i in range(0, self.num_asset):
            for j in range(i, self.num_asset):
                if i == j: continue
                
                # 基于论文中，设定λ为2
                # 论文中的初始权重就是各个节点的特征，后续会被替换为GCN的嵌入
                # 注意Heat_kernel的输入应为np.array
                if init:
                    e_i, e_j = self.x[i].numpy(), self.x[j].numpy()
                else:
                    if embedding is None:
                        raise ValueError("Embedding mustn't be none for updating.")
                    e_i = embedding[i].detach().numpy()
                    e_j = embedding[j].detach().numpy()
                
                similarity = heat_kernel(e_i, e_j, lambda_param=2)
                # 相似度低于下限时认为不存在边
                if similarity < self.threshold: continue
                
                # 无向图，双向更新
                self.weighted_adj[i, j] = similarity
                self.weighted_adj[j, i] = similarity

    def update(self, node_features: np.array):
        """
        Update the node features using the data from pm env.
        
        features: np.array of shape (state=4, action=1, reward=1, next_state=4)
        """
        # 展开、合并传入的节点特征
        obs = torch.from_numpy(node_features[0].reshape(self.num_asset, -1))        # (4,4)
        action = np.delete(node_features[1], 0)                                     # 删除掉表示现金的第一个维度
        action = torch.from_numpy(action.reshape(self.num_asset, 1))                # (4,1)
        reward = np.delete(node_features[2], 0)                                     # 删除掉表示现金的第一个维度
        reward = torch.from_numpy(reward.reshape(self.num_asset, 1))                # (4,1)
        next_obs = torch.from_numpy(node_features[3].reshape(self.num_asset, -1))   # (4,1,4) -> (4,4)
        x = torch.cat([obs, action, reward, next_obs], dim=1)                       # (4,10)
        # PyTorch官方的建议写法：
        # 与GCN的weights保持类型一致，均为float32
        self.x = x.clone().detach().type(torch.float32)

        # GCN前向传播
        z_context, _ = self.gcn(self.x, self.weighted_adj)

        # 计算MI损失
        mi_loss = self.gcn.compute_mi_loss(self.x, z_context)

        # 反向传播与优化
        optimizer = torch.optim.Adam(self.gcn.parameters(), lr=0.01)
        optimizer.zero_grad()
        mi_loss.backward()
        optimizer.step()

        # 更新TPG边权重
        self._update_similarity(embedding=z_context.cpu())

    def observe(self):
        """
        返回的观察值即GCN生成的全局嵌入向量 (维度为d/2=128)
        """
        z_context, global_context = self.gcn(self.x, self.weighted_adj)
        # TODO 此处的返回值还需要结合task embedding
        # 这些返回值给MADDPG时不再需要继续追踪梯度
        return global_context.cpu().detach().numpy()
    
    def get_pyg_object(self):
        """
        Builds and returns the graph as a PyG Data object.

        仅用于初始化整张TPG图
        
        Returns:
            data: PyG Data object representing the graph.
        """
        # Convert adjacency matrix to edge_index and edge_attr
        edge_index = torch.nonzero(self.weighted_adj, as_tuple=False).t().contiguous()
        edge_attr = self.weighted_adj[edge_index[0], edge_index[1]]
        
        # Create PyG Data object
        data = Data(x=self.x, edge_index=edge_index, edge_attr=edge_attr)
        return data


# Visualization function for NX graph or PyTorch tensor
def visualize(h, color="blue", epoch=None, loss=None, accuracy=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None and accuracy['train'] is not None and accuracy['val'] is not None:
            plt.xlabel((f'Epoch: {epoch}, Loss: {loss.item():.4f} \n'
                       f'Training Accuracy: {accuracy["train"]*100:.2f}% \n'
                       f' Validation Accuracy: {accuracy["val"]*100:.2f}%'),
                       fontsize=16)
    else:
        nx.draw_networkx(h, pos=nx.spring_layout(h, seed=42), with_labels=False,
                         node_color=color, cmap="Set2")
    plt.show()


def example_run():
    target_history, target_stocks = get_history_and_abb()
    data_gen = DataGenerator(target_history, 
                             target_stocks, 
                             trade_steps=730, 
                             window_length=1,
                             start_idx=0,
                             start_date=None)
    data_gen.reset()
    
    num_asset = len(data_gen.asset_names)
    cur_price = data_gen.observe()
    next_price = data_gen.next_observe()
    # 仅仅为了匹配mse_loss
    tpg = TemporalPortfolioGraph(output_dim=10)
    tpg.reset(num_asset, [cur_price, next_price])

    # 更新10步
    for _ in range(0, 10):
        _, truncation, _ = data_gen.step(True)
        if truncation: break
        
        cur_price = data_gen.observe()
        act = np.random.rand(num_asset+1, 1)
        e_x = np.exp(act - np.max(act))
        act = e_x / e_x.sum(axis=0)
        rwd = np.random.rand(num_asset+1, 1)
        next_price = data_gen.next_observe()
        tpg.update([cur_price, act, rwd, next_price])

    # Build the PyG Data object
    data = tpg.get_pyg_object()
    print(data)

    # 转化为networkx对象以可视化
    # G = to_networkx(data, to_undirected=True)
    # visualize(G)


if __name__ == '__main__':
    cProfile.run('example_run()', 'profiling/output.prof')