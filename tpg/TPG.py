import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from tpg.GCN import GCNwithAttention
from simple_env.data_gen_sim import DataGenerator
from tpg.heat_kernel import heat_kernel
from utils.common import get_history_and_abb

import networkx as nx
import matplotlib.pyplot as plt


def reset_check(method):
    """
    用于检查方法调用前TPG是否已经被正确地初始化
    """
    def wrapper(self, *args, **kwargs):
        if not self.is_reset:
            raise RuntimeError("TPG need to be reset first.")
        return method(self, *args, **kwargs)
    return wrapper


# TODO TPG与DG的脱轨处理，TPG应该可以只依赖PM_Env传递的信息进行初始化、更新
class TemporalPortfolioGraph:
    """
    构建一个资产之间的时序相似度图，并且依据时间戳进行更新
    
    每一个TPG应当唯一对应一个DataGenerator
    """
    def __init__(self,
                 data_gen: DataGenerator,
                 input_dim=12,
                 output_dim=128, 
                 hidden_dim=128,
                 device='cpu'):
        """
        Args:
        - input_dim: [state=5, action=1, reward=1, next_state=5]
        - output_dim: the dimension of e_i
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.src = data_gen
        self.device = device

        # 初始化边邻接矩阵
        self.weighted_adj = None

        self.is_reset = False
    
    def reset(self):
        """
        重置图相关数据
        """
        # 利用src进行初始化的变量需要严格等待DataGenerator.reset()后再进行
        if not self.src.is_reset:
            try:
                self.src.reset()
                raise RuntimeWarning("The reset of DataGenerator should be conducted outside TPG.")
            except Exception as e:
                raise RuntimeError(f"{e}: TPG can only be reset after DataGenerator is reset!")
        
        self.num_asset = self.src.data.shape[0]
        self.weighted_adj = torch.zeros((self.num_asset, self.num_asset), dtype=torch.float).to(self.device)

        # TODO 处理输入维度算法
        self.gcn = GCNwithAttention(self.num_asset, 
                                    self.input_dim, 
                                    self.hidden_dim, 
                                    self.output_dim)

        # 初始化节点信息，最终的self.x形状应为(num_assets, node_features)，例如(4, 5+1+1+5=12)
        # TODO Volume值过大的问题
        next_obs = torch.from_numpy(self.src.observe().reshape(self.num_asset, -1)) # (4,1,4) -> (4,4)
        obs = torch.zeros(next_obs.shape)                                           # (4,4)
        action = torch.zeros((self.num_asset,1))                                    # (4,1)
        action[0] = 1                                                               # 初始化动作值默认选择第一个资产100%
        reward = torch.zeros((self.num_asset,1))                                    # (4,1)
        x = torch.cat([obs, action, reward, next_obs], dim=1)                       # (4,10)，即在第二个维度上合并
        # self.x = torch.tensor(x, dtype=torch.float)
        # PyTorch官方的建议写法：
        self.x = x.clone().detach()

        self.is_reset = True

    # TODO 这个函数没啥必要单独设出来了
    @reset_check
    def _add_similarity(self, asset1, asset2, similarity):
        """
        Adds a similarity between two assets (asset1 and asset2) into the graph.
        
        应当只在初始化图的时候使用
        
        Args:
            asset1: Index of the first asset (node).
            asset2: Index of the second asset (node).
            similarity: Similarity value between asset1 and asset2.
        """
        # Add edges and weights in both directions for an undirected graph
        self.weighted_adj[asset1, asset2] = similarity
        self.weighted_adj[asset2, asset1] = similarity

    @reset_check
    def build_graph(self):
        """
        Builds and returns the graph as a PyG Data object.

        仅用于初始化整张TPG图
        
        Returns:
            data: PyG Data object representing the graph.
        """
        # Add similarities between assets
        # TODO 大规模矩阵处理的问题，此处暂时以初始化全部边进行计算，此处复杂度为o(n^2)
        for i in range(0, self.num_asset):
            for j in range(i, self.num_asset):
                if i == j: continue
                # 基于论文中，设定λ为2
                # 论文中的初始权重就是各个节点的特征，后续会被替换为GCN的嵌入
                # 注意Heat_kernel的输入应为np.array
                similarity = heat_kernel(self.x[i].numpy(), self.x[j].numpy(), lambda_param=2)
                self._add_similarity(i, j, similarity)

        # Convert adjacency matrix to edge_index and edge_attr
        edge_index = torch.nonzero(self.weighted_adj, as_tuple=False).t().contiguous()
        edge_attr = self.weighted_adj[edge_index[0], edge_index[1]]
        
        # Create PyG Data object
        data = Data(x=self.x, edge_index=edge_index, edge_attr=edge_attr)
        return data

    @reset_check
    def update_similarity(self, features: np.array):
        """
        Update the node features using the data from pm env.
        
        features: np.array of shape (num_assets, observe_window, num_features)
        """
        # 将时序数据转化为一维数据
        flattened_observation = features.reshape(self.num_asset, -1)
        
        # 转化为张量
        # self.x = torch.tensor(flattened_observation, dtype=torch.float)

    @reset_check
    def observe(self):
        """
        返回的观察值即GCN生成的嵌入向量 (维度为d/2=128)
        """
        z_context, global_context = self.gcn(self.x, self.weighted_adj)
        # TODO 此处的返回值还需要结合task embedding
        return global_context


# Visualization function for NX graph or PyTorch tensor
def visualize(h, color=None, epoch=None, loss=None, accuracy=None):
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


if __name__ == "__main__":
    # Example usage
    target_history, target_stocks = get_history_and_abb()
    data_gen = DataGenerator(target_history, 
                             target_stocks, 
                             trade_steps=730, 
                             window_length=1,
                             start_idx=0,
                             start_date=None)
    asset_graph = TemporalPortfolioGraph(data_gen)
    
    data_gen.reset()
    asset_graph.reset()

    # Build the PyG Data object
    data = asset_graph.build_graph()

    # 转化为networkx对象以可视化
    G = to_networkx(data, to_undirected=True)
    visualize(G)
