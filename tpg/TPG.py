import torch
import numpy as np
from torch_geometric.data import Data
from tpg.GCN import GCNWithAttention
from simple_env.data_gen_sim import DataGenerator
from tpg.heat_kernel import heat_kernel
from utils.common import get_history_and_abb


def reset_check(method):
    """
    用于检查方法调用前TPG是否已经被正确地初始化
    """
    def wrapper(self, *args, **kwargs):
        if not self.is_reset:
            raise RuntimeError("TPG need to be reset first.")
        return method(self, *args, **kwargs)
    return wrapper


class TemporalPortfolioGraph:
    """
    构建一个资产之间的时序相似度图，并且依据时间戳进行更新
    
    每一个TPG应当唯一对应一个DataGenerator
    """
    def __init__(self, data_gen: DataGenerator, device='cpu'):
        self.device = device
        self.src = data_gen

        # 初始化边向量与权重
        self.edge_index = [[], []]
        self.edge_weights = []

        self.is_reset = False

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
        # Add edges in both directions for an undirected graph
        self.edge_index[0].extend([asset1, asset2])
        self.edge_index[1].extend([asset2, asset1])
        # Add similarity to edge attributes (weight of edges)
        self.edge_weights.extend([similarity, similarity])

    @reset_check
    def build_graph(self):
        """
        Builds and returns the graph as a PyG Data object.
        
        Returns:
            data: PyG Data object representing the graph.
        """
        # Add similarities between assets
        # TODO 大规模矩阵处理的问题，此处暂时以初始化全部边进行计算，此处复杂度为o(n^2)
        for i in range(0, self.num_asset):
            for j in range(i, self.num_asset):
                if i == j: continue
                # 基于论文中，设定λ为2
                weight = heat_kernel(self.x[i].numpy(), self.x[j].numpy(), lambda_param=2)
                self._add_similarity(i, j, weight)

        # Convert edge_index and edge_attr to tensors
        edge_index_tensor = torch.tensor(self.edge_index, dtype=torch.long).to(self.device)
        edge_attr_tensor = torch.tensor(self.edge_weights, dtype=torch.float).to(self.device)
        # Create PyG Data object
        data = Data(x=self.x, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor)
        return data

    @reset_check
    def update_similarity(self, features: np.array):
        """
        Update the node features using the observation from DataGenerator.
        
        features: np.array of shape (num_assets, observe_window, num_features)
        """
        # 将时序数据转化为一维数据
        flattened_observation = features.reshape(self.num_asset, -1)
        
        # 转化为张量
        self.x = torch.tensor(flattened_observation, dtype=torch.float)

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
        # 初始化节点信息
        # TODO 暂时只使用各个价格作为节点属性
        obs = self.src.observe().reshape(self.num_asset, -1)
        self.x = torch.tensor(obs, dtype=torch.float)   # shape of (num_assets, wnd_size * features)

        self.is_reset = True


if __name__ == "__main__":
    # Example usage
    target_history, target_stocks = get_history_and_abb()
    data_gen = DataGenerator(target_history, 
                             target_stocks, 
                             trade_steps=730, 
                             window_length=5,
                             start_idx=0,
                             start_date=None)
    asset_graph = TemporalPortfolioGraph(data_gen)
    
    data_gen.reset()
    asset_graph.reset()

    # Build the PyG Data object
    data = asset_graph.build_graph()

    # Now you can use 'data' for downstream tasks
    print(data)
