import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNWeightedLayer(nn.Module):
    """
    支持对带权边的处理
    """
    def __init__(self, in_feature, out_feature) -> None:
        super(GCNWeightedLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_feature, out_feature))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        """
        x: Node features (num_nodes, in_features)
        adj: Weighted Adjacency matrix (num_nodes, num_nodes)
        """     
        # 邻接矩阵标准化
        # TODO 对于一个全为0的初始化带权邻接矩阵，算法会出现问题
        degree = torch.sum(adj, dim=1)     # 对加权邻接矩阵按行求和
        degree_inv_sqrt = torch.diag(degree.pow(-0.5))  # D^(-1/2)
        normalized_adj = degree_inv_sqrt @ adj @ degree_inv_sqrt   # B̃ = D^(-1/2) * B * D^(-1/2)

        # 图卷积操作
        support = torch.matmul(x, self.weight)
        out = torch.matmul(normalized_adj, support) # Ã * X * W
        return out


class GCNwithAttention(nn.Module):
    """
    GCN with Attention.

    Based of [Yang 2023](https://doi.org/10.1016/j.knosys.2023.110905)
    """
    def __init__(self, num_assets, input_dim, hidden_dim, output_dim):
        super(GCNwithAttention, self).__init__()
        # Define 3 GCN layers
        self.conv1 = GCNWeightedLayer(input_dim, hidden_dim)
        self.conv2 = GCNWeightedLayer(hidden_dim, hidden_dim)
        self.conv3 = GCNWeightedLayer(hidden_dim, output_dim)
        
        # Define attention mechanism parameters
        self.q = nn.Parameter(torch.randn(output_dim, 1))  # Attention vector
        self.W = nn.Linear(output_dim, output_dim)  # Parameter matrix
        self.b = nn.Parameter(torch.randn(1))  # Bias term
        
        # Number of assets
        self.num_assets = num_assets

    def forward(self, x, weighted_adj):
        """
        Forward pass through the GCN and attention mechanism.
        
        Args:
        - x: Input features of nodes (e.g., features of assets) [num_assets, feature_dim]
        - edge_index: Edge connections between nodes (e.g., asset correlation graph)
        """
        # Step 1: GCN layers (Eq 10)
        x = self.conv1(x, weighted_adj)            # 第一层
        x = F.relu(x)
        x = self.conv2(x, weighted_adj)            # 第二层
        x = F.relu(x)
        z_context = self.conv3(x, weighted_adj)    # 第三层（即最终Z_context）
        
        # TODO 这里的Global_Context感觉有点不对
        # Step 2: Attention mechanism (Eq 11)
        attn_scores = self.compute_attention(z_context)
        global_context = torch.sum(attn_scores * z_context, dim=0)  # Global context embedding

        return z_context, global_context

    def compute_attention(self, z_context):
        """
        Compute attention scores for each node embedding.
        
        Args:
        - z_context: Embeddings for all nodes (assets) [num_assets, embedding_dim]
        
        Returns:
        - attn_scores: Attention weights [num_assets, 1]
        """
        # Apply attention mechanism: q^T * tanh(W * z_context + b)
        attn_scores = torch.tanh(self.W(z_context)) @ self.q + self.b
        # Normalize using softmax
        attn_weights = F.softmax(attn_scores, dim=0)
        return attn_weights

    def compute_mi_loss(self, local_info, z_context):
        """
        Compute the MI loss between local info and global context embeddings.
        
        Args:
        - local_info: Local information (s_i, a_i, r_i, s'_i) [num_assets, local_info_dim]
        - z_context: Node embeddings for all assets [num_assets, embedding_dim]
        
        Returns:
        - mi_loss: Mutual Information loss
        """
        # Combine local information into a single vector per asset
        local_features = torch.cat(local_info, dim=1)  # Concatenate (s_i, a_i, r_i, s'_i)
        
        # TODO 使用论文中公式21~25所使用的MI损失函数
        # 此处由于并未使用真正的MI损失函数，故需要通过输出层参数out_dim来使local_features和z_context的维度匹配
        # Mutual Information Loss: -MI([(s_i, a_i, r_i, s'_i), z_i])
        # We'll compute a similarity measure between local_features and z_context
        mi_loss = F.mse_loss(local_features, z_context)  # Use MSE as a proxy for MI
        return mi_loss


if __name__ == '__main__':
    # Define parameters
    num_assets = 4
    input_dim = 12    # For example, the feature vector size for each node is [s_t=5, a_t=1, r_t=1, s_t+1=5]
    hidden_dim = 512
    output_dim = 128  # Size of the final embeddings

    # Create a GCNwithAttention instance
    model = GCNwithAttention(num_assets, input_dim, hidden_dim, output_dim)

    # Example data
    # Input features (num_assets, feature_size)
    x = torch.randn((num_assets, input_dim))  # Asset features
    edge_weights = torch.rand(num_assets, num_assets)
    local_info = [torch.randn((num_assets, int(output_dim/4))) for _ in range(4)]  # Local (s_i, a_i, r_i, s'_i)

    # Forward pass
    z_context, global_context = model(x, edge_weights)

    # Compute MI loss
    mi_loss = model.compute_mi_loss(local_info, z_context)

    # Backward pass and optimization (example)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    mi_loss.backward()
    optimizer.step()
