import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# 用于防止0成为除数的微小量
eps = 1e-8

class GCNWeightedLayer(nn.Module):
    """
    支持对带权边的处理
    """
    def __init__(self, in_feature, out_feature, num_assets, device) -> None:
        super(GCNWeightedLayer, self).__init__()
        self.device = device
        self.weight = nn.Parameter(torch.FloatTensor(in_feature, out_feature)).to(self.device)
        nn.init.xavier_uniform_(self.weight)

        # 使用可学习的参数矩阵来规避初始化时全0边权重导致的问题
        self.learnable_adj = nn.Parameter(torch.rand(num_assets, num_assets)).to(self.device)

    def forward(self, x, adj):
        """
        x: Node features (num_nodes, in_features)
        adj: Weighted Adjacency matrix (num_nodes, num_nodes)
        """     
        # 添加噪声，也是为减轻全0问题
        noise_adj = (adj + self.learnable_adj + eps * torch.rand_like(adj)).to(self.device)

        # 添加自循环，保证即便边全为0时也至少能将节点自身的特征传递给下一层
        looped_adj = noise_adj + torch.eye(noise_adj.size(0)).to(self.device)

        # 邻接矩阵标准化
        degree = torch.sum(looped_adj, dim=1)                                           # 对加权邻接矩阵按行求和
        degree_inv_sqrt = torch.diag(torch.pow(degree + eps, -0.5)).to(self.device)     # D^(-1/2)
        normalized_adj = degree_inv_sqrt @ looped_adj @ degree_inv_sqrt                 # B̃ = D^(-1/2) * B * D^(-1/2)

        # 图卷积操作
        support = torch.matmul(x, self.weight)
        out = torch.matmul(normalized_adj, support)     # Ã * X * W
        return out


class GCNwithAttention(nn.Module):
    """
    GCN with Attention.

    Based of [Yang 2023](https://doi.org/10.1016/j.knosys.2023.110905)
    """
    def __init__(self, num_assets, input_dim, hidden_dim, output_dim, device):
        super(GCNwithAttention, self).__init__()
        # Define 3 GCN layers
        self.conv1 = GCNWeightedLayer(input_dim, hidden_dim, num_assets, device)    
        self.conv2 = GCNWeightedLayer(hidden_dim, hidden_dim, num_assets, device)
        self.conv3 = GCNWeightedLayer(hidden_dim, output_dim, num_assets, device)
        
        # Define attention mechanism parameters
        self.q = nn.Parameter(torch.randn(output_dim, 1)).to(device)   # Attention vector
        self.W = nn.Linear(output_dim, output_dim).to(device)          # Parameter matrix
        self.b = nn.Parameter(torch.randn(1)).to(device)               # Bias term
        
        # Number of assets
        self.num_assets = num_assets
        self.device = device

    def forward(self, x, weighted_adj):
        """
        Forward pass through the GCN and attention mechanism.
        
        Args:
        - x: Input features of nodes (e.g., features of assets) [num_assets, feature_dim]
        - edge_index: Edge connections between nodes (e.g., asset correlation graph)
        """
        x = x.to(self.device)
        weighted_adj = weighted_adj.to(self.device)

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
        z_context = z_context.to(self.device)
        
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
        local_info = local_info.to(self.device)
        z_context = z_context.to(self.device)

        # Combine local information into a single vector per asset
        local_features = local_info.view(-1)  # Concatenate (s_i, a_i, r_i, s'_i)
        z_context = z_context.view(-1)
        
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
    weighted_adj = torch.rand(num_assets, num_assets)
    local_info = [torch.randn((num_assets, int(output_dim/4))) for _ in range(4)]  # Local (s_i, a_i, r_i, s'_i)

    # Forward pass
    z_context, global_context = model(x, weighted_adj)

    # Compute MI loss
    mi_loss = model.compute_mi_loss(local_info, z_context)

    # Backward pass and optimization (example)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    mi_loss.backward()
    optimizer.step()
