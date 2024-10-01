import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNwithAttention(torch.nn.Module):
    """
    GCN with Attention.

    Based of [Yang 2023](https://doi.org/10.1016/j.knosys.2023.110905)
    """
    def __init__(self, num_assets, input_dim, hidden_dim, output_dim):
        super(GCNwithAttention, self).__init__()
        # Define 3 GCN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        
        # Define attention mechanism parameters
        self.q = torch.nn.Parameter(torch.randn(output_dim, 1))  # Attention vector
        self.W = torch.nn.Linear(output_dim, output_dim)  # Parameter matrix
        self.b = torch.nn.Parameter(torch.randn(1))  # Bias term
        
        # Number of assets
        self.num_assets = num_assets

    def forward(self, x, edge_index):
        """
        Forward pass through the GCN and attention mechanism.
        
        Args:
        - x: Input features of nodes (e.g., features of assets) [num_assets, feature_dim]
        - edge_index: Edge connections between nodes (e.g., asset correlation graph)
        """
        # Step 1: GCN layers (Eq 10)
        x = F.relu(self.conv1(x, edge_index))   # First GCN layer
        x = F.relu(self.conv2(x, edge_index))   # Second GCN layer
        z_context = self.conv3(x, edge_index)   # Third Layer (final embeddings Z_context)
        
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
    input_dim = 4  # For example, the feature vector size (Open/Close/High/Low)
    hidden_dim = 16
    output_dim = 16  # Size of the final embeddings

    # Create a GCNwithAttention instance
    model = GCNwithAttention(num_assets, input_dim, hidden_dim, output_dim)

    # Example data
    x = torch.randn((num_assets, input_dim))  # Asset features
    edge_index = torch.tensor([[0, 1], [1, 0]])  # Asset similarity edges (example)
    local_info = [torch.randn((num_assets, 4)) for _ in range(4)]  # Local (s_i, a_i, r_i, s'_i)

    # Forward pass
    z_context, global_context = model(x, edge_index)

    # Compute MI loss
    mi_loss = model.compute_mi_loss(local_info, z_context)

    # Backward pass and optimization (example)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    mi_loss.backward()
    optimizer.step()
