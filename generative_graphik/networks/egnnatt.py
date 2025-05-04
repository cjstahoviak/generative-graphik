from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import degree, softmax
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import OptTensor
from generative_graphik.utils.torch_utils import get_norm_layer

class ResWrapper(nn.Module):
    def __init__(self, module, dim_res=2):
        super(ResWrapper, self).__init__()
        self.module = module
        self.dim_res = dim_res

    def forward(self, x):
        res = x[:, :self.dim_res]
        out = self.module(x)
        return out + res

class EGNNAttLayer(MessagePassing):
    def __init__(
        self,
        non_linearity,
        channels_h,
        channels_m, 
        channels_a,
        aggr: str = 'add', 
        norm_layer: str = 'None',
        hidden_channels: int = 64,
        mlp_layers=2,
        **kwargs
    ):
        super(EGNNAttLayer, self).__init__(aggr=aggr, **kwargs)

        self.m_len = channels_m

        # Attention MLP: [h_i, h_j, edge_attr (4D), d_ij (3D)]
        self.attn_mlp = nn.Sequential(
            nn.Linear(2 * channels_h + channels_a + 3, hidden_channels),
            non_linearity,
            nn.Linear(hidden_channels, 1)
        )

        # phi_e MLP: [h_i, h_j, dist_sq (1D), edge_attr (4D)] = total dim: 2h + 5
        phi_e_layers = [
            nn.Linear(2 * channels_h + 5, hidden_channels),
            get_norm_layer(hidden_channels, layer_type=norm_layer, layer_dim="1d"),
            non_linearity
        ]
        for _ in range(mlp_layers - 2):
            phi_e_layers += [
                nn.Linear(hidden_channels, hidden_channels),
                get_norm_layer(hidden_channels, layer_type=norm_layer, layer_dim="1d"),
                non_linearity
            ]
        phi_e_layers += [
            nn.Linear(hidden_channels, channels_m),
            get_norm_layer(channels_m, layer_type=norm_layer, layer_dim="1d"),
            non_linearity
        ]
        self.phi_e = nn.Sequential(*phi_e_layers)

        # phi_x MLP
        phi_x_layers = [
            nn.Linear(channels_m, hidden_channels),
            get_norm_layer(hidden_channels, layer_type=norm_layer, layer_dim="1d"),
            non_linearity
        ]
        for _ in range(mlp_layers - 2):
            phi_x_layers += [
                nn.Linear(hidden_channels, hidden_channels),
                get_norm_layer(hidden_channels, layer_type=norm_layer, layer_dim="1d"),
                non_linearity
            ]
        phi_x_layers.append(nn.Linear(hidden_channels, 1))
        self.phi_x = nn.Sequential(*phi_x_layers)

        # phi_h MLP
        phi_h_layers = [
            nn.Linear(channels_h + channels_m, hidden_channels),
            get_norm_layer(hidden_channels, layer_type=norm_layer, layer_dim="1d"),
            non_linearity
        ]
        for _ in range(mlp_layers - 2):
            phi_h_layers += [
                nn.Linear(hidden_channels, hidden_channels),
                get_norm_layer(hidden_channels, layer_type=norm_layer, layer_dim="1d"),
                non_linearity
            ]
        phi_h_layers.append(nn.Linear(hidden_channels, channels_h))
        self.phi_h = ResWrapper(nn.Sequential(*phi_h_layers), dim_res=channels_h)

    def forward(self, x: Tensor, h: Tensor, edge_attr: Tensor, edge_index: Tensor, c: OptTensor = None) -> Tuple[Tensor, Tensor]:
        if c is None:
            c = degree(edge_index[0], x.shape[0]).unsqueeze(-1)
        return self.propagate(edge_index=edge_index, x=x, h=h, edge_attr=edge_attr, c=c)

    def message(self, x_i: Tensor, x_j: Tensor, h_i: Tensor, h_j: Tensor, edge_attr: Tensor, index: Tensor) -> Tensor:
        d_ij = x_j - x_i  # Directional vector

        # Attention uses directional vector directly
        attn_input = torch.cat([h_i, h_j, edge_attr], dim=-1)
        scores = self.attn_mlp(attn_input)
        attn_weights = custom_softmax(scores, index=index)  # shape [E, 1]

        # Use squared norm of direction for phi_e
        dist_sq = torch.sum(d_ij ** 2, dim=-1, keepdim=True)
        mh_ij = self.phi_e(torch.cat([h_i, h_j, dist_sq, edge_attr], dim=-1))

        # Directional message scaled by learned function
        mx_ij = d_ij * self.phi_x(mh_ij)

        # Apply attention weights
        mx_ij = mx_ij * attn_weights
        mh_ij = mh_ij * attn_weights

        return torch.cat((mx_ij, mh_ij), dim=-1)

    def update(self, aggr_out: Tensor, x: Tensor, h: Tensor, edge_attr: Tensor, c: Tensor) -> Tuple[Tensor, Tensor]:
        m_x, m_h = aggr_out[:, :self.m_len], aggr_out[:, self.m_len:]
        h_l1 = self.phi_h(torch.cat([h, m_h], dim=-1))
        x_l1 = x + (m_x / c)
        return x_l1, h_l1

def custom_softmax(src, index, dim=0):
    """
    Custom implementation of scatter softmax to replace torch_geometric.utils.softmax
    
    Args:
        src: Source tensor of shape [E, *] where E is the number of edges
        index: Index tensor of shape [E] that maps each edge to its target node
        dim: Dimension along which to perform softmax, default is 0
        
    Returns:
        Softmax normalized tensor of shape [E, *]
    """
    from torch_scatter import scatter_max, scatter_add
    
    # Get max value per target node for numerical stability
    max_value = scatter_max(src, index, dim=dim)[0]
    max_value = max_value[index]  # Broadcast max values back to edges
    
    # Subtract max and compute exponentials
    exp_scores = torch.exp(src - max_value)
    
    # Sum exponentials per target node
    sum_exp = scatter_add(exp_scores, index, dim=dim)
    sum_exp = sum_exp[index]  # Broadcast sums back to edges
    
    # Compute softmax
    return exp_scores / (sum_exp + 1e-12)  # Add small epsilon for numerical stability