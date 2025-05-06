from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import degree, softmax
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import OptTensor
from generative_graphik.utils.torch_utils import get_norm_layer
from torch_scatter import scatter

class DebugLinear(nn.Linear):
    def forward(self, input):
        #print(f"[phi_h] DebugLinear input shape: {input.shape}, expected: ({input.shape[0]}, {self.in_features})")
        return super().forward(input)

class ResWrapper(nn.Module):
    def __init__(self, module, dim_res=2):
        super(ResWrapper, self).__init__()
        self.module = module
        self.dim_res = dim_res

    def forward(self, x):
        res = x[:, :self.dim_res]
        #print(f"[ResWrapper] input to wrapped module: {x.shape}") 
        out = self.module(x)
        return out + res

class EGNNAttLayer(MessagePassing):
    def __init__(
        self,
        channels_h: int,
        channels_m: int,
        aggr: str = 'add',
        norm_layer: str = 'None',
        hidden_channels: int = 64,
        mlp_layers: int = 2,
        non_linearity: nn.Module = nn.SiLU(),
        **kwargs
    ):
        super().__init__(aggr=aggr, **kwargs)

        #print(f"[EGNNAttLayer] Initialized with channels_h = {channels_h}, channels_m = {channels_m}")

        self.channels_h = channels_h
        self.channels_m = channels_m
        self.hidden_channels = hidden_channels
        self.norm_layer = norm_layer
        self.mlp_layers = mlp_layers
        self.non_linearity = non_linearity

        self.phi_e = None
        self.phi_x = None
        self.phi_h = None
        self.attn_mlp = None
        self._mlps_built = False

    def _build_mlps(self, h_dim, edge_attr_dim, input_dim_phi_h, device=None):

         # Get device from inputs if not provided
        if device is None and hasattr(self, '_x'):
            device = self._x.device

        channels_a = edge_attr_dim
        #print(f"[DEBUG] channels_h = {self.channels_h}, channels_m = {self.channels_m}, input_dim_phi_h = {input_dim_phi_h}")
        self.attn_mlp = nn.Sequential(
            nn.Linear(2 * self.channels_h + channels_a, self.hidden_channels),
            self.non_linearity,
            nn.Linear(self.hidden_channels, 1)
        ).to(device)

        phi_e_layers = [
            nn.Linear(2 * self.channels_h + 1 + channels_a, self.hidden_channels),
            get_norm_layer(self.hidden_channels, layer_type=self.norm_layer),
            self.non_linearity
        ]
        for _ in range(self.mlp_layers - 2):
            phi_e_layers += [
                nn.Linear(self.hidden_channels, self.hidden_channels),
                get_norm_layer(self.hidden_channels, layer_type=self.norm_layer),
                self.non_linearity
            ]
        phi_e_layers += [
            nn.Linear(self.hidden_channels, self.channels_m),
            get_norm_layer(self.channels_m, layer_type=self.norm_layer),
            self.non_linearity
        ]
        self.phi_e = nn.Sequential(*phi_e_layers).to(device)

        phi_x_layers = [
            nn.Linear(self.channels_m + 3, self.hidden_channels),
            get_norm_layer(self.hidden_channels, layer_type=self.norm_layer),
            self.non_linearity
        ]
        for _ in range(self.mlp_layers - 2):
            phi_x_layers += [
                nn.Linear(self.hidden_channels, self.hidden_channels),
                get_norm_layer(self.hidden_channels, layer_type=self.norm_layer),
                self.non_linearity
            ]
        phi_x_layers.append(nn.Linear(self.hidden_channels, self.channels_h)) 
        self.phi_x = nn.Sequential(*phi_x_layers).to(device)

        phi_h_layers = [
            DebugLinear(input_dim_phi_h, self.hidden_channels),
            get_norm_layer(self.hidden_channels, layer_type=self.norm_layer),
            self.non_linearity
        ]
        for _ in range(self.mlp_layers - 2):
            phi_h_layers += [
                nn.Linear(self.hidden_channels, self.hidden_channels),
                get_norm_layer(self.hidden_channels, layer_type=self.norm_layer),
                self.non_linearity
            ]
        phi_h_layers.append(DebugLinear(self.hidden_channels, self.channels_h))
        self.phi_h = ResWrapper(nn.Sequential(*phi_h_layers), dim_res=self.channels_h).to(device)

        self._mlps_built = True

    def forward(self, x: Tensor, h: Tensor, edge_attr: Tensor, edge_index: Tensor, c: OptTensor = None) -> Tuple[Tensor, Tensor]:
        if not self._mlps_built:
            # Use the actual message dimension after phi_e
            h_dim = h.shape[-1]
            edge_attr_dim = edge_attr.shape[-1]
            dist_dim = 1  # dist_sq is always 1D
            phi_e_input_dim = 2 * h_dim + dist_dim + edge_attr_dim
            m_dim = self.channels_m
            self._build_mlps(h_dim, edge_attr.shape[-1], input_dim_phi_h=h_dim + m_dim, device=x.device)
        # Store extra tensors for update()
        self._x = x
        self._h = h
        self._edge_attr = edge_attr
        self._c = c

        aggr_out = self.propagate(edge_index=edge_index, x=x, h=h, edge_attr=edge_attr, c=c)
        return self.update(aggr_out, x, h, edge_attr, c)

    def message(self, x_i: Tensor, x_j: Tensor, h_i: Tensor, h_j: Tensor, edge_attr: Tensor, index: Tensor) -> Tensor:
        dist_sq = torch.sum((x_j - x_i) ** 2, dim=-1, keepdim=True)
        d_ij = edge_attr[:, 1:4]

        attn_input = torch.cat([h_i, h_j, edge_attr], dim=-1)
        scores = self.attn_mlp(attn_input)
        attn_weights = custom_softmax(scores, index=index)

        phi_e_input = torch.cat([h_i, h_j, dist_sq, edge_attr], dim=-1)
        #print(f"[DEBUG] phi_e input shape: {phi_e_input.shape}")
        mh_ij = self.phi_e(phi_e_input)
        #print(f"[DEBUG] mh_ij shape: {mh_ij.shape}, channels_m: {self.channels_m}")
        #print(f"[DEBUG] phi_e_input shape: {phi_e_input.shape}, mh_ij shape: {mh_ij.shape}, channels_m: {self.channels_m}")
        #print(f"[message] mh_ij shape: {mh_ij.shape}, expected: (?, {self.channels_m})")
        phi_x_input = torch.cat([mh_ij, d_ij], dim=-1)
        mx_ij = self.phi_x(phi_x_input)

        mx_ij = mx_ij * attn_weights
        mh_ij = mh_ij * attn_weights

        return mx_ij, mh_ij  

    def update(self, aggr_out: Tuple[Tensor, Tensor], x: Tensor, h: Tensor, edge_attr: Tensor, c: Tensor) -> Tuple[Tensor, Tensor]:
        m_x, m_h = aggr_out
        #print(f"h shape: {h.shape}, m_h shape: {m_h.shape}")
        h_input = torch.cat([h, m_h], dim=-1)
        #print(f"[update] h shape: {h.shape}, m_h shape: {m_h.shape}, expecting m_h dim = {self.channels_m}")
        #print(f"phi_h input shape: {h_input.shape}")
        h_l1 = self.phi_h(h_input)
        x_l1 = x + (m_x / c) if c is not None else x + m_x
        return x_l1, h_l1

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        # inputs is a tuple: (mx_ij, mh_ij)
        mx_ij, mh_ij = inputs
        mx = scatter(mx_ij, index, dim=0, dim_size=dim_size, reduce=self.aggr)
        mh = scatter(mh_ij, index, dim=0, dim_size=dim_size, reduce=self.aggr)
        return mx, mh

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