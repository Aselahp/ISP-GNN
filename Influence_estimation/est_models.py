import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import GINConv


class ISP_GIN(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes, dropout, num_layers,
                 isp_hidden_dim=64, invariant_type='degree', num_strata=None):
        super(ISP_GIN, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.num_layers = num_layers
        self.isp_hidden_dim = isp_hidden_dim
        self.num_strata = num_layers if num_strata is None else num_strata

        self.convs = nn.ModuleList()

        input_mlp = Sequential(
            Linear(in_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels)
        )
        self.convs.append(GINConv(input_mlp))

        for _ in range(num_layers - 2):
            hidden_mlp = Sequential(
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINConv(hidden_mlp))

        if num_layers > 1:
            output_mlp = Sequential(
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINConv(output_mlp))

        self.isp_wl_gnn = ISP_WL_GNN(
            in_channels, isp_hidden_dim, num_layers,
            invariant_type=invariant_type,
            num_strata=self.num_strata
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_channels + isp_hidden_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.isp_wl_gnn.reset_parameters()
        for layer in self.fusion_mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x, edge_index):
        h_gin = x
        for i, conv in enumerate(self.convs):
            h_gin = conv(h_gin, edge_index)
            if i < len(self.convs) - 1:
                h_gin = F.relu(h_gin)
                h_gin = F.dropout(h_gin, p=self.dropout, training=self.training)

        h_isp = self.isp_wl_gnn(x, edge_index)

        h_combined = torch.cat([h_gin, h_isp], dim=-1)
        out = self.fusion_mlp(h_combined)
        return out, 0


class ISP_WL_GNN(nn.Module):

    def __init__(self, in_channels, hidden_channels, num_layers, invariant_type='degree', num_strata=None):
        super(ISP_WL_GNN, self).__init__()

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.invariant_type = invariant_type
        self.num_strata = num_layers if num_strata is None else num_strata

        if invariant_type == 'learnable':
            self.mlp_phi = nn.Sequential(
                nn.Linear(3, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
            self.register_buffer('beta', torch.tensor(0.5))

        self.invariant_embedding = nn.Embedding(self.num_strata + 1, hidden_channels)

        self.wl_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            ) for _ in range(num_layers)
        ])

        self.mlp_struct = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.mlp_tri = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            ) for _ in range(num_layers)
        ])

        self.epsilon = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(num_layers)])

        self.output_proj = nn.Linear(2 * hidden_channels, hidden_channels)

    def reset_parameters(self):
        self.invariant_embedding.reset_parameters()
        if self.invariant_type == 'learnable':
            for layer in self.mlp_phi:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        for mlp in self.wl_mlps:
            for layer in mlp:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        for layer in self.mlp_struct:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for mlp in self.mlp_tri:
            for layer in mlp:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        self.output_proj.reset_parameters()

    def set_beta(self, beta_value):
        self.beta = torch.tensor(beta_value, device=self.beta.device)

    def compute_base_invariants(self, edge_index, num_nodes):
        device = edge_index.device

        degree = torch.zeros(num_nodes, dtype=torch.float, device=device)
        degree.scatter_add_(0, edge_index[1],
                            torch.ones(edge_index.size(1), dtype=torch.float, device=device))

        k_core = torch.zeros(num_nodes, dtype=torch.float, device=device)
        remaining = torch.ones(num_nodes, dtype=torch.bool, device=device)
        edge_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=device)
        for k in range(num_nodes):
            degrees = torch.zeros(num_nodes, dtype=torch.long, device=device)
            valid_edges = edge_index[:, edge_mask]
            if valid_edges.size(1) == 0:
                break
            degrees.scatter_add_(0, valid_edges[1],
                                 torch.ones(valid_edges.size(1), dtype=torch.long, device=device))
            in_core = (degrees > k) & remaining
            if in_core.sum() == 0:
                break
            k_core[in_core] = k + 1
            edge_mask = in_core[edge_index[0]] & in_core[edge_index[1]]

        onion = torch.zeros(num_nodes, dtype=torch.float, device=device)
        remaining = torch.ones(num_nodes, dtype=torch.bool, device=device)
        edge_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=device)
        layer_idx = 0
        while edge_mask.sum() > 0:
            degrees = torch.zeros(num_nodes, dtype=torch.long, device=device)
            valid_edges = edge_index[:, edge_mask]
            if valid_edges.size(1) == 0:
                break
            degrees.scatter_add_(0, valid_edges[1],
                                 torch.ones(valid_edges.size(1), dtype=torch.long, device=device))
            active_degrees = degrees[degrees > 0]
            if active_degrees.numel() == 0:
                break
            min_deg = active_degrees.min()
            is_layer = (degrees == min_deg) & (degrees > 0) & remaining
            if is_layer.sum() == 0:
                break
            onion[is_layer] = layer_idx
            remaining[is_layer] = False
            edge_mask = remaining[edge_index[0]] & remaining[edge_index[1]]
            layer_idx += 1
            if layer_idx > num_nodes:
                break

        degree_norm = (degree - degree.min()) / (degree.max() - degree.min() + 1e-8)
        k_core_norm = (k_core - k_core.min()) / (k_core.max() - k_core.min() + 1e-8)
        onion_norm = (onion - onion.min()) / (onion.max() - onion.min() + 1e-8)

        return torch.stack([degree_norm, k_core_norm, onion_norm], dim=1)

    def quantile_bin(self, values, num_strata, device):
        n = values.size(0)
        sorted_vals, sorted_idx = torch.sort(values)
        ranks = torch.zeros(n, dtype=torch.long, device=device)
        ranks[sorted_idx] = torch.arange(n, device=device)
        binned = (ranks * num_strata // n).clamp(0, num_strata - 1) + 1
        return binned

    def compute_predefined_invariant(self, edge_index, num_nodes):
        device = edge_index.device

        if self.invariant_type == 'degree':
            raw = torch.zeros(num_nodes, dtype=torch.float, device=device)
            raw.scatter_add_(0, edge_index[1],
                             torch.ones(edge_index.size(1), dtype=torch.float, device=device))

        elif self.invariant_type == 'k_core':
            raw = torch.zeros(num_nodes, dtype=torch.float, device=device)
            remaining = torch.ones(num_nodes, dtype=torch.bool, device=device)
            edge_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=device)
            for k in range(num_nodes):
                degrees = torch.zeros(num_nodes, dtype=torch.long, device=device)
                valid_edges = edge_index[:, edge_mask]
                if valid_edges.size(1) == 0:
                    break
                degrees.scatter_add_(0, valid_edges[1],
                                     torch.ones(valid_edges.size(1), dtype=torch.long, device=device))
                in_core = (degrees > k) & remaining
                if in_core.sum() == 0:
                    break
                raw[in_core] = k + 1
                edge_mask = in_core[edge_index[0]] & in_core[edge_index[1]]

        elif self.invariant_type == 'onion':
            raw = torch.zeros(num_nodes, dtype=torch.float, device=device)
            remaining = torch.ones(num_nodes, dtype=torch.bool, device=device)
            edge_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=device)
            layer_idx = 0
            while edge_mask.sum() > 0:
                degrees = torch.zeros(num_nodes, dtype=torch.long, device=device)
                valid_edges = edge_index[:, edge_mask]
                if valid_edges.size(1) == 0:
                    break
                degrees.scatter_add_(0, valid_edges[1],
                                     torch.ones(valid_edges.size(1), dtype=torch.long, device=device))
                active_degrees = degrees[degrees > 0]
                if active_degrees.numel() == 0:
                    break
                min_deg = active_degrees.min()
                is_layer = (degrees == min_deg) & (degrees > 0) & remaining
                if is_layer.sum() == 0:
                    break
                raw[is_layer] = layer_idx
                remaining[is_layer] = False
                edge_mask = remaining[edge_index[0]] & remaining[edge_index[1]]
                layer_idx += 1
                if layer_idx > num_nodes:
                    break
        else:
            raw = torch.zeros(num_nodes, dtype=torch.float, device=device)
            raw.scatter_add_(0, edge_index[1],
                             torch.ones(edge_index.size(1), dtype=torch.float, device=device))

        return self.quantile_bin(raw, self.num_strata, device)

    def compute_learnable_invariant(self, edge_index, num_nodes):
        base_invariants = self.compute_base_invariants(edge_index, num_nodes)
        phi_raw = self.mlp_phi(base_invariants).squeeze(-1)
        phi_learn = (self.num_strata - 1) * torch.sigmoid(phi_raw) + 1
        return phi_learn

    def compute_soft_membership(self, phi_learn, device):
        S = self.num_strata
        k_values = torch.arange(1, S + 1, dtype=torch.float, device=device)
        distances = torch.abs(phi_learn.unsqueeze(1) - k_values.unsqueeze(0))
        weights = torch.exp(-self.beta * distances)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        return weights

    def build_triangle_index(self, edge_index, num_nodes):
        device = edge_index.device
        adj = [[] for _ in range(num_nodes)]
        for i in range(edge_index.size(1)):
            u = edge_index[0, i].item()
            v = edge_index[1, i].item()
            adj[u].append(v)

        adj_sets = [set(neighbors) for neighbors in adj]

        triangles = [[] for _ in range(num_nodes)]
        for v in range(num_nodes):
            nbrs = list(adj_sets[v])
            for i in range(len(nbrs)):
                u = nbrs[i]
                for j in range(i + 1, len(nbrs)):
                    w = nbrs[j]
                    if w in adj_sets[u]:
                        triangles[v].append((u, w))

        return triangles

    def compute_gap_encoding(self, phi_v, phi_u, phi_w):
        gap1 = max(phi_v - phi_u, phi_v - phi_w)
        gap2 = min(phi_v - phi_u, phi_v - phi_w)
        gap3 = abs(phi_u - phi_w)
        return torch.tensor([gap1, gap2, gap3], dtype=torch.float)

    def aggregate_triangle_messages(self, v, triangles_v, h_isp, phi_vals, layer_idx, device):
        if len(triangles_v) == 0:
            return None

        messages = []
        phi_v = phi_vals[v].item() if torch.is_tensor(phi_vals[v]) else phi_vals[v]

        for (u, w) in triangles_v:
            h_u = h_isp[u]
            h_w = h_isp[w]

            phi_u = phi_vals[u].item() if torch.is_tensor(phi_vals[u]) else phi_vals[u]
            phi_w = phi_vals[w].item() if torch.is_tensor(phi_vals[w]) else phi_vals[w]

            d_vuw = self.compute_gap_encoding(phi_v, phi_u, phi_w).to(device)
            alpha = self.mlp_struct(d_vuw)

            h_uw_agg = (h_u + h_w) / 2.0
            msg = alpha * self.mlp_tri[layer_idx](h_uw_agg)
            messages.append(msg)

        if len(messages) == 0:
            return None

        return torch.stack(messages, dim=0).mean(dim=0)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        device = x.device

        triangles = self.build_triangle_index(edge_index, num_nodes)

        if self.invariant_type == 'learnable':
            phi_learn = self.compute_learnable_invariant(edge_index, num_nodes)

            if self.training:
                soft_weights = self.compute_soft_membership(phi_learn, device)

                h_wl = torch.zeros(num_nodes, self.hidden_channels, device=device)
                for k in range(1, self.num_strata + 1):
                    k_embed = self.invariant_embedding(
                        torch.full((num_nodes,), k, dtype=torch.long, device=device)
                    )
                    h_wl = h_wl + soft_weights[:, k - 1].unsqueeze(1) * k_embed

                h_isp = torch.zeros(num_nodes, self.hidden_channels, device=device)
                phi_vals = phi_learn

                for layer_idx in range(self.num_layers):
                    h_comb = torch.cat([h_wl, h_isp], dim=-1)

                    row, col = edge_index
                    h_agg = torch.zeros_like(h_comb)
                    h_agg.scatter_add_(0, col.unsqueeze(1).expand(-1, h_comb.size(1)), h_comb[row])

                    h_wl = self.wl_mlps[layer_idx](
                        (1 + self.epsilon[layer_idx]) * h_comb + h_agg
                    )

                    h_isp_new = torch.zeros(num_nodes, self.hidden_channels, device=device)
                    for v in range(num_nodes):
                        tri_msg = self.aggregate_triangle_messages(
                            v, triangles[v], h_isp, phi_vals, layer_idx, device
                        )
                        if tri_msg is not None:
                            h_isp_new[v] = tri_msg

                    h_isp = h_isp + torch.stack([
                        soft_weights[:, k].unsqueeze(1) * h_isp_new
                        for k in range(min(layer_idx + 1, self.num_strata))
                    ], dim=0).sum(dim=0)

            else:
                phi_discrete = torch.round(phi_learn).long().clamp(1, self.num_strata)
                phi_binned = self.quantile_bin(phi_learn, self.num_strata, device)

                h_wl = self.invariant_embedding(phi_binned)
                h_isp = torch.zeros(num_nodes, self.hidden_channels, device=device)
                assigned = torch.zeros(num_nodes, dtype=torch.bool, device=device)
                phi_vals = phi_learn

                for layer_idx in range(self.num_layers):
                    h_comb = torch.cat([h_wl, h_isp], dim=-1)

                    row, col = edge_index
                    h_agg = torch.zeros_like(h_comb)
                    h_agg.scatter_add_(0, col.unsqueeze(1).expand(-1, h_comb.size(1)), h_comb[row])

                    h_wl = self.wl_mlps[layer_idx](
                        (1 + self.epsilon[layer_idx]) * h_comb + h_agg
                    )

                    target_stratum = layer_idx + 1
                    gate_mask = (phi_binned == target_stratum) & (~assigned)

                    if gate_mask.any():
                        nodes_to_update = gate_mask.nonzero(as_tuple=True)[0]
                        for node in nodes_to_update:
                            node = node.item()
                            tri_msg = self.aggregate_triangle_messages(
                                node, triangles[node], h_isp, phi_vals, layer_idx, device
                            )
                            if tri_msg is not None:
                                h_isp[node] = tri_msg
                            else:
                                h_isp[node] = self.invariant_embedding(phi_binned[node])
                            assigned[node] = True

        else:
            phi_binned = self.compute_predefined_invariant(edge_index, num_nodes)
            phi_vals_float = phi_binned.float()

            h_wl = self.invariant_embedding(phi_binned)
            h_isp = torch.zeros(num_nodes, self.hidden_channels, device=device)
            assigned = torch.zeros(num_nodes, dtype=torch.bool, device=device)

            for layer_idx in range(self.num_layers):
                h_comb = torch.cat([h_wl, h_isp], dim=-1)

                row, col = edge_index
                h_agg = torch.zeros_like(h_comb)
                h_agg.scatter_add_(0, col.unsqueeze(1).expand(-1, h_comb.size(1)), h_comb[row])

                h_wl = self.wl_mlps[layer_idx](
                    (1 + self.epsilon[layer_idx]) * h_comb + h_agg
                )

                target_stratum = layer_idx + 1
                gate_mask = (phi_binned == target_stratum) & (~assigned)

                if gate_mask.any():
                    nodes_to_update = gate_mask.nonzero(as_tuple=True)[0]
                    for node in nodes_to_update:
                        node = node.item()
                        tri_msg = self.aggregate_triangle_messages(
                            node, triangles[node], h_isp, phi_vals_float, layer_idx, device
                        )
                        if tri_msg is not None:
                            h_isp[node] = tri_msg
                        else:
                            h_isp[node] = self.invariant_embedding(phi_binned[node])
                        assigned[node] = True

        h_final = torch.cat([h_wl, h_isp], dim=-1)
        output = self.output_proj(h_final)
        return output