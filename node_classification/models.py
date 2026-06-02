import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, Dropout, ReLU, Embedding
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, DirGNNConv


class ISP_GNN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout, iterations, adj,
                 backbone='gcn', invariant_type='degree', num_strata=None):
        super().__init__()

        self.hidden_dim = nhid
        self.num_layers = iterations
        self.dropout = dropout
        self.backbone = backbone
        self.invariant_type = invariant_type
        self.num_strata = iterations if num_strata is None else num_strata

        self.register_buffer('edge_index', adj)
        num_nodes = adj.max().item() + 1
        self._precompute_structure(adj, num_nodes)

        if invariant_type == 'learnable':
            self.mlp_phi = nn.Sequential(
                nn.Linear(3, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
            self.register_buffer('beta', torch.tensor(1.0))

        self.input_embed = Linear(nfeat, nhid)

        self.backbone_convs = nn.ModuleList()
        for _ in range(iterations):
            if backbone == 'gcn':
                self.backbone_convs.append(GCNConv(nhid, nhid))
            elif backbone == 'gat':
                self.backbone_convs.append(GATConv(nhid, nhid, heads=1, concat=False))
            elif backbone == 'sage':
                self.backbone_convs.append(SAGEConv(nhid, nhid))
            elif backbone == 'gin':
                mlp = Sequential(Linear(nhid, nhid), ReLU(), Linear(nhid, nhid))
                self.backbone_convs.append(GINConv(mlp))
            elif backbone == 'dirgnn':
                self.backbone_convs.append(DirGNNConv(GCNConv(nhid, nhid)))
            else:
                raise ValueError(f"Unknown backbone: {backbone}")

        self.backbone_norms = nn.ModuleList([nn.LayerNorm(nhid) for _ in range(iterations)])

        self.mlp_struct = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3, nhid // 4),
                nn.ReLU(),
                nn.Linear(nhid // 4, 1),
                nn.Sigmoid()
            ) for _ in range(iterations)
        ])

        self.mlp_tri = nn.ModuleList([
            nn.Sequential(
                nn.Linear(nhid, nhid),
                nn.ReLU(),
                nn.Linear(nhid, nhid)
            ) for _ in range(iterations)
        ])

        self.stratum_embed = Embedding(self.num_strata + 1, nhid)

        self.output_proj = nn.Sequential(
            Linear(2 * nhid, nhid),
            nn.LayerNorm(nhid),
            nn.ReLU(),
            Dropout(dropout),
            Linear(nhid, nclass)
        )

    def _precompute_structure(self, edge_index, num_nodes):
        device = edge_index.device

        adj_dict = {i: set() for i in range(num_nodes)}
        for i in range(edge_index.size(1)):
            u = edge_index[0, i].item()
            v = edge_index[1, i].item()
            adj_dict[u].add(v)
            adj_dict[v].add(u)

        self._adj_dict = adj_dict

        degree_raw = torch.zeros(num_nodes, dtype=torch.float, device=device)
        for node in range(num_nodes):
            degree_raw[node] = len(adj_dict[node])
        self.register_buffer('_base_degree_raw', degree_raw)

        if self.invariant_type != 'learnable':
            raw = self._compute_raw_invariant(adj_dict, num_nodes, device)
            binned = self._quantile_bin(raw, self.num_strata, device)
            self.register_buffer('phi', binned)

        self._precompute_triangles(adj_dict, num_nodes, device)

    def _quantile_bin(self, values, num_strata, device):
        n = values.float().size(0)
        _, sorted_idx = torch.sort(values.float())
        ranks = torch.zeros(n, dtype=torch.long, device=device)
        ranks[sorted_idx] = torch.arange(n, device=device)
        return (ranks * num_strata // n).clamp(0, num_strata - 1) + 1

    def _compute_raw_invariant(self, adj_dict, num_nodes, device):
        if self.invariant_type == 'degree':
            return self._raw_degree(adj_dict, num_nodes, device)
        elif self.invariant_type == 'k_core':
            return self._raw_k_core(adj_dict, num_nodes, device)
        elif self.invariant_type == 'onion':
            return self._raw_onion(adj_dict, num_nodes, device)
        else:
            return self._raw_degree(adj_dict, num_nodes, device)

    def _raw_degree(self, adj_dict, num_nodes, device):
        deg = torch.zeros(num_nodes, dtype=torch.float, device=device)
        for node in range(num_nodes):
            deg[node] = len(adj_dict[node])
        return deg

    def _raw_k_core(self, adj_dict, num_nodes, device):
        k_core = torch.zeros(num_nodes, dtype=torch.float, device=device)
        edges = {(u, v) for u in adj_dict for v in adj_dict[u]}
        for k in range(num_nodes):
            degrees = {node: sum(1 for v in adj_dict[node] if (node, v) in edges or (v, node) in edges)
                       for node in range(num_nodes)}
            if max(degrees.values(), default=0) == 0:
                break
            in_core = {node for node, deg in degrees.items() if deg > k}
            for node in in_core:
                k_core[node] = k + 1
            edges = {(u, v) for u, v in edges if u in in_core and v in in_core}
        return k_core

    def _raw_onion(self, adj_dict, num_nodes, device):
        onion = torch.zeros(num_nodes, dtype=torch.float, device=device)
        edges = {(u, v) for u in adj_dict for v in adj_dict[u]}
        layer = 0
        while edges:
            degrees = {node: sum(1 for v in adj_dict[node] if (node, v) in edges or (v, node) in edges)
                       for node in range(num_nodes)}
            if max(degrees.values(), default=0) == 0:
                break
            min_deg = min(deg for deg in degrees.values() if deg > 0)
            is_layer = {node for node, deg in degrees.items() if deg == min_deg and deg > 0}
            for node in is_layer:
                onion[node] = layer
            edges = {(u, v) for u, v in edges if u not in is_layer and v not in is_layer}
            layer += 1
        return onion

    def _precompute_triangles(self, adj_dict, num_nodes, device):
        triangle_pairs = [[] for _ in range(num_nodes)]
        processed = set()
        for u in range(num_nodes):
            for v in adj_dict[u]:
                if (u, v) in processed or (v, u) in processed:
                    continue
                processed.add((u, v))
                for w in adj_dict[u] & adj_dict[v]:
                    triangle_pairs[u].append((v, w))
                    triangle_pairs[v].append((u, w))
                    triangle_pairs[w].append((u, v))
        self.triangle_pairs = [
            torch.tensor(pairs, dtype=torch.long, device=device) if pairs
            else torch.zeros(0, 2, dtype=torch.long, device=device)
            for pairs in triangle_pairs
        ]

    def _compute_learnable_phi(self, num_nodes, device):
        degree_norm = self._base_degree_raw
        degree_norm = (degree_norm - degree_norm.min()) / (degree_norm.max() - degree_norm.min() + 1e-8)

        k_core_raw = self._raw_k_core(self._adj_dict, num_nodes, device)
        k_core_norm = (k_core_raw - k_core_raw.min()) / (k_core_raw.max() - k_core_raw.min() + 1e-8)

        onion_raw = self._raw_onion(self._adj_dict, num_nodes, device)
        onion_norm = (onion_raw - onion_raw.min()) / (onion_raw.max() - onion_raw.min() + 1e-8)

        base = torch.stack([degree_norm, k_core_norm, onion_norm], dim=1)
        phi_raw = self.mlp_phi(base).squeeze(-1)
        return (self.num_strata - 1) * torch.sigmoid(phi_raw) + 1

    def _soft_membership(self, phi_learn, device):
        S = self.num_strata
        k_vals = torch.arange(1, S + 1, dtype=torch.float, device=device)
        distances = torch.abs(phi_learn.unsqueeze(1) - k_vals.unsqueeze(0))
        weights = torch.exp(-self.beta * distances)
        return weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

    def _compute_gap_encoding(self, phi_v, phi_u, phi_w):
        gap_u = phi_v - phi_u
        gap_w = phi_v - phi_w
        delta_1 = torch.maximum(gap_u, gap_w)
        delta_2 = torch.minimum(gap_u, gap_w)
        delta_3 = torch.abs(phi_u - phi_w)
        return torch.stack([delta_1, delta_2, delta_3], dim=1)

    def _aggregate_isp_layer(self, h_isp, phi_float, layer_idx, device):
        N = h_isp.size(0)
        has_color = (torch.norm(h_isp, p=2, dim=-1) > 1e-6)
        h_isp_new = torch.zeros_like(h_isp)
        has_triangles = torch.zeros(N, dtype=torch.bool, device=device)

        for v in range(N):
            pairs = self.triangle_pairs[v]
            if pairs.size(0) == 0:
                continue
            u_idx = pairs[:, 0]
            w_idx = pairs[:, 1]
            valid = has_color[u_idx] | has_color[w_idx]
            if not valid.any():
                continue
            u_idx = u_idx[valid]
            w_idx = w_idx[valid]
            phi_v = phi_float[v].expand(u_idx.size(0))
            phi_u = phi_float[u_idx]
            phi_w = phi_float[w_idx]
            delta = self._compute_gap_encoding(phi_v, phi_u, phi_w)
            alpha = self.mlp_struct[layer_idx](delta)
            h_uw = (h_isp[u_idx] + h_isp[w_idx]) / 2.0
            msg = alpha * self.mlp_tri[layer_idx](h_uw)
            h_isp_new[v] = msg.mean(dim=0)
            has_triangles[v] = True

        return h_isp_new, has_triangles

    def forward(self, x, adj):
        edge_index = self.edge_index
        N = x.size(0)
        device = x.device

        X_wl = self.input_embed(x)
        X_isp = torch.zeros(N, self.hidden_dim, device=device)
        assigned = torch.zeros(N, dtype=torch.bool, device=device)

        if self.invariant_type == 'learnable':
            phi_learn = self._compute_learnable_phi(N, device)
            phi_float = phi_learn

            if self.training:
                soft_w = self._soft_membership(phi_learn, device)

                h_wl_init = torch.zeros(N, self.hidden_dim, device=device)
                for k in range(1, self.num_strata + 1):
                    k_embed = self.stratum_embed(
                        torch.full((N,), k, dtype=torch.long, device=device)
                    )
                    h_wl_init = h_wl_init + soft_w[:, k - 1].unsqueeze(1) * k_embed
                X_isp = h_wl_init

                for layer_idx in range(self.num_layers):
                    X_wl = self.backbone_convs[layer_idx](X_wl, edge_index)
                    X_wl = F.relu(X_wl)
                    X_wl = self.backbone_norms[layer_idx](X_wl)
                    if layer_idx < self.num_layers - 1:
                        X_wl = F.dropout(X_wl, self.dropout, training=self.training)

                    h_isp_new, has_triangles = self._aggregate_isp_layer(
                        X_isp, phi_float, layer_idx, device
                    )

                    X_isp_update = torch.zeros_like(X_isp)
                    for k in range(min(layer_idx + 1, self.num_strata)):
                        X_isp_update = X_isp_update + soft_w[:, k].unsqueeze(1) * h_isp_new

                    use_tri = has_triangles.float().unsqueeze(1)
                    stratum_fallback = self.stratum_embed(
                        torch.full((N,), layer_idx + 1, dtype=torch.long, device=device)
                    )
                    X_isp = use_tri * X_isp_update + (1 - use_tri) * stratum_fallback

            else:
                phi_binned = self._quantile_bin(phi_learn, self.num_strata, device)
                X_isp = self.stratum_embed(phi_binned)

                for layer_idx in range(self.num_layers):
                    X_wl = self.backbone_convs[layer_idx](X_wl, edge_index)
                    X_wl = F.relu(X_wl)
                    X_wl = self.backbone_norms[layer_idx](X_wl)
                    if layer_idx < self.num_layers - 1:
                        X_wl = F.dropout(X_wl, self.dropout, training=self.training)

                    target = layer_idx + 1
                    gate_mask = (phi_binned == target) & (~assigned)
                    if gate_mask.any():
                        h_isp_new, has_triangles = self._aggregate_isp_layer(
                            X_isp, phi_float, layer_idx, device
                        )
                        for v in gate_mask.nonzero(as_tuple=True)[0].tolist():
                            if has_triangles[v]:
                                X_isp[v] = h_isp_new[v]
                            else:
                                X_isp[v] = self.stratum_embed(phi_binned[v])
                            assigned[v] = True

        else:
            phi_binned = self.phi
            phi_float = phi_binned.float()

            for layer_idx in range(self.num_layers):
                X_wl = self.backbone_convs[layer_idx](X_wl, edge_index)
                X_wl = F.relu(X_wl)
                X_wl = self.backbone_norms[layer_idx](X_wl)
                if layer_idx < self.num_layers - 1:
                    X_wl = F.dropout(X_wl, self.dropout, training=self.training)

                target = layer_idx + 1
                gate_mask = (phi_binned == target) & (~assigned)
                if gate_mask.any():
                    h_isp_new, has_triangles = self._aggregate_isp_layer(
                        X_isp, phi_float, layer_idx, device
                    )
                    for v in gate_mask.nonzero(as_tuple=True)[0].tolist():
                        if has_triangles[v]:
                            X_isp[v] = h_isp_new[v]
                        else:
                            X_isp[v] = self.stratum_embed(phi_binned[v])
                        assigned[v] = True

        X_combined = torch.cat([X_wl, X_isp], dim=-1)
        logits = self.output_proj(X_combined)
        return F.log_softmax(logits, dim=-1), 0


def ISP_GCN(nfeat, nhid, nclass, dropout, iterations, adj,
            invariant_type='degree', num_strata=None):
    return ISP_GNN(nfeat, nhid, nclass, dropout, iterations, adj,
                   backbone='gcn', invariant_type=invariant_type, num_strata=num_strata)


def ISP_GAT(nfeat, nhid, nclass, dropout, iterations, adj,
            invariant_type='degree', num_strata=None):
    return ISP_GNN(nfeat, nhid, nclass, dropout, iterations, adj,
                   backbone='gat', invariant_type=invariant_type, num_strata=num_strata)


def ISP_GraphSAGE(nfeat, nhid, nclass, dropout, iterations, adj,
                  invariant_type='degree', num_strata=None):
    return ISP_GNN(nfeat, nhid, nclass, dropout, iterations, adj,
                   backbone='sage', invariant_type=invariant_type, num_strata=num_strata)


def ISP_GIN(nfeat, nhid, nclass, dropout, iterations, adj,
            invariant_type='degree', num_strata=None):
    return ISP_GNN(nfeat, nhid, nclass, dropout, iterations, adj,
                   backbone='gin', invariant_type=invariant_type, num_strata=num_strata)


def ISP_DirGNN(nfeat, nhid, nclass, dropout, iterations, adj,
               invariant_type='degree', num_strata=None):
    return ISP_GNN(nfeat, nhid, nclass, dropout, iterations, adj,
                   backbone='dirgnn', invariant_type=invariant_type, num_strata=num_strata)