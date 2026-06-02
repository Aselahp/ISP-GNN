import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, Dropout, ReLU, LayerNorm, Embedding, Parameter
import math


class ISP_GIN(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, batchnorm_dim, dropout, 
                 invariant_type='degree', num_strata=10):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.invariant_type = invariant_type
        self.num_strata = num_strata
        
        self.structure_precomputed = False
        self.triangle_pairs = None
        
        self.input_proj = Linear(input_dim, hidden_dim)
        
        self.gin_mlp = Sequential(
            Linear(2 * hidden_dim, hidden_dim),
            Dropout(dropout),
            ReLU(), 
            LayerNorm(hidden_dim),
            Linear(hidden_dim, hidden_dim), 
            Dropout(dropout),
            ReLU(), 
            LayerNorm(hidden_dim)
        )
        
        self.eps = Parameter(torch.FloatTensor(1))
        
        self.isp_mlp = Sequential(
            Linear(2 * hidden_dim + 3, hidden_dim),
            Dropout(dropout),
            ReLU(),
            LayerNorm(hidden_dim),
            Linear(hidden_dim, hidden_dim),
            Dropout(dropout),
            ReLU(),
            LayerNorm(hidden_dim)
        )
        
        self.heterogeneity_attention = Sequential(
            Linear(3, hidden_dim // 4),
            ReLU(),
            Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.stratum_embed = Embedding(num_strata + 1, hidden_dim)
        
        self.gate = Sequential(
            Linear(2 * hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.combine = Sequential(
            Linear(2 * hidden_dim, hidden_dim),
            LayerNorm(hidden_dim)
        )
        
        if invariant_type == 'learnable':
            self.invariant_mlp = Sequential(
                Linear(3, hidden_dim // 2),
                ReLU(),
                Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            self.register_buffer('beta', torch.tensor(2.0))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv_eps = 0.1 / math.sqrt(self.eps.size(0))
        nn.init.constant_(self.eps, stdv_eps)
        
        if self.invariant_type == 'learnable':
            for module in self.invariant_mlp:
                if isinstance(module, Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def set_temperature(self, beta):
        if self.invariant_type == 'learnable':
            self.beta.fill_(beta)
    
    def precompute_structure(self, A):
        if self.structure_precomputed:
            return
        
        adj = A[0].cpu()
        num_nodes = adj.shape[0]
        device = A.device
        
        adj_dict = self._build_adj_dict(adj, num_nodes)
        
        # Raw, unclamped graph invariants. These are kept for the learnable branch,
        # which normalises them, and as the source values for predefined stratification.
        degree_inv = self._compute_degree(adj_dict, num_nodes, device)
        k_core_inv = self._compute_k_core(adj_dict, num_nodes, device)
        onion_inv = self._compute_onion(adj_dict, num_nodes, device)
        
        self.register_buffer('degree_invariant', degree_inv)
        self.register_buffer('k_core_invariant', k_core_inv)
        self.register_buffer('onion_invariant', onion_inv)
        
        if self.invariant_type == 'degree':
            raw_invariant = degree_inv
        elif self.invariant_type == 'k_core':
            raw_invariant = k_core_inv
        elif self.invariant_type == 'onion':
            raw_invariant = onion_inv
        else:
            raw_invariant = degree_inv
        
        # Predefined phi: map the global invariant ranking into num_strata quantile bins
        # (Sec. 4.2). Binning is a deterministic function of the invariant value, so
        # structurally equivalent nodes share a bin and phi stays a graph invariant.
        invariant = self._quantile_bin(raw_invariant, device)
        self.register_buffer('invariant', invariant)
        
        if self.invariant_type == 'learnable':
            phi_min_val = 1
            phi_max_val = self.num_strata
        else:
            phi_min_val = 0
            phi_max_val = self.num_strata - 1
        
        self.register_buffer('phi_min', torch.tensor(phi_min_val, dtype=torch.long, device=device))
        self.register_buffer('phi_max', torch.tensor(phi_max_val, dtype=torch.long, device=device))
        
        self.triangle_pairs = self._precompute_triangles(adj_dict, num_nodes, device)
        self.structure_precomputed = True
    
    def _build_adj_dict(self, adj, num_nodes):
        adj_dict = {i: set() for i in range(num_nodes)}
        edges = (adj > 0).nonzero(as_tuple=False).tolist()
        for u, v in edges:
            if u != v:
                adj_dict[u].add(v)
                adj_dict[v].add(u)
        return adj_dict
    
    def _quantile_bin(self, values, device):
        vals = values.float()
        N = vals.shape[0]
        if N == 0:
            return torch.zeros(0, dtype=torch.long, device=device)
        
        uniq, counts = torch.unique(vals, sorted=True, return_counts=True)
        cum = torch.cumsum(counts, dim=0).float()
        cdf_lower = (cum - counts.float()) / N
        cdf_upper = cum / N
        cdf_mid = 0.5 * (cdf_lower + cdf_upper)
        
        bin_per_uniq = torch.floor(cdf_mid * self.num_strata).long()
        bin_per_uniq = bin_per_uniq.clamp(0, self.num_strata - 1)
        
        idx = torch.searchsorted(uniq, vals)
        bins = bin_per_uniq[idx]
        return bins.to(device)
    
    def _precompute_triangles(self, adj_dict, num_nodes, device):
        node_triangle_pairs = [[] for _ in range(num_nodes)]
        processed = set()
        
        for u in range(num_nodes):
            for v in adj_dict[u]:
                if (u, v) in processed or (v, u) in processed:
                    continue
                processed.add((u, v))
                
                common = adj_dict[u] & adj_dict[v]
                for w in common:
                    node_triangle_pairs[u].append((v, w))
                    node_triangle_pairs[v].append((u, w))
                    node_triangle_pairs[w].append((u, v))
        
        triangle_pairs = []
        for node in range(num_nodes):
            if node_triangle_pairs[node]:
                pairs = torch.tensor(node_triangle_pairs[node], dtype=torch.long, device=device)
                triangle_pairs.append(pairs)
            else:
                triangle_pairs.append(torch.zeros(0, 2, dtype=torch.long, device=device))
        
        return triangle_pairs
    
    def _compute_degree(self, adj_dict, num_nodes, device):
        degree = torch.zeros(num_nodes, dtype=torch.long, device=device)
        for node in range(num_nodes):
            degree[node] = len(adj_dict[node])
        return degree
    
    def _compute_k_core(self, adj_dict, num_nodes, device):
        # Core number via Batagelj-Zaversnik peeling. core[v] is the largest k such that v
        # belongs to a subgraph in which every node has degree >= k.
        if num_nodes == 0:
            return torch.zeros(0, dtype=torch.long, device=device)
        
        cur_deg = [len(adj_dict[v]) for v in range(num_nodes)]
        max_deg = max(cur_deg)
        bins = [[] for _ in range(max_deg + 1)]
        for v in range(num_nodes):
            bins[cur_deg[v]].append(v)
        
        core = [0] * num_nodes
        processed = [False] * num_nodes
        
        for d in range(max_deg + 1):
            i = 0
            while i < len(bins[d]):
                v = bins[d][i]
                i += 1
                if processed[v] or cur_deg[v] != d:
                    continue
                processed[v] = True
                core[v] = d
                for u in adj_dict[v]:
                    if not processed[u] and cur_deg[u] > d:
                        cur_deg[u] -= 1
                        bins[cur_deg[u]].append(u)
        
        return torch.tensor(core, dtype=torch.long, device=device)
    
    def _compute_onion(self, adj_dict, num_nodes, device):
        onion = torch.zeros(num_nodes, dtype=torch.long, device=device)
        if num_nodes == 0:
            return onion
        
        neighbors = {v: set(adj_dict[v]) for v in range(num_nodes)}
        degrees = {v: len(adj_dict[v]) for v in range(num_nodes)}
        od = {}
        current_core = 1
        current_layer = 1
        
        isolated = [v for v in range(num_nodes) if degrees[v] == 0]
        if isolated:
            for v in isolated:
                od[v] = current_layer
                degrees.pop(v)
            current_layer = 2
        
        while degrees:
            min_degree = min(degrees.values())
            if min_degree > current_core:
                current_core = min_degree
            this_layer = [v for v in degrees if degrees[v] <= current_core]
            for v in this_layer:
                od[v] = current_layer
                for u in neighbors[v]:
                    if u in degrees:
                        degrees[u] -= 1
                degrees.pop(v)
            current_layer += 1
        
        for v in range(num_nodes):
            onion[v] = od[v] - 1
        return onion
    
    def compute_learnable_invariant(self, batch_size):
        N = self.degree_invariant.shape[0]
        
        degree_norm = self.degree_invariant.float() / (self.degree_invariant.max() + 1e-8)
        k_core_norm = self.k_core_invariant.float() / (self.k_core_invariant.max() + 1e-8)
        onion_norm = self.onion_invariant.float() / (self.onion_invariant.max() + 1e-8)
        
        invariant_features = torch.stack([degree_norm, k_core_norm, onion_norm], dim=1)
        phi_continuous = self.invariant_mlp(invariant_features).squeeze(-1)
        phi_learn = phi_continuous * (self.num_strata - 1) + 1
        phi_learn = phi_learn.unsqueeze(0).expand(batch_size, -1)
        
        return phi_learn
    
    def compute_soft_stratum_weights(self, phi_learn):
        batch_size, N = phi_learn.shape
        device = phi_learn.device
        
        strata = torch.arange(1, self.num_strata + 1, device=device).float()
        distances = torch.abs(phi_learn.unsqueeze(-1) - strata.view(1, 1, -1))
        weights = F.softmax(-self.beta * distances, dim=-1)
        
        return weights
    
    def compute_hierarchical_heterogeneity_encoding(self, v_phi, u_phi, w_phi):
        gap_u = v_phi - u_phi
        gap_w = v_phi - w_phi
        
        delta_1 = torch.maximum(gap_u, gap_w)
        delta_2 = torch.minimum(gap_u, gap_w)
        delta_3 = torch.abs(u_phi - w_phi)
        
        return torch.stack([delta_1, delta_2, delta_3], dim=-1)
    
    def aggregate_triangle_structures(self, X_isp, invariant_values):
        batch_size, N, feat_dim = X_isp.shape
        device = X_isp.device
        
        has_color = (torch.norm(X_isp, p=2, dim=-1) > 1e-6).float()
        
        X_agg_mean = torch.zeros_like(X_isp)
        X_agg_sum = torch.zeros_like(X_isp)
        delta_agg = torch.zeros(batch_size, N, 3, device=device)
        has_triangles = torch.zeros(batch_size, N, dtype=torch.bool, device=device)
        
        for node in range(N):
            pairs = self.triangle_pairs[node]
            if len(pairs) == 0:
                continue
            
            u_idx, w_idx = pairs[:, 0], pairs[:, 1]
            
            for b in range(batch_size):
                valid = (has_color[b, u_idx] * has_color[b, w_idx]) > 0
                
                if valid.sum() == 0:
                    continue
                
                has_triangles[b, node] = True
                u_idx_valid, w_idx_valid = u_idx[valid], w_idx[valid]
                
                v_phi = invariant_values[b, node].expand(len(u_idx_valid))
                u_phi = invariant_values[b, u_idx_valid]
                w_phi = invariant_values[b, w_idx_valid]
                delta = self.compute_hierarchical_heterogeneity_encoding(v_phi, u_phi, w_phi)
                
                alpha = self.heterogeneity_attention(delta)
                
                neighbor_avg = (X_isp[b, u_idx_valid] + X_isp[b, w_idx_valid]) / 2.0
                weighted_embeddings = alpha * neighbor_avg
                
                X_agg_sum[b, node] = weighted_embeddings.sum(dim=0)
                X_agg_mean[b, node] = weighted_embeddings.mean(dim=0)
                delta_agg[b, node] = delta.mean(dim=0)
        
        X_combined = torch.cat([X_agg_mean, X_agg_sum, delta_agg], dim=-1)
        X_isp_new = self.isp_mlp(X_combined)
        
        return X_isp_new, has_triangles
    
    def forward(self, A, X):
        if not self.structure_precomputed:
            self.precompute_structure(A)
        
        batch_size, N, _ = X.shape
        device = X.device
        
        X = self.input_proj(X)
        
        # Invariant / stratum setup.
        if self.invariant_type == 'learnable':
            if self.training:
                phi_continuous = self.compute_learnable_invariant(batch_size)
                stratum_weights = self.compute_soft_stratum_weights(phi_continuous)
            else:
                phi_continuous = self.compute_learnable_invariant(batch_size)
                invariant = torch.floor(phi_continuous).long().clamp(self.phi_min.item(), self.phi_max.item())
        else:
            invariant = self.invariant.unsqueeze(0).expand(batch_size, -1)
        
        X_isp = torch.zeros(batch_size, N, self.hidden_dim, device=device)
        
        if self.invariant_type == 'learnable' and self.training:
            is_min_stratum = stratum_weights[:, :, 0].unsqueeze(-1)
            stratum_embed = self.stratum_embed(torch.ones(batch_size, N, dtype=torch.long, device=device))
            X_isp = is_min_stratum * stratum_embed
        else:
            is_min_stratum = (invariant == self.phi_min).float().unsqueeze(-1)
            stratum_embed = self.stratum_embed(invariant)
            X_isp = is_min_stratum * stratum_embed
        
        for k in range(self.num_strata):
            if self.invariant_type == 'learnable':
                phi_k = k + 1
            else:
                phi_k = min(self.phi_min.item() + k, self.phi_max.item())
            
            if self.invariant_type == 'learnable' and self.training:
                is_target_stratum = stratum_weights[:, :, k]
            else:
                is_target_stratum = (invariant == phi_k).float()
            
            has_isp_color = (torch.norm(X_isp, p=2, dim=-1) > 1e-6).float()
            gate = (is_target_stratum * (1 - has_isp_color)).unsqueeze(-1)
            
            if gate.sum() > 0:
                if self.invariant_type == 'learnable' and self.training:
                    current_phi = phi_continuous
                else:
                    current_phi = invariant.float()
                
                X_isp_new, has_triangles = self.aggregate_triangle_structures(X_isp, current_phi)
                
                X_stratum = self.stratum_embed(
                    torch.full((batch_size, N), phi_k, dtype=torch.long, device=device)
                )
                
                use_triangles = has_triangles.float().unsqueeze(-1)
                X_isp_candidate = use_triangles * X_isp_new + (1 - use_triangles) * X_stratum
                
                X_isp = gate * X_isp_candidate + (1 - gate) * X_isp
        
        X_comb = torch.cat([X, X_isp], dim=-1)
        identity = torch.eye(N, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        A_gin = (1 + self.eps) * identity + A
        X_backbone = A_gin @ X_comb
        X_backbone = self.gin_mlp(X_backbone)
        
        X_combined = torch.cat([X_backbone, X_isp], dim=-1)
        X_out = self.combine(X_combined)
        
        return X_out


def ISP_GIN_Degree(input_dim, hidden_dim, batchnorm_dim, dropout, num_strata=10):
    return ISP_GIN(input_dim, hidden_dim, batchnorm_dim, dropout, 
                   invariant_type='degree', num_strata=num_strata)

def ISP_GIN_Core(input_dim, hidden_dim, batchnorm_dim, dropout, num_strata=10):
    return ISP_GIN(input_dim, hidden_dim, batchnorm_dim, dropout, 
                   invariant_type='k_core', num_strata=num_strata)

def ISP_GIN_Onion(input_dim, hidden_dim, batchnorm_dim, dropout, num_strata=10):
    return ISP_GIN(input_dim, hidden_dim, batchnorm_dim, dropout, 
                   invariant_type='onion', num_strata=num_strata)

def ISP_GIN_Learnable(input_dim, hidden_dim, batchnorm_dim, dropout, num_strata=10):
    return ISP_GIN(input_dim, hidden_dim, batchnorm_dim, dropout, 
                   invariant_type='learnable', num_strata=num_strata)