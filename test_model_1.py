import math, contextlib
from typing import List, Tuple
from torch_geometric.nn import Set2Set
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter_add, scatter_mean
from e3nn import o3
from e3nn.o3 import Irreps, FullyConnectedTensorProduct
from e3nn.nn import Gate, BatchNorm
from rdkit import Chem
from scipy import special, optimize
import numpy as np
from torch_geometric.nn import GlobalAttention
from hierarchical_gnn_1 import HierarchicalGNN
from ogb.graphproppred.mol_encoder import AtomEncoder

def get_device():
    return torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

def no_amp():
    if hasattr(torch, "amp") and torch.cuda.is_available():
        return torch.amp.autocast('cuda', enabled=False)
    return contextlib.nullcontext()

class AtomRef(nn.Module):
    def __init__(self, max_z=200, atom_refs=None):
        super().__init__()
        self.atom_ref = nn.Embedding(max_z, 1)
        nn.init.zeros_(self.atom_ref.weight)
        if atom_refs is not None:
            self.atom_ref.weight.data.copy_(atom_refs.view(-1, 1))
            self.atom_ref.weight.requires_grad = False 

    def forward(self, z, batch):
        per_atom_ref = self.atom_ref(z) 
        return scatter_add(per_atom_ref, batch, dim=0)

class ExpNormalSmearing(nn.Module):
    def __init__(self, cutoff=5.0, num_rbf=50, trainable=True):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.trainable = trainable
        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 5.0 / cutoff

        start_value = torch.exp(torch.scalar_tensor(-self.cutoff))
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf)

        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        dist = dist.clamp(min=1e-6, max=self.cutoff)
        self.betas.data = self.betas.data.clamp(min=1e-6)

        return self.cutoff_fn(dist) * torch.exp(
            -self.betas * (torch.exp(self.alpha * (-dist)) - self.means) ** 2
        )

class ViSNetRadial(nn.Module):
    def __init__(self, L: int, N: int, r_cut: float, hidden_dim: int = 128):
        super().__init__()
        self.L = L
        self.N = N 
        self.r_cut = r_cut
        self.smearing = ExpNormalSmearing(cutoff=r_cut, num_rbf=N, trainable=True)
        self.radial_mlps = nn.ModuleDict({
            str(l): nn.Sequential(
                nn.Linear(N, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1) 
            ) for l in range(L + 1)
        })

    def forward(self, d: torch.Tensor) -> List[torch.Tensor]:
        rbf_feat = self.smearing(d)
        out = []
        for l in range(self.L + 1):
            raw_coeff = self.radial_mlps[str(l)](rbf_feat)
            out.append(raw_coeff)
        return out
    
class CosineCutoff(nn.Module):
    def __init__(self, cutoff):
        super(CosineCutoff, self).__init__()
        self.cutoff = cutoff

    def forward(self, distances):
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).float()
        return cutoffs

class NeighborEmbedding(nn.Module):
    def __init__(self, hidden_channels, num_rbf, cutoff, max_z=100):
        super(NeighborEmbedding, self).__init__()
        self.embedding = nn.Embedding(max_z, hidden_channels)
        self.distance_proj = nn.Linear(num_rbf, hidden_channels)
        self.combine = nn.Linear(hidden_channels * 2, hidden_channels)
        self.cutoff = CosineCutoff(cutoff)

    def forward(self, z, x, edge_index, edge_weight, edge_attr):
        row, col = edge_index
        C = self.cutoff(edge_weight)
        W = self.distance_proj(edge_attr) * C.view(-1, 1)
        x_neighbors = self.embedding(z) 
        out_neighbors = x_neighbors[col] * W
        aggr_out = scatter_add(out_neighbors, row, dim=0, dim_size=x.size(0))
        out = self.combine(torch.cat([x, aggr_out], dim=1))
        return out

def _hat(v: Tensor) -> Tensor:
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    O = torch.zeros_like(x)
    return torch.stack([
        torch.stack([ O, -z,  y], dim=-1),
        torch.stack([ z,  O, -x], dim=-1),
        torch.stack([-y,  x,  O], dim=-1)], dim=-2)

def so3_exp(omega: Tensor, step: float = 1.0) -> Tensor:
    th = omega.norm(dim=-1, keepdim=True).clamp(min=1e-12)        
    axis = omega / th                                             
    K = _hat(axis)                                                
    th = th * step                                                
    I = torch.eye(3, device=omega.device, dtype=omega.dtype)
    return I + torch.sin(th)[..., None] * K + (1 - torch.cos(th))[..., None] * (K @ K)

def reorthonormalize(R: Tensor) -> Tensor:
    x = F.normalize(R[..., :, 0], dim=-1)
    y = R[..., :, 1] - (x * R[..., :, 1]).sum(-1, keepdim=True) * x
    y = F.normalize(y, dim=-1)
    z = torch.cross(x, y, dim=-1)
    return torch.stack([x, y, z], dim=-1)

class EdgeEncoder(nn.Module):
    def __init__(self, L: int, radial: ViSNetRadial):
        super().__init__()
        self.L = L
        self.radial = radial
        self.irreps = o3.Irreps.spherical_harmonics(L)

    def forward(self, rhat: Tensor, d: Tensor) -> Tensor:
        Y = o3.spherical_harmonics(list(range(self.L + 1)), rhat,
                                   normalize=False, normalization='component')
        coeffs = self.radial(d)
        blocks, idx = [], 0
        for l in range(self.L + 1):
            dim = 2 * l + 1
            blk = Y[:, idx: idx + dim] * coeffs[l]
            blocks.append(blk)
            idx += dim
        eattr = torch.cat(blocks, dim=-1)
        return torch.nan_to_num(eattr, nan=0.0, posinf=0.0, neginf=0.0)

class EquivariantLayer(nn.Module):
    def __init__(self, irreps_node: Irreps, irreps_edge: Irreps,
                 mul0: int, mul1: int, mul2: int):
        super().__init__()
        self.irreps_node = irreps_node
        self.irreps_edge = irreps_edge
        self.mul0 = mul0
        self.mul1 = mul1
        self.mul2 = mul2

        self.tp = FullyConnectedTensorProduct(irreps_node, irreps_edge, irreps_node)
        self.lin_self = o3.Linear(irreps_node, irreps_node)

        irreps_scalars = Irreps(f"{mul0}x0e")
        irreps_gates   = Irreps(f"{mul1 + mul2}x0e")
        
        gated_list = []
        if mul1 > 0: gated_list.append(f"{mul1}x1o")
        if mul2 > 0: gated_list.append(f"{mul2}x2e")
        irreps_gated = Irreps("+".join(gated_list))
        
        self.pre = o3.Linear(irreps_node, irreps_scalars + irreps_gates + irreps_gated)
        self.gate = Gate(irreps_scalars, [F.silu],
                         irreps_gates,   [torch.sigmoid],
                         irreps_gated)
        self.post = o3.Linear(self.gate.irreps_out, irreps_node)
        self.layernorm = nn.LayerNorm(mul0)

        geom_channels = 0
        if mul1 > 0: geom_channels += mul1
        if mul2 > 0: geom_channels += mul2

        if geom_channels > 0:
            self.geom2s_proj = nn.Sequential(
                nn.Linear(geom_channels, geom_channels),
                nn.SiLU(),
                nn.Linear(geom_channels, mul0)
            )
        else:
            self.geom2s_proj = None

    def forward(self, x: Tensor, src: Tensor, dst: Tensor, eattr: Tensor, alpha: Tensor):
        m_ij = self.tp(x[src], eattr) * alpha

        m = scatter_add(m_ij, dst, dim=0, dim_size=x.size(0))
        
        x = self.lin_self(x) + m
        x = self.pre(x)
        x = self.gate(x)
        x = self.post(x)
        
        s = x[:, :self.mul0]
        v = x[:, self.mul0:]
        
        s = self.layernorm(s)
        
        x = torch.cat([s, v], dim=-1)

        if self.geom2s_proj is not None:
            s = x[:, :self.mul0]
            
            geom_norms = []

            if self.mul1 > 0:
                start = self.mul0
                end = self.mul0 + self.mul1 * 3
                v_flat = x[:, start : end]
                
                v_vec = v_flat.view(-1, self.mul1, 3)
                v_norm = torch.norm(v_vec, dim=-1) 
                geom_norms.append(v_norm)

            if self.mul2 > 0:
                start = self.mul0 + self.mul1 * 3
                end = start + self.mul2 * 5
                q_flat = x[:, start : end]
                
                q_vec = q_flat.view(-1, self.mul2, 5)
                q_norm = torch.norm(q_vec, dim=-1) 
                geom_norms.append(q_norm)

            geom_input = torch.cat(geom_norms, dim=-1)
            
            delta_s = self.geom2s_proj(geom_input)
            
            s_mixed = s + delta_s  
            
            x_rest = x[:, self.mul0:]
            x = torch.cat([s_mixed, x_rest], dim=-1)

        return x, m_ij

class ProductManifoldGNN(nn.Module):
    def __init__(
        self,
        L=2,
        Nrad=8,
        r_cut=3.5,
        n_layers=4,
        n_hire_layers=1, 
        mul_l0=32,
        mul_l1=16,
        mul_l2=0,
        so3_step=0.2,
        num_z_embeddings=100,
        hidden_dim=128,
        task_num=12,
        atom_refs=None,
        target_name='mu',
    ):
        super().__init__()

        self.L = L
        self.Nrad = Nrad
        self.r_cut = r_cut
        self.n_layers = n_layers
        self.n_hire_layers = n_hire_layers 
        self.mul_l0 = mul_l0
        self.mul_l1 = mul_l1
        self.mul_l2 = mul_l2
        self.so3_step = so3_step
        self.target_name = target_name 
        self.task_num = task_num
        self.hidden_dim = hidden_dim 

        irreps_list = []
        if mul_l0 > 0: irreps_list.append(f"{mul_l0}x0e")
        if mul_l1 > 0: irreps_list.append(f"{mul_l1}x1o")
        if mul_l2 > 0: irreps_list.append(f"{mul_l2}x2e")
        
        self.irreps_node = Irreps("+".join(irreps_list))
        self.irreps_edge = o3.Irreps.spherical_harmonics(L)
        
        self.l1_slices: List[Tuple[int, slice]] = []
        for mir, sl in zip(self.irreps_node, self.irreps_node.slices()):
            if mir.ir.l == 1:
                self.l1_slices.append((mir.mul, sl))
        if not self.l1_slices:
            raise RuntimeError("No ℓ=1 channels present; cannot drive SO(3) updates.")

        self.atom_emb = AtomEncoder(mul_l0)
        self.neighbor_emb = NeighborEmbedding(
            hidden_channels=mul_l0,
            num_rbf=Nrad,  
            cutoff=r_cut,
            max_z=100
        )
        
        self.radial = ViSNetRadial(L, Nrad, r_cut, hidden_dim=256)
        self.edge_enc = EdgeEncoder(L, self.radial)

        self.layers = nn.ModuleList([
            EquivariantLayer(self.irreps_node, self.irreps_edge,
                             mul_l0, mul_l1, mul_l2)
            for _ in range(n_layers)
        ])

        self.hire_layer = HierarchicalGNN(self.hidden_dim)
        
        self.hire_proj = nn.Linear(self.hidden_dim, self.mul_l0)
        self.gate_proj = nn.Linear(self.mul_l0, self.mul_l0)

        gate_in_dim = 5
        self.alpha_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 1)
        )

        self.node_feat_dim = self.mul_l0 + self.hidden_dim + self.hidden_dim

        self.att_pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(self.node_feat_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
        )

        readout_in = self.node_feat_dim * 3 
        
        self.readout = nn.Sequential(
            nn.Linear(readout_in, 128), nn.SiLU(),
            nn.LayerNorm(128), 
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, self.task_num)
        )


        self.prior_model = None
        if atom_refs is not None:
            self.prior_model = AtomRef(max_z=100, atom_refs=atom_refs)
        
    def reset_parameters(self):
        if hasattr(self.atom_emb, "atom_embedding_list"):
            for emb in self.atom_emb.atom_embedding_list:
                nn.init.xavier_uniform_(emb.weight.data)

        for l in range(self.L + 1):
            mlp = self.radial.radial_mlps[str(l)]
            for m in mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                        
        for layer in self.layers:
            if hasattr(layer.tp, "reset_parameters"):
                layer.tp.reset_parameters()
            if hasattr(layer.lin_self, "reset_parameters"):
                layer.lin_self.reset_parameters()
            if hasattr(layer.pre, "reset_parameters"):
                layer.pre.reset_parameters()
            if hasattr(layer.post, "reset_parameters"):
                layer.post.reset_parameters()
            if hasattr(layer, "layernorm") and hasattr(layer.layernorm, "reset_parameters"):
                layer.layernorm.reset_parameters()
            if hasattr(layer, 'v2s_proj') and layer.v2s_proj is not None:
                for m in layer.v2s_proj:
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

        self.hire_layer.reset_parameters()
        nn.init.xavier_uniform_(self.hire_proj.weight)
        nn.init.zeros_(self.hire_proj.bias)
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)
        
        self.att_pool.reset_parameters()

        def init_mlp(m):
            for module in m.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                    if module.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                        bound = 1 / math.sqrt(fan_in)
                        nn.init.uniform_(module.bias, -bound, bound)

        init_mlp(self.alpha_mlp)
        init_mlp(self.readout)


    def _init_latents(self, n: int, dev):
        with no_amp():
            R = torch.eye(3, device=dev).repeat(n, 1, 1)
        return R

    def forward(self, data):
        apos = data.coords if hasattr(data, 'coords') else data.pos
        x = data.x
        batch = data.batch
        
        atom2u = data.atom2u
        cliques = data.frag_cliques

        if hasattr(data, 'cutoff_edge_index') and data.cutoff_edge_index is not None:
            calc_edge_index = data.cutoff_edge_index
        else:
            calc_edge_index = data.edge_index

        num_nodes = x.size(0)
        mask_src = (calc_edge_index[0] < num_nodes) & (calc_edge_index[0] >= 0)
        mask_dst = (calc_edge_index[1] < num_nodes) & (calc_edge_index[1] >= 0)
        mask_valid = mask_src & mask_dst

        if not mask_valid.all():
            num_bad = (~mask_valid).sum().item()
            calc_edge_index = calc_edge_index[:, mask_valid]

        dev = apos.device
        
        scalars0 = self.atom_emb.atom_embedding_list[0](x[:, 0].long()) 

        src, dst = calc_edge_index
        rij = apos[src] - apos[dst]
        dij = rij.norm(dim=-1)

        rbf_feat = self.radial.smearing(dij) 
        
        z = x[:, 0].long()
        scalars0 = self.neighbor_emb(z, scalars0, calc_edge_index, dij, rbf_feat)

        zeros = apos.new_zeros(
            scalars0.size(0),
            self.mul_l1 * 3 + self.mul_l2 * 5
        )
        x_all = torch.cat([scalars0, zeros], dim=-1)

        eps = 1e-8
        rhat = rij / (dij[:, None] + eps) 
        zmask = dij <= eps
        if zmask.any():
            rhat[zmask] = 0.0
            rhat[zmask, 0] = 1.0

        eattr = self.edge_enc(rhat, dij.clamp(min=0.0))

        if hasattr(data, 'geo_cos_phi') and data.geo_cos_phi is not None:
            cos_phi = data.geo_cos_phi
            sin2_phi = data.geo_sin2_phi
            if not mask_valid.all():
                if cos_phi.size(0) == mask_valid.size(0):
                    cos_phi = cos_phi[mask_valid]
                    sin2_phi = sin2_phi[mask_valid]
                else:
                    cos_phi = torch.zeros(dij.size(0), 1, device=dev)
                    sin2_phi = torch.zeros(dij.size(0), 1, device=dev)
        else:
            raise RuntimeError("Missing geometric features. Please re-run preprocess.")
        
        size_u = data.frag_sizes.to(dev)
        
        c0 = self.radial(dij)[0] 

        if hasattr(data, 'cutoff_curvature'):
            cutoff_curv = data.cutoff_curvature
        elif hasattr(data, 'edge_curvature'):
            cutoff_curv = data.edge_curvature
        else:
            cutoff_curv = torch.zeros(dij.size(0), device=dev)

        if not mask_valid.all():
            if cutoff_curv.size(0) == mask_valid.size(0):
                cutoff_curv = cutoff_curv[mask_valid]

        mol_curv = getattr(data, 'mol_curvature', None)

        R_i = self._init_latents(x_all.size(0), dev)

        hire_x = None 
        for layer in self.layers:
            curv_term = 1.0 - torch.sigmoid(cutoff_curv.unsqueeze(-1))
            size_dst = size_u[atom2u[dst]].unsqueeze(-1).to(x.dtype)
            
            alpha_in = torch.cat([c0, cos_phi, sin2_phi, size_dst, curv_term], dim=-1)
            
            alpha = torch.sigmoid(self.alpha_mlp(alpha_in))
            alpha = torch.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)

            ex, m_ij = layer(x_all, src, dst, eattr, alpha)

            s0_hat = ex[:, :self.mul_l0]

            alpha_node = scatter_add(alpha, dst, dim=0, dim_size=x_all.size(0))
            frag_weight = getattr(data, 'frag_edge_weight', None)

            hire_x, x_f = self.hire_layer(data, s0_hat, alpha_node=alpha_node, frag_edge_weight=frag_weight, mol_edge_curvature=mol_curv)
            
            if hire_x is not None:
                hire_x_proj = self.hire_proj(hire_x) 
                gate = torch.sigmoid(self.gate_proj(s0_hat))
                s0_hat = s0_hat + gate * hire_x_proj
                
                ex = torch.cat([s0_hat, ex[:, self.mul_l0:]], dim=-1)
            
            x_all = ex 
            
            node_msgs = scatter_add(m_ij, dst, dim=0, dim_size=x.size(0))
            M_R = torch.zeros(x.size(0), 3, device=x.device, dtype=node_msgs.dtype)
            for mul, sl in self.l1_slices:
                chunk = node_msgs[:, sl].view(-1, mul, 3).sum(dim=1)
                M_R = M_R + chunk

            R_i = torch.einsum("nij,njk->nik", R_i, so3_exp(M_R, step=self.so3_step))
            R_i = reorthonormalize(R_i)

        s0 = ex[:, :self.mul_l0]

        if hire_x is None:
            hire_x = torch.zeros(x.size(0), self.hidden_dim, device=dev)
        
        hire_x_proj = self.hire_proj(hire_x)

        if x_f is not None and atom2u is not None:
            frag_feat_atom = x_f[atom2u]
        else:
            frag_feat_atom = torch.zeros(x.size(0), self.hidden_dim, device=dev)

        node_features = torch.cat([s0, hire_x, frag_feat_atom], dim=-1)

        batch_size = int(batch.max().item()) + 1
        graph_sum = scatter_add(node_features, batch, dim=0, dim_size=batch_size)
        graph_mean = scatter_mean(node_features, batch, dim=0, dim_size=batch_size)
        graph_att = self.att_pool(node_features, batch)
        
        parts = torch.cat([graph_sum, graph_mean, graph_att], dim=-1)
        
        res_pred = self.readout(parts)

        if self.prior_model is not None:
            z = data.x[:, 0].long()
            base_energy = self.prior_model(z, batch)
            y_pred = res_pred + base_energy
        else:
            y_pred = res_pred

        return y_pred, s0, hire_x_proj