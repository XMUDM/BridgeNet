import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean 
from torch_geometric.nn import GCNConv, Linear

class HierarchicalGNN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.atom_gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.frag_gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.liner1 = Linear(9, hidden_dim)
        self.liner2 = Linear(9, hidden_dim)
        self.proj_atom_to_frag = nn.Linear(hidden_dim, hidden_dim)
        self.proj_frag_to_atom = nn.Linear(hidden_dim, hidden_dim)

    def reset_parameters(self):
        self.atom_gcn1.reset_parameters()
        self.frag_gcn2.reset_parameters()
        self.liner1.reset_parameters()
        self.liner2.reset_parameters()
        self.proj_atom_to_frag.reset_parameters()
        self.proj_frag_to_atom.reset_parameters()

    def forward(self, data, s0, alpha_node=None, frag_edge_weight=None, mol_edge_curvature=None):
        x_atom = data.x.float()           
        edge_index_atom = data.edge_index 
        x_frag = data.frag_h              
        edge_index_frag = data.frag_edge_index
        atom2frag = data.atom2u           
        N_atom = x_atom.size(0)
        N_frag = x_frag.size(0)

        x_frag = self.liner2(x_frag.float())
        x_atom = self.liner1(x_atom) + s0

        atom_edge_weight = None
        if mol_edge_curvature is not None:
            atom_edge_weight = 1.0 - torch.sigmoid(mol_edge_curvature)

        x_atom = F.relu(self.atom_gcn1(x_atom, edge_index_atom, edge_weight=atom_edge_weight))

        x_atom_proj = self.proj_atom_to_frag(x_atom)
        
        if alpha_node is not None:
            x_atom_proj = x_atom_proj * alpha_node
        
        x_frag_update = scatter_mean(x_atom_proj, atom2frag, dim=0, dim_size=N_frag)
        x_frag = x_frag + x_frag_update

        frag_weight = None
        if frag_edge_weight is not None:
            frag_weight = 1.0 - torch.sigmoid(frag_edge_weight)

        x_frag = F.relu(self.frag_gcn2(x_frag, edge_index_frag, edge_weight=frag_weight))

        x_frag_proj = self.proj_frag_to_atom(x_frag)
        x_atom_update = x_frag_proj[atom2frag]
        
        if alpha_node is not None:
            x_atom_update = x_atom_update * alpha_node

        x_atom = x_atom + x_atom_update

        return x_atom, x_frag