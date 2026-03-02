from utils import *

import os
import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.utils import to_networkx
from typing import Callable, List, Optional
from tqdm import tqdm
import argparse
import time
import numpy as np
import networkx as nx
from torch_geometric.utils import to_undirected
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.datasets import ZINC
from torch_geometric.data import InMemoryDataset
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download
from torch_geometric.data import Data, download_url, extract_zip
from torch_geometric.nn import radius_graph
import pickle
import hashlib
import os.path as osp
import shutil
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict
from mol import smiles2graph
from sklearn.utils import shuffle
import sys
from torch_scatter import scatter
from ogb.utils.features import (allowable_features, atom_to_feature_vector,
 bond_to_feature_vector, atom_feature_vector_to_dict, bond_feature_vector_to_dict)


HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])


def compute_forman_curvature(edge_index, num_nodes):
    row, col = edge_index
    device = edge_index.device
    deg = torch.zeros(num_nodes, dtype=torch.float, device=device)
    deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
    row_cpu = row.cpu().tolist()
    col_cpu = col.cpu().tolist()
    adj_list = [set() for _ in range(num_nodes)]
    for i, j in zip(row_cpu, col_cpu):
        adj_list[i].add(j)
    num_edges = edge_index.size(1)
    deg_cpu = deg.cpu()
    curvature_cpu = torch.zeros(num_edges, dtype=torch.float)
    for k in range(num_edges):
        u = row_cpu[k]
        v = col_cpu[k]
        d_u = deg_cpu[u]
        d_v = deg_cpu[v]
        common_neighbors = len(adj_list[u].intersection(adj_list[v]))
        curv = 4 - d_u - d_v + 3 * common_neighbors
        curvature_cpu[k] = curv
    return curvature_cpu.to(device)

def dihedral_angle(p0, p1, p2, p3):
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    b1n = b1 / (torch.norm(b1) + 1e-8)

    v = b0 - torch.dot(b0, b1n) * b1n
    w = b2 - torch.dot(b2, b1n) * b1n

    x = torch.dot(v, w)
    y = torch.dot(torch.cross(b1n, v), w)
    
    return torch.atan2(y, x)

def calculate_torsion_features(pos, edge_index):
    src_list, dst_list = edge_index[0], edge_index[1]
    num_edges = src_list.size(0)
    num_nodes = pos.size(0)

    adj = [[] for _ in range(num_nodes)]
    for i in range(num_edges):
        u, v = src_list[i].item(), dst_list[i].item()
        adj[u].append(v)

    cos_phi = torch.zeros(num_edges, 1, dtype=torch.float32)
    sin2_phi = torch.zeros(num_edges, 1, dtype=torch.float32)

    for e in range(num_edges):
        j = src_list[e].item()
        i = dst_list[e].item()

        k_idx = -1
        for neighbor in adj[i]:
            if neighbor != j:
                k_idx = neighbor
                break
        
        l_idx = -1
        for neighbor in adj[j]:
            if neighbor != i:
                l_idx = neighbor
                break
        
        if k_idx != -1 and l_idx != -1:
            angle = dihedral_angle(pos[k_idx], pos[i], pos[j], pos[l_idx])
            cos_phi[e] = torch.cos(angle)
            sin_val = torch.sin(angle)
            sin2_phi[e] = sin_val * sin_val
    
    return cos_phi, sin2_phi

def process_cliques(cliques, n_atoms):
    assigned_atoms = set()
    processed = []
    for c in cliques:
        new_c = [atom for atom in c if atom not in assigned_atoms]
        if new_c:
            processed.append(new_c)
            assigned_atoms.update(new_c)
    unassigned = [atom for atom in range(n_atoms) if atom not in assigned_atoms]
    for atom in unassigned:
        processed.append([atom])
    return processed

class FragData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'frag_edge_index':
            return self.num_frags
        if key == 'atom2u':
            return self.num_frags
        if key == 'cutoff_edge_index':
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

class Qm9dataset(InMemoryDataset):
    raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
               'molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    processed_url = 'https://data.pyg.org/datasets/qm9_v3.zip'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.task_type = "mse_regression"
        self.num_tasks = 1
        self.eval_metric = 'mae'
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
        target_idx = 0 
        
        if self.data.y.dim() > 1 and self.data.y.size(1) > 1:
            print(f"Selecting target index {target_idx} from {self.data.y.size(1)} tasks for QM9.")
            self.data.y = self.data.y[:, target_idx:target_idx+1]

    @property
    def raw_file_names(self) -> List[str]:
        try:
            import rdkit  # noqa
            return ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']
        except ImportError:
            return ['qm9_v3.pt']

    @property
    def processed_file_names(self) -> str:
        return 'data_v3.pt'

    def download(self):
        try:
            import rdkit  # noqa
            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

            file_path = download_url(self.raw_url2, self.raw_dir)
            os.rename(osp.join(self.raw_dir, '3195404'),
                      osp.join(self.raw_dir, 'uncharacterized.txt'))
        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

    def process(self):
        print(f"Processing QM9 dataset in {self.root}...")
        try:
            import rdkit
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit.Chem.rdchem import HybridizationType
            RDLogger.DisableLog('rdApp.*')

        except ImportError:
            rdkit = None

        if rdkit is None:
            print(("Using a pre-processed version of the dataset. Please "
                   "install 'rdkit' to alternatively process the raw data."),
                  file=sys.stderr)
            data_list = torch.load(self.raw_paths[0])
            data_list = [Data(**data_dict) for data_dict in data_list]
            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]
            torch.save(self.collate(data_list), self.processed_paths[0])
            return

        with open(self.raw_paths[1], 'r') as f:
            target = f.read().split('\n')[1:-1]
            target = [[float(x) for x in line.split(',')[1:20]]
                      for line in target]
            target = torch.tensor(target, dtype=torch.float)
            target = torch.cat([target[:, 3:], target[:, :3]], dim=-1)
            target = target * conversion.view(1, -1)
        with open(self.raw_paths[2], 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
                                   sanitize=False)

        cutoff = 5

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip:
                continue
            if mol is None:
                continue

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)
            
            smiles = Chem.MolToSmiles(mol)

            atom_features_list = []
            for atom in mol.GetAtoms():
                atom_features_list.append(atom_to_feature_vector(atom))

            num_bond_features = 3
            if len(mol.GetBonds()) > 0:
                edges_list = []
                edge_features_list = []
                for bond in mol.GetBonds():
                    m = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()

                    edge_feature = bond_to_feature_vector(bond)
                    edges_list.append((m, j))
                    edge_features_list.append(edge_feature)
                    edges_list.append((j, m))
                    edge_features_list.append(edge_feature)

                edge_index = torch.tensor(edges_list, dtype=torch.int64).T
                edge_attr = torch.tensor(edge_features_list, dtype=torch.int64)
            else: 
                edge_index = torch.tensor((2, 0), dtype=torch.int64)
                edge_attr = torch.tensor((0, num_bond_features), dtype=torch.int64)

            x = torch.tensor(atom_features_list, dtype=torch.int64)
            y = target[i].unsqueeze(0)
            name = mol.GetProp('_Name')
            num_nodes = x.size(0)

            data = FragData(x=x, edge_index=edge_index, pos=pos,
                            edge_attr=edge_attr, y=y, name=name, idx=i)
            
            data.coords = pos

            cutoff_edge_index = radius_graph(pos, r=cutoff, batch=None, loop=False, max_num_neighbors=100)
            data.cutoff_edge_index = cutoff_edge_index
            
            if cutoff_edge_index.size(1) > 0:
                cos_phi, sin2_phi = calculate_torsion_features(pos, cutoff_edge_index)
            else:
                cos_phi = torch.empty((0, 1), dtype=torch.float32)
                sin2_phi = torch.empty((0, 1), dtype=torch.float32)
            data.geo_cos_phi = cos_phi
            data.geo_sin2_phi = sin2_phi
            
            if data.cutoff_edge_index is not None and data.cutoff_edge_index.size(1) > 0:
                data.cutoff_curvature = compute_forman_curvature(data.cutoff_edge_index, data.num_nodes)
            else:
                data.cutoff_curvature = torch.zeros(0, dtype=torch.float)

            if data.edge_index is not None and data.edge_index.size(1) > 0:
                data.mol_curvature = compute_forman_curvature(data.edge_index, data.num_nodes)
            else:
                data.mol_curvature = torch.zeros(0, dtype=torch.float)

            data.edge_curvature = data.cutoff_curvature
            
            data.__num_node__ = num_nodes
            data.num_nodes = num_nodes

            try:
                cliques = motif_decomp(smiles)
                frag_cliques = process_cliques(cliques, num_nodes)
                n_frags = len(frag_cliques)
                
                frag_sizes = torch.tensor([len(c) for c in frag_cliques], dtype=torch.float32)
                atom2frag = torch.full((num_nodes,), -1, dtype=torch.long)
                for f_idx, frag_atoms in enumerate(frag_cliques):
                    atom2frag[frag_atoms] = f_idx

                frag_h_list, frag_pos_list = [], []
                for frag_atoms in frag_cliques:
                    frag_pos_list.append(pos[frag_atoms].mean(axis=0))
                    frag_h_list.append(data.x[frag_atoms].float().mean(dim=0))
                
                data.frag_pos = torch.tensor(torch.stack(frag_pos_list), dtype=torch.float32)
                data.frag_h = torch.stack(frag_h_list, dim=0)

                edge_lookup = {}
                if hasattr(data, 'mol_curvature'):
                    rows, cols = data.edge_index
                    curvs = data.mol_curvature
                    for k in range(rows.size(0)):
                        u, v = rows[k].item(), cols[k].item()
                        edge_lookup[(u, v)] = curvs[k].item()
                
                frag_edges = []
                frag_edge_attrs = []
                visited_frag_edges = set()

                if len(mol.GetBonds()) > 0:
                    for bond in mol.GetBonds():
                        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                        fu, fv = atom2frag[u], atom2frag[v]
                        if fu >= 0 and fv >= 0 and fu != fv:
                            if (fu, fv) not in visited_frag_edges:
                                frag_edges.append((fu, fv))
                                frag_edges.append((fv, fu))
                                
                                c_val = edge_lookup.get((u, v), edge_lookup.get((v, u), 0.0))
                                frag_edge_attrs.append(c_val)
                                frag_edge_attrs.append(c_val)

                                visited_frag_edges.add((fu, fv))
                                visited_frag_edges.add((fv, fu))
                
                if len(frag_edges) > 0:
                    data.frag_edge_index = torch.tensor(frag_edges, dtype=torch.long).t().contiguous()
                    data.frag_edge_weight = torch.tensor(frag_edge_attrs, dtype=torch.float32)
                else:
                    data.frag_edge_index = torch.empty((2, 0), dtype=torch.long)
                    data.frag_edge_weight = torch.empty((0,), dtype=torch.float32)

                data.frag_cliques = frag_cliques
                data.frag_sizes = frag_sizes
                data.num_frags = n_frags
                data.atom2u = atom2frag

            except Exception as e:
                print(f"Error decomposing molecule {i} ({smiles}): {e}")
                continue

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict


class BBBPDataset(InMemoryDataset):
    def __init__(self, root="dataset", smiles2graph=smiles2graph, transform=None, pre_transform=None):
        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "BBBP")
        self.task_type = "classification"
        self.num_tasks = 1
        self.eval_metric = "rocauc"
        self.root = root
        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "BBBP.csv"

    @property
    def processed_file_names(self):
        return "BBBP_precessed.pt"

    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        return file_path

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, "BBBP.csv"))
        smiles_list = data_df["smiles"]
        print("Converting SMILES string into graphs (BBBP)...")
        cutoff = 3.5
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            smiles = smiles_list[i]
            graph = self.smiles2graph(smiles)
            if graph is None: continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None: continue
            n_atoms = mol.GetNumAtoms()

            temp_mol = Chem.AddHs(mol)
            coords = None
            try:
                if temp_mol.GetNumConformers() == 0:
                    AllChem.EmbedMolecule(temp_mol, randomSeed=42)
                    AllChem.MMFFOptimizeMolecule(temp_mol)
                conf = temp_mol.GetConformer()
                coords = np.array([[conf.GetAtomPosition(atom.GetIdx()).x,
                                    conf.GetAtomPosition(atom.GetIdx()).y,
                                    conf.GetAtomPosition(atom.GetIdx()).z]
                                   for atom in mol.GetAtoms()], dtype=np.float32)
            except:
                continue

            pos = torch.tensor(coords, dtype=torch.float)
            if pos.shape[0] == 0 or pos.shape[0] != int(graph["num_nodes"]): continue

            data = FragData()
            num_nodes = int(graph["num_nodes"])
            data.__num_node__ = num_nodes
            data.num_nodes = num_nodes
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            data.y = torch.Tensor([data_df["p_np"].iloc[i]])
            data.pos = pos
            data.coords = pos

            cutoff_edge_index = radius_graph(pos, r=cutoff, batch=None, loop=False)
            data.cutoff_edge_index = cutoff_edge_index
            
            cos_phi, sin2_phi = calculate_torsion_features(pos, cutoff_edge_index)
            data.geo_cos_phi = cos_phi
            data.geo_sin2_phi = sin2_phi
            
            target_edge_index = data.cutoff_edge_index if data.cutoff_edge_index is not None else data.edge_index
            if target_edge_index is not None and target_edge_index.size(1) > 0:
                curv = compute_forman_curvature(target_edge_index, data.num_nodes)
                data.edge_curvature = curv
            else:
                data.edge_curvature = torch.zeros(0, dtype=torch.float)

            cliques = motif_decomp(smiles)
            frag_cliques = process_cliques(cliques, n_atoms)
            n_frags = len(frag_cliques)
            frag_sizes = torch.tensor([len(c) for c in frag_cliques], dtype=torch.float32)
            atom2frag = torch.full((n_atoms,), -1, dtype=torch.long)
            for f_idx, frag_atoms in enumerate(frag_cliques):
                atom2frag[frag_atoms] = f_idx

            frag_h_list, frag_pos_list = [], []
            for frag_atoms in frag_cliques:
                frag_pos_list.append(coords[frag_atoms].mean(axis=0))
                frag_h_list.append(data.x[frag_atoms].float().mean(dim=0))
            data.frag_pos = torch.tensor(frag_pos_list, dtype=torch.float32)
            data.frag_h = torch.stack(frag_h_list, dim=0)

            edge_lookup = {}
            if hasattr(data, 'edge_curvature'):
                rows, cols = target_edge_index
                curvs = data.edge_curvature
                for k in range(rows.size(0)):
                    u, v = rows[k].item(), cols[k].item()
                    edge_lookup[(u, v)] = curvs[k].item()
                    
            frag_edges = []
            frag_edge_attrs = []
            visited_frag_edges = set()
            
            for bond in mol.GetBonds():
                u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                fu, fv = atom2frag[u], atom2frag[v]
                if fu >= 0 and fv >= 0 and fu != fv:
                    if (fu, fv) not in visited_frag_edges:
                        frag_edges.append((fu, fv))
                        frag_edges.append((fv, fu))
                        
                        c_val = edge_lookup.get((u, v), edge_lookup.get((v, u), 0.0))
                        frag_edge_attrs.append(c_val)
                        frag_edge_attrs.append(c_val)
                        
                        visited_frag_edges.add((fu, fv))
                        visited_frag_edges.add((fv, fu))

            if len(frag_edges) > 0:
                data.frag_edge_index = torch.tensor(frag_edges, dtype=torch.long).t().contiguous()
                data.frag_edge_weight = torch.tensor(frag_edge_attrs, dtype=torch.float32)
            else:
                data.frag_edge_index = torch.empty((2, 0), dtype=torch.long)
                data.frag_edge_weight = torch.empty((0,), dtype=torch.float32)
            
            data.frag_cliques = frag_cliques
            data.frag_sizes = frag_sizes
            data.num_frags = n_frags
            data.atom2u = atom2frag

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        data.y = data.y.view(len(data.y), 1)
        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict


class BBBPDataset_cutoff(InMemoryDataset):
    def __init__(self, root="dataset", smiles2graph=smiles2graph, transform=None, pre_transform=None):
        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "BBBP")
        self.task_type = "classification"
        self.num_tasks = 1
        self.eval_metric = "rocauc"
        self.root = root
        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self): return "BBBP.csv"
    @property
    def processed_file_names(self): return "BBBP_precessed.pt"
    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        return file_path

    def process(self):
        import numpy as np
        cutoff = 3.5
        data_df = pd.read_csv(osp.join(self.raw_dir, "BBBP.csv"))
        smiles_list = data_df["smiles"]
        print("Converting SMILES string into graphs (BBBP_cutoff)...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            smiles = smiles_list[i]
            graph = self.smiles2graph(smiles)
            if graph is None: continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: continue
            n_atoms = mol.GetNumAtoms()
            temp_mol = Chem.AddHs(mol)
            coords = None
            try:
                if temp_mol.GetNumConformers() == 0:
                    AllChem.EmbedMolecule(temp_mol, randomSeed=42)
                    AllChem.MMFFOptimizeMolecule(temp_mol)
                conf = temp_mol.GetConformer()
                coords = np.array([[conf.GetAtomPosition(atom.GetIdx()).x,
                                    conf.GetAtomPosition(atom.GetIdx()).y,
                                    conf.GetAtomPosition(atom.GetIdx()).z]
                                   for atom in mol.GetAtoms()], dtype=np.float32)
            except: continue
            pos = torch.tensor(coords, dtype=torch.float)
            if pos.shape[0] == 0 or pos.shape[0] != int(graph["num_nodes"]): continue
            
            num_nodes = pos.shape[0]
            data = FragData()
            data.__num_node__ = num_nodes
            data.num_nodes = num_nodes
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64) 
            data.y = torch.Tensor([data_df["p_np"].iloc[i]])
            data.pos = pos
            data.coords = pos

            dist = torch.cdist(pos, pos, p=2) 
            mask = (dist < cutoff) & (dist > 0) 
            edge_index = mask.nonzero(as_tuple=False).t() 
            data.edge_index = edge_index
            data.cutoff_edge_index = edge_index
            data.edge_attr = None

            cos_phi, sin2_phi = calculate_torsion_features(pos, edge_index)
            data.geo_cos_phi = cos_phi
            data.geo_sin2_phi = sin2_phi
            
            target_edge_index = data.cutoff_edge_index
            if target_edge_index is not None and target_edge_index.size(1) > 0:
                curv = compute_forman_curvature(target_edge_index, data.num_nodes)
                data.edge_curvature = curv
            else:
                data.edge_curvature = torch.zeros(0, dtype=torch.float)

            cliques = motif_decomp(smiles)
            frag_cliques = process_cliques(cliques, n_atoms)
            n_frags = len(frag_cliques)
            frag_sizes = torch.tensor([len(c) for c in frag_cliques], dtype=torch.float32)
            atom2frag = torch.full((n_atoms,), -1, dtype=torch.long)
            for f_idx, frag_atoms in enumerate(frag_cliques):
                atom2frag[frag_atoms] = f_idx

            frag_h_list, frag_pos_list = [], []
            for frag_atoms in frag_cliques:
                frag_pos_list.append(coords[frag_atoms].mean(axis=0))
                frag_h_list.append(data.x[frag_atoms].float().mean(dim=0))
            data.frag_pos = torch.tensor(frag_pos_list, dtype=torch.float32)
            data.frag_h = torch.stack(frag_h_list, dim=0)

            edge_lookup = {}
            if hasattr(data, 'edge_curvature'):
                rows, cols = target_edge_index
                curvs = data.edge_curvature
                for k in range(rows.size(0)):
                    u, v = rows[k].item(), cols[k].item()
                    edge_lookup[(u, v)] = curvs[k].item()

            frag_edges = []
            frag_edge_attrs = []
            visited_frag_edges = set()
            
            for bond in mol.GetBonds():
                u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                fu, fv = atom2frag[u], atom2frag[v]
                if fu >= 0 and fv >= 0 and fu != fv:
                    if (fu, fv) not in visited_frag_edges:
                        frag_edges.append((fu, fv))
                        frag_edges.append((fv, fu))
                        
                        c_val = edge_lookup.get((u, v), edge_lookup.get((v, u), 0.0))
                        frag_edge_attrs.append(c_val)
                        frag_edge_attrs.append(c_val)
                        
                        visited_frag_edges.add((fu, fv))
                        visited_frag_edges.add((fv, fu))

            if len(frag_edges) > 0:
                data.frag_edge_index = torch.tensor(frag_edges, dtype=torch.long).t().contiguous()
                data.frag_edge_weight = torch.tensor(frag_edge_attrs, dtype=torch.float32)
            else:
                data.frag_edge_index = torch.empty((2, 0), dtype=torch.long)
                data.frag_edge_weight = torch.empty((0,), dtype=torch.float32)
                
            data.frag_cliques = frag_cliques
            data.frag_sizes = frag_sizes
            data.num_frags = n_frags
            data.atom2u = atom2frag

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        data.y = data.y.view(len(data.y), 1)
        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict


class BaceDataset(InMemoryDataset):
    def __init__(self, root="dataset", smiles2graph=smiles2graph, transform=None, pre_transform=None):
        self.smiles2graph = smiles2graph
        self.folder = os.path.join(root, "Bace") 
        self.task_type = "classification"
        self.num_tasks = 1
        self.eval_metric = "rocauc"
        self.root = root
        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self): return "bace.csv"
    @property
    def processed_file_names(self): return "bace_processed.pt"
    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        return file_path

    def process(self):
        print(f"Start Processing Bace Dataset in {self.raw_dir} ...")
        data_df = pd.read_csv(os.path.join(self.raw_dir, "bace.csv"))
        smiles_list = data_df["mol"]
        label = ["Class"]
        cutoff = 3.5
        data_list = []
        
        for i in tqdm(range(len(smiles_list))):
            smiles = smiles_list[i]
            y = data_df.iloc[i][label].values.astype('float32')
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: continue
            n_atoms = mol.GetNumAtoms()
            temp_mol = Chem.AddHs(mol)
            coords = None
            try:
                if temp_mol.GetNumConformers() == 0:
                    AllChem.EmbedMolecule(temp_mol, randomSeed=42)
                    AllChem.MMFFOptimizeMolecule(temp_mol)
                conf = temp_mol.GetConformer()
                coords = np.array([[conf.GetAtomPosition(atom.GetIdx()).x,
                                    conf.GetAtomPosition(atom.GetIdx()).y,
                                    conf.GetAtomPosition(atom.GetIdx()).z]
                                   for atom in mol.GetAtoms()], dtype=np.float32)
            except: continue

            graph = self.smiles2graph(smiles)
            if graph is None: continue

            data = FragData()
            
            data.__num_node__ = int(graph["num_nodes"])
            data.num_nodes = int(graph["num_nodes"])
            data.edge_index = torch.unique(torch.from_numpy(graph["edge_index"]).to(torch.int64), dim=1)
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            data.y = torch.from_numpy(y).view(1, -1)
            data.coords = torch.from_numpy(coords)
            
            cutoff_edge_index = radius_graph(data.coords, r=cutoff, batch=None, loop=False)
            data.cutoff_edge_index = cutoff_edge_index
            
            cos_phi, sin2_phi = calculate_torsion_features(data.coords, cutoff_edge_index)
            data.geo_cos_phi = cos_phi
            data.geo_sin2_phi = sin2_phi

            target_edge_index = data.cutoff_edge_index if data.cutoff_edge_index is not None else data.edge_index
            
            if target_edge_index is not None and target_edge_index.size(1) > 0:
                curv = compute_forman_curvature(target_edge_index, data.num_nodes)
                data.edge_curvature = curv
            else:
                data.edge_curvature = torch.zeros(0, dtype=torch.float)

            cliques = motif_decomp(smiles)
            frag_cliques = process_cliques(cliques, n_atoms)
            n_frags = len(frag_cliques)
            frag_sizes = torch.tensor([len(c) for c in frag_cliques], dtype=torch.float32)
            
            atom2frag = torch.full((n_atoms,), -1, dtype=torch.long)
            for f_idx, frag_atoms in enumerate(frag_cliques):
                atom2frag[frag_atoms] = f_idx

            frag_h_list, frag_pos_list = [], []
            for frag_atoms in frag_cliques:
                frag_pos_list.append(coords[frag_atoms].mean(axis=0))
                frag_h_list.append(data.x[frag_atoms].float().mean(dim=0))
            data.frag_pos = torch.tensor(frag_pos_list, dtype=torch.float32)
            data.frag_h = torch.stack(frag_h_list, dim=0)

            edge_lookup = {}
            if hasattr(data, 'edge_curvature'):
                rows, cols = target_edge_index
                curvs = data.edge_curvature
                for k in range(rows.size(0)):
                    u, v = rows[k].item(), cols[k].item()
                    edge_lookup[(u, v)] = curvs[k].item()
            
            frag_edges = []
            frag_edge_attrs = [] 
            visited_frag_edges = set()

            for bond in mol.GetBonds():
                u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                fu, fv = atom2frag[u], atom2frag[v]
                if fu >= 0 and fv >= 0 and fu != fv:
                    if (fu, fv) not in visited_frag_edges:
                        frag_edges.append((fu, fv))
                        frag_edges.append((fv, fu))
                        
                        c_val = edge_lookup.get((u, v), edge_lookup.get((v, u), 0.0))
                        frag_edge_attrs.append(c_val)
                        frag_edge_attrs.append(c_val)

                        visited_frag_edges.add((fu, fv))
                        visited_frag_edges.add((fv, fu))
            
            if len(frag_edges) > 0:
                data.frag_edge_index = torch.tensor(frag_edges, dtype=torch.long).t().contiguous()
                data.frag_edge_weight = torch.tensor(frag_edge_attrs, dtype=torch.float32)
            else:
                data.frag_edge_index = torch.empty((2, 0), dtype=torch.long)
                data.frag_edge_weight = torch.empty((0,), dtype=torch.float32)
                
            data.frag_cliques = frag_cliques
            data.frag_sizes = frag_sizes
            data.num_frags = n_frags
            data.atom2u = atom2frag

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Processed file saved to: {self.processed_paths[0]}")

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict


class Tox21Dataset(InMemoryDataset):
    def __init__(self, root="dataset", smiles2graph=smiles2graph, transform=None, pre_transform=None):
        self.smiles2graph = smiles2graph
        self.folder = root 
        self.task_type = "classification" 
        self.num_tasks = 12
        self.eval_metric = "rocauc"
        self.root = root
        
        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
        
        self.tox21_tasks = [
            'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
            'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
        ]

        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["tox21.csv"]

    @property
    def processed_file_names(self):
        return "tox21_processed.pt"

    def download(self):
        gz_file_path = download_url(self.url, self.raw_dir)
        final_file_path = osp.join(self.raw_dir, "tox21.csv")

        print(f"Reading and extracting {gz_file_path} to {final_file_path}")
        
        try:
            data_df = pd.read_csv(gz_file_path, compression='gzip')
            
            if 'smiles' not in data_df.columns and 'mol_id' not in data_df.columns:
                print("Header not found, retrying with skiprows...")
                data_df = pd.read_csv(gz_file_path, compression='gzip', skiprows=2)

        except Exception as e:
            print(f"Error reading Tox21 GZ file: {e}")
            os.unlink(gz_file_path)
            raise e
        
        data_df.to_csv(final_file_path, index=False)
        
        if os.path.exists(gz_file_path):
            os.unlink(gz_file_path)

    def process(self):
        print(f"Start Processing Tox21 Dataset in {self.raw_dir} ...")
        
        data_df = pd.read_csv(os.path.join(self.raw_dir, "tox21.csv"))
        smiles_list = data_df["smiles"]
        
        labels = data_df[self.tox21_tasks]
        labels = labels.replace(np.nan, float('nan')) 
        
        cutoff = 3.5

        data_list = []
        for i in tqdm(range(len(smiles_list))):
            smiles = smiles_list[i]
            y_np = labels.iloc[i].values.astype('float32')

            mol = Chem.MolFromSmiles(smiles)
            if mol is None: continue
            n_atoms = mol.GetNumAtoms()

            temp_mol = Chem.AddHs(mol)
            coords = None
            try:
                if temp_mol.GetNumConformers() == 0:
                    AllChem.EmbedMolecule(temp_mol, randomSeed=42)
                    AllChem.MMFFOptimizeMolecule(temp_mol)
                conf = temp_mol.GetConformer()
                coords = np.array([[conf.GetAtomPosition(atom.GetIdx()).x,
                                    conf.GetAtomPosition(atom.GetIdx()).y,
                                    conf.GetAtomPosition(atom.GetIdx()).z]
                                   for atom in mol.GetAtoms()], dtype=np.float32)
            except:
                continue
            
            pos = torch.tensor(coords, dtype=torch.float)

            graph = self.smiles2graph(smiles)
            if graph is None: continue
            
            if pos.shape[0] != int(graph["num_nodes"]):
                continue

            data = FragData()
            data.__num_node__ = int(graph["num_nodes"])
            data.num_nodes = int(graph["num_nodes"])
            data.edge_index = torch.unique(torch.from_numpy(graph["edge_index"]).to(torch.int64), dim=1)
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            
            data.y= torch.from_numpy(y_np).view(1, -1)
            
            data.coords = pos
            data.pos = pos

            cutoff_edge_index = radius_graph(pos, r=cutoff, batch=None, loop=False)
            data.cutoff_edge_index = cutoff_edge_index
            
            cos_phi, sin2_phi = calculate_torsion_features(pos, cutoff_edge_index)
            data.geo_cos_phi = cos_phi
            data.geo_sin2_phi = sin2_phi
            
            target_edge_index = data.cutoff_edge_index if data.cutoff_edge_index is not None else data.edge_index
            if target_edge_index is not None and target_edge_index.size(1) > 0:
                curv = compute_forman_curvature(target_edge_index, data.num_nodes)
                data.edge_curvature = curv
            else:
                data.edge_curvature = torch.zeros(0, dtype=torch.float)

            cliques = motif_decomp(smiles)
            frag_cliques = process_cliques(cliques, n_atoms)
            n_frags = len(frag_cliques)
            
            frag_sizes = torch.tensor([len(c) for c in frag_cliques], dtype=torch.float32)

            atom2frag = torch.full((n_atoms,), -1, dtype=torch.long)
            for f_idx, frag_atoms in enumerate(frag_cliques):
                atom2frag[frag_atoms] = f_idx

            frag_h_list, frag_pos_list = [], []
            for frag_atoms in frag_cliques:
                frag_pos_list.append(coords[frag_atoms].mean(axis=0))
                frag_h_list.append(data.x[frag_atoms].float().mean(dim=0))
            data.frag_pos = torch.tensor(frag_pos_list, dtype=torch.float32)
            data.frag_h = torch.stack(frag_h_list, dim=0)

            edge_lookup = {}
            if hasattr(data, 'edge_curvature'):
                rows, cols = target_edge_index
                curvs = data.edge_curvature
                for k in range(rows.size(0)):
                    u, v = rows[k].item(), cols[k].item()
                    edge_lookup[(u, v)] = curvs[k].item()
            
            frag_edges = []
            frag_edge_attrs = []
            visited_frag_edges = set()

            for bond in mol.GetBonds():
                u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                fu, fv = atom2frag[u], atom2frag[v]
                if fu >= 0 and fv >= 0 and fu != fv:
                    if (fu, fv) not in visited_frag_edges:
                        frag_edges.append((fu, fv))
                        frag_edges.append((fv, fu))
                        
                        c_val = edge_lookup.get((u, v), edge_lookup.get((v, u), 0.0))
                        frag_edge_attrs.append(c_val)
                        frag_edge_attrs.append(c_val)
                        
                        visited_frag_edges.add((fu, fv))
                        visited_frag_edges.add((fv, fu))

            if len(frag_edges) > 0:
                data.frag_edge_index = torch.tensor(frag_edges, dtype=torch.long).t().contiguous()
                data.frag_edge_weight = torch.tensor(frag_edge_attrs, dtype=torch.float32)
            else:
                data.frag_edge_index = torch.empty((2, 0), dtype=torch.long)
                data.frag_edge_weight = torch.empty((0,), dtype=torch.float32)

            data.frag_cliques = frag_cliques
            data.frag_sizes = frag_sizes
            data.num_frags = n_frags
            data.atom2u = atom2frag

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Processed file saved to: {self.processed_paths[0]}")

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict

class HIVDataset(InMemoryDataset):
    def __init__(self, root="dataset", smiles2graph=smiles2graph, transform=None, pre_transform=None):
        self.smiles2graph = smiles2graph
        self.folder = os.path.join(root, "HIV")
        self.task_type = "classification"
        self.num_tasks = 1
        self.eval_metric = "rocauc"
        self.root = root
        
        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv"

        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "HIV.csv"

    @property
    def processed_file_names(self):
        return "hiv_processed.pt"

    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        return file_path

    def process(self):
        print(f"Start Processing HIV Dataset in {self.raw_dir} ...")
        
        data_df = pd.read_csv(os.path.join(self.raw_dir, "HIV.csv"))
        smiles_list = data_df["smiles"]
        label_col = "HIV_active"
        
        cutoff = 3.5

        data_list = []
        for i in tqdm(range(len(smiles_list))):
            smiles = smiles_list[i]
            y = data_df.iloc[i][label_col]

            mol = Chem.MolFromSmiles(smiles)
            if mol is None: continue
            n_atoms = mol.GetNumAtoms()

            temp_mol = Chem.AddHs(mol)
            coords = None
            try:
                if temp_mol.GetNumConformers() == 0:
                    res = AllChem.EmbedMolecule(
                        temp_mol, 
                        randomSeed=42, 
                        useRandomCoords=True, 
                        maxAttempts=30
                    )
                    if res < 0: continue
                    
                    AllChem.MMFFOptimizeMolecule(temp_mol, maxIters=50)
                    
                conf = temp_mol.GetConformer()
                coords = np.array([[conf.GetAtomPosition(atom.GetIdx()).x,
                                    conf.GetAtomPosition(atom.GetIdx()).y,
                                    conf.GetAtomPosition(atom.GetIdx()).z]
                                   for atom in mol.GetAtoms()], dtype=np.float32)
            except:
                continue
            
            pos = torch.tensor(coords, dtype=torch.float)

            graph = self.smiles2graph(smiles)
            if graph is None: continue
            
            if pos.shape[0] != int(graph["num_nodes"]):
                continue

            data = FragData()
            data.__num_node__ = int(graph["num_nodes"])
            data.num_nodes = int(graph["num_nodes"])
            data.edge_index = torch.unique(torch.from_numpy(graph["edge_index"]).to(torch.int64), dim=1)
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            data.y = torch.Tensor([y])
            data.coords = pos
            data.pos = pos

            cutoff_edge_index = radius_graph(pos, r=cutoff, batch=None, loop=False)
            data.cutoff_edge_index = cutoff_edge_index
            
            if cutoff_edge_index.size(1) > 0:
                cos_phi, sin2_phi = calculate_torsion_features(pos, cutoff_edge_index)
            else:
                cos_phi = torch.empty((0, 1), dtype=torch.float32)
                sin2_phi = torch.empty((0, 1), dtype=torch.float32)

            data.geo_cos_phi = cos_phi
            data.geo_sin2_phi = sin2_phi
            
            target_edge_index = data.cutoff_edge_index if data.cutoff_edge_index is not None else data.edge_index
            if target_edge_index is not None and target_edge_index.size(1) > 0:
                curv = compute_forman_curvature(target_edge_index, data.num_nodes)
                data.edge_curvature = curv
            else:
                data.edge_curvature = torch.zeros(0, dtype=torch.float)

            cliques = motif_decomp(smiles)
            frag_cliques = process_cliques(cliques, n_atoms)
            n_frags = len(frag_cliques)
            
            frag_sizes = torch.tensor([len(c) for c in frag_cliques], dtype=torch.float32)

            atom2frag = torch.full((n_atoms,), -1, dtype=torch.long)
            for f_idx, frag_atoms in enumerate(frag_cliques):
                atom2frag[frag_atoms] = f_idx

            frag_h_list, frag_pos_list = [], []
            for frag_atoms in frag_cliques:
                frag_pos_list.append(coords[frag_atoms].mean(axis=0))
                frag_h_list.append(data.x[frag_atoms].float().mean(dim=0))
            data.frag_pos = torch.tensor(frag_pos_list, dtype=torch.float32)
            data.frag_h = torch.stack(frag_h_list, dim=0)

            edge_lookup = {}
            if hasattr(data, 'edge_curvature'):
                rows, cols = target_edge_index
                curvs = data.edge_curvature
                for k in range(rows.size(0)):
                    u, v = rows[k].item(), cols[k].item()
                    edge_lookup[(u, v)] = curvs[k].item()
                    
            frag_edges = []
            frag_edge_attrs = []
            visited_frag_edges = set()

            for bond in mol.GetBonds():
                u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                fu, fv = atom2frag[u], atom2frag[v]
                if fu >= 0 and fv >= 0 and fu != fv:
                    if (fu, fv) not in visited_frag_edges:
                        frag_edges.append((fu, fv))
                        frag_edges.append((fv, fu))
                        
                        c_val = edge_lookup.get((u, v), edge_lookup.get((v, u), 0.0))
                        frag_edge_attrs.append(c_val)
                        frag_edge_attrs.append(c_val)
                        
                        visited_frag_edges.add((fu, fv))
                        visited_frag_edges.add((fv, fu))

            if len(frag_edges) > 0:
                data.frag_edge_index = torch.tensor(frag_edges, dtype=torch.long).t().contiguous()
                data.frag_edge_weight = torch.tensor(frag_edge_attrs, dtype=torch.float32)
            else:
                data.frag_edge_index = torch.empty((2, 0), dtype=torch.long)
                data.frag_edge_weight = torch.empty((0,), dtype=torch.float32)

            data.frag_cliques = frag_cliques
            data.frag_sizes = frag_sizes
            data.num_frags = n_frags
            data.atom2u = atom2frag

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Processed file saved to: {self.processed_paths[0]}")

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict

class SiderDataset(InMemoryDataset):
    def __init__(self, root="dataset", smiles2graph=smiles2graph, transform=None, pre_transform=None):
        self.smiles2graph = smiles2graph
        self.folder = root 
        self.task_type = "classification" 
        self.num_tasks = 27
        self.eval_metric = "rocauc"
        self.root = root
        
        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz"
        
        self.sider_tasks = [
            'Hepatobiliary disorders', 'Metabolism and nutrition disorders', 'Product issues', 
            'Eye disorders', 'Investigations', 'Musculoskeletal and connective tissue disorders', 
            'Gastrointestinal disorders', 'Social circumstances', 'Immune system disorders', 
            'Reproductive system and breast disorders', 'Neoplasms benign, malignant and unspecified (incl cysts and polyps)', 
            'General disorders and administration site conditions', 'Endocrine disorders', 
            'Surgical and medical procedures', 'Immune system disorders.1', 'Congenital, familial and genetic disorders', 
            'Infections and infestations', 'Respiratory, thoracic and mediastinal disorders', 
            'Psychiatric disorders', 'Renal and urinary disorders', 'Vascular disorders', 
            'Blood and lymphatic system disorders', 'Nervous system disorders', 
            'Skin and subcutaneous tissue disorders', 'Cardiac disorders', 
            'Ear and labyrinth disorders', 'Injury, poisoning and procedural complications'
        ]

        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["sider.csv"]

    @property
    def processed_file_names(self):
        return "sider_processed.pt"

    def download(self):
        gz_file_path = download_url(self.url, self.raw_dir)
        final_file_path = osp.join(self.raw_dir, "sider.csv")

        print(f"Reading and extracting {gz_file_path} to {final_file_path}")
        
        try:
            data_df = pd.read_csv(gz_file_path, compression='gzip')
        except Exception as e:
            print(f"Error reading Sider GZ file: {e}")
            os.unlink(gz_file_path)
            raise e
        
        data_df.to_csv(final_file_path, index=False)
        if os.path.exists(gz_file_path):
            os.unlink(gz_file_path)

    def process(self):
        print(f"Start Processing Sider Dataset in {self.raw_dir} ...")
        
        data_df = pd.read_csv(os.path.join(self.raw_dir, "sider.csv"))
        smiles_list = data_df["smiles"]
        
        try:
            labels = data_df[self.sider_tasks]
        except KeyError:
            labels = data_df.iloc[:, 1:28]
        labels = labels.replace(np.nan, float('nan')) 
        
        cutoff = 3.5
        data_list = []

        skip_count = 0

        for i in tqdm(range(len(smiles_list)), desc="Processing Sider"):
            smiles = smiles_list[i]
            y_np = labels.iloc[i].values.astype('float32')

            mol = Chem.MolFromSmiles(smiles)
            if mol is None: 
                skip_count += 1
                continue
            
            n_atoms = mol.GetNumAtoms()
            
            if n_atoms > 150:
                skip_count += 1
                continue

            temp_mol = Chem.AddHs(mol)
            coords = None
            try:
                if temp_mol.GetNumConformers() == 0:
                    res = AllChem.EmbedMolecule(
                        temp_mol, 
                        randomSeed=42, 
                        useRandomCoords=True, 
                        maxAttempts=30, 
                        clearConfs=True
                    )
                    if res < 0: 
                        skip_count += 1
                        continue 
                    
                    AllChem.MMFFOptimizeMolecule(temp_mol, maxIters=50)
                    
                conf = temp_mol.GetConformer()
                coords = np.array([[conf.GetAtomPosition(atom.GetIdx()).x,
                                    conf.GetAtomPosition(atom.GetIdx()).y,
                                    conf.GetAtomPosition(atom.GetIdx()).z]
                                   for atom in mol.GetAtoms()], dtype=np.float32)
            except Exception:
                skip_count += 1
                continue
            
            if coords is None: 
                skip_count += 1
                continue
            
            pos = torch.tensor(coords, dtype=torch.float)

            try:
                graph = self.smiles2graph(smiles)
            except: 
                skip_count += 1
                continue
            
            if graph is None or pos.shape[0] != int(graph["num_nodes"]):
                skip_count += 1
                continue

            data = FragData()
            data.__num_node__ = int(graph["num_nodes"])
            data.num_nodes = int(graph["num_nodes"])
            data.edge_index = torch.unique(torch.from_numpy(graph["edge_index"]).to(torch.int64), dim=1)
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            data.y = torch.from_numpy(y_np).view(1, -1)
            data.coords = pos
            data.pos = pos

            cutoff_edge_index = radius_graph(pos, r=cutoff, batch=None, loop=False)
            data.cutoff_edge_index = cutoff_edge_index
            if cutoff_edge_index.size(1) > 0:
                cos_phi, sin2_phi = calculate_torsion_features(pos, cutoff_edge_index)
            else:
                cos_phi = torch.empty((0, 1), dtype=torch.float32)
                sin2_phi = torch.empty((0, 1), dtype=torch.float32)
            data.geo_cos_phi = cos_phi
            data.geo_sin2_phi = sin2_phi
            
            target_edge_index = data.cutoff_edge_index if data.cutoff_edge_index is not None else data.edge_index
            if target_edge_index is not None and target_edge_index.size(1) > 0:
                curv = compute_forman_curvature(target_edge_index, data.num_nodes)
                data.edge_curvature = curv
            else:
                data.edge_curvature = torch.zeros(0, dtype=torch.float)

            try:
                cliques = motif_decomp(smiles)
            except Exception:
                skip_count += 1
                continue

            frag_cliques = process_cliques(cliques, n_atoms)
            n_frags = len(frag_cliques)
            frag_sizes = torch.tensor([len(c) for c in frag_cliques], dtype=torch.float32)
            atom2frag = torch.full((n_atoms,), -1, dtype=torch.long)
            for f_idx, frag_atoms in enumerate(frag_cliques):
                atom2frag[frag_atoms] = f_idx

            frag_h_list, frag_pos_list = [], []
            for frag_atoms in frag_cliques:
                frag_pos_list.append(coords[frag_atoms].mean(axis=0))
                frag_h_list.append(data.x[frag_atoms].float().mean(dim=0))
            data.frag_pos = torch.tensor(frag_pos_list, dtype=torch.float32)
            data.frag_h = torch.stack(frag_h_list, dim=0)

            edge_lookup = {}
            if hasattr(data, 'edge_curvature'):
                rows, cols = target_edge_index
                curvs = data.edge_curvature
                for k in range(rows.size(0)):
                    u, v = rows[k].item(), cols[k].item()
                    edge_lookup[(u, v)] = curvs[k].item()
            
            frag_edges = []
            frag_edge_attrs = []
            visited_frag_edges = set()
            
            for bond in mol.GetBonds():
                u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                fu, fv = atom2frag[u], atom2frag[v]
                if fu >= 0 and fv >= 0 and fu != fv:
                    if (fu, fv) not in visited_frag_edges:
                        frag_edges.append((fu, fv))
                        frag_edges.append((fv, fu))
                        
                        c_val = edge_lookup.get((u, v), edge_lookup.get((v, u), 0.0))
                        frag_edge_attrs.append(c_val)
                        frag_edge_attrs.append(c_val)
                        
                        visited_frag_edges.add((fu, fv))
                        visited_frag_edges.add((fv, fu))
                        
            if len(frag_edges) > 0:
                data.frag_edge_index = torch.tensor(frag_edges, dtype=torch.long).t().contiguous()
                data.frag_edge_weight = torch.tensor(frag_edge_attrs, dtype=torch.float32)
            else:
                data.frag_edge_index = torch.empty((2, 0), dtype=torch.long)
                data.frag_edge_weight = torch.empty((0,), dtype=torch.float32)

            data.frag_cliques = frag_cliques
            data.frag_sizes = frag_sizes
            data.num_frags = n_frags
            data.atom2u = atom2frag

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        if len(data_list) == 0:
            print("Error: No data was processed successfully!")
            return

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Processed file saved to: {self.processed_paths[0]}")
        print(f"Total skipped molecules: {skip_count}")
    
    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict

class ClinToxDataset(InMemoryDataset):
    def __init__(self, root="dataset", smiles2graph=smiles2graph, transform=None, pre_transform=None):
        self.smiles2graph = smiles2graph
        self.folder = root 
        self.task_type = "classification" 
        self.num_tasks = 2 
        self.eval_metric = "rocauc"
        self.root = root
        
        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"
        
        self.clintox_tasks = ['FDA_APPROVED', 'CT_TOX_FREE']

        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["clintox.csv"]

    @property
    def processed_file_names(self):
        return "clintox_processed.pt"

    def download(self):
        gz_file_path = download_url(self.url, self.raw_dir)
        final_file_path = osp.join(self.raw_dir, "clintox.csv")

        print(f"Reading and extracting {gz_file_path} to {final_file_path}")
        
        try:
            data_df = pd.read_csv(gz_file_path, compression='gzip')
        except Exception as e:
            print(f"Error reading ClinTox GZ file: {e}")
            os.unlink(gz_file_path)
            raise e
        
        data_df.to_csv(final_file_path, index=False)
        if os.path.exists(gz_file_path):
            os.unlink(gz_file_path)

    def process(self):
        print(f"Start Processing ClinTox Dataset in {self.raw_dir} ...")
        
        data_df = pd.read_csv(os.path.join(self.raw_dir, "clintox.csv"))
        smiles_list = data_df["smiles"]
        
        actual_tasks = []
        possible_tox_cols = ['CT_TOX', 'CT_TOX_FREE']
        
        if 'FDA_APPROVED' in data_df.columns:
            actual_tasks.append('FDA_APPROVED')
        else:
            print("Warning: 'FDA_APPROVED' column not found, using index 1.")
            actual_tasks.append(data_df.columns[1])

        tox_col_found = False
        for col in possible_tox_cols:
            if col in data_df.columns:
                actual_tasks.append(col)
                tox_col_found = True
                break
        
        if not tox_col_found:
            print(f"Warning: Neither {possible_tox_cols} found, using index 2.")
            actual_tasks.append(data_df.columns[2])
            
        print(f"Using columns for labels: {actual_tasks}")
        labels = data_df[actual_tasks]

        labels = labels.replace(np.nan, float('nan')) 
        
        cutoff = 3.5
        data_list = []
        skip_count = 0

        for i in tqdm(range(len(smiles_list)), desc="Processing ClinTox"):
            smiles = smiles_list[i]
            y_np = labels.iloc[i].values.astype('float32')

            mol = Chem.MolFromSmiles(smiles)
            if mol is None: 
                skip_count += 1
                continue
            
            n_atoms = mol.GetNumAtoms()
            if n_atoms > 150:
                skip_count += 1
                continue

            temp_mol = Chem.AddHs(mol)
            coords = None
            try:
                if temp_mol.GetNumConformers() == 0:
                    res = AllChem.EmbedMolecule(
                        temp_mol, 
                        randomSeed=42, 
                        useRandomCoords=True, 
                        maxAttempts=30,
                        clearConfs=True
                    )
                    if res < 0: 
                        skip_count += 1
                        continue 
                    
                    AllChem.MMFFOptimizeMolecule(temp_mol, maxIters=50)
                    
                conf = temp_mol.GetConformer()
                coords = np.array([[conf.GetAtomPosition(atom.GetIdx()).x,
                                    conf.GetAtomPosition(atom.GetIdx()).y,
                                    conf.GetAtomPosition(atom.GetIdx()).z]
                                   for atom in mol.GetAtoms()], dtype=np.float32)
            except Exception:
                skip_count += 1
                continue
            
            if coords is None:
                skip_count += 1
                continue
                
            pos = torch.tensor(coords, dtype=torch.float)

            try:
                graph = self.smiles2graph(smiles)
            except: 
                skip_count += 1
                continue
            
            if graph is None: 
                skip_count += 1
                continue
            
            if pos.shape[0] != int(graph["num_nodes"]):
                skip_count += 1
                continue

            data = FragData()
            data.__num_node__ = int(graph["num_nodes"])
            data.num_nodes = int(graph["num_nodes"])
            data.edge_index = torch.unique(torch.from_numpy(graph["edge_index"]).to(torch.int64), dim=1)
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            
            data.y = torch.from_numpy(y_np).view(1, -1)
            
            data.coords = pos
            data.pos = pos

            cutoff_edge_index = radius_graph(pos, r=cutoff, batch=None, loop=False)
            data.cutoff_edge_index = cutoff_edge_index
            
            if cutoff_edge_index.size(1) > 0:
                cos_phi, sin2_phi = calculate_torsion_features(pos, cutoff_edge_index)
            else:
                cos_phi = torch.empty((0, 1), dtype=torch.float32)
                sin2_phi = torch.empty((0, 1), dtype=torch.float32)
            
            data.geo_cos_phi = cos_phi
            data.geo_sin2_phi = sin2_phi
            
            target_edge_index = data.cutoff_edge_index if data.cutoff_edge_index is not None else data.edge_index
            if target_edge_index is not None and target_edge_index.size(1) > 0:
                curv = compute_forman_curvature(target_edge_index, data.num_nodes)
                data.edge_curvature = curv
            else:
                data.edge_curvature = torch.zeros(0, dtype=torch.float)

            try:
                cliques = motif_decomp(smiles)
            except Exception:
                skip_count += 1
                continue

            frag_cliques = process_cliques(cliques, n_atoms)
            n_frags = len(frag_cliques)
            
            frag_sizes = torch.tensor([len(c) for c in frag_cliques], dtype=torch.float32)
            atom2frag = torch.full((n_atoms,), -1, dtype=torch.long)
            for f_idx, frag_atoms in enumerate(frag_cliques):
                atom2frag[frag_atoms] = f_idx

            frag_h_list, frag_pos_list = [], []
            for frag_atoms in frag_cliques:
                frag_pos_list.append(coords[frag_atoms].mean(axis=0))
                frag_h_list.append(data.x[frag_atoms].float().mean(dim=0))
            data.frag_pos = torch.tensor(frag_pos_list, dtype=torch.float32)
            data.frag_h = torch.stack(frag_h_list, dim=0)

            edge_lookup = {}
            if hasattr(data, 'edge_curvature'):
                rows, cols = target_edge_index
                curvs = data.edge_curvature
                for k in range(rows.size(0)):
                    u, v = rows[k].item(), cols[k].item()
                    edge_lookup[(u, v)] = curvs[k].item()
            
            frag_edges = []
            frag_edge_attrs = []
            visited_frag_edges = set()

            for bond in mol.GetBonds():
                u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                fu, fv = atom2frag[u], atom2frag[v]
                if fu >= 0 and fv >= 0 and fu != fv:
                    if (fu, fv) not in visited_frag_edges:
                        frag_edges.append((fu, fv))
                        frag_edges.append((fv, fu))
                        
                        c_val = edge_lookup.get((u, v), edge_lookup.get((v, u), 0.0))
                        frag_edge_attrs.append(c_val)
                        frag_edge_attrs.append(c_val)
                        
                        visited_frag_edges.add((fu, fv))
                        visited_frag_edges.add((fv, fu))

            if len(frag_edges) > 0:
                data.frag_edge_index = torch.tensor(frag_edges, dtype=torch.long).t().contiguous()
                data.frag_edge_weight = torch.tensor(frag_edge_attrs, dtype=torch.float32)
            else:
                data.frag_edge_index = torch.empty((2, 0), dtype=torch.long)
                data.frag_edge_weight = torch.empty((0,), dtype=torch.float32)

            data.frag_cliques = frag_cliques
            data.frag_sizes = frag_sizes
            data.num_frags = n_frags
            data.atom2u = atom2frag

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        if len(data_list) == 0:
            print("Error: No data was processed successfully!")
            return

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Processed file saved to: {self.processed_paths[0]}")
        print(f"Total skipped: {skip_count}")
    
    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict

class EsolDataset(InMemoryDataset):
    def __init__(self, root="dataset", smiles2graph=smiles2graph, transform=None, pre_transform=None):
        self.smiles2graph = smiles2graph
        self.folder = os.path.join(root, "Esol")
        self.task_type = "mse_regression"
        self.num_tasks = 1
        self.eval_metric = "rmse"
        self.root = root
        
        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"

        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "delaney-processed.csv"

    @property
    def processed_file_names(self):
        return "esol_processed.pt"

    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        return file_path

    def process(self):
        print(f"Start Processing ESOL Dataset in {self.raw_dir} ...")
        
        data_df = pd.read_csv(os.path.join(self.raw_dir, "delaney-processed.csv"))
        smiles_list = data_df["smiles"]
        
        target_col = "measured log solubility in mols per litre"
        if target_col not in data_df.columns:
            target_col = data_df.columns[-1]
            print(f"Warning: Target column not found, using last column: {target_col}")

        labels = data_df[target_col]
        
        cutoff = 3.5
        data_list = []
        skip_count = 0

        for i in tqdm(range(len(smiles_list)), desc="Processing ESOL"):
            smiles = smiles_list[i]
            y_val = float(labels.iloc[i])

            mol = Chem.MolFromSmiles(smiles)
            if mol is None: 
                skip_count += 1
                continue
            
            n_atoms = mol.GetNumAtoms()
            temp_mol = Chem.AddHs(mol)
            coords = None
        
            try:
                if temp_mol.GetNumConformers() == 0:
                    res = AllChem.EmbedMolecule(
                        temp_mol, 
                        randomSeed=42, 
                        useRandomCoords=True, 
                        maxAttempts=50
                    )
                    if res < 0: 
                        skip_count += 1
                        continue 
                    AllChem.MMFFOptimizeMolecule(temp_mol, maxIters=50)
                    
                conf = temp_mol.GetConformer()
                coords = np.array([[conf.GetAtomPosition(atom.GetIdx()).x,
                                    conf.GetAtomPosition(atom.GetIdx()).y,
                                    conf.GetAtomPosition(atom.GetIdx()).z]
                                   for atom in mol.GetAtoms()], dtype=np.float32)
            except Exception:
                skip_count += 1
                continue
            
            if coords is None:
                skip_count += 1
                continue
                
            pos = torch.tensor(coords, dtype=torch.float)

            try:
                graph = self.smiles2graph(smiles)
            except:
                skip_count += 1
                continue

            if graph is None or pos.shape[0] != int(graph["num_nodes"]):
                skip_count += 1
                continue

            data = FragData()
            data.__num_node__ = int(graph["num_nodes"])
            data.num_nodes = int(graph["num_nodes"])
            data.edge_index = torch.unique(torch.from_numpy(graph["edge_index"]).to(torch.int64), dim=1)
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            
            data.y = torch.tensor([[y_val]], dtype=torch.float32)
            
            data.coords = pos
            data.pos = pos

            cutoff_edge_index = radius_graph(pos, r=cutoff, batch=None, loop=False)
            data.cutoff_edge_index = cutoff_edge_index
            
            if cutoff_edge_index.size(1) > 0:
                cos_phi, sin2_phi = calculate_torsion_features(pos, cutoff_edge_index)
            else:
                cos_phi = torch.empty((0, 1), dtype=torch.float32)
                sin2_phi = torch.empty((0, 1), dtype=torch.float32)
                
            data.geo_cos_phi = cos_phi
            data.geo_sin2_phi = sin2_phi
            
            target_edge_index = data.cutoff_edge_index if data.cutoff_edge_index is not None else data.edge_index
            if target_edge_index is not None and target_edge_index.size(1) > 0:
                curv = compute_forman_curvature(target_edge_index, data.num_nodes)
                data.edge_curvature = curv
            else:
                data.edge_curvature = torch.zeros(0, dtype=torch.float)

            try:
                cliques = motif_decomp(smiles)
            except Exception:
                skip_count += 1
                continue

            frag_cliques = process_cliques(cliques, n_atoms)
            n_frags = len(frag_cliques)
            
            frag_sizes = torch.tensor([len(c) for c in frag_cliques], dtype=torch.float32)
            atom2frag = torch.full((n_atoms,), -1, dtype=torch.long)
            for f_idx, frag_atoms in enumerate(frag_cliques):
                atom2frag[frag_atoms] = f_idx

            frag_h_list, frag_pos_list = [], []
            for frag_atoms in frag_cliques:
                frag_pos_list.append(coords[frag_atoms].mean(axis=0))
                frag_h_list.append(data.x[frag_atoms].float().mean(dim=0))
            
            if len(frag_pos_list) > 0:
                data.frag_pos = torch.tensor(frag_pos_list, dtype=torch.float32)
                data.frag_h = torch.stack(frag_h_list, dim=0)
            else:
                data.frag_pos = torch.empty((0, 3), dtype=torch.float32)
                data.frag_h = torch.empty((0, data.x.size(1)), dtype=torch.float32)

            edge_lookup = {}
            if hasattr(data, 'edge_curvature'):
                rows, cols = target_edge_index
                curvs = data.edge_curvature
                for k in range(rows.size(0)):
                    u, v = rows[k].item(), cols[k].item()
                    edge_lookup[(u, v)] = curvs[k].item()
            
            frag_edges = []
            frag_edge_attrs = []
            visited_frag_edges = set()
            
            for bond in mol.GetBonds():
                u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                fu, fv = atom2frag[u], atom2frag[v]
                if fu >= 0 and fv >= 0 and fu != fv:
                    if (fu, fv) not in visited_frag_edges:
                        frag_edges.append((fu, fv))
                        frag_edges.append((fv, fu))
                        
                        c_val = edge_lookup.get((u, v), edge_lookup.get((v, u), 0.0))
                        frag_edge_attrs.append(c_val)
                        frag_edge_attrs.append(c_val)
                        
                        visited_frag_edges.add((fu, fv))
                        visited_frag_edges.add((fv, fu))

            if len(frag_edges) > 0:
                data.frag_edge_index = torch.tensor(frag_edges, dtype=torch.long).t().contiguous()
                data.frag_edge_weight = torch.tensor(frag_edge_attrs, dtype=torch.float32)
            else:
                data.frag_edge_index = torch.empty((2, 0), dtype=torch.long)
                data.frag_edge_weight = torch.empty((0,), dtype=torch.float32)

            data.frag_cliques = frag_cliques
            data.frag_sizes = frag_sizes
            data.num_frags = n_frags
            data.atom2u = atom2frag

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        if len(data_list) == 0:
            print("Error: No data was processed successfully!")
            return

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Processed file saved to: {self.processed_paths[0]}")
        print(f"Total skipped: {skip_count}")
    
    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict

class FreesolvDataset(InMemoryDataset):
    def __init__(self, root="dataset", smiles2graph=smiles2graph, transform=None, pre_transform=None):
        self.smiles2graph = smiles2graph
        self.folder = os.path.join(root, "Freesolv")
        self.task_type = "mse_regression"
        self.num_tasks = 1
        self.eval_metric = "rmse"
        self.root = root
        
        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv"

        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "SAMPL.csv"

    @property
    def processed_file_names(self):
        return "freesolv_processed.pt"

    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        return file_path

    def process(self):
        print(f"Start Processing FreeSolv Dataset in {self.raw_dir} ...")
        
        data_df = pd.read_csv(os.path.join(self.raw_dir, "SAMPL.csv"))
        smiles_list = data_df["smiles"]
        
        target_col = "expt"
        if target_col not in data_df.columns:
            print(f"Warning: '{target_col}' column not found in FreeSolv. Columns are: {data_df.columns}")
            target_col = data_df.columns[-1]
            
        labels = data_df[target_col]
        
        cutoff = 3.5
        data_list = []
        skip_count = 0

        for i in tqdm(range(len(smiles_list)), desc="Processing FreeSolv"):
            smiles = smiles_list[i]
            try:
                y_val = float(labels.iloc[i])
            except ValueError:
                skip_count += 1
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None: 
                skip_count += 1
                continue
            
            n_atoms = mol.GetNumAtoms()
            temp_mol = Chem.AddHs(mol)
            coords = None
            try:
                if temp_mol.GetNumConformers() == 0:
                    res = AllChem.EmbedMolecule(
                        temp_mol, 
                        randomSeed=42, 
                        useRandomCoords=True, 
                        maxAttempts=50
                    )
                    if res < 0: 
                        skip_count += 1
                        continue 
                    AllChem.MMFFOptimizeMolecule(temp_mol, maxIters=50)
                    
                conf = temp_mol.GetConformer()
                coords = np.array([[conf.GetAtomPosition(atom.GetIdx()).x,
                                    conf.GetAtomPosition(atom.GetIdx()).y,
                                    conf.GetAtomPosition(atom.GetIdx()).z]
                                   for atom in mol.GetAtoms()], dtype=np.float32)
            except Exception:
                skip_count += 1
                continue
            
            if coords is None:
                skip_count += 1
                continue
                
            pos = torch.tensor(coords, dtype=torch.float)

            try:
                graph = self.smiles2graph(smiles)
            except:
                skip_count += 1
                continue

            if graph is None or pos.shape[0] != int(graph["num_nodes"]):
                skip_count += 1
                continue

            data = FragData()
            data.__num_node__ = int(graph["num_nodes"])
            data.num_nodes = int(graph["num_nodes"])
            data.edge_index = torch.unique(torch.from_numpy(graph["edge_index"]).to(torch.int64), dim=1)
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            
            data.y = torch.tensor([[y_val]], dtype=torch.float32)
            
            data.coords = pos
            data.pos = pos

            cutoff_edge_index = radius_graph(pos, r=cutoff, batch=None, loop=False)
            data.cutoff_edge_index = cutoff_edge_index
            
            if cutoff_edge_index.size(1) > 0:
                cos_phi, sin2_phi = calculate_torsion_features(pos, cutoff_edge_index)
            else:
                cos_phi = torch.empty((0, 1), dtype=torch.float32)
                sin2_phi = torch.empty((0, 1), dtype=torch.float32)
                
            data.geo_cos_phi = cos_phi
            data.geo_sin2_phi = sin2_phi

            target_edge_index = data.cutoff_edge_index if data.cutoff_edge_index is not None else data.edge_index
            if target_edge_index is not None and target_edge_index.size(1) > 0:
                curv = compute_forman_curvature(target_edge_index, data.num_nodes)
                data.edge_curvature = curv
            else:
                data.edge_curvature = torch.zeros(0, dtype=torch.float)

            try:
                cliques = motif_decomp(smiles)
            except Exception:
                skip_count += 1
                continue

            frag_cliques = process_cliques(cliques, n_atoms)
            n_frags = len(frag_cliques)
            
            frag_sizes = torch.tensor([len(c) for c in frag_cliques], dtype=torch.float32)
            
            atom2frag = torch.full((n_atoms,), -1, dtype=torch.long)
            for f_idx, frag_atoms in enumerate(frag_cliques):
                atom2frag[frag_atoms] = f_idx

            frag_h_list, frag_pos_list = [], []
            for frag_atoms in frag_cliques:
                frag_pos_list.append(coords[frag_atoms].mean(axis=0))
                frag_h_list.append(data.x[frag_atoms].float().mean(dim=0))
            
            if len(frag_pos_list) > 0:
                data.frag_pos = torch.tensor(frag_pos_list, dtype=torch.float32)
                data.frag_h = torch.stack(frag_h_list, dim=0)
            else:
                data.frag_pos = torch.empty((0, 3), dtype=torch.float32)
                data.frag_h = torch.empty((0, data.x.size(1)), dtype=torch.float32)

            edge_lookup = {}
            if hasattr(data, 'edge_curvature'):
                rows, cols = target_edge_index
                curvs = data.edge_curvature
                for k in range(rows.size(0)):
                    u, v = rows[k].item(), cols[k].item()
                    edge_lookup[(u, v)] = curvs[k].item()
                    
            frag_edges = []
            frag_edge_attrs = []
            visited_frag_edges = set()
            
            for bond in mol.GetBonds():
                u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                fu, fv = atom2frag[u], atom2frag[v]
                if fu >= 0 and fv >= 0 and fu != fv:
                    if (fu, fv) not in visited_frag_edges:
                        frag_edges.append((fu, fv))
                        frag_edges.append((fv, fu))
                        
                        c_val = edge_lookup.get((u, v), edge_lookup.get((v, u), 0.0))
                        frag_edge_attrs.append(c_val)
                        frag_edge_attrs.append(c_val)
                        
                        visited_frag_edges.add((fu, fv))
                        visited_frag_edges.add((fv, fu))

            if len(frag_edges) > 0:
                data.frag_edge_index = torch.tensor(frag_edges, dtype=torch.long).t().contiguous()
                data.frag_edge_weight = torch.tensor(frag_edge_attrs, dtype=torch.float32)
            else:
                data.frag_edge_index = torch.empty((2, 0), dtype=torch.long)
                data.frag_edge_weight = torch.empty((0,), dtype=torch.float32)

            data.frag_cliques = frag_cliques
            data.frag_sizes = frag_sizes
            data.num_frags = n_frags
            data.atom2u = atom2frag

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        if len(data_list) == 0:
            print("Error: No data was processed successfully!")
            return

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Processed file saved to: {self.processed_paths[0]}")
        print(f"Total skipped: {skip_count}")
    
    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict 
   

class LipopDataset(InMemoryDataset):
    def __init__(self, root="dataset", smiles2graph=smiles2graph, transform=None, pre_transform=None):
        self.smiles2graph = smiles2graph
        self.folder = os.path.join(root, "Lipop")
        self.task_type = "mse_regression"
        self.num_tasks = 1
        self.eval_metric = "rmse"
        self.root = root
        
        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv"

        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "Lipophilicity.csv"

    @property
    def processed_file_names(self):
        return "lipop_processed.pt"

    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        return file_path

    def process(self):
        print(f"Start Processing Lipop Dataset in {self.raw_dir} ...")
        
        data_df = pd.read_csv(os.path.join(self.raw_dir, "Lipophilicity.csv"))
        smiles_list = data_df["smiles"]
        
        target_col = "exp"
        if target_col not in data_df.columns:
            print(f"Warning: '{target_col}' column not found in Lipop. Columns are: {data_df.columns}")
            target_col = data_df.columns[-1]
            
        labels = data_df[target_col]
        
        cutoff = 3.5
        data_list = []
        skip_count = 0

        for i in tqdm(range(len(smiles_list)), desc="Processing Lipop"):
            smiles = smiles_list[i]
            
            try:
                y_val = float(labels.iloc[i])
            except ValueError:
                skip_count += 1
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None: 
                skip_count += 1
                continue
            
            n_atoms = mol.GetNumAtoms()
            if n_atoms > 150:
                skip_count += 1
                continue

            temp_mol = Chem.AddHs(mol)
            coords = None
            try:
                if temp_mol.GetNumConformers() == 0:
                    res = AllChem.EmbedMolecule(
                        temp_mol, 
                        randomSeed=42, 
                        useRandomCoords=True, 
                        maxAttempts=50
                    )
                    if res < 0: 
                        skip_count += 1
                        continue 
                    AllChem.MMFFOptimizeMolecule(temp_mol, maxIters=50)
                    
                conf = temp_mol.GetConformer()
                coords = np.array([[conf.GetAtomPosition(atom.GetIdx()).x,
                                    conf.GetAtomPosition(atom.GetIdx()).y,
                                    conf.GetAtomPosition(atom.GetIdx()).z]
                                   for atom in mol.GetAtoms()], dtype=np.float32)
            except Exception:
                skip_count += 1
                continue
            
            if coords is None:
                skip_count += 1
                continue
                
            pos = torch.tensor(coords, dtype=torch.float)

            try:
                graph = self.smiles2graph(smiles)
            except:
                skip_count += 1
                continue

            if graph is None or pos.shape[0] != int(graph["num_nodes"]):
                skip_count += 1
                continue

            data = FragData()
            data.__num_node__ = int(graph["num_nodes"])
            data.num_nodes = int(graph["num_nodes"])
            data.edge_index = torch.unique(torch.from_numpy(graph["edge_index"]).to(torch.int64), dim=1)
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            
            data.y = torch.tensor([[y_val]], dtype=torch.float32)
            
            data.coords = pos
            data.pos = pos

            cutoff_edge_index = radius_graph(pos, r=cutoff, batch=None, loop=False)
            data.cutoff_edge_index = cutoff_edge_index
            
            if cutoff_edge_index.size(1) > 0:
                cos_phi, sin2_phi = calculate_torsion_features(pos, cutoff_edge_index)
            else:
                cos_phi = torch.empty((0, 1), dtype=torch.float32)
                sin2_phi = torch.empty((0, 1), dtype=torch.float32)
                
            data.geo_cos_phi = cos_phi
            data.geo_sin2_phi = sin2_phi

            target_edge_index = data.cutoff_edge_index if data.cutoff_edge_index is not None else data.edge_index
            if target_edge_index is not None and target_edge_index.size(1) > 0:
                curv = compute_forman_curvature(target_edge_index, data.num_nodes)
                data.edge_curvature = curv
            else:
                data.edge_curvature = torch.zeros(0, dtype=torch.float)

            try:
                cliques = motif_decomp(smiles)
            except Exception:
                skip_count += 1
                continue

            frag_cliques = process_cliques(cliques, n_atoms)
            n_frags = len(frag_cliques)
            
            frag_sizes = torch.tensor([len(c) for c in frag_cliques], dtype=torch.float32)
            atom2frag = torch.full((n_atoms,), -1, dtype=torch.long)
            for f_idx, frag_atoms in enumerate(frag_cliques):
                atom2frag[frag_atoms] = f_idx

            frag_h_list, frag_pos_list = [], []
            for frag_atoms in frag_cliques:
                frag_pos_list.append(coords[frag_atoms].mean(axis=0))
                frag_h_list.append(data.x[frag_atoms].float().mean(dim=0))
            
            if len(frag_pos_list) > 0:
                data.frag_pos = torch.tensor(frag_pos_list, dtype=torch.float32)
                data.frag_h = torch.stack(frag_h_list, dim=0)
            else:
                data.frag_pos = torch.empty((0, 3), dtype=torch.float32)
                data.frag_h = torch.empty((0, data.x.size(1)), dtype=torch.float32)

            edge_lookup = {}
            if hasattr(data, 'edge_curvature'):
                rows, cols = target_edge_index
                curvs = data.edge_curvature
                for k in range(rows.size(0)):
                    u, v = rows[k].item(), cols[k].item()
                    edge_lookup[(u, v)] = curvs[k].item()
                    
            frag_edges = []
            frag_edge_attrs = []
            visited_frag_edges = set()
            
            for bond in mol.GetBonds():
                u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                fu, fv = atom2frag[u], atom2frag[v]
                if fu >= 0 and fv >= 0 and fu != fv:
                    if (fu, fv) not in visited_frag_edges:
                        frag_edges.append((fu, fv))
                        frag_edges.append((fv, fu))
                        
                        c_val = edge_lookup.get((u, v), edge_lookup.get((v, u), 0.0))
                        frag_edge_attrs.append(c_val)
                        frag_edge_attrs.append(c_val)
                        
                        visited_frag_edges.add((fu, fv))
                        visited_frag_edges.add((fv, fu))

            if len(frag_edges) > 0:
                data.frag_edge_index = torch.tensor(frag_edges, dtype=torch.long).t().contiguous()
                data.frag_edge_weight = torch.tensor(frag_edge_attrs, dtype=torch.float32)
            else:
                data.frag_edge_index = torch.empty((2, 0), dtype=torch.long)
                data.frag_edge_weight = torch.empty((0,), dtype=torch.float32)

            data.frag_cliques = frag_cliques
            data.frag_sizes = frag_sizes
            data.num_frags = n_frags
            data.atom2u = atom2frag

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        if len(data_list) == 0:
            print("Error: No data was processed successfully!")
            return

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Processed file saved to: {self.processed_paths[0]}")
        print(f"Total skipped: {skip_count}")
    
    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict

def get_dataset(dataset, output_dir="./"):
    print(f"Preprocessing {dataset}".upper())
    if not os.path.exists(os.path.join(output_dir, "datasets")):
        os.makedirs(os.path.join(output_dir, "datasets"))
    
    root = os.path.join(output_dir, "datasets", dataset)

    if dataset == "BBBP":
        data = BBBPDataset(root=root, pre_transform=None)
    elif dataset == "BBBP_cutoff":
        data = BBBPDataset_cutoff(root=root, pre_transform=None)
    elif dataset == "Esol":
        data = EsolDataset(root=root, pre_transform=None)
    elif dataset == "Freesolv":
        data = FreesolvDataset(root=root, pre_transform=None)
    elif dataset == "Lipop":
        data = LipopDataset(root=root, pre_transform=None)
    elif dataset == "qm9":
        data = Qm9dataset(root=root, pre_transform=None)
    elif dataset == "Bace":
        data = BaceDataset(root=root, pre_transform=None)
    elif dataset == "Tox21":
        data = Tox21Dataset(root=root, pre_transform=None)
    elif dataset == "HIV":
        data = HIVDataset(root=root, pre_transform=None)
    elif dataset == "Sider":
        data = SiderDataset(root=root, pre_transform=None)
    elif dataset == "ClinTox":
        data = ClinToxDataset(root=root, pre_transform=None)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return data