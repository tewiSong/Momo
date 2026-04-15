#!/usr/bin/env python

# momo/tools/preprocess_pcqm4mv2.py
import os
import sys
import argparse
import pickle
from typing import List, Tuple, Dict, Optional

import numpy as np
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import BRICS, AllChem
import torch


def _ensure_conformer(mol: Chem.Mol) -> Chem.Conformer:
    if mol is None:
        raise RuntimeError("Molecule is None")
    if mol.GetNumConformers() == 0:
        raise RuntimeError("No conformer present (no 3D coordinates)")
    return mol.GetConformer()


def _fragment_motifs_brics(mol: Chem.Mol) -> List[Tuple[int, ...]]:
    """Decompose a molecule into BRICS fragments and return a tuple of atom index tuples per motif.

    We find bonds to break via BRICS, then fragment on those bonds without adding dummies,
    and collect connected components (fragments) as atom index tuples.
    """
    # Identify BRICS bonds (pairs of atom indices)
    brics_bonds = list(BRICS.FindBRICSBonds(mol))  # yields ((u, v), (u_label, v_label))
    if not brics_bonds:
        # No BRICS cut; return one fragment containing all atoms
        return [tuple(range(mol.GetNumAtoms()))]

    # Map atom pairs to bond indices to cut
    pair_set = {tuple(sorted(pair)) for pair, _ in brics_bonds}
    bond_indices_to_cut: List[int] = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if tuple(sorted((i, j))) in pair_set:
            bond_indices_to_cut.append(b.GetIdx())

    if not bond_indices_to_cut:
        return [tuple(range(mol.GetNumAtoms()))]

    # Strict: fragment on identified bonds without dummies; raise if this fails
    fragged = Chem.FragmentOnBonds(mol, bondIndices=bond_indices_to_cut, addDummies=False)

    frags = Chem.GetMolFrags(fragged, asMols=False, sanitizeFrags=False)
    # RDKit returns a tuple of tuples of atom indices
    frags_list = list(frags)
    return frags_list


def _motif_batch_from_frags(frags: List[Tuple[int, ...]], n_atoms: int) -> np.ndarray:
    """Build per-atom motif ids; requires full coverage by frags.

    This function assumes every atom is included in exactly one fragment.
    """
    motif_batch = np.full((n_atoms,), -1, dtype=np.int32)
    for m_id, atoms in enumerate(frags):
        for a in atoms:
            motif_batch[a] = m_id
    # Fail fast if any atom wasn't assigned (should not happen since frags was fixed to cover all atoms)
    if (motif_batch < 0).any():
        missing = np.where(motif_batch < 0)[0].tolist()
        raise RuntimeError(f"Fragments do not cover all atoms; missing atom indices: {missing}")
    return motif_batch


def _compute_bond_lengths(mol: Chem.Mol, conf: Chem.Conformer, atom_set: set) -> np.ndarray:
    vals = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in atom_set and j in atom_set:
            pi = np.array(conf.GetAtomPosition(i))
            pj = np.array(conf.GetAtomPosition(j))
            vals.append(np.linalg.norm(pi - pj))
    return np.asarray(vals, dtype=float)


def _compute_bond_angles(mol: Chem.Mol, conf: Chem.Conformer, atom_set: set) -> np.ndarray:
    angles = []
    for center in atom_set:
        nbrs = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(center).GetNeighbors() if nbr.GetIdx() in atom_set]
        # All unordered pairs of neighbors
        for i_idx in range(len(nbrs)):
            for j_idx in range(i_idx + 1, len(nbrs)):
                i = nbrs[i_idx]
                j = nbrs[j_idx]
                pi = np.array(conf.GetAtomPosition(i))
                pc = np.array(conf.GetAtomPosition(center))
                pj = np.array(conf.GetAtomPosition(j))
                v1 = pi - pc
                v2 = pj - pc
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 < 1e-8 or n2 < 1e-8:
                    continue
                cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                ang = np.arccos(cosang)  # radians
                angles.append(ang)
    return np.asarray(angles, dtype=float)


def _compute_inertia_eigs(mol: Chem.Mol, conf: Chem.Conformer, atom_indices: List[int]) -> np.ndarray:
    if len(atom_indices) == 0:
        return np.zeros(3, dtype=float)
    coords = np.array([list(conf.GetAtomPosition(i)) for i in atom_indices], dtype=float)
    masses = np.array([Chem.GetPeriodicTable().GetAtomicWeight(mol.GetAtomWithIdx(i).GetAtomicNum()) for i in atom_indices], dtype=float)
    # Center of mass
    total_mass = masses.sum() if masses.sum() > 0 else 1.0
    com = (coords * masses[:, None]).sum(axis=0) / total_mass
    rel = coords - com[None, :]
    I = np.zeros((3, 3), dtype=float)
    for r, m in zip(rel, masses):
        x, y, z = r
        r2 = np.dot(r, r)
        I += m * (r2 * np.eye(3) - np.outer(r, r))
    # Eigenvalues (principal moments), sorted ascending for consistency
    w = np.linalg.eigvalsh(I)
    w = np.sort(np.real(w))
    return w


def _principal_axes(coords: np.ndarray) -> np.ndarray:
    """Return a 3x3 orthonormal basis (rows) as principal axes for coords (N,3).

    Uses covariance eigendecomposition to always produce a 3x3 matrix, even when N < 3
    or points are collinear/degenerate. Ensures a right‑handed frame.
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise RuntimeError("coords must be [N,3]")
    c = coords.mean(axis=0, keepdims=True)
    X = coords - c  # [N,3]
    C = X.T @ X  # [3,3]
    # eigh returns eigenvalues ascending and column‑stacked eigenvectors
    w, V = np.linalg.eigh(C)
    # Orthonormal basis rows
    R = V.T  # [3,3]
    # Enforce right‑handedness
    if np.linalg.det(R) < 0:
        R[2, :] *= -1.0
    return R


def _bond_type_onehot(b: Chem.Bond) -> List[float]:
    bt = b.GetBondType()
    # SINGLE, DOUBLE, TRIPLE, AROMATIC
    vals = [0.0, 0.0, 0.0, 0.0]
    if bt == Chem.BondType.SINGLE:
        vals[0] = 1.0
    elif bt == Chem.BondType.DOUBLE:
        vals[1] = 1.0
    elif bt == Chem.BondType.TRIPLE:
        vals[2] = 1.0
    elif bt == Chem.BondType.AROMATIC:
        vals[3] = 1.0
    else:
        # Map other types to SINGLE bucket deterministically（不引入容错，只按固定规则编码）
        vals[0] = 1.0
    return vals


def _build_motif_graph_and_targets(
    mol: Chem.Mol,
    conf: Chem.Conformer,
    heavy_idx: List[int],
    motif_id_heavy: np.ndarray,
    frags_heavy: List[Tuple[int, ...]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct motif-level graph and build targets per motif.

    同时返回 motif 内每个重原子的局部坐标（atom_local_coord），目前仅作调试/兼容保留，
    不再用于离线 template 聚类或训练标签。
    逐边 attachment 目标：每条有向 motif 边生成 7 维几何向量（外键方向3+邻居方向3+扭转1）。

    Returns:
      motif_edge_index        [2, E]
      motif_edge_attr         [E, 8]  (4 bond onehot + 2 aromatic flags + 2 rank_norm)
      atom_local_coord        [N_heavy, 3]
      motif_edge_target       [E, 7]  (per directed edge attachment target)
      motif_edge_attach_pos_src [E]   (local index of source atom within src motif; -1 if unknown)
      motif_edge_attach_pos_dst [E]   (local index of dest atom within dst motif; -1 if unknown)
    """
    # Canonical ranks for normalization (on original atoms)
    ranks = list(Chem.CanonicalRankAtoms(mol))
    rank_map = {old: ranks[old] for old in heavy_idx}
    rank_max = float(max(rank_map.values()) if rank_map else 1.0)

    # Helper: positions array aligned with heavy_idx contiguous order
    coords = np.array([list(conf.GetAtomPosition(i)) for i in heavy_idx], dtype=float)
    M = int(len(frags_heavy))

    # Build motif-level edges from inter-motif bonds among heavy atoms
    rows: List[int] = []
    cols: List[int] = []
    eattrs: List[List[float]] = []
    edge_src_atom: List[int] = []
    edge_dst_atom: List[int] = []
    # Map heavy atom original id -> contiguous index
    index_map = {old: new for new, old in enumerate(heavy_idx)}
    # For attachment vectors per motif, collect per-neighbor edge info
    motif_attach_info: Dict[int, List[Tuple[int, int, int]]] = {m: [] for m in range(M)}  # m -> list of (a_heavy, b_heavy, m_nbr)

    for b in mol.GetBonds():
        i0, j0 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i0 not in index_map or j0 not in index_map:
            continue
        ii = index_map[i0]
        jj = index_map[j0]
        mu = int(motif_id_heavy[ii])
        mv = int(motif_id_heavy[jj])
        if mu == mv:
            continue
        # Edge mu -> mv and mv -> mu
        for s_old, t_old, s_idx, t_idx, ms, mt in (
            (i0, j0, ii, jj, mu, mv),
            (j0, i0, jj, ii, mv, mu),
        ):
            rows.append(ms)
            cols.append(mt)
            # Edge attributes: bond type onehot + src/dst aromatic + src/dst canonical rank norm
            bt = _bond_type_onehot(b)
            src_arom = 1.0 if mol.GetAtomWithIdx(s_old).GetIsAromatic() else 0.0
            dst_arom = 1.0 if mol.GetAtomWithIdx(t_old).GetIsAromatic() else 0.0
            src_rank = rank_map[s_old] / max(rank_max, 1.0)
            dst_rank = rank_map[t_old] / max(rank_max, 1.0)
            eattrs.append(bt + [src_arom, dst_arom, float(src_rank), float(dst_rank)])
            motif_attach_info[ms].append((s_old, t_old, mt))
            edge_src_atom.append(s_old)
            edge_dst_atom.append(t_old)

    motif_edge_index = np.asarray([rows, cols], dtype=np.int64) if rows else np.zeros((2, 0), dtype=np.int64)
    motif_edge_attr = np.asarray(eattrs, dtype=float) if eattrs else np.zeros((0, 8), dtype=float)

    # Build local coordinates per atom in motif frames
    atom_local_coord = np.zeros((len(heavy_idx), 3), dtype=float)
    # no per‑motif slot matrices; compute per‑edge attach positions instead

    # 先计算每个 motif 的局部坐标系（R、中心），并填充 atom_local_coord
    R_list: List[np.ndarray] = [np.eye(3, dtype=float) for _ in range(M)]
    center_list: List[np.ndarray] = [np.zeros(3, dtype=float) for _ in range(M)]
    for m_id, atoms in enumerate(frags_heavy):
        # coords in this motif (heavy-atom contiguous indices)
        atom_ids = [index_map[a] for a in atoms]
        pts = coords[atom_ids]  # [Nm,3]
        # principal axes
        R = _principal_axes(pts)  # [3,3]
        center = pts.mean(axis=0)
        R_list[m_id] = R
        center_list[m_id] = center
        # fill atom local coords
        for a in atoms:
            pid = index_map[a]
            v = coords[pid] - center
            atom_local_coord[pid] = R @ v
    # Edge-level targets aligned with motif_edge_index order
    E = motif_edge_index.shape[1]
    edge_target = np.zeros((E, 7), dtype=float)
    edge_attach_pos_src = -np.ones((E,), dtype=np.int64)
    edge_attach_pos_dst = -np.ones((E,), dtype=np.int64)
    for k in range(E):
        ms = int(rows[k])
        mt = int(cols[k])
        a_old = int(edge_src_atom[k])
        b_old = int(edge_dst_atom[k])
        a_pos = np.array(conf.GetAtomPosition(a_old), dtype=float)
        b_pos = np.array(conf.GetAtomPosition(b_old), dtype=float)
        d_ext = b_pos - a_pos
        # neighbor motif centroid in global coords
        nb_atoms = [index_map[x] for x in frags_heavy[mt]]
        nb_center = coords[nb_atoms].mean(axis=0)
        d_nb = nb_center - a_pos
        # torsion sign via normals of src/dst motifs
        R = R_list[ms]
        R_nb = R_list[mt]
        n_self = R[2, :] / (np.linalg.norm(R[2, :]) + 1e-12)
        n_nb = R_nb[2, :] / (np.linalg.norm(R_nb[2, :]) + 1e-12)
        axis = d_ext / (np.linalg.norm(d_ext) + 1e-12)
        cross = np.cross(n_self, n_nb)
        sinv = float(np.dot(axis, cross))
        cosv = float(np.dot(n_self, n_nb))
        phi = float(np.arctan2(sinv, cosv))
        # project into src motif frame
        d_ext_loc = R @ d_ext
        d_nb_loc = R @ d_nb
        edge_target[k, :] = np.concatenate([d_ext_loc, d_nb_loc, np.asarray([phi], dtype=float)], axis=0)
        # per-edge attach positions (src within ms, dst within mt)
        atoms_src = frags_heavy[ms]
        atoms_dst = frags_heavy[mt]
        if a_old in atoms_src:
            edge_attach_pos_src[k] = int(atoms_src.index(a_old))
        else:
            import warnings
            warnings.warn(f"attachment atom {a_old} not in motif {ms}; set edge_attach_pos_src=-1")
            edge_attach_pos_src[k] = -1
        if b_old in atoms_dst:
            edge_attach_pos_dst[k] = int(atoms_dst.index(b_old))
        else:
            import warnings
            warnings.warn(f"attachment atom {b_old} not in motif {mt}; set edge_attach_pos_dst=-1")
            edge_attach_pos_dst[k] = -1
    return motif_edge_index, motif_edge_attr, atom_local_coord, edge_target, edge_attach_pos_src, edge_attach_pos_dst


def _motif_3d_features(mol: Chem.Mol, conf: Chem.Conformer, atoms: List[int]) -> np.ndarray:
    atom_set = set(atoms)
    bl = _compute_bond_lengths(mol, conf, atom_set)
    ba = _compute_bond_angles(mol, conf, atom_set)
    eigs = _compute_inertia_eigs(mol, conf, atoms)

    def stats(x: np.ndarray) -> Tuple[float, float]:
        if x.size == 0:
            return 0.0, 0.0
        if x.size == 1:
            return float(x[0]), 0.0
        return float(x.mean()), float(x.var())

    bl_mean, bl_var = stats(bl)
    ba_mean, ba_var = stats(ba)
    feat = np.array([bl_mean, bl_var, ba_mean, ba_var, eigs[0], eigs[1], eigs[2]], dtype=float)
    return feat


def _atom_numbers(mol: Chem.Mol, heavy_idx: List[int]) -> np.ndarray:
    z = []
    for i in heavy_idx:
        z.append(mol.GetAtomWithIdx(i).GetAtomicNum())
    return np.asarray(z, dtype=np.int64)


def _edge_index_undirected(mol: Chem.Mol, heavy_idx: List[int]) -> np.ndarray:
    index_map = {old: new for new, old in enumerate(heavy_idx)}
    rows, cols = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in index_map and j in index_map:
            ii = index_map[i]
            jj = index_map[j]
            rows += [ii, jj]
            cols += [jj, ii]
    return np.asarray([rows, cols], dtype=np.int64)


def process_sdf(
    input_sdf: str,
    output_path: str,
    limit: int = -1,
) -> None:
    if not os.path.exists(input_sdf):
        raise FileNotFoundError(f"Input SDF not found: {input_sdf}")

    supplier = Chem.SDMolSupplier(input_sdf, removeHs=False, sanitize=True)
    results: List[Dict] = []
    # Motif 类型词表（跨分子一致）
    motif_type_vocab: Dict[str, int] = {}

    for idx in tqdm(range(len(supplier)), desc="Processing molecules"):
        if limit >= 0 and len(results) >= limit:
            break
        mol = supplier[idx]
        if mol is None:
            raise RuntimeError(f"SDF molecule at index {idx} failed to parse")
        conf = _ensure_conformer(mol)

        # Heavy atom indices (Z>1)
        heavy_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
        if not heavy_idx:
            raise RuntimeError(f"Molecule at index {idx} has no heavy atoms")

        # BRICS-based motifs on original mol, then filter to heavy atoms per fragment
        frags_all = _fragment_motifs_brics(mol)
        frags_heavy: List[Tuple[int, ...]] = []
        heavy_set = set(heavy_idx)
        for tup in frags_all:
            sub = tuple(i for i in tup if i in heavy_set)
            if len(sub) > 0:
                frags_heavy.append(sub)

        # Validate coverage and non-overlap across heavy atoms
        assigned_heavy = [a for tup in frags_heavy for a in tup]
        if len(set(assigned_heavy)) != len(assigned_heavy):
            raise RuntimeError("Heavy-atom fragments overlap on atoms")
        if set(assigned_heavy) != heavy_set:
            missing = sorted(heavy_set - set(assigned_heavy))
            raise RuntimeError(f"Fragments do not cover all heavy atoms; missing {missing}")

        # Map heavy atom ids to contiguous [0..n-1]
        index_map = {old: new for new, old in enumerate(heavy_idx)}
        frags_heavy_mapped = [tuple(index_map[i] for i in tup) for tup in frags_heavy]

        motif_id = _motif_batch_from_frags(frags_heavy_mapped, n_atoms=len(heavy_idx))
        # Build motif-level graph and edge-level geometry targets
        motif_edge_index, motif_edge_attr, _atom_local_coord, edge_target, edge_attach_pos_src, edge_attach_pos_dst = _build_motif_graph_and_targets(
            mol, conf, heavy_idx, motif_id, frags_heavy
        )
        # 计算每个 motif 的类型 id（基于子图 canonical SMILES）
        from rdkit.Chem.rdmolfiles import MolFragmentToSmiles
        motif_type_ids: List[int] = []
        for atoms in frags_heavy:
            smi = MolFragmentToSmiles(mol, atoms, canonical=True, isomericSmiles=True)
            if smi not in motif_type_vocab:
                motif_type_vocab[smi] = len(motif_type_vocab) + 1
            motif_type_ids.append(motif_type_vocab[smi])

        # Precompute 2D heavy-atom graph to avoid RDKit in training
        z = _atom_numbers(mol, heavy_idx)
        edge_index = _edge_index_undirected(mol, heavy_idx)
        # Save heavy-atom 3D coordinates aligned with contiguous indexing
        coords = np.array([list(conf.GetAtomPosition(i)) for i in heavy_idx], dtype=float)

        # Optional SMILES for reference
        smi = Chem.MolToSmiles(mol)

        rec = {
            "smiles": smi,
            "num_atoms": int(len(heavy_idx)),
            "num_motifs": int(len(frags_heavy_mapped)),
            "motif_id": motif_id.astype(int).tolist(),
            "atom_local_coord": _atom_local_coord.astype(float).tolist(),
            # 边级目标：每条有向 motif 边一个 7 维 attachment 向量
            "motif_edge_target": edge_target.astype(float).tolist(),
            # motif 级图
            "motif_edge_index": motif_edge_index.astype(int).tolist(),
            "motif_edge_attr": motif_edge_attr.astype(float).tolist(),
            "motif_edge_neighbor_type": [motif_type_ids[j] for j in motif_edge_index[1].tolist()],
            "motif_type_id": motif_type_ids,
            "motif_edge_attach_pos_src": edge_attach_pos_src.astype(int).tolist(),
            "motif_edge_attach_pos_dst": edge_attach_pos_dst.astype(int).tolist(),
            "z": z.astype(int).tolist(),
            "edge_index": edge_index.astype(int).tolist(),
            "pos": coords.astype(float).tolist(),
        }
        results.append(rec)

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Processed molecules: {len(results)}")
    print(f"Saved to: {output_path}")


# ---------------- MoleculeNet support below (inline in this file) ---------------- #

def _embed_smiles_to_3d(
    smiles: str,
    max_attempts: int = 5,
    random_seed: int = 0,
    optimize: bool = True,
) -> Optional[Chem.Mol]:
    """Embed a 3D conformer for a SMILES string using RDKit ETKDGv3.

    Returns a molecule with at least one conformer on success; otherwise None.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol, addCoords=False)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)
    params.useSmallRingTorsions = True
    params.useRandomCoords = False

    for _ in range(max_attempts):
        res = AllChem.EmbedMolecule(mol, params)
        if res == 0:
            if optimize:
                try:
                    if AllChem.MMFFHasAllMoleculeParams(mol):
                        AllChem.MMFFOptimizeMolecule(mol)
                    else:
                        AllChem.UFFOptimizeMolecule(mol)
                except Exception:
                    pass
            mol = Chem.RemoveHs(mol)
            return mol
        params.randomSeed += 17
        params.useRandomCoords = True

    # last fallback: no hydrogens
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed) + 123
    params.useRandomCoords = True
    res = AllChem.EmbedMolecule(mol, params)
    if res == 0:
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol)
            else:
                AllChem.UFFOptimizeMolecule(mol)
        except Exception:
            pass
        return mol
    return None


def _read_smiles_csv(path: str) -> List[str]:
    smiles_list: List[str] = []
    with open(path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if ',' in s:
                s = s.split(',')[0]
            smiles_list.append(s)
    return smiles_list


def _load_y_from_pt(ds_dir: str) -> Optional[List[List[float]]]:
    """Load per-sample labels from GraphMVP/SNAP processed .pt in a way that is robust to
    older/newer PyG Data layouts. Returns None if labels not available.
    """
    import warnings
    pt_path = os.path.join(ds_dir, 'processed', 'geometric_data_processed.pt')
    if not os.path.exists(pt_path):
        return None
    try:
        obj = torch.load(pt_path, map_location='cpu')
    except Exception as e:
        warnings.warn(f"failed to load labels from {pt_path}: {e}")
        return None
    if not isinstance(obj, (tuple, list)) or len(obj) != 2:
        warnings.warn("unexpected PT format (expect (data, slices)); skipping labels")
        return None
    data, slices = obj
    if not isinstance(slices, dict) or 'y' not in slices:
        warnings.warn("no 'y' in slices; skipping labels")
        return None
    # Avoid triggering Data.__getattr__; read raw storage if present
    try:
        y_storage = getattr(data, '__dict__', {}).get('y', None)
    except Exception:
        y_storage = None
    if y_storage is None or not torch.is_tensor(y_storage):
        warnings.warn("could not access raw 'y' storage; skipping labels")
        return None
    idx = slices['y']
    n = int(idx.numel() - 1) if torch.is_tensor(idx) else (len(idx) - 1)
    ys: List[List[float]] = []
    for i in range(n):
        s = int(idx[i].item() if torch.is_tensor(idx) else idx[i])
        e = int(idx[i + 1].item() if torch.is_tensor(idx) else idx[i + 1])
        y_i = y_storage[s:e]
        y_i = y_i.view(-1).cpu().numpy().tolist()
        ys.append([float(v) for v in y_i])
    return ys


def process_moleculenet(
    root: str,
    dataset: str,
    out_path: Optional[str] = None,
    limit: int = -1,
    attach_y: bool = False,
) -> None:
    ds_dir = os.path.join(root, dataset)
    smiles_path = os.path.join(ds_dir, 'processed', 'smiles.csv')
    if not os.path.exists(smiles_path):
        raise FileNotFoundError(f"smiles.csv not found: {smiles_path}")
    if out_path is None:
        out_path = os.path.join(ds_dir, f"{dataset}_motif_preprocessed.pkl")

    smiles_list = _read_smiles_csv(smiles_path)
    y_list: Optional[List[List[float]]] = _load_y_from_pt(ds_dir) if attach_y else None

    results: List[Dict] = []
    failures: List[int] = []
    # 独立的 motif 类型词表（MoleculeNet 分支本地维护）
    motif_type_vocab: Dict[str, int] = {}
    for idx, smi in enumerate(tqdm(smiles_list, desc=f"{dataset}")):
        if limit >= 0 and len(results) >= limit:
            break
        try:
            mol = _embed_smiles_to_3d(smi, max_attempts=6, random_seed=idx)
            if mol is None:
                failures.append(idx)
                continue
            conf = mol.GetConformer()
            # heavy atoms only (Z>1)
            heavy_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
            if not heavy_idx:
                failures.append(idx)
                continue

            frags_all = _fragment_motifs_brics(mol)
            heavy_set = set(heavy_idx)
            frags_heavy: List[Tuple[int, ...]] = []
            for tup in frags_all:
                sub = tuple(i for i in tup if i in heavy_set)
                if len(sub) > 0:
                    frags_heavy.append(sub)

            assigned_heavy = [a for tup in frags_heavy for a in tup]
            if len(set(assigned_heavy)) != len(assigned_heavy) or set(assigned_heavy) != heavy_set:
                # fallback to single motif if coverage/overlap bad
                frags_heavy = [tuple(heavy_idx)]

            index_map = {old: new for new, old in enumerate(heavy_idx)}
            frags_heavy_mapped = [tuple(index_map[i] for i in tup) for tup in frags_heavy]

            motif_id = _motif_batch_from_frags(frags_heavy_mapped, n_atoms=len(heavy_idx))
            # 统一与 PCQM 预处理：构建 motif 图和边级目标
            motif_edge_index, motif_edge_attr, _atom_local_coord, edge_target, edge_attach_pos_src, edge_attach_pos_dst = _build_motif_graph_and_targets(
                mol, conf, heavy_idx, motif_id, frags_heavy
            )
            # 邻居 motif 类型 id（基于 fragment canonical SMILES，与 PCQM 分支一致）
            from rdkit.Chem.rdmolfiles import MolFragmentToSmiles
            motif_type_ids: List[int] = []
            for atoms in frags_heavy:
                smi = MolFragmentToSmiles(mol, atoms, canonical=True, isomericSmiles=True)
                if smi not in motif_type_vocab:
                    motif_type_vocab[smi] = len(motif_type_vocab) + 1
                motif_type_ids.append(motif_type_vocab[smi])

            z = _atom_numbers(mol, heavy_idx)
            edge_index = _edge_index_undirected(mol, heavy_idx)
            coords = np.array([list(conf.GetAtomPosition(i)) for i in heavy_idx], dtype=float)

            rec = {
                "smiles": smi,
                "orig_idx": int(idx),
                "num_atoms": int(len(heavy_idx)),
                "num_motifs": int(len(frags_heavy_mapped)),
                "motif_id": motif_id.astype(int).tolist(),
                "atom_local_coord": _atom_local_coord.astype(float).tolist(),
                "motif_edge_target": edge_target.astype(float).tolist(),
                # motif 级图
                "motif_edge_index": motif_edge_index.astype(int).tolist(),
                "motif_edge_attr": motif_edge_attr.astype(float).tolist(),
                "motif_edge_neighbor_type": [motif_type_ids[j] for j in motif_edge_index[1].tolist()],
                "motif_type_id": motif_type_ids,
                "motif_edge_attach_pos_src": edge_attach_pos_src.astype(int).tolist(),
                "motif_edge_attach_pos_dst": edge_attach_pos_dst.astype(int).tolist(),
                # 原子级图
                "z": z.astype(int).tolist(),
                "edge_index": edge_index.astype(int).tolist(),
                "pos": coords.astype(float).tolist(),
            }
            if y_list is not None and idx < len(y_list):
                rec["y"] = y_list[idx]
            results.append(rec)
        except Exception:
            failures.append(idx)
            continue

    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    with open(out_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Dataset: {dataset}")
    print(f"Total SMILES: {len(smiles_list)}")
    print(f"Processed: {len(results)}")
    print(f"Failed: {len(failures)}")
    if failures:
        print(f"Failed indices (first 50): {failures[:50]}{' ...' if len(failures) > 50 else ''}")
    print(f"Saved to: {out_path}")


# split generation aligned with GraphMVP
from collections import defaultdict
from rdkit.Chem.Scaffolds import MurckoScaffold


def _generate_scaffold(smiles: str, include_chirality: bool = True) -> str:
    return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)


def scaffold_split_indices(smiles_list: List[str], frac_train: float, frac_valid: float, frac_test: float):
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    scaffold_to_indices = {}
    for i, smi in enumerate(smiles_list):
        scaf = _generate_scaffold(smi, include_chirality=True)
        scaffold_to_indices.setdefault(scaf, []).append(i)
    all_sets = [
        sorted(v) for (k, v) in sorted(scaffold_to_indices.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    n = len(smiles_list)
    train_cut = frac_train * n
    valid_cut = (frac_train + frac_valid) * n
    train_idx: List[int] = []
    valid_idx: List[int] = []
    test_idx: List[int] = []
    for sset in all_sets:
        if len(train_idx) + len(sset) > train_cut:
            if len(train_idx) + len(valid_idx) + len(sset) > valid_cut:
                test_idx.extend(sset)
            else:
                valid_idx.extend(sset)
        else:
            train_idx.extend(sset)
    return train_idx, valid_idx, test_idx


def random_scaffold_split_indices(smiles_list: List[str], frac_train: float, frac_valid: float, frac_test: float, seed: int):
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    scaffolds = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        scaf = _generate_scaffold(smi, include_chirality=True)
        scaffolds[scaf].append(i)
    rng = np.random.RandomState(seed)
    scaffold_sets = list(scaffolds.values())
    rng.shuffle(scaffold_sets)

    n_total = len(smiles_list)
    n_valid = int(np.floor(frac_valid * n_total))
    n_test = int(np.floor(frac_test * n_total))
    train_idx: List[int] = []
    valid_idx: List[int] = []
    test_idx: List[int] = []
    for sset in scaffold_sets:
        if len(valid_idx) + len(sset) <= n_valid:
            valid_idx.extend(sset)
        elif len(test_idx) + len(sset) <= n_test:
            test_idx.extend(sset)
        else:
            train_idx.extend(sset)
    return train_idx, valid_idx, test_idx


def random_split_indices(n: int, frac_train: float, frac_valid: float, frac_test: float, seed: int):
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    rng = np.random.RandomState(seed)
    all_idx = np.arange(n)
    rng.shuffle(all_idx)
    n_train = int(frac_train * n)
    n_valid = int(frac_valid * n)
    train_idx = all_idx[:n_train].tolist()
    valid_idx = all_idx[n_train:n_train + n_valid].tolist()
    test_idx = all_idx[n_train + n_valid:].tolist()
    return train_idx, valid_idx, test_idx


def write_split_files(base_dir: str, mode: str, seed: int, tr: List[int], va: List[int], te: List[int]) -> str:
    if mode == 'scaffold':
        out_dir = os.path.join(base_dir, 'splits', mode)
    else:
        out_dir = os.path.join(base_dir, 'splits', f"{mode}_seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    for name, idx in [('train.txt', tr), ('val.txt', va), ('test.txt', te)]:
        with open(os.path.join(out_dir, name), 'w') as f:
            for i in idx:
                f.write(str(int(i)) + '\n')
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="PCQM4Mv2 & MoleculeNet motif-aware preprocessing")
    # PCQM4Mv2 mode
    parser.add_argument("--sdf", type=str, default="", help="Path to PCQM4Mv2 train SDF (enable PCQM mode if set)")
    parser.add_argument("--out", type=str, default="", help="Output pickle path (PCQM mode). Default: data/pcqm4m_v2_motif_preprocessed.pkl")
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of molecules to process (-1 for all)")

    # MoleculeNet mode
    parser.add_argument("--molnet-root", type=str, default="", help="Path to MoleculeNet root (contains dataset subfolders)")
    parser.add_argument("--molnet-dataset", type=str, default="", help="Single dataset name (e.g., bbbp)")
    parser.add_argument("--molnet-list", type=str, default="", help="Comma-separated dataset list (e.g., 'bbbp,esol')")
    parser.add_argument("--molnet-out", type=str, default="", help="Output path for single dataset; default to {root}/{dataset}/{dataset}_motif_preprocessed.pkl")
    parser.add_argument("--molnet-attach-y", action='store_true', help="Attach labels from processed/geometric_data_processed.pt if available")

    # Split generation (MoleculeNet)
    parser.add_argument("--gen-splits", action='store_true', help="Generate split files for MoleculeNet datasets")
    parser.add_argument("--split-mode", type=str, default='scaffold', choices=['scaffold', 'random', 'random_scaffold'])
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--frac-train", type=float, default=0.8)
    parser.add_argument("--frac-valid", type=float, default=0.1)
    parser.add_argument("--frac-test", type=float, default=0.1)
    args = parser.parse_args()

    ran_any = False

    # PCQM4Mv2 mode
    if args.sdf:
        out = args.out or "data/pcqm4m_v2_motif_preprocessed.pkl"
        try:
            process_sdf(
                args.sdf,
                out,
                limit=args.limit,
            )
        except FileNotFoundError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)
        ran_any = True

    # MoleculeNet mode
    if args.molnet_root and (args.molnet_dataset or args.molnet_list):
        datasets: List[str] = []
        if args.molnet_dataset:
            datasets = [args.molnet_dataset]
        else:
            datasets = [s.strip() for s in args.molnet_list.split(',') if s.strip()]
        for ds in datasets:
            out_path = args.molnet_out or os.path.join(args.molnet_root, ds, f"{ds}_motif_preprocessed.pkl")
            process_moleculenet(
                args.molnet_root,
                ds,
                out_path=out_path,
                limit=args.limit,
                attach_y=bool(args.molnet_attach_y),
            )
        ran_any = True

        if args.gen_splits:
            for ds in datasets:
                ds_dir = os.path.join(args.molnet_root, ds)
                smiles_path = os.path.join(ds_dir, 'processed', 'smiles.csv')
                if not os.path.exists(smiles_path):
                    print(f"Skip splits for {ds}: missing {smiles_path}")
                    continue
                smiles = _read_smiles_csv(smiles_path)
                if args.split_mode == 'scaffold':
                    tr, va, te = scaffold_split_indices(smiles, args.frac_train, args.frac_valid, args.frac_test)
                elif args.split_mode == 'random_scaffold':
                    tr, va, te = random_scaffold_split_indices(smiles, args.frac_train, args.frac_valid, args.frac_test, args.split_seed)
                else:
                    tr, va, te = random_split_indices(len(smiles), args.frac_train, args.frac_valid, args.frac_test, args.split_seed)
                out_dir = write_split_files(ds_dir, args.split_mode, args.split_seed, tr, va, te)
                print(f"{ds}: wrote splits to {out_dir} (train={len(tr)}, val={len(va)}, test={len(te)})")

    if not ran_any:
        parser.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
