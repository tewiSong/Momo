#!/usr/bin/env python
"""
Preprocess MoleculeNet datasets into Momo's motif-aware pickle format.

参考 momo/tools/preprocess_pcqm4mv2.py 的实现，但数据来源为 MoleculeNet 的 SMILES
（位于 GraphMVP/SNAP 预处理包的 processed/smiles.csv），我们为每条分子：
  - 以 RDKit ETKDGv3 生成 3D 构型（失败时多次重试，并尝试 UFF/MMFF 优化）
  - 使用 BRICS 对分子进行子结构切分，得到每原子的 motif_id（每图内从 0 连续编号）
  - 为每个 motif 计算 3D 几何特征（均值/方差的键长、夹角、转动惯量特征等，维度=7）
  - 导出与 PCQM4Mv2 预处理一致的数据结构（见 momo/data/pcqm4mv2.py 所需字段）

输出文件：/path/to/dataset/{dataset}/{dataset}_motif_preprocessed.pkl

注意：本脚本不依赖 GraphMVP 的 processed/geometric_data_processed.pt 内容，仅以
smiles.csv 为索引顺序。如需附带 y 标签，可扩展读取 pt 并在 rec 中加入 'y' 字段。
"""

import os
import sys
import argparse
import pickle
from typing import List, Tuple, Dict, Optional

import numpy as np
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, BRICS
import torch


def _embed_smiles_to_3d(
    smiles: str,
    max_attempts: int = 5,
    random_seed: int = 0,
    optimize: bool = True,
) -> Optional[Chem.Mol]:
    """Embed a 3D conformer for a SMILES string using RDKit ETKDGv3.

    Returns a molecule with a single conformer on success; otherwise None.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol, addCoords=False)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)
    params.useSmallRingTorsions = True
    params.useRandomCoords = False

    for attempt in range(max_attempts):
        res = AllChem.EmbedMolecule(mol, params)
        if res == 0:
            if optimize:
                # Try MMFF first; fallback to UFF if needed
                try:
                    if AllChem.MMFFHasAllMoleculeParams(mol):
                        AllChem.MMFFOptimizeMolecule(mol)
                    else:
                        AllChem.UFFOptimizeMolecule(mol)
                except Exception:
                    # Optimization failure is non-fatal
                    pass
            mol = Chem.RemoveHs(mol)
            return mol
        # tweak seed for the next attempt
        params.randomSeed += 17
        params.useRandomCoords = True

    # As a last resort, attempt without hydrogens altogether
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


def _ensure_conformer(mol: Chem.Mol) -> Chem.Conformer:
    if mol is None:
        raise RuntimeError("Molecule is None")
    if mol.GetNumConformers() == 0:
        raise RuntimeError("No conformer present (no 3D coordinates)")
    return mol.GetConformer()


def _fragment_motifs_brics(mol: Chem.Mol) -> List[Tuple[int, ...]]:
    """Use BRICS to define motif fragments; return a list of atom-index tuples per motif.
    If no BRICS bonds exist, return a single fragment containing all atoms.
    """
    brics_bonds = list(BRICS.FindBRICSBonds(mol))
    if not brics_bonds:
        return [tuple(range(mol.GetNumAtoms()))]
    pair_set = {tuple(sorted(pair)) for pair, _ in brics_bonds}
    bond_indices_to_cut: List[int] = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if tuple(sorted((i, j))) in pair_set:
            bond_indices_to_cut.append(b.GetIdx())
    if not bond_indices_to_cut:
        return [tuple(range(mol.GetNumAtoms()))]
    fragged = Chem.FragmentOnBonds(mol, bondIndices=bond_indices_to_cut, addDummies=False)
    frags = Chem.GetMolFrags(fragged, asMols=False, sanitizeFrags=False)
    return list(frags)


def _motif_batch_from_frags(frags: List[Tuple[int, ...]], n_atoms: int) -> np.ndarray:
    motif_batch = np.full((n_atoms,), -1, dtype=np.int32)
    for m_id, atoms in enumerate(frags):
        for a in atoms:
            motif_batch[a] = m_id
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
                ang = np.arccos(cosang)
                angles.append(ang)
    return np.asarray(angles, dtype=float)


def _compute_inertia_eigs(mol: Chem.Mol, conf: Chem.Conformer, atom_indices: List[int]) -> np.ndarray:
    if len(atom_indices) == 0:
        return np.zeros(3, dtype=float)
    coords = np.array([list(conf.GetAtomPosition(i)) for i in atom_indices], dtype=float)
    masses = np.array([Chem.GetPeriodicTable().GetAtomicWeight(mol.GetAtomWithIdx(i).GetAtomicNum()) for i in atom_indices], dtype=float)
    total_mass = masses.sum() if masses.sum() > 0 else 1.0
    com = (coords * masses[:, None]).sum(axis=0) / total_mass
    rel = coords - com[None, :]
    I = np.zeros((3, 3), dtype=float)
    for r, m in zip(rel, masses):
        r2 = np.dot(r, r)
        I += m * (r2 * np.eye(3) - np.outer(r, r))
    w = np.linalg.eigvalsh(I)
    w = np.sort(np.real(w))
    return w


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


def _atom_numbers(mol: Chem.Mol, selected_idx: List[int]) -> np.ndarray:
    return np.asarray([mol.GetAtomWithIdx(i).GetAtomicNum() for i in selected_idx], dtype=np.int64)


def _edge_index_undirected(mol: Chem.Mol, selected_idx: List[int]) -> np.ndarray:
    index_map = {old: new for new, old in enumerate(selected_idx)}
    rows, cols = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in index_map and j in index_map:
            ii = index_map[i]
            jj = index_map[j]
            rows += [ii, jj]
            cols += [jj, ii]
    return np.asarray([rows, cols], dtype=np.int64)


def _load_y_from_pt(ds_dir: str) -> Optional[List[List[float]]]:
    pt_path = os.path.join(ds_dir, 'processed', 'geometric_data_processed.pt')
    if not os.path.exists(pt_path):
        return None
    try:
        data, slices = torch.load(pt_path, map_location='cpu')
    except Exception:
        return None
    # 优先根据 slices 判断是否包含 y，避免触发 Data 的 __getattr__
    if not isinstance(slices, dict) or 'y' not in slices:
        return None
    # 尝试从底层 __dict__ 直接取出拼接后的 y Tensor，避免触发新版本 PyG 的属性访问
    y_storage = None
    try:
        y_storage = getattr(data, '__dict__', {}).get('y', None)
    except Exception:
        y_storage = None
    if y_storage is None or not torch.is_tensor(y_storage):
        # 无法安全读取，放弃附带标签
        return None
    # 从 slices 重建逐样本 y
    idx_tensor = slices['y']
    # 兼容张量/列表
    n = int((len(idx_tensor) - 1) if not hasattr(idx_tensor, 'numel') else idx_tensor.numel() - 1)
    ys: List[List[float]] = []
    for i in range(n):
        s = int(idx_tensor[i].item() if hasattr(idx_tensor[i], 'item') else idx_tensor[i])
        e = int(idx_tensor[i + 1].item() if hasattr(idx_tensor[i + 1], 'item') else idx_tensor[i + 1])
        y_i = y_storage[s:e]
        y_i = y_i.view(-1).cpu().numpy().tolist()
        ys.append([float(v) for v in y_i])
    return ys


def process_dataset(root: str, dataset: str, out_path: Optional[str] = None, limit: int = -1, attach_y: bool = False) -> None:
    ds_dir = os.path.join(root, dataset)
    smiles_path = os.path.join(ds_dir, 'processed', 'smiles.csv')
    if not os.path.exists(smiles_path):
        raise FileNotFoundError(f"smiles.csv not found: {smiles_path}")

    if out_path is None:
        out_path = os.path.join(ds_dir, f"{dataset}_motif_preprocessed.pkl")

    smiles_list: List[str] = []
    with open(smiles_path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # GraphMVP 的 smiles.csv 通常是单列；安全起见遇到逗号取首段
            if ',' in s:
                s = s.split(',')[0]
            smiles_list.append(s)

    results: List[Dict] = []
    failures: List[int] = []
    y_list: Optional[List[List[float]]] = None
    if attach_y:
        y_list = _load_y_from_pt(ds_dir)

    for idx, smi in enumerate(tqdm(smiles_list, desc=f"{dataset}")):
        if limit >= 0 and len(results) >= limit:
            break
        try:
            mol = _embed_smiles_to_3d(smi, max_attempts=6, random_seed=idx)
            if mol is None:
                failures.append(idx)
                continue
            conf = _ensure_conformer(mol)

            # 仅保留重原子（Z>1）
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
            if len(set(assigned_heavy)) != len(assigned_heavy):
                failures.append(idx)
                continue
            if set(assigned_heavy) != heavy_set:
                # 用单一 motif 兜底
                frags_heavy = [tuple(heavy_idx)]

            index_map = {old: new for new, old in enumerate(heavy_idx)}
            frags_heavy_mapped = [tuple(index_map[i] for i in tup) for tup in frags_heavy]

            motif_id = _motif_batch_from_frags(frags_heavy_mapped, n_atoms=len(heavy_idx))
            motif_feats: List[List[float]] = []
            for atoms in frags_heavy:
                feat = _motif_3d_features(mol, conf, list(atoms))
                motif_feats.append(feat.tolist())

            z = _atom_numbers(mol, heavy_idx)
            edge_index = _edge_index_undirected(mol, heavy_idx)
            coords = np.array([list(conf.GetAtomPosition(i)) for i in heavy_idx], dtype=float)

            rec = {
                "smiles": smi,
                "orig_idx": int(idx),
                "num_atoms": int(len(heavy_idx)),
                "num_motifs": int(len(frags_heavy_mapped)),
                "motif_id": motif_id.astype(int).tolist(),
                "motif_features": motif_feats,
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
        print(f"Failed indices: {failures[:50]}{' ...' if len(failures) > 50 else ''}")
    print(f"Saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess MoleculeNet (GraphMVP/SNAP) into Momo motif pickle")
    parser.add_argument("--root", type=str, required=True, help="Path to MoleculeNet root (contains subfolders like bbbp/, esol/, ...)")
    parser.add_argument("--dataset", type=str, default="", help="Dataset name (e.g., bbbp). If empty, use --list")
    parser.add_argument("--list", type=str, default="", help="Comma-separated dataset list, e.g. 'bbbp,esol,lipophilicity'")
    parser.add_argument("--out", type=str, default="", help="Output path for a single dataset; default to {root}/{dataset}/{dataset}_motif_preprocessed.pkl")
    parser.add_argument("--limit", type=int, default=-1, help="Limit processed molecules per dataset (-1 for all)")
    parser.add_argument("--attach-y", action='store_true', help="Attach labels from processed/geometric_data_processed.pt if available")
    args = parser.parse_args()

    targets: List[str] = []
    if args.dataset:
        targets = [args.dataset]
    elif args.list:
        targets = [s.strip() for s in args.list.split(',') if s.strip()]
    else:
        print("Either --dataset or --list must be provided", file=sys.stderr)
        sys.exit(2)

    for ds in targets:
        out_path = args.out or os.path.join(args.root, ds, f"{ds}_motif_preprocessed.pkl")
        process_dataset(args.root, ds, out_path=out_path, limit=args.limit, attach_y=bool(args.attach_y))


if __name__ == "__main__":
    main()
