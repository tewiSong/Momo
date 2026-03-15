#!/usr/bin/env python
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


def process_sdf(input_sdf: str, output_path: str, limit: int = -1) -> None:
    if not os.path.exists(input_sdf):
        raise FileNotFoundError(f"Input SDF not found: {input_sdf}")

    supplier = Chem.SDMolSupplier(input_sdf, removeHs=False, sanitize=True)
    results: List[Dict] = []

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
        motif_feats: List[List[float]] = []
        for atoms in frags_heavy:
            feat = _motif_3d_features(mol, conf, list(atoms))
            motif_feats.append(feat.tolist())

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
            "motif_features": motif_feats,
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
    pt_path = os.path.join(ds_dir, 'processed', 'geometric_data_processed.pt')
    if not os.path.exists(pt_path):
        return None
    try:
        data, slices = torch.load(pt_path)
    except Exception:
        return None
    if not hasattr(data, 'y') or 'y' not in slices:
        return None
    n = int(slices['y'].numel() - 1)
    ys: List[List[float]] = []
    for i in range(n):
        s = int(slices['y'][i].item())
        e = int(slices['y'][i + 1].item())
        y_i = data.y[s:e]
        y_i = y_i.view(-1).cpu().numpy().tolist()
        ys.append([float(v) for v in y_i])
    return ys


def process_moleculenet(root: str, dataset: str, out_path: Optional[str] = None, limit: int = -1, attach_y: bool = False) -> None:
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
            process_sdf(args.sdf, out, limit=args.limit)
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
            process_moleculenet(args.molnet_root, ds, out_path=out_path, limit=args.limit, attach_y=bool(args.molnet_attach_y))
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
