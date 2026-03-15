#!/usr/bin/env python
"""
Generate MoleculeNet train/valid/test splits aligned with GraphMVP's splitters.

- Input: {root}/{dataset}/processed/smiles.csv (single-column SMILES, same顺序 as processed dataset)
- Modes:
  - scaffold (deterministic Bemis–Murcko scaffold split)
  - random_scaffold (scaffold groups shuffled by seed)
  - random (pure random split by seed)
- Output: {root}/{dataset}/splits/{mode}[_seed{seed}]/train.txt, val.txt, test.txt

参考 GraphMVP/src_classification/splitters.py 的实现，去除对 PyG dataset 的依赖，直接输出索引。
"""

import os
import sys
import argparse
from typing import List, Tuple

import numpy as np
from collections import defaultdict
from rdkit.Chem.Scaffolds import MurckoScaffold


def _read_smiles(path: str) -> List[str]:
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


def _generate_scaffold(smiles: str, include_chirality: bool = True) -> str:
    return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)


def scaffold_split_indices(
    smiles_list: List[str], frac_train: float, frac_valid: float, frac_test: float
) -> Tuple[List[int], List[int], List[int]]:
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    # group indices by scaffold
    scaffold_to_indices = {}
    for i, smi in enumerate(smiles_list):
        scaf = _generate_scaffold(smi, include_chirality=True)
        scaffold_to_indices.setdefault(scaf, []).append(i)
    # sort by size desc, then by smallest index desc (match GraphMVP key lambda)
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


def random_scaffold_split_indices(
    smiles_list: List[str], frac_train: float, frac_valid: float, frac_test: float, seed: int
) -> Tuple[List[int], List[int], List[int]]:
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


def random_split_indices(n: int, frac_train: float, frac_valid: float, frac_test: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
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


def _write_indices(base_dir: str, mode: str, seed: int, tr: List[int], va: List[int], te: List[int]) -> str:
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
    ap = argparse.ArgumentParser(description='Generate MoleculeNet splits (GraphMVP-aligned)')
    ap.add_argument('--root', type=str, required=True, help='MoleculeNet root (contains dataset subfolders)')
    ap.add_argument('--dataset', type=str, default='', help='Single dataset name (e.g., bbbp)')
    ap.add_argument('--list', type=str, default='', help="Comma-separated dataset list (e.g., 'bbbp,esol')")
    ap.add_argument('--mode', type=str, default='scaffold', choices=['scaffold', 'random', 'random_scaffold'])
    ap.add_argument('--seed', type=int, default=0, help='Seed for random/random_scaffold modes')
    ap.add_argument('--frac-train', type=float, default=0.8)
    ap.add_argument('--frac-valid', type=float, default=0.1)
    ap.add_argument('--frac-test', type=float, default=0.1)
    args = ap.parse_args()

    if args.dataset:
        datasets = [args.dataset]
    elif args.list:
        datasets = [s.strip() for s in args.list.split(',') if s.strip()]
    else:
        print('Either --dataset or --list must be provided', file=sys.stderr)
        sys.exit(2)

    for ds in datasets:
        ds_dir = os.path.join(args.root, ds)
        smiles_path = os.path.join(ds_dir, 'processed', 'smiles.csv')
        if not os.path.exists(smiles_path):
            print(f"Skip {ds}: missing {smiles_path}")
            continue
        smiles = _read_smiles(smiles_path)
        if args.mode == 'scaffold':
            tr, va, te = scaffold_split_indices(smiles, args.frac_train, args.frac_valid, args.frac_test)
        elif args.mode == 'random_scaffold':
            tr, va, te = random_scaffold_split_indices(smiles, args.frac_train, args.frac_valid, args.frac_test, args.seed)
        else:
            tr, va, te = random_split_indices(len(smiles), args.frac_train, args.frac_valid, args.frac_test, args.seed)
        out_dir = _write_indices(ds_dir, args.mode, args.seed, tr, va, te)
        print(f"{ds}: wrote splits to {out_dir} (train={len(tr)}, val={len(va)}, test={len(te)})")


if __name__ == '__main__':
    main()

