import numpy as np
import copy
import torch
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import AllChem
from scipy.spatial.transform import Rotation
from sklearn.cluster import KMeans
from src.utils.dist_to_coords_utils import get_mask_rotate, modify_conformer
from src.utils.docking_utils import set_coord


def single_conf_gen(tgt_mol, num_confs=1000, seed=42, mmff=False):
    mol = copy.deepcopy(tgt_mol)
    mol = Chem.AddHs(mol)
    allconformers = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, randomSeed=seed, clearConfs=True, numThreads=0)
    try:
        if mmff:
            AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)
    except:
        pass
    # sz = len(allconformers)
    # for i in range(sz):
    #     try:
    #         AllChem.MMFFOptimizeMolecule(mol, confId=i)
    #     except:
    #         continue
    mol = Chem.RemoveHs(mol)
    return mol


def get_torsions(m):
    m = Chem.RemoveHs(m)
    torsionList = []
    torsionSmarts = "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"
    torsionQuery = Chem.MolFromSmarts(torsionSmarts)
    matches = m.GetSubstructMatches(torsionQuery)
    for match in matches:
        idx2 = match[0]
        idx3 = match[1]
        bond = m.GetBondBetweenAtoms(idx2, idx3)
        jAtom = m.GetAtomWithIdx(idx2)
        kAtom = m.GetAtomWithIdx(idx3)
        for b1 in jAtom.GetBonds():
            if b1.GetIdx() == bond.GetIdx():
                continue
            idx1 = b1.GetOtherAtomIdx(idx2)
            for b2 in kAtom.GetBonds():
                if (b2.GetIdx() == bond.GetIdx()) or (b2.GetIdx() == b1.GetIdx()):
                    continue
                idx4 = b2.GetOtherAtomIdx(idx3)
                # skip 3-membered rings
                if idx4 == idx1:
                    continue
                # skip torsions that include hydrogens
                if (m.GetAtomWithIdx(idx1).GetAtomicNum() == 1) or (
                        m.GetAtomWithIdx(idx4).GetAtomicNum() == 1
                ):
                    continue
                if m.GetAtomWithIdx(idx4).IsInRing():
                    torsionList.append((idx4, idx3, idx2, idx1))
                    break
                else:
                    torsionList.append((idx1, idx2, idx3, idx4))
                    break
            break
    return torsionList


def SetDihedral(conf, atom_idx, new_vale):
    rdMolTransforms.SetDihedralRad(
        conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale
    )


def single_conf_gen_bonds(tgt_mol, num_confs=1000, seed=42):
    mol = copy.deepcopy(tgt_mol)
    mol = Chem.AddHs(mol)
    allconformers = AllChem.EmbedMultipleConfs(
        mol, numConfs=num_confs, randomSeed=seed, clearConfs=True, numThreads=40
    )
    mol = Chem.RemoveHs(mol)
    # rotable_bonds = get_torsions(mol)
    # for i in range(len(allconformers)):
    #     np.random.seed(i)
    #     values = 3.1415926 * 2 * np.random.rand(len(rotable_bonds))
    #     for idx in range(len(rotable_bonds)):
    #         SetDihedral(mol.GetConformers()[i], rotable_bonds[idx], values[idx])
    #     Chem.rdMolTransforms.CanonicalizeConformer(mol.GetConformers()[i])
    torsions, masks = get_mask_rotate(mol)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if len(torsions) > 0:
        with torch.no_grad():
            coords = torch.from_numpy(np.array([c.GetPositions() for c in mol.GetConformers()])).to(torch.float)
            values = torch.zeros(coords.shape[0], 6 + len(torsions))
            values[:, 6:] = torch.rand(coords.shape[0], len(torsions)) * np.pi * 2
            for i, (coord, value) in enumerate(zip(coords, values)):
                new_coord = modify_conformer(coord, value, torsions, masks).cpu().data.numpy()
                set_coord(mol, new_coord, i)
    return mol


def single_conf_gen_no_MMFF(tgt_mol, num_confs=1000, seed=42):
    mol = copy.deepcopy(tgt_mol)
    mol = Chem.AddHs(mol)
    allconformers = AllChem.EmbedMultipleConfs(
        mol, numConfs=num_confs, randomSeed=seed, clearConfs=True, numThreads=40
    )
    mol = Chem.RemoveHs(mol)
    return mol

def clustering2(mol, M=100, N=5, conf_threshold=60):
    Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)
    atom_num = len(mol.GetAtoms())
    M = min(N * 5, M)
    rdkit_mol = single_conf_gen(mol, num_confs=M, mmff=(atom_num <= conf_threshold))
    rdkit_mol = Chem.RemoveAllHs(rdkit_mol)
    total_sz = 0
    sz = len(rdkit_mol.GetConformers())
    if sz == 0:
        rdkit_mol = copy.deepcopy(mol)
        rdkit_mol = Chem.RemoveAllHs(rdkit_mol)
        sz = len(rdkit_mol.GetConformers())
    tgt_coords = rdkit_mol.GetConformers()[0].GetPositions().astype(np.float32)
    tgt_coords = tgt_coords - np.mean(tgt_coords, axis=0)
    rdkit_coords_list = []
    for i in range(sz):
        _coords = rdkit_mol.GetConformers()[i].GetPositions().astype(np.float32)
        _coords = _coords - _coords.mean(axis=0)  # need to normalize first
        _R, _score = Rotation.align_vectors(_coords, tgt_coords)
        rdkit_coords_list.append(np.dot(_coords, _R.as_matrix()))
    total_sz += sz

    # 部分小分子生成的构象少于聚类数目
    if len(rdkit_coords_list) > N:
        rdkit_coords_flatten = np.array(rdkit_coords_list).reshape(total_sz, -1)
        cluster_size = N
        ids = (
            KMeans(n_clusters=cluster_size, random_state=42, n_init=10)
            .fit_predict(rdkit_coords_flatten)
            .tolist()
        )
        # 部分小分子仅可聚出较少的类
        ids_set = set(ids)
        coords_list = [rdkit_coords_list[ids.index(i)] for i in range(cluster_size) if i in ids_set]
    else:
        coords_list = rdkit_coords_list[:N]
    return coords_list
