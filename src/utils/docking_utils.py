import io
import os
import re

import numpy as np
from rdkit.Chem import AllChem
from rdkit import RDLogger
from scipy.optimize import differential_evolution

RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import rdMolTransforms
import copy
import lmdb
import pickle
import pandas as pd
import torch
from multiprocessing import Pool
from tqdm import tqdm
import glob
import torch.nn.functional as F
import prody
from rdkit import Chem
from spyrmsd import rmsd, molecule


def get_torsions(m, removeHs=True):
    if removeHs:
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


def single_conf_gen_bonds(tgt_mol, num_confs=1000, seed=42, removeHs=True):
    mol = copy.deepcopy(tgt_mol)
    mol = Chem.AddHs(mol)
    allconformers = AllChem.EmbedMultipleConfs(
        mol, numConfs=num_confs, randomSeed=seed, clearConfs=True, numThreads=40
    )
    if removeHs:
        mol = Chem.RemoveHs(mol)
    rotable_bonds = get_torsions(mol, removeHs=removeHs)
    for i in range(len(allconformers)):
        np.random.seed(i)
        values = 3.1415926 * 2 * np.random.rand(len(rotable_bonds))
        for idx in range(len(rotable_bonds)):
            SetDihedral(mol.GetConformers()[i], rotable_bonds[idx], values[idx])
        Chem.rdMolTransforms.CanonicalizeConformer(mol.GetConformers()[i])
    return mol


def load_lmdb_data(lmdb_path, key):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    txn = env.begin()
    _keys = list(txn.cursor().iternext(values=False))
    collects = []
    for idx in range(len(_keys)):
        datapoint_pickled = txn.get(f"{idx}".encode("ascii"))
        data = pickle.loads(datapoint_pickled)
        collects.append(data[key])
    return collects


def docking_data_pre(raw_data_path, predict):
    mol_list = load_lmdb_data(os.path.join(raw_data_path, 'test.lmdb'), "mol_list")
    mol_list = [Chem.RemoveHs(mol) for items in mol_list for mol in items]
    (
        smi_list,
        pocket_list,
        pocket_coords_list,
        distance_predict_list,
        holo_distance_predict_list,
        holo_coords_list,
        holo_center_coords_list,
    ) = ([], [], [], [], [], [], [])
    for batch in predict:
        sz = batch["atoms"].size(0)

        for i in range(sz):
            smi_list.append(batch["smi_name"][i])
            pocket_list.append(batch["pocket_name"][i])

            distance_predict = batch["cross_distance_predict"][i]
            token_mask = batch["atoms"][i] > 2
            pocket_token_mask = batch["pocket_atoms"][i] > 2
            distance_predict = distance_predict[token_mask][:, pocket_token_mask]
            pocket_coords = batch["pocket_coordinates"][i]
            pocket_coords = pocket_coords[pocket_token_mask, :]

            holo_distance_predict = batch["holo_distance_predict"][i]
            holo_distance_predict = holo_distance_predict[token_mask][:, token_mask]

            holo_coordinates = batch["holo_coordinates"][i]
            holo_coordinates = holo_coordinates[token_mask, :]
            holo_center_coordinates = batch["holo_center_coordinates"][i][:3]

            pocket_coords = pocket_coords.numpy().astype(np.float32)
            distance_predict = distance_predict.numpy().astype(np.float32)
            holo_distance_predict = holo_distance_predict.numpy().astype(np.float32)
            holo_coords = holo_coordinates.numpy().astype(np.float32)

            pocket_coords_list.append(pocket_coords)
            distance_predict_list.append(distance_predict)
            holo_distance_predict_list.append(holo_distance_predict)
            holo_coords_list.append(holo_coords)
            holo_center_coords_list.append(holo_center_coordinates)

    return (
        mol_list,
        smi_list,
        pocket_list,
        pocket_coords_list,
        distance_predict_list,
        holo_distance_predict_list,
        holo_coords_list,
        holo_center_coords_list,
    )


def ensemble_iterations(
        mol_list,
        smi_list,
        pocket_list,
        pocket_coords_list,
        distance_predict_list,
        holo_distance_predict_list,
        holo_coords_list,
        holo_center_coords_list,
        tta_times=10,
):
    sz = len(mol_list)
    hash_mol_list = dict()
    # smi_set = set()
    for i in range(sz // tta_times):
        start_idx, end_idx = i * tta_times, (i + 1) * tta_times
        smi = Chem.MolToSmiles(mol_list[start_idx])
        hash_mol_list[smi] = start_idx
    #     smi_set.add(smi)
    # smi_list = list(smi_set)

    for i in range(sz // tta_times):
        # indices = [j for j, smi in enumerate(smi_list) if smi == smi_list[i]]
        start_idx, end_idx = i * tta_times, (i + 1) * tta_times
        distance_predict_tta = distance_predict_list[start_idx:end_idx]
        # distance_predict_tta = [distance_predict_list[j] for j in indices]
        holo_distance_predict_tta = holo_distance_predict_list[start_idx:end_idx]
        mol_index = hash_mol_list.get(smi_list[start_idx])
        mol = copy.deepcopy(mol_list[mol_index])
        rdkit_mol = single_conf_gen_bonds(
            mol, num_confs=tta_times, seed=42, removeHs=True
        )
        sz = len(rdkit_mol.GetConformers())
        initial_coords_list = [
            rdkit_mol.GetConformers()[i].GetPositions().astype(np.float32)
            for i in range(sz)
        ]

        yield [
            initial_coords_list,
            mol,
            smi_list[start_idx],
            pocket_list[start_idx],
            pocket_coords_list[start_idx],
            distance_predict_tta,
            holo_distance_predict_tta,
            holo_coords_list[start_idx],
            holo_center_coords_list[start_idx],
        ]


def rmsd_func(holo_coords, predict_coords):
    if predict_coords is not np.nan:
        sz = holo_coords.shape
        rmsd = np.sqrt(np.sum((predict_coords - holo_coords) ** 2) / sz[0])
        return rmsd
    return 1000.0


def print_results(rmsd_results):
    print("RMSD < 1.0 : ", np.mean(rmsd_results < 1.0))
    print("RMSD < 1.5 : ", np.mean(rmsd_results < 1.5))
    print("RMSD < 2.0 : ", np.mean(rmsd_results < 2.0))
    print("RMSD < 3.0 : ", np.mean(rmsd_results < 3.0))
    print("RMSD < 5.0 : ", np.mean(rmsd_results < 5.0))
    print("avg RMSD : ", np.mean(rmsd_results))


def single_SF_loss(
        predict_coords,
        pocket_coords,
        distance_predict,
        holo_distance_predict,
        dist_threshold=6,
        # dist_threshold=10,

):
    # dist = dist.unsqueeze(0)
    dist = torch.norm(predict_coords.unsqueeze(1) - pocket_coords.unsqueeze(0), dim=-1)
    holo_dist = torch.norm(
        predict_coords.unsqueeze(1) - predict_coords.unsqueeze(0), dim=-1
    )
    distance_mask = distance_predict < dist_threshold
    cross_dist_score = F.smooth_l1_loss(distance_predict[distance_mask], dist[distance_mask])
    dist_score = F.smooth_l1_loss(holo_distance_predict, holo_dist)
    loss = cross_dist_score * 1.0 + dist_score * 1.0
    return loss


def scoring(
        predict_coords,
        pocket_coords,
        distance_predict,
        holo_distance_predict,
        dist_threshold=4.5,
):
    predict_coords = predict_coords.detach()
    dist = torch.norm(predict_coords.unsqueeze(1) - pocket_coords.unsqueeze(0), dim=-1)
    # dist = dist.unsqueeze(0)
    holo_dist = torch.norm(
        predict_coords.unsqueeze(1) - predict_coords.unsqueeze(0), dim=-1
    )
    distance_mask = distance_predict < dist_threshold
    cross_dist_score = (
            (dist[distance_mask] - distance_predict[distance_mask]) ** 2
    ).mean()
    dist_score = ((holo_dist - holo_distance_predict) ** 2).mean()
    return cross_dist_score.cpu().numpy(), dist_score.cpu().numpy()


def dock_with_gradient(
        coords,
        pocket_coords,
        distance_predict_tta,
        holo_distance_predict_tta,
        loss_func=single_SF_loss,
        holo_coords=None,
        iterations=20000,
        early_stoping=5,
        return_best=True,
):
    bst_loss, bst_coords, bst_meta_info = 10000.0, coords, None

    all_coords = []
    for i, (distance_predict, holo_distance_predict) in enumerate(
            zip(distance_predict_tta, holo_distance_predict_tta)
    ):
        new_coords = copy.deepcopy(coords)
        _coords, _loss, _meta_info = single_dock_with_gradient(
            new_coords,
            pocket_coords,
            distance_predict,
            holo_distance_predict,
            loss_func=loss_func,
            holo_coords=holo_coords,
            iterations=iterations,
            early_stoping=early_stoping,
        )
        if bst_loss > _loss:
            bst_coords = _coords
            bst_loss = _loss
            bst_meta_info = _meta_info
        all_coords.append((_coords, _loss, _meta_info))
    #     all_coords.append(_coords)
    # return all_coords
    if return_best:
        return bst_coords, bst_loss, bst_meta_info
    else:
        return all_coords


def single_dock_with_gradient(
        coords,
        pocket_coords,
        distance_predict,
        holo_distance_predict,
        loss_func=single_SF_loss,
        holo_coords=None,
        iterations=20000,
        early_stoping=5,
):
    # coords = torch.from_numpy(coords).float()
    # pocket_coords = torch.from_numpy(pocket_coords).float()
    # distance_predict = torch.from_numpy(distance_predict).float()
    # holo_distance_predict = torch.from_numpy(holo_distance_predict).float()

    if holo_coords is not None:
        holo_coords = torch.from_numpy(holo_coords).float()
    # coords = coords[0]
    coords.requires_grad = True
    optimizer = torch.optim.LBFGS([coords], lr=0.1)
    bst_loss, times = 10000.0, 0
    for i in range(iterations):

        def closure():
            optimizer.zero_grad()
            loss = loss_func(
                coords, pocket_coords, distance_predict, holo_distance_predict
            )
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        if loss.item() < bst_loss:
            bst_loss = loss.item()
            times = 0
        else:
            times += 1
            if times > early_stoping:
                break

    meta_info = scoring(coords, pocket_coords, distance_predict, holo_distance_predict)
    return coords.detach().cpu().numpy(), loss.detach().cpu().item(), meta_info


def set_coord(mol, coords, idx):
    for i in range(coords.shape[0]):
        mol.GetConformer(idx).SetAtomPosition(i, coords[i].tolist())
    return mol


def add_coord(mol, xyz, idx=0):
    x, y, z = xyz
    conf = mol.GetConformer()
    pos = conf.GetPositions()
    pos[:, 0] += x
    pos[:, 1] += y
    pos[:, 2] += z
    for i in range(pos.shape[0]):
        conf.SetAtomPosition(
            i, Chem.rdGeometry.Point3D(pos[i][0], pos[i][1], pos[i][2])
        )
    return mol


def multithreading_process_single_docking(input_path, output_path, output_ligand_path):
    content = pd.read_pickle(input_path)
    (
        init_coords_tta,
        mol,
        smi,
        pocket,
        pocket_coords,
        distance_predict_tta,
        holo_distance_predict_tta,
        holo_coords,
        holo_cener_coords,
    ) = content
    sample_times = len(init_coords_tta)
    bst_predict_coords, bst_loss, bst_meta_info = None, 1000.0, None
    for i in range(sample_times):
        init_coords = init_coords_tta[i]
        predict_coords, loss, meta_info = dock_with_gradient(
            init_coords,
            pocket_coords,
            distance_predict_tta,
            holo_distance_predict_tta,
            holo_coords=holo_coords,
            loss_func=single_SF_loss,
        )
        if loss < bst_loss:
            bst_loss = loss
            bst_predict_coords = predict_coords
            bst_meta_info = meta_info

    try:
        _rmsd = round(rmsd_func(holo_coords, bst_predict_coords), 4)
    except:
        print(input_path)
    _cross_score = round(float(bst_meta_info[0]), 4)
    _self_score = round(float(bst_meta_info[1]), 4)
    print(f"{pocket}-{smi}-RMSD:{_rmsd}-{_cross_score}-{_self_score}")
    mol = Chem.RemoveHs(mol)
    mol = set_coord(mol, bst_predict_coords, 0)

    if output_path is not None:
        with open(output_path, "wb") as f:
            pickle.dump(
                [bst_predict_coords, holo_coords, bst_loss, smi, pocket, pocket_coords],
                f,
            )
    if output_ligand_path is not None:
        mol = add_coord(mol, holo_cener_coords.numpy())
        Chem.MolToMolFile(mol, output_ligand_path)

    return True


def result_log(dir_path):
    ### result logging ###
    output_dir = os.path.join(dir_path, "cache")
    rmsd_results = []
    for path in glob.glob(os.path.join(output_dir, "*.docking.pkl")):
        (
            bst_predict_coords,
            holo_coords,
            bst_loss,
            smi,
            pocket,
            pocket_coords,
        ) = pd.read_pickle(path)
        rmsd = rmsd_func(holo_coords, bst_predict_coords)
        rmsd_results.append(rmsd)
    rmsd_results = np.array(rmsd_results)
    print_results(rmsd_results)


class MultiProcess():
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def dump(self, content):
        pocket = content[3]
        output_name = os.path.join(self.output_dir, "{}.pkl".format(pocket))
        try:
            os.remove(output_name)
        except:
            pass
        pd.to_pickle(content, output_name)
        return True

    def single_docking(self, pocket_name):
        input_name = os.path.join(self.output_dir, "{}.pkl".format(pocket_name))
        output_path = os.path.join(self.output_dir, "{}.docking.pkl".format(pocket_name))
        output_ligand_path = os.path.join(
            self.output_dir, "{}.ligand.sdf".format(pocket_name)
        )
        try:
            os.remove(output_path)
        except:
            pass
        try:
            os.remove(output_ligand_path)
        except:
            pass
        cmd = "/home/abtion/miniconda3/bin/python src/utils/coordinate_model.py --input {} --output {} --output-ligand {}".format(
            input_name, output_path, output_ligand_path
        )
        os.system(cmd)
        return True


def docking(raw_data_path, predict, nthreads):
    tta_times = 10
    (
        mol_list,
        smi_list,
        pocket_list,
        pocket_coords_list,
        distance_predict_list,
        holo_distance_predict_list,
        holo_coords_list,
        holo_center_coords_list,
    ) = docking_data_pre(raw_data_path, predict)
    iterations = ensemble_iterations(
        mol_list,
        smi_list,
        pocket_list,
        pocket_coords_list,
        distance_predict_list,
        holo_distance_predict_list,
        holo_coords_list,
        holo_center_coords_list,
        tta_times=tta_times,
    )

    sz = len(mol_list) // tta_times
    new_pocket_list = pocket_list[::tta_times]
    output_dir = os.path.join(raw_data_path, "cache")
    os.makedirs(output_dir, exist_ok=True)
    MP = MultiProcess(output_dir)

    with Pool(nthreads) as pool:
        for inner_output in tqdm(pool.imap(MP.dump, iterations), total=sz):
            if not inner_output:
                print("fail to dump")

    with Pool(nthreads) as pool:
        for inner_output in tqdm(
                pool.imap(MP.single_docking, new_pocket_list), total=len(new_pocket_list)
        ):
            if not inner_output:
                print("fail to docking")

    result_log(raw_data_path)


class OptimizeConformer:
    def __init__(self, mol, true_mol, rotable_bonds, probe_id=-1, ref_id=-1, seed=None):
        super(OptimizeConformer, self).__init__()
        if seed:
            np.random.seed(seed)
        self.rotable_bonds = rotable_bonds
        self.mol = mol
        self.true_mol = true_mol
        self.probe_id = probe_id
        self.ref_id = ref_id

    def score_conformation(self, values):
        for i, r in enumerate(self.rotable_bonds):
            SetDihedral(self.mol.GetConformer(self.probe_id), r, values[i])
        return AllChem.AlignMol(self.mol, self.true_mol, self.probe_id, self.ref_id)


def apply_changes(mol, values, rotable_bonds, conf_id):
    opt_mol = copy.copy(mol)
    [SetDihedral(opt_mol.GetConformer(conf_id), rotable_bonds[r], values[r]) for r in range(len(rotable_bonds))]
    return opt_mol


def optimize_rotatable_bonds(mol, true_mol, rotable_bonds, probe_id=-1, ref_id=-1, seed=0, popsize=15, maxiter=500,
                             mutation=(0.5, 1), recombination=0.8):
    opt = OptimizeConformer(mol, true_mol, rotable_bonds, seed=seed, probe_id=probe_id, ref_id=ref_id)
    max_bound = [np.pi] * len(opt.rotable_bonds)
    min_bound = [-np.pi] * len(opt.rotable_bonds)
    bounds = (min_bound, max_bound)
    bounds = list(zip(bounds[0], bounds[1]))

    # Optimize conformations
    result = differential_evolution(opt.score_conformation, bounds,
                                    maxiter=maxiter, popsize=popsize,
                                    mutation=mutation, recombination=recombination, disp=False, seed=seed)
    opt_mol = apply_changes(opt.mol, result['x'], opt.rotable_bonds, conf_id=probe_id)

    return opt_mol


def align_conformer(mol_rdkit, mol):
    mol.AddConformer(mol_rdkit.GetConformer())
    rms_list = []
    AllChem.AlignMolConformers(mol, RMSlist=rms_list)
    mol_rdkit.RemoveAllConformers()
    mol_rdkit.AddConformer(mol.GetConformers()[1])
    return mol_rdkit


def prepare_log_data(mol_list, pocket_coords, distance_predict_tta, holo_distance_predict_tta):
    rst = []
    _pocket_coords = torch.from_numpy(pocket_coords)
    _distance_predict_tta = [torch.from_numpy(d) for d in distance_predict_tta]
    _holo_distance_predict_tta = [torch.from_numpy(d) for d in holo_distance_predict_tta]
    for mol in mol_list:
        coords = mol.GetConformer().GetPositions()
        _coords = torch.from_numpy(coords)
        loss = np.mean([single_SF_loss(_coords, _pocket_coords, d1, d2).item() for d1, d2 in
                        zip(_distance_predict_tta, _holo_distance_predict_tta)])
        rst.append((coords, loss))
    rst = sorted(zip(rst, mol_list), key=lambda x: x[0][1])
    log_data = [i[0] for i in rst]
    mol_list = [i[1] for i in rst]
    for mol, d in zip(mol_list, log_data):
        mol.SetProp('loss', f'{d[1]:.4f}')
    return log_data, mol_list


def save_sdf(mol_list, output_path):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    with Chem.SDWriter(output_path) as w:
        for i, mol in enumerate(mol_list):
            w.write(mol)


def read_mol(mol_path, sanitize=True):
    if re.search(r'.pdb$', mol_path):
        mol = Chem.MolFromPDBFile(mol_path, sanitize=sanitize, removeHs=True)
    elif re.search(r'.mol2$', mol_path):
        mol = Chem.MolFromMol2File(mol_path, sanitize=sanitize, removeHs=True)
    elif re.search(r'.mol$', mol_path):
        mol = Chem.MolFromMolFile(mol_path, sanitize=sanitize, removeHs=True)
    else:
        mol = Chem.SDMolSupplier(mol_path, sanitize=sanitize, removeHs=True)
    return mol


def extract_pocket(pdb_file, ligand_file):
    with open(pdb_file, 'r') as pdb_file:
        # supp = Chem.SDMolSupplier(ligand_file)
        # ligand = Chem.RemoveAllHs(supp[0])
        ligand = read_mol(ligand_file)
        if ligand is None:
            ligand = read_mol(ligand_file, sanitize=False)
            positions = ligand.GetConformer().GetPositions()
            atoms = np.array([a.GetSymbol() for a in ligand.GetAtoms()])
            positions = positions[atoms!='H']
        else:
            if ligand_file.endswith('.sdf'):
                ligand = ligand[0]
            ligand = Chem.RemoveAllHs(ligand)
            positions = ligand.GetConformer().GetPositions()
        distance = 6
        protein = prody.parsePDBStream(pdb_file).select('protein or water')
        selected = protein.select(f'same residue as within {distance} of ligand',
                                  ligand=positions)

        f = io.StringIO()
        prody.writePDBStream(f, selected)
        pocket = Chem.MolFromPDBBlock(f.getvalue(), sanitize=False, removeHs=True)
        # pocket = Chem.RemoveHs(pocket)
    return pocket, ligand


def extract_pocket_core(pdb_pre, reflig_mol, cutoff=6):
    protein = prody.parsePDBStream(pdb_pre).select('protein or water')
    selected = protein.select(f'same residue as within {cutoff} of ligand',
                              ligand=reflig_mol.GetConformer().GetPositions())
    fm = io.StringIO()
    prody.writePDBStream(fm, selected)
    pocket = Chem.MolFromPDBBlock(fm.getvalue())
    return pocket


def extract_pocket_extra(pdb_file, ligand_file):
    with open(pdb_file, 'r') as pdb_:
        pdb_pre = pdb_.read()
    with open(pdb_file, 'r') as pdb_file:
        ligand = Chem.MolFromMolFile(ligand_file)
        if not ligand:
            mol2 = os.path.join(ligand_file.rsplit('.')[0] + '.mol2')
            ligand = Chem.MolFromMol2File(mol2)
        if not ligand:
            ligand_mol = Chem.MolFromMolFile(ligand_file, sanitize=False)
            rw_mol = Chem.RWMol(ligand_mol)
            deleteH_num = 0
            for i, atom in enumerate(ligand_mol.GetAtoms()):
                if atom.GetSymbol() == "H":
                    rw_mol.RemoveAtom(i - deleteH_num)
                    deleteH_num = deleteH_num + 1
            ligand = rw_mol.GetMol()
        if not ligand:
            print("无法读到口袋配体")
        pocket = extract_pocket_core(pdb_file, ligand, cutoff=6)
        return pocket, ligand, pdb_pre


def get_symmetry_rmsd(mol, coords1, coords2, mol2=None):
    mol = molecule.Molecule.from_rdkit(mol)
    mol2 = molecule.Molecule.from_rdkit(mol2) if mol2 is not None else mol2
    mol2_atomicnums = mol2.atomicnums if mol2 is not None else mol.atomicnums
    mol2_adjacency_matrix = mol2.adjacency_matrix if mol2 is not None else mol.adjacency_matrix
    RMSD = rmsd.symmrmsd(
        coords1,
        coords2,
        mol.atomicnums,
        mol2_atomicnums,
        mol.adjacency_matrix,
        mol2_adjacency_matrix,
    )
    return RMSD
