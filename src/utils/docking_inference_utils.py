import copy
import time
from multiprocessing import Pool
from typing import List

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.autograd import Variable
from torch.distributions import Normal
import math

from src.utils.conf_gen import clustering2, get_torsions
from rdkit.Chem.rdchem import Conformer

from src.utils.dist_to_coords_utils import modify_conformer, get_mask_rotate
from src.utils.docking_utils import optimize_rotatable_bonds, prepare_log_data, add_coord, save_sdf, dock_with_gradient, \
    single_SF_loss, get_symmetry_rmsd


def set_coord(mol, coords):
    _mol = copy.deepcopy(mol)
    if len(_mol.GetConformers()) == 0:
        conf = Conformer(len(_mol.GetAtoms()))
        for i in range(len(_mol.GetAtoms())):
            conf.SetAtomPosition(i, coords[i].tolist())
        _mol.AddConformer(conf)
    else:
        for i in range(coords.shape[0]):
            _mol.GetConformer().SetAtomPosition(i, coords[i].tolist())
    return _mol


def prepare_data_from_mol(mol_list, dictionary, prefix='mol', max_atoms=384, device='cuda', center_mode=False):
    atoms = [a.GetSymbol().upper() for a in mol_list[0].GetAtoms()]
    atoms = [a if '[' not in a else a[1] for a in atoms]
    indices = np.array(atoms) != 'H'
    # 限制口袋最大原子个数
    if (np.sum(indices) > max_atoms) and (prefix != 'mol'):
        _indices = np.random.choice(np.sum(indices), max_atoms, replace=False)
        drop_indices = np.zeros(np.sum(indices), dtype=bool)
        drop_indices[_indices] = True
        indices[indices] = drop_indices
    tokens = torch.from_numpy(dictionary.vec_index(atoms)[indices]).long()
    ori_coordinates = torch.from_numpy(np.array([mol.GetConformer().GetPositions()[indices] for mol in mol_list]))
    coordinates = ori_coordinates.clone()
    bsz, sz = coordinates.shape[:2]
    if center_mode:
        center = coordinates.mean(dim=1).unsqueeze(1)
        coordinates = torch.cat([center, coordinates, center], dim=1)
        distance = (coordinates.unsqueeze(2) - coordinates.unsqueeze(1)).norm(dim=-1)
    else:
        distance = torch.zeros(bsz, sz + 2, sz + 2)
        distance[:, 1:-1, 1:-1] = (coordinates.unsqueeze(2) - coordinates.unsqueeze(1)).norm(dim=-1)
    # sos & eos
    tokens = torch.cat([torch.full((1,), dictionary.bos()), tokens, torch.full((1,), dictionary.eos())], dim=0)
    # coordinates = torch.cat([torch.full((1, 3), 0.0), coordinates, torch.full((1, 3), 0.0)], dim=0)
    edge_type = tokens.view(-1, 1) * len(dictionary) + tokens.view(1, -1)
    tokens = tokens.unsqueeze(0).repeat(bsz, 1)
    edge_type = edge_type.unsqueeze(0).repeat(bsz, 1, 1)
    return {
        f'{prefix}_src_tokens': tokens.to(device=device),
        f'{prefix}_src_distance': distance.to(device=device, dtype=torch.float32),
        f'{prefix}_src_edge_type': edge_type.to(device=device),
    }, ori_coordinates.to(device=device, dtype=torch.float32)


class MultiProcess:
    def __init__(self, mol, pocket_coords, pocket_center, pi, mu, sigma, iterations=20000, early_stoping=5, **unused):
        self.mol = Chem.RemoveHs(mol)
        self.pocket_coords = pocket_coords
        self.pocket_center = pocket_center
        self.pi = pi
        self.mu = mu
        self.sigma = sigma
        self.iterations = iterations
        self.early_stoping = early_stoping
        torsions, masks = get_mask_rotate(mol)
        self.torsions = torsions
        self.masks = masks
        self.rotable_bonds = get_torsions(mol)

    def dist_to_coords(self, init_coord, pred_cross_dist, pred_holo_dist):
        coords = copy.deepcopy(init_coord)
        coords.requires_grad = True
        optimizer = torch.optim.Adam([coords], lr=0.1)
        best_loss, times, best_coords = 10000.0, 0, None
        for i in range(self.iterations):
            def closure():
                optimizer.zero_grad()
                loss = single_SF_loss(coords, self.pocket_coords, pred_cross_dist, pred_holo_dist)
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_coords = copy.deepcopy(coords.detach())
                times = 0
            else:
                times += 1
                if times > self.early_stoping:
                    break
        if best_loss < 100:
            mol = set_coord(self.mol, best_coords.cpu().data.numpy())
            rdkit_mol = set_coord(self.mol, init_coord.cpu().data.numpy())
            if len(self.rotable_bonds) > 0:
                opt_mol = optimize_rotatable_bonds(rdkit_mol, mol, self.rotable_bonds)
                with torch.no_grad():
                    best_coords = torch.from_numpy(opt_mol.GetConformer().GetPositions())
                    best_loss = single_SF_loss(best_coords, self.pocket_coords, pred_cross_dist, pred_holo_dist).item()
            else:
                AllChem.AlignMol(rdkit_mol, mol)
                opt_mol = rdkit_mol
            with torch.no_grad():
                score = mdn_score(self.pi, self.mu, self.sigma, best_coords, self.pocket_coords).item()
            opt_mol = add_coord(opt_mol, self.pocket_center.cpu().data.numpy())
            opt_mol.SetProp('loss', f'{best_loss}')
            opt_mol.SetProp('score', f'{score}')
            return opt_mol, best_loss, score

    def dist_to_coords_with_tor(self, init_coord, pred_cross_dist, pred_holo_dist):
        values = Variable(torch.zeros(6 + len(self.torsions), device=init_coord.device), requires_grad=True)
        optimizer = torch.optim.LBFGS([values], lr=0.1)
        best_loss, times, best_values, best_score, best_coords = 10000.0, 0, None, 0, None
        for i in range(self.iterations):
            def closure():
                optimizer.zero_grad()
                new_pos = modify_conformer(init_coord, values, self.torsions, self.masks)
                loss = single_SF_loss(new_pos, self.pocket_coords, pred_cross_dist, pred_holo_dist)
                loss.backward()
                return loss

            def fn(v):
                new_pos = modify_conformer(init_coord, torch.from_numpy(v).float(), self.torsions, self.masks)
                return single_SF_loss(new_pos, self.pocket_coords, pred_cross_dist, pred_holo_dist)

            def der(v):
                eps = 0.001
                y = fn(v)
                g = np.zeros_like(v, dtype=np.float32)
                for i in range(v.shape[0]):
                    save = v[i]
                    v[i] += eps
                    yd = fn(v)
                    g[i] = (yd.item() - y.item()) / eps
                    v[i] = save
                return y, g

            def closure_Finite_Difference():
                optimizer.zero_grad()
                # 计算梯度并将其存储在values.grad中
                loss, grad = der(values.detach().numpy())
                values.grad = torch.tensor(grad, dtype=torch.float32, device=values.device)
                return loss

            loss = optimizer.step(closure)
            # loss = optimizer.step(closure_Finite_Difference)
            if loss.item() < best_loss:
                best_loss = loss.item()
                times = 0
                best_values = copy.deepcopy(values).detach()
            else:
                times += 1
                if times > self.early_stoping:
                    break
        if best_loss < 100:
            with torch.no_grad():
                best_coords = modify_conformer(init_coord, best_values, self.torsions, self.masks)
                score = mdn_score(self.pi, self.mu, self.sigma, best_coords, self.pocket_coords).item()
                best_coords = best_coords.cpu().data.numpy()
            opt_mol = set_coord(self.mol, best_coords)
            opt_mol = add_coord(opt_mol, self.pocket_center.cpu().data.numpy())
            opt_mol.SetProp('loss', f'{best_loss}')
            opt_mol.SetProp('score', f'{score}')
            return opt_mol, best_loss, score

    def dist_to_coords_with_tor_cuda(self, init_coord, pred_cross_dist, pred_holo_dist):
        import pydock
        values = torch.zeros(6 + len(self.torsions))
        best_values, best_loss, ok = pydock.lbfgsb(init_coord, self.torsions, self.masks, self.pocket_coords,
                                                   pred_cross_dist, pred_holo_dist, values, eps=0.01)
        assert ok
        if best_loss < 100:
            with torch.no_grad():
                best_coords = modify_conformer(init_coord, best_values, self.torsions, self.masks)
                score = mdn_score(self.pi, self.mu, self.sigma, best_coords, self.pocket_coords).item()
                best_coords = best_coords.cpu().data.numpy()
            opt_mol = set_coord(self.mol, best_coords)
            opt_mol = add_coord(opt_mol, self.pocket_center.cpu().data.numpy())
            opt_mol.SetProp('loss', f'{best_loss}')
            opt_mol.SetProp('score', f'{score}')
            return opt_mol, best_loss, score

    def dist_to_coords_with_tor_culbfgsb(self, srv, args):
        values = torch.zeros(6 + len(self.torsions))
        seqs = {}  # save request and its init coord for score calculation

        # post optimize requests
        for arg in args:
            init_coord, pred_cross_dist, pred_holo_dist = arg
            seq, ok = srv.dock_optimize(init_coord, self.torsions, self.masks, self.pocket_coords, pred_cross_dist,
                                        pred_holo_dist, values, eps=0.01)
            if not ok:
                print('dock optimize failure')
            else:
                seqs[seq] = init_coord

        # poll response one by one and calc scores
        sz = len(seqs)
        mol_list = []
        while sz > 0:
            sz -= 1
            rsp = srv.poll()
            if rsp is None:
                print('error: no responses are expected in server')
            else:
                best_values, best_loss, seq, ok = rsp
                if not ok:
                    print('optimize response failure')
                elif best_loss < 100 and seq in seqs:
                    with torch.no_grad():
                        best_coords = modify_conformer(seqs[seq], best_values, self.torsions, self.masks)
                        score = mdn_score(self.pi, self.mu, self.sigma, best_coords, self.pocket_coords).item()
                        best_coords = best_coords.cpu().data.numpy()
                    opt_mol = set_coord(self.mol, best_coords)
                    opt_mol = add_coord(opt_mol, self.pocket_center.cpu().data.numpy())
                    opt_mol.SetProp('loss', f'{best_loss}')
                    opt_mol.SetProp('score', f'{score}')
                    mol_list.append((opt_mol, best_loss, score,))
        return mol_list


def convert_dist2coord(infer_output, ligands: List, target_mol=None, output_path=None, one_step=True, num_threads=24,
                       lbfgsbsrv=None):
    mp = MultiProcess(mol=ligands[0], **infer_output)
    args = [(ic, pcd, phd) for ic in infer_output['ligands_coords'] for pcd, phd in
            zip(infer_output['distance_predict_tta'], infer_output['holo_distance_predict_tta'])]
    if lbfgsbsrv is not None:
        mol_list = mp.dist_to_coords_with_tor_culbfgsb(lbfgsbsrv, args)
    elif one_step:
        if num_threads == -1:
            mol_list = [mp.dist_to_coords_with_tor_cuda(*arg) for arg in args]
        elif num_threads > 1:
            with Pool(num_threads) as pool:
                mol_list = pool.starmap(mp.dist_to_coords_with_tor, args)
        else:
            mol_list = [mp.dist_to_coords_with_tor(*arg) for arg in args]
    else:
        if num_threads > 1:
            with Pool(num_threads) as pool:
                mol_list = pool.starmap(mp.dist_to_coords, args)
        else:
            mol_list = [mp.dist_to_coords(*arg) for arg in args]

    mol_list = sorted([m for m in mol_list if m is not None], key=lambda x: 5 * x[1] - x[2])
    rdkit_mol_list = [item[0] for item in mol_list]

    rmsd = None
    if target_mol is not None:
        target_coords = Chem.RemoveHs(target_mol).GetConformer().GetPositions()
        pred_coords = [Chem.RemoveHs(mol).GetConformer().GetPositions() for mol in rdkit_mol_list]
        try:
            rmsd = get_symmetry_rmsd(target_mol, target_coords, pred_coords, rdkit_mol_list[0])
        except:
            print("rmsd 计算失败。")

    rdkit_mol_list = [Chem.rdmolops.AddHs(mol, addCoords=True) for mol in rdkit_mol_list]
    if output_path is not None:
        save_sdf(rdkit_mol_list, output_path)

    return mol_list, rdkit_mol_list, rmsd


def mdn_score(pi, mu, sigma, predict_coords=None, pocket_coords=None, dist=None, threshold=5, reduction='sum'):
    if dist is None:
        dist = torch.norm(predict_coords.unsqueeze(1) - pocket_coords.unsqueeze(0), dim=-1)
    dist_mask = dist < threshold
    normal = Normal(mu, sigma)
    # [BSZ, N, M, 10]
    loglik = normal.log_prob(dist.unsqueeze(-1))
    logprob = loglik + torch.log(pi)
    # [BSZ, N, M]
    prob = logprob.exp().sum(-1)
    if reduction == 'mean':
        score = (prob[dist_mask] / (dist[dist_mask] ** 2 + 1e-6)).mean() * 1000
    else:
        # score = torch.stack([p[m].sum() for p, m in zip(prob, dist_mask)])
        score = prob[dist_mask].sum() / dist.shape[0]
        # score = prob[dist_mask].sum()
    conf_mask = dist < 1.5
    print(dist.min())
    if conf_mask.sum() > 0:
        score += torch.log(dist[conf_mask] / 1.5).sum() * 10
    return score


def mdn_score_list(pi, mu, sigma, dist=None, threshold=5, reduction='sum'):
    dist_mask = dist < threshold
    normal = Normal(mu, sigma)
    # [BSZ, N, M, 10]
    loglik = normal.log_prob(dist.unsqueeze(-1))
    logprob = loglik + torch.log(pi)
    # [BSZ, N, M]
    prob = logprob.exp().sum(-1)
    if reduction == 'mean':
        score = (prob[dist_mask] / (dist[dist_mask] ** 2 + 1e-6)).mean() * 1000
    else:
        # score = torch.stack([p[m].sum() / dist.shape[1] for p, m in zip(prob, dist_mask)])
        score = torch.stack([p[m].sum() for p, m in zip(prob, dist_mask)])
        # score = prob[dist_mask].sum() / dist.shape[0]
        # score = prob[dist_mask].sum()
    return score


def read_ligands(mol_list=None, smiles=None, num_gen_conf=100, num_use_conf=5):
    """
    读取小分子，如为smiles则生成构象
    """
    if mol_list is None:
        assert smiles is not None
        mol_list = [Chem.MolFromSmiles(smi) for smi in smiles]
    # print(Chem.MolToInchiKey(mol_list[0]))
    mol_list = [Chem.RemoveAllHs(mol) for mol in mol_list if mol is not None]
    total_coords = [clustering2(mol, num_gen_conf, num_use_conf) for mol in mol_list]
    total_mol_list = [[set_coord(mol, coord) for coord in coords] for mol, coords in zip(mol_list, total_coords)]
    return total_mol_list


@torch.no_grad()
def model_inference(model, pocket, ligands: List, ligand_dict, pocket_dict, device='cuda', bsz=8, center_mode=False):
    model.eval()
    ligand_nums = len(ligands)
    # print('prepare data...')
    l_data, p_data = [], []
    for i in range(int(math.ceil(ligand_nums / bsz))):
        l_data.append(
            prepare_data_from_mol(ligands[i * bsz:(i + 1) * bsz], ligand_dict, device=device, center_mode=center_mode))
        length = len(ligands[i * bsz:(i + 1) * bsz])
        p_data.append(prepare_data_from_mol([pocket for _ in range(length)], pocket_dict, 'pocket', device=device,
                                            center_mode=center_mode))
    # print('inference distance matrix...')
    with torch.no_grad():
        outputs = [model(**pocket_data[0], **ligand_data[0]) for pocket_data, ligand_data in zip(p_data, l_data)]
        pocket_coords = p_data[0][1][0]
        ligands_coords = torch.cat([l[1] for l in l_data], dim=0)

        affinity = torch.cat([output.affinity_predict for output in outputs], dim=0).mean(dim=0).cpu().data.numpy()
        pocket_center = pocket_coords.mean(dim=0)
        pocket_coords = pocket_coords - pocket_center
        ligands_coords = ligands_coords - ligands_coords.mean(dim=1, keepdim=True)
        distance_predict_tta = torch.cat([output.cross_distance_predict for output in outputs])[:, 1:-1, 1:-1]
        holo_distance_predict_tta = torch.cat([output.holo_distance_predict for output in outputs])[:, 1:-1, 1:-1]
        pi = torch.cat([output.mdn[0] for output in outputs], dim=0).mean(dim=0)[1:-1, 1:-1]
        mu = torch.cat([output.mdn[1] for output in outputs], dim=0).mean(dim=0)[1:-1, 1:-1]
        sigma = torch.cat([output.mdn[2] for output in outputs], dim=0).mean(dim=0)[1:-1, 1:-1]
        mean_cross_dist = torch.mean(distance_predict_tta, dim=0)
        score = mdn_score(pi, mu, sigma, dist=mean_cross_dist).item()

        # 后续操作均使用cpu计算
        affinity = np.power(10, -affinity) * 1e6
        distance_predict_tta = distance_predict_tta.cpu()
        holo_distance_predict_tta = holo_distance_predict_tta.cpu()
        pi, mu, sigma = pi.cpu(), mu.cpu(), sigma.cpu()
        ligands_coords = ligands_coords.cpu()
        pocket_coords = pocket_coords.cpu()
        pocket_center = pocket_center.cpu()

        inference_output = {
            'affinity': affinity,
            'score': score,
            'distance_predict_tta': distance_predict_tta,
            'holo_distance_predict_tta': holo_distance_predict_tta,
            'pi': pi,
            'mu': mu,
            'sigma': sigma,
            'ligands_coords': ligands_coords,
            'pocket_coords': pocket_coords,
            'pocket_center': pocket_center,
        }
        return inference_output


def scoring(model, pocket, init_ligands, ligand_dict, pocket_dict, docked_ligands, device='cuda', center_mode=False,
            add_loss=True):
    # assert Chem.MolToSmiles(init_ligands[0]) == Chem.MolToSmiles(docked_ligands[0])
    init_ligands = [Chem.RemoveHs(mol) for mol in init_ligands]
    docked_ligands = [Chem.RemoveHs(mol) for mol in docked_ligands]

    bsz = len(init_ligands)
    pocket_data, pocket_coords = prepare_data_from_mol([pocket for _ in range(bsz)], pocket_dict, 'pocket',
                                                       device=device, center_mode=center_mode)
    ligand_data, _ = prepare_data_from_mol(init_ligands, ligand_dict, device=device, center_mode=center_mode)
    with torch.no_grad():
        outputs = model(**pocket_data, **ligand_data)
        pocket_coords = pocket_coords[0]
        ligands_coords = torch.from_numpy(np.array([mol.GetConformer().GetPositions() for mol in docked_ligands])).to(
            device)
        dists = (ligands_coords.unsqueeze(-2) - pocket_coords.view(1, 1, *pocket_coords.shape)).norm(dim=-1)
        pi, mu, sigma = outputs.mdn
        pi = torch.mean(pi[:, 1:-1, 1:-1], dim=0)
        mu = torch.mean(mu[:, 1:-1, 1:-1], dim=0)
        sigma = torch.mean(sigma[:, 1:-1, 1:-1], dim=0)
        # scores = [mdn_score(pi, mu, sigma, dist=dist).item() for dist in dists]
        scores = mdn_score_list(pi, mu, sigma, dist=dists)
        if add_loss:
            losses = []
            distance_predict = outputs.cross_distance_predict[:, 1:-1, 1:-1].mean(dim=0)
            holo_distance_predict = outputs.holo_distance_predict[:, 1:-1, 1:-1].mean(dim=0)
            for ligand_coord in ligands_coords:
                loss = single_SF_loss(ligand_coord, pocket_coords, distance_predict, holo_distance_predict)
                losses.append(loss)
            scores = scores - 5 * torch.stack(losses)
        scores = scores.cpu().data.numpy().tolist()
    return scores


def docking(model, pocket, ligands: List, ligand_dict, pocket_dict, dock=True,
            output_path=None, target_mol=None, one_step=True, device='cuda', num_threads=24, bsz=8, center_mode=False,
            lbfgsbsrv=None):
    """
    model: 用于推理的模型
    pocket: rdkit的口袋 mol 对象
    ligands: list[rdkit mol], 每个mol对象均有不同的初始构象，强烈建议列表长度大于等于5
    ligand_dict: 小分子token字典
    pocket_dict: 口袋token字典
    dock: 是否进行对接
    output_path：对接构象输出路径，如不指定则不输出小分子构象
    target_mol: 带ground truth构象的小分子mol对象，如给定则打印每个构象的rmsd
    one_step：距离矩阵转坐标是否仅通过一步进行，设为True时更快，设为False时更准。
    device: 在进行模型推理时使用的设备。（距离矩阵转坐标阶段仅支持cpu）
    num_threads: 在距离矩阵转坐标阶段的进程池中进程的数量, -1表示使用cuda kernel计算。
    bsz: 模型进行inference时的batch size
    center_mode: 补充蛋白和小分子构象中心作为sos/eos的坐标，默认为false，0408及往后的模型需置为True
    """
    assert len(ligands) > 0, "初始构象个数至少为1"
    assert bsz > 0, "batch size 至少为1"
    start_time = time.time()
    infer_output = model_inference(model, pocket, ligands, ligand_dict, pocket_dict, device=device, bsz=bsz,
                                   center_mode=center_mode)
    infer_end_time = time.time()

    convert_start_time = time.time()
    if dock:
        print('convert distance matrix to coordinates...')
        mol_list, rdkit_mol_list, rmsd = convert_dist2coord(infer_output, ligands, target_mol, output_path, one_step,
                                                            num_threads, lbfgsbsrv=lbfgsbsrv)
    else:
        mol_list, rdkit_mol_list, rmsd = [], [], []
    end_time = time.time()

    rst = dict()
    rst['screening_score'] = infer_output['score']
    rst['mol_list'] = rdkit_mol_list
    rst['conf_ranking_score'] = [5 * item[1] - item[2] for item in mol_list]
    rst['conf_convert_loss'] = [item[1] for item in mol_list]
    rst['conf_mdn_score'] = [item[2] for item in mol_list]
    rst['affinity'] = {'ki': infer_output['affinity'][0],
                       'kd': infer_output['affinity'][1],
                       'ic50': infer_output['affinity'][2]}
    rst['infer_time'] = infer_end_time - start_time
    rst['convert_time'] = end_time - convert_start_time
    rst['total_time'] = end_time - start_time
    rst['rmsd'] = rmsd
    rst['smiles'] = Chem.MolToSmiles(ligands[0])
    return rst