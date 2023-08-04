"""
本脚本用于对单个口袋和多个小分子（smiles或带初始构象的mol文件）进行对接,
"""
import argparse
import os

import numpy as np
import torch
from rdkit import Chem
from tqdm import tqdm
import multiprocessing as mp
from src.data.dictionary import Dictionary
from src.modeling.modeling_foldock2 import FoldDockingForPredict
from src.modeling.modeling_base_model import BetaConfig
from src.utils.docking_inference_utils import read_ligands, docking
from src.utils.docking_utils import extract_pocket
from src.utils.utils import get_abs_path
import pytorch_lightning as pl


def print_results(rmsd_results):
    print("RMSD < 0.5 : ", np.mean(rmsd_results < 0.5))
    print("RMSD < 1.0 : ", np.mean(rmsd_results < 1.0))
    print("RMSD < 1.5 : ", np.mean(rmsd_results < 1.5))
    print("RMSD < 2.0 : ", np.mean(rmsd_results < 2.0))
    print("RMSD < 2.5 : ", np.mean(rmsd_results < 2.5))
    print("RMSD < 3.0 : ", np.mean(rmsd_results < 3.0))
    print("RMSD < 4.0 : ", np.mean(rmsd_results < 4.0))
    print("RMSD < 5.0 : ", np.mean(rmsd_results < 5.0))
    print("avg RMSD : ", np.mean(rmsd_results))


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device_index)
    DEVICE = torch.device(f'cuda')


    if args.cuda_convert:
        import pydock
        lbfgsbsrv = pydock.LBFGSBServer(args.num_threads, args.cuda_device_index)
    else:
        lbfgsbsrv = None

    ligand_dict = Dictionary.load(get_abs_path('example_data/molecule/dict.txt'))
    pocket_dict = Dictionary.load(get_abs_path('example_data/pocket/dict.txt'))
    model_config = BetaConfig(num_hidden_layers=6,
                              recycling=3,
                              hidden_size=768,
                              num_attention_heads=16,
                              mol_config=BetaConfig(num_hidden_layers=6,
                                                    vocab_size=len(ligand_dict) + 1,
                                                    hidden_size=768,
                                                    num_attention_heads=16),
                              pocket_config=BetaConfig(num_hidden_layers=6,
                                                       vocab_size=len(pocket_dict) + 1,
                                                       hidden_size=768,
                                                       num_attention_heads=16))
    model = FoldDockingForPredict(model_config).to(DEVICE)
    model.load_state_dict(torch.load(get_abs_path(args.ckpt_path))['state_dict'], strict=False)
    model.eval()

    with open(get_abs_path(args.pdb_id_file), 'r', encoding='utf8') as f:
        pdb_ids = [line.strip() for line in f.readlines()]

    rmsds = []
    for pdb_id in tqdm(pdb_ids):
        pocket_file = get_abs_path(args.pdb_file.format(pdb_id=pdb_id))
        ligand_file = get_abs_path(args.sdf_file.format(pdb_id=pdb_id))
        pocket, ligand = extract_pocket(pocket_file, ligand_file)
        init_mol_list = read_ligands(mol_list=[ligand], num_use_conf=10)[0]
        torch.cuda.empty_cache()
        if args.output_dir:
            output_path = get_abs_path(args.output_dir, f'{pdb_id}.sdf')
        else:
            output_path = None
        outputs = docking(model, pocket, init_mol_list, ligand_dict, pocket_dict, device=DEVICE,
                          output_path=output_path, num_threads=args.num_threads, target_mol=ligand, lbfgsbsrv=lbfgsbsrv)

        print(pdb_id, 'rmsd: top1/top100', outputs['rmsd'][0], min(outputs['rmsd']))
        rmsd = outputs['rmsd']
        if len(rmsd) < 100:
            rmsd += [100.0] * (100 - len(rmsd))
        rmsds.append(rmsd)

    rmsds_np = np.array(rmsds)
    print('===============top1==============')
    print_results(rmsds_np[:, 0])
    print('===============top100=============')
    print_results(np.min(rmsds_np, axis=-1))


if __name__ == '__main__':
    pl.seed_everything(42)
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_id_file', default="example_data/coreset_pdb_ids.txt",
                        help='coreset pdbid列表文件')
    parser.add_argument('--pdb_file', default="data/casf2016/{pdb_id}_protein.pdb",
                        help='口袋或蛋白pdb文件模板，在传入sdf_file时会以改sdf割口袋')
    parser.add_argument('--sdf_file', default='data/casf2016/{pdb_id}_ligand.sdf',
                        help='配体sdf文件，会以该sdf文件割口袋，在无smiles输入时，会进行该小分子的对接')
    parser.add_argument('--output_dir', default='outputs/casf')
    parser.add_argument('--ckpt_path', default='checkpoints/0409-v1-best-val.ckpt')
    parser.add_argument('--num_threads', default=1, help='线程数量，默认为1')
    parser.add_argument('--cuda_convert', action='store_true',
                        help='使用cuda进行距离矩阵转坐标，默认为False，即默认不使用cuda进行距离矩阵转坐标')
    parser.add_argument('--cuda_device_index', default=0, type=int, help="gpu设备的索引")
    args = parser.parse_args()
    main(args)
