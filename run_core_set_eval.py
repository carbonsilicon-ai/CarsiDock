"""
evaluation on pdbbind coreset.
"""
import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
import multiprocessing as mp
from src.utils.docking_inference_utils import read_ligands, docking
from src.utils.docking_utils import extract_carsidock_pocket
from src.utils.utils import get_abs_path, get_carsidock_model
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
    final_conformers = args.num_conformer

    if args.cuda_convert:
        import pydock
        lbfgsbsrv = pydock.LBFGSBServer(args.num_threads, args.cuda_device_index)
    else:
        lbfgsbsrv = None

    model, ligand_dict, pocket_dict = get_carsidock_model(args.ckpt_path, DEVICE)

    with open(get_abs_path(args.pdb_id_file), 'r', encoding='utf8') as f:
        pdb_ids = [line.strip() for line in f.readlines()]

    rmsds = []
    for pdb_id in tqdm(pdb_ids):
        pocket_file = get_abs_path(args.pdb_file.format(pdb_id=pdb_id))
        ligand_file = get_abs_path(args.sdf_file.format(pdb_id=pdb_id))
        pocket, ligand = extract_carsidock_pocket(pocket_file, ligand_file)
        init_mol_list = read_ligands(mol_list=[ligand], num_use_conf=args.num_conformer)[0]
        torch.cuda.empty_cache()
        if args.output_dir:
            output_path = get_abs_path(args.output_dir, f'{pdb_id}.sdf')
        else:
            output_path = None
        outputs = docking(model, pocket, init_mol_list, ligand_dict, pocket_dict, device=DEVICE,
                          output_path=output_path, num_threads=args.num_threads, target_mol=ligand, lbfgsbsrv=lbfgsbsrv)

        print(pdb_id, 'rmsd: top1/top100', outputs['rmsd'][0], min(outputs['rmsd']))
        rmsd = outputs['rmsd']
        if len(rmsd) < final_conformers:
            rmsd += [10.0] * (final_conformers - len(rmsd))
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
                        help='coreset pdbid list')
    parser.add_argument('--pdb_file', default="data/casf2016/{pdb_id}_protein.pdb",
                        help='protein file location template.')
    parser.add_argument('--sdf_file', default='data/casf2016/{pdb_id}_ligand.sdf',
                        help='ligand file location template.')
    parser.add_argument('--output_dir', default='outputs/casf')
    parser.add_argument('--ckpt_path', default='checkpoints/carsidock_230731.ckpt')
    parser.add_argument('--num_conformer', default=5, type=int, help='number of initial conformer, resulting in num_conformer * num_conformer docking conformations.')
    parser.add_argument('--num_threads', default=1, help='recommend 1')
    parser.add_argument('--cuda_convert', action='store_true',
                        help='use cuda to accelerate distance matrix to coordinate.')
    parser.add_argument('--cuda_device_index', default=0, type=int, help="cuda device index")
    args = parser.parse_args()
    main(args)
