"""
本脚本用于对单个口袋和多个小分子（smiles或带初始构象的mol文件）进行对接,
"""
import argparse
import os
import multiprocessing as mp
import torch
from rdkit import Chem
from src.data.dictionary import Dictionary
from src.modeling.modeling_foldock2 import FoldDockingForPredict
from src.modeling.modeling_base_model import BetaConfig
from src.utils.docking_inference_utils import docking, read_ligands
from src.utils.docking_utils import extract_pocket, read_pocket
from src.utils.utils import get_abs_path

DEVICE = torch.device('cuda')


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device_index)

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

    print('read data...')
    if args.sdf_file is None:
        pocket = read_pocket(get_abs_path(args.pdb_file))
        ligand = None
    else:
        pocket, ligand = extract_pocket(get_abs_path(args.pdb_file),
                                        get_abs_path(args.sdf_file))

    if args.smiles_file is not None:
        with open(args.smiles_file, 'r') as f:
            smiles = [s.strip() for s in f.readlines()]
        all_mol_list = read_ligands(smiles=smiles, num_use_conf=args.num_conformer)
    elif ligand is not None:
        all_mol_list = read_ligands([ligand], num_use_conf=args.num_conformer)
    else:
        raise ValueError('Where are the ligands?')
        # all_mol_list = read_ligands()

    for i, mol_list in enumerate(all_mol_list):
        print(f'docking...{i}')
        if args.output_dir:
            output_path = get_abs_path(args.output_dir, f'{os.path.basename(args.pdb_file).split(".")[0]}_{i}.sdf')
        else:
            output_path = None
        outputs = docking(model, pocket, mol_list, ligand_dict, pocket_dict, output_path=output_path, device=DEVICE,
                          num_threads=args.num_threads,
                          lbfgsbsrv=lbfgsbsrv, target_mol=ligand)
        print(f"rmsd: {outputs['rmsd']}")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_file', default="example_data/4YKQ_hsp90_40_water.pdb",
                        help='crystal protein .pdb file.')
    parser.add_argument('--sdf_file', default='example_data/4YKQ_hsp90_40.sdf',
                        help='crystal ligand .sdf file, we need this to get the pocket.')
    parser.add_argument('--smiles_file', default=None,
                        help='smiles file to docking, txt file with One smiles per line. You dont need to provide it when redocking.')
    parser.add_argument('--output_dir', default='outputs/conformer')
    parser.add_argument('--num_conformer', default=5, help='number of initial conformer, resulting in num_conformer * num_conformer docking conformations.')
    parser.add_argument('--ckpt_path', default='checkpoints/0409-v1-best-val.ckpt')
    parser.add_argument('--num_threads', default=1, help='recommend 1')
    parser.add_argument('--cuda_convert', action='store_true',
                        help='use cuda to accelerate distance matrix to coordinate.')
    parser.add_argument('--cuda_device_index', default=0, type=int, help="gpu device index")
    args = parser.parse_args()
    main(args)
