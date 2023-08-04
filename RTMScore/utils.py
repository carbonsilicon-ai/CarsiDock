"""
@Time: 4/8/2023 下午2:25
@Author: Heng Cai
@FileName: utils.py
@Copyright: 2020-2023 CarbonSilicon.ai
@Description:
"""
import torch as th
import MDAnalysis as mda
from torch.utils.data import DataLoader
from RTMScore.data.data import VSDataset
from RTMScore.model.utils import collate, run_an_eval_epoch
from RTMScore.model.model2 import RTMScore, DGLGraphTransformer


def get_rtmscore_model(modpath, **kwargs):
    args = {}
    args["batch_size"] = 128
    args["dist_threhold"] = 5
    args['device'] = 'cuda'
    args["num_workers"] = 10
    args["num_node_featsp"] = 41
    args["num_node_featsl"] = 41
    args["num_edge_featsp"] = 5
    args["num_edge_featsl"] = 10
    args["hidden_dim0"] = 128
    args["hidden_dim"] = 128
    args["n_gaussians"] = 10
    args["dropout_rate"] = 0.10
    kwargs.update(args)
    ligmodel = DGLGraphTransformer(in_channels=kwargs["num_node_featsl"],
                                   edge_features=kwargs["num_edge_featsl"],
                                   num_hidden_channels=kwargs["hidden_dim0"],
                                   activ_fn=th.nn.SiLU(),
                                   transformer_residual=True,
                                   num_attention_heads=4,
                                   norm_to_apply='batch',
                                   dropout_rate=0.15,
                                   num_layers=6
                                   )

    protmodel = DGLGraphTransformer(in_channels=kwargs["num_node_featsp"],
                                    edge_features=kwargs["num_edge_featsp"],
                                    num_hidden_channels=kwargs["hidden_dim0"],
                                    activ_fn=th.nn.SiLU(),
                                    transformer_residual=True,
                                    num_attention_heads=4,
                                    norm_to_apply='batch',
                                    dropout_rate=0.15,
                                    num_layers=6
                                    )

    model = RTMScore(ligmodel, protmodel,
                     in_channels=kwargs["hidden_dim0"],
                     hidden_dim=kwargs["hidden_dim"],
                     n_gaussians=kwargs["n_gaussians"],
                     dropout_rate=kwargs["dropout_rate"],
                     dist_threhold=kwargs["dist_threhold"]).to(kwargs['device'])

    checkpoint = th.load(modpath, map_location=th.device(kwargs['device']))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def scoring(prot, lig, model,
            cut=10.0,
            gen_pocket=False,
            reflig=None,
            atom_contribution=False,
            res_contribution=False,
            explicit_H=False,
            use_chirality=True,
            parallel=False,
            **kwargs
            ):
    """
    prot: The input protein file ('.pdb')
    lig: The input ligand file ('.sdf|.mol2', multiple ligands are supported)
    modpath: The path to store the pre-trained model
    gen_pocket: whether to generate the pocket from the protein file.
    reflig: The reference ligand to determine the pocket.
    cut: The distance within the reference ligand to determine the pocket.
    atom_contribution: whether the decompose the score at atom level.
    res_contribution: whether the decompose the score at residue level.
    explicit_H: whether to use explicit hydrogen atoms to represent the molecules.
    use_chirality: whether to adopt the information of chirality to represent the molecules.
    parallel: whether to generate the graphs in parallel. (This argument is suitable for the situations when there are lots of ligands/poses)
    kwargs: other arguments related with model
    """
    args = {}
    args["batch_size"] = 128
    args["dist_threhold"] = 5
    args['device'] = 'cuda'
    args["num_workers"] = 10
    kwargs.update(args)
    # try:
    data = VSDataset(ligs=lig,
                     prot=prot,
                     cutoff=cut,
                     gen_pocket=gen_pocket,
                     reflig=reflig,
                     explicit_H=explicit_H,
                     use_chirality=use_chirality,
                     parallel=parallel)

    test_loader = DataLoader(dataset=data,
                             batch_size=kwargs["batch_size"],
                             shuffle=False,
                             num_workers=kwargs["num_workers"],
                             collate_fn=collate)

    if atom_contribution:
        preds, at_contrs, _ = run_an_eval_epoch(model,
                                                test_loader,
                                                pred=True,
                                                atom_contribution=True,
                                                res_contribution=False,
                                                dist_threhold=kwargs['dist_threhold'], device=kwargs['device'])

        atids = ["%s%s" % (a.GetSymbol(), a.GetIdx()) for a in data.ligs[0].GetAtoms()]
        return data.ids, preds, atids, at_contrs

    elif res_contribution:
        preds, _, res_contrs = run_an_eval_epoch(model,
                                                 test_loader,
                                                 pred=True,
                                                 atom_contribution=False,
                                                 res_contribution=True,
                                                 dist_threhold=kwargs['dist_threhold'], device=kwargs['device'])
        u = mda.Universe(data.prot)
        resids = ["%s_%s%s" % (x[0], y, z) for x, y, z in
                  zip(u.residues.chainIDs, u.residues.resnames, u.residues.resids)]
        return data.ids, preds, resids, res_contrs
    else:
        preds = run_an_eval_epoch(model, test_loader, pred=True, dist_threhold=kwargs['dist_threhold'],
                                  device=kwargs['device'])
        return data.ids, preds
