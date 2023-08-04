import pandas as pd
import numpy as np
from rdkit import Chem
import torch as th
import re, os
import dgl
from itertools import product, groupby, permutations
from scipy.spatial import distance_matrix
from dgl.data.utils import save_graphs, load_graphs, load_labels
from joblib import Parallel, delayed
import MDAnalysis as mda
from MDAnalysis.analysis import dihedrals
from MDAnalysis.analysis import distances

METAL = ["LI","NA","K","RB","CS","MG","TL","CU","AG","BE","NI","PT","ZN","CO","PD","AG","CR","FE","V","MN","HG",'GA', 
		"CD","YB","CA","SN","PB","EU","SR","SM","BA","RA","AL","IN","TL","Y","LA","CE","PR","ND","GD","TB","DY","ER",
		"TM","LU","HF","ZR","CE","U","PU","TH"] 
RES_MAX_NATOMS=24

def prot_to_graph(prot, cutoff):
	"""obtain the residue graphs"""
	u = mda.Universe(prot)
	g = dgl.DGLGraph()
	# Add nodes
	num_residues = len(u.residues)
	g.add_nodes(num_residues)
	
	res_feats = np.array([calc_res_features(res) for res in u.residues])
	g.ndata["feats"] = th.tensor(res_feats)
	edgeids, distm = obatin_edge(u, cutoff)	
	src_list, dst_list = zip(*edgeids)
	g.add_edges(src_list, dst_list)
	
	g.ndata["ca_pos"] = th.tensor(np.array([obtain_ca_pos(res) for res in u.residues]))	
	g.ndata["center_pos"] = th.tensor(u.atoms.center_of_mass(compound='residues'))
	dis_matx_ca = distance_matrix(g.ndata["ca_pos"], g.ndata["ca_pos"])
	cadist = th.tensor([dis_matx_ca[i,j] for i,j in edgeids]) * 0.1
	dis_matx_center = distance_matrix(g.ndata["center_pos"], g.ndata["center_pos"])
	cedist = th.tensor([dis_matx_center[i,j] for i,j in edgeids]) * 0.1
	edge_connect =  th.tensor(np.array([check_connect(u, x, y) for x,y in zip(src_list, dst_list)]))
	g.edata["feats"] = th.cat([edge_connect.view(-1,1), cadist.view(-1,1), cedist.view(-1,1), th.tensor(distm)], dim=1)
	g.ndata.pop("ca_pos")
	g.ndata.pop("center_pos")
	#res_max_natoms = max([len(res.atoms) for res in u.residues])
	g.ndata["pos"] = th.tensor(np.array([np.concatenate([res.atoms.positions, np.full((RES_MAX_NATOMS-len(res.atoms), 3), np.nan)],axis=0) for res in u.residues]))
	#g.ndata["posmask"] = th.tensor([[1]* len(res.atoms)+[0]*(RES_MAX_NATOMS-len(res.atoms)) for res in u.residues]).bool()
	#g.ndata["atnum"] = th.tensor([len(res.atoms) for res in u.residues])
	return g


def obtain_ca_pos(res):
	if obtain_resname(res) == "M":
		return res.atoms.positions[0]
	else:
		try:
			pos = res.atoms.select_atoms("name CA").positions[0]
			return pos
		except:  ##some residues loss the CA atoms
			return res.atoms.positions.mean(axis=0)



def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def obtain_self_dist(res):
	try:
		#xx = res.atoms.select_atoms("not name H*")
		xx = res.atoms
		dists = distances.self_distance_array(xx.positions)
		ca = xx.select_atoms("name CA")
		c = xx.select_atoms("name C")
		n = xx.select_atoms("name N")
		o = xx.select_atoms("name O")
		return [dists.max()*0.1, dists.min()*0.1, distances.dist(ca,o)[-1][0]*0.1, distances.dist(o,n)[-1][0]*0.1, distances.dist(n,c)[-1][0]*0.1]
	except:
		return [0, 0, 0, 0, 0]


def obtain_dihediral_angles(res):
	try:
		if res.phi_selection() is not None:
			phi = res.phi_selection().dihedral.value()
		else:
			phi = 0
		if res.psi_selection() is not None:
			psi = res.psi_selection().dihedral.value()
		else:
			psi = 0
		if res.omega_selection() is not None:
			omega = res.omega_selection().dihedral.value()
		else:
			omega = 0
		if res.chi1_selection() is not None:
			chi1 = res.chi1_selection().dihedral.value()
		else:
			chi1 = 0
		return [phi*0.01, psi*0.01, omega*0.01, chi1*0.01]
	except:
		return [0, 0, 0, 0]

def calc_res_features(res):
	return np.array(one_of_k_encoding_unk(obtain_resname(res), 
										['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 
										'TRP', 'SER', 'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 
										'GLU', 'LYS', 'ARG', 'HIS', 'MSE', 'CSO', 'PTR', 'TPO',
										'KCX', 'CSD', 'SEP', 'MLY', 'PCA', 'LLP', 'M', 'X']) +          #32  residue type	
			obtain_self_dist(res) +  #5
			obtain_dihediral_angles(res) #4		
			)

def obtain_resname(res):
	if res.resname[:2] == "CA":
		resname = "CA"
	elif res.resname[:2] == "FE":
		resname = "FE"
	elif res.resname[:2] == "CU":
		resname = "CU"
	else:
		resname = res.resname.strip()
	
	if resname in METAL:
		return "M"
	else:
		return resname

##'FE', 'SR', 'GA', 'IN', 'ZN', 'CU', 'MN', 'SR', 'K' ,'NI', 'NA', 'CD' 'MG','CO','HG', 'CS', 'CA',

def obatin_edge(u, cutoff=10.0):
	edgeids = []
	dismin = []
	dismax = []
	for res1, res2 in permutations(u.residues, 2):
		dist = calc_dist(res1, res2)
		if dist.min() <= cutoff:
			edgeids.append([res1.ix, res2.ix])
			dismin.append(dist.min()*0.1)
			dismax.append(dist.max()*0.1)
	return edgeids, np.array([dismin, dismax]).T



def check_connect(u, i, j):
	if abs(i-j) != 1:
		return 0
	else:
		if i > j:
			i = j
		nb1 = len(u.residues[i].get_connections("bonds"))
		nb2 = len(u.residues[i+1].get_connections("bonds"))
		nb3 = len(u.residues[i:i+2].get_connections("bonds"))
		if nb1 + nb2 == nb3 + 1:
			return 1
		else:
			return 0
		
	

def calc_dist(res1, res2):
	#xx1 = res1.atoms.select_atoms('not name H*')
	#xx2 = res2.atoms.select_atoms('not name H*')
	#dist_array = distances.distance_array(xx1.positions,xx2.positions)
	dist_array = distances.distance_array(res1.atoms.positions,res2.atoms.positions)
	return dist_array
	#return dist_array.max()*0.1, dist_array.min()*0.1



def calc_atom_features(atom, explicit_H=False):
    """
    atom: rdkit.Chem.rdchem.Atom
    explicit_H: whether to use explicit H
    use_chirality: whether to use chirality
    """
    results = one_of_k_encoding_unk(
      atom.GetSymbol(),
      [
       'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 
		'Br', 'I', 'B', 'Si', 'Fe', 'Zn', 
		'Cu', 'Mn', 'Mo', 'other'
      ]) + one_of_k_encoding(atom.GetDegree(),
                             [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,'other']) + [atom.GetIsAromatic()]
                # [atom.GetIsAromatic()] # set all aromaticity feature blank.
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])	
    return np.array(results)


def calc_bond_features(bond, use_chirality=True):
    """
    bond: rdkit.Chem.rdchem.Bond
    use_chirality: whether to use chirality
    """
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats).astype(int)


	
def load_mol(molpath, explicit_H=False, use_chirality=True):
	# load mol
	if re.search(r'.pdb$', molpath):
		mol = Chem.MolFromPDBFile(molpath, removeHs=not explicit_H)
	elif re.search(r'.mol2$', molpath):
		mol = Chem.MolFromMol2File(molpath, removeHs=not explicit_H)
	elif re.search(r'.sdf$', molpath):			
		mol = Chem.MolFromMolFile(molpath, removeHs=not explicit_H)
	else:
		raise IOError("only the molecule files with .pdb|.sdf|.mol2 are supported!")	
	
	if use_chirality:
		Chem.AssignStereochemistryFrom3D(mol)
	return mol


def mol_to_graph(mol, explicit_H=False, use_chirality=True):
	"""
	mol: rdkit.Chem.rdchem.Mol
	explicit_H: whether to use explicit H
	use_chirality: whether to use chirality
	"""   	
				
	g = dgl.DGLGraph()
	# Add nodes
	num_atoms = mol.GetNumAtoms()
	g.add_nodes(num_atoms)
	
	atom_feats = np.array([calc_atom_features(a, explicit_H=explicit_H) for a in mol.GetAtoms()])
	if use_chirality:
		chiralcenters = Chem.FindMolChiralCenters(mol,force=True,includeUnassigned=True, useLegacyImplementation=False)
		chiral_arr = np.zeros([num_atoms,3]) 
		for (i, rs) in chiralcenters:
			if rs == 'R':
				chiral_arr[i, 0] =1 
			elif rs == 'S':
				chiral_arr[i, 1] =1 
			else:
				chiral_arr[i, 2] =1 
		atom_feats = np.concatenate([atom_feats,chiral_arr],axis=1)
			
	g.ndata["atom"] = th.tensor(atom_feats)
	
	# obtain the positions of the atoms
	atomCoords = mol.GetConformer().GetPositions()
	g.ndata["pos"] = th.tensor(atomCoords)
	
	# Add edges
	src_list = []
	dst_list = []
	bond_feats_all = []
	num_bonds = mol.GetNumBonds()
	for i in range(num_bonds):
		bond = mol.GetBondWithIdx(i)
		u = bond.GetBeginAtomIdx()
		v = bond.GetEndAtomIdx()
		bond_feats = calc_bond_features(bond, use_chirality=use_chirality)
		src_list.extend([u, v])
		dst_list.extend([v, u])		
		bond_feats_all.append(bond_feats)
		bond_feats_all.append(bond_feats)
	
	g.add_edges(src_list, dst_list)
	#normal_all = []
	#for i in etype_feature_all:
	#	normal = etype_feature_all.count(i)/len(etype_feature_all)
	#	normal = round(normal, 1)
	#	normal_all.append(normal)
	
	g.edata["bond"] = th.tensor(np.array(bond_feats_all))
	#g.edata["normal"] = th.tensor(normal_all)
	
	#dis_matx = distance_matrix(g.ndata["pos"], g.ndata["pos"])
	#g.edata["dist"] = th.tensor([dis_matx[i,j] for i,j in zip(*g.edges())]) * 0.1	
	return g



def mol_to_graph2(prot_path, lig_path, cutoff=10.0, explicit_H=False, use_chirality=True):
	prot = load_mol(prot_path, explicit_H=explicit_H, use_chirality=use_chirality) 
	lig = load_mol(lig_path, explicit_H=explicit_H, use_chirality=use_chirality)
	#gm = obtain_inter_graphs(prot, lig, cutoff=cutoff)
	#return gm
	#up = mda.Universe(prot)
	gp = prot_to_graph(prot, cutoff)
	gl = mol_to_graph(lig, explicit_H=explicit_H, use_chirality=use_chirality)
	return gp, gl



def pdbbind_handle(pdbid, args):
	prot_path = "%s/%s/%s_prot/%s_p_pocket_%s.pdb"%(args.dir, pdbid, pdbid, pdbid, args.cutoff)
	lig_path = "%s/%s/%s_prot/%s_l.sdf"%(args.dir, pdbid, pdbid, pdbid)
	try: 
		gp, gl = mol_to_graph2(prot_path, 
							lig_path, 
							cutoff=args.cutoff,
							explicit_H=args.useH, 
							use_chirality=args.use_chirality)
	except:
		print("%s failed to generare the graph"%pdbid)
		gp, gl = None, None
		#gm = None
	return pdbid, gp, gl


def UserInput():
	import argparse
	p = argparse.ArgumentParser()
	p.add_argument('-d', '--dir', default=".",
						help='The directory to store the protein-ligand complexes.')	
	p.add_argument('-c', '--cutoff', default=None, type=float,
						help='the cutoff to determine the pocket')	
	p.add_argument('-o', '--outprefix', default="out",
						help='The output bin file.')	
	p.add_argument('-usH', '--useH', default=False, action="store_true",
						help='whether to use the explicit H atoms.')
	p.add_argument('-uschi', '--use_chirality', default=False, action="store_true",
						help='whether to use chirality.')							
	p.add_argument('-p', '--parallel', default=False, action="store_true",
						help='whether to obtain the graphs in parallel (When the dataset is too large,\
						 it may be out of memory when conducting the parallel mode).')	
	
	args = p.parse_args()	
	return args



def main():
	args = UserInput()
	pdbids = [x for x in os.listdir(args.dir) if os.path.isdir("%s/%s"%(args.dir, x))]
	if args.parallel:
		results = Parallel(n_jobs=-1)(delayed(pdbbind_handle)(pdbid, args) for pdbid in pdbids)
	else:
		results = []
		for pdbid in pdbids:
			results.append(pdbbind_handle(pdbid, args))
	results = list(filter(lambda x: x[1] != None, results))
	#ids, graphs =  list(zip(*results))
	#np.save("%s_idsresx.npy"%args.outprefix, ids)
	#save_graphs("%s_plresx.bin"%args.outprefix, list(graphs))	
	ids, graphs_p, graphs_l =  list(zip(*results))
	np.save("%s_idsresz.npy"%args.outprefix, ids)
	save_graphs("%s_presz.bin"%args.outprefix, list(graphs_p))
	save_graphs("%s_lresz.bin"%args.outprefix, list(graphs_l))
	


if __name__ == '__main__':
	main()

