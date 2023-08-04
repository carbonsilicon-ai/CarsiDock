import torch as th
import dgl
from dgl.data.utils import load_graphs
from torch.utils.data import Dataset #, DataLoader
import pandas as pd
import numpy as np    
#from copy import deepcopy
from rdkit import Chem
from joblib import Parallel, delayed
import os
import tempfile
import shutil
from ..feats.mol2graph_rdmda_res import mol_to_graph, load_mol, prot_to_graph
from ..feats.extract_pocket_prody import extract_pocket




class PDBbindDataset(Dataset):
	def __init__(self,  
				ids=None,
				ligs=None,
				prots=None
				):
		if isinstance(ids,np.ndarray) or isinstance(ids,list):
			self.pdbids = ids
		else:
			try:
				self.pdbids = np.load(ids)
			except:
				raise ValueError('the variable "ids" should be numpy.ndarray or list or a file to store numpy.ndarray')
		if isinstance(ligs,np.ndarray) or isinstance(ligs,tuple) or isinstance(ligs,list):
			if isinstance(ligs[0],dgl.DGLGraph):
				self.graphsl = ligs
			else:
				raise ValueError('the variable "ligs" should be a set of (or a file to store) dgl.DGLGraph objects.')
		else:
			try:
				self.graphsl, _ = load_graphs(ligs) 
			except:
				raise ValueError('the variable "ligs" should be a set of (or a file to store) dgl.DGLGraph objects.')
		
		if isinstance(prots,np.ndarray) or isinstance(prots,tuple) or isinstance(prots,list):
			if isinstance(prots[0],dgl.DGLGraph):
				self.graphsp = prots
			else:
				raise ValueError('the variable "prots" should be a set of (or a file to store) dgl.DGLGraph objects.')	
		else:
			try:
				self.graphsp, _ = load_graphs(prots) 
			except:
				raise ValueError('the variable "prots" should be a set of (or a file to store) dgl.DGLGraph objects.')
		
		self.graphsl = list(self.graphsl)
		self.graphsp = list(self.graphsp)
		assert len(self.pdbids) == len(self.graphsl) == len(self.graphsp)
		
	def __getitem__(self, idx): 
		""" Get graph and label by index
		
        Parameters
        ----------
        idx : int
            Item index
	
		Returns
		-------
		(dgl.DGLGraph, Tensor)
		"""
		return self.pdbids[idx], self.graphsl[idx], self.graphsp[idx]
	
	
	def __len__(self):
		"""Number of graphs in the dataset"""
		return len(self.pdbids)			
	
	
	def train_and_test_split(self, valfrac=0.2, valnum=None, seed=0):
		#random.seed(seed)
		np.random.seed(seed)
		if valnum is None:
			valnum = int(valfrac * len(self.pdbids))
		val_inds = np.random.choice(np.arange(len(self.pdbids)),valnum, replace=False)
		train_inds = np.setdiff1d(np.arange(len(self.pdbids)),val_inds)
		return train_inds, val_inds
		
		
class VSDataset(Dataset):
	def __init__(self,  
				ids=None,
				ligs=None,
				prot=None,
				gen_pocket=False,
				cutoff=None,
				reflig=None,
				explicit_H=False, 
				use_chirality=True,
				parallel=True			
				):
		self.graphp=None
		self.graphsl=None
		self.pocketdir = None
		self.prot = None
		self.ligs = None
		self.cutoff = cutoff
		self.explicit_H=explicit_H
		self.use_chirality=use_chirality
		self.parallel=parallel
		
		if isinstance(prot, Chem.rdchem.Mol):
			assert gen_pocket == False
			self.prot = prot
			self.graphp = prot_to_graph(self.prot, cutoff)
		else:
			if gen_pocket:
				if cutoff is None or reflig is None:
					raise ValueError('If you want to generate the pocket, the cutoff and the reflig should be given')
				try:
					self.pocketdir = tempfile.mkdtemp()
					extract_pocket(prot, reflig, cutoff, 
								protname="temp",
								workdir=self.pocketdir)
					pocket = load_mol("%s/temp_pocket_%s.pdb"%(self.pocketdir, cutoff), 
								explicit_H=explicit_H, use_chirality=use_chirality)
					self.prot = pocket
					self.graphp = prot_to_graph(self.prot, cutoff)
				except:
					raise ValueError('The graph of pocket cannot be generated')
			else:
				try:
					pocket = load_mol(prot, explicit_H=explicit_H, use_chirality=use_chirality)
					#self.graphp = mol_to_graph(pocket, explicit_H=explicit_H, use_chirality=use_chirality)	
					self.prot = pocket
					self.graphp = prot_to_graph(self.prot, cutoff)
				except:
					raise ValueError('The graph of pocket cannot be generated')
			
		if isinstance(ligs,np.ndarray) or isinstance(ligs,list):
			if isinstance(ligs[0], Chem.rdchem.Mol):
				self.ligs = ligs
				self.graphsl = self._mol_to_graph()
			elif isinstance(ligs[0], dgl.DGLGraph):
				self.graphsl = ligs
			else:
				raise ValueError('Ligands should be a list of rdkit.Chem.rdchem.Mol objects')
		else:
			if ligs.endswith(".mol2"):
				lig_blocks = self._mol2_split(ligs)	
				self.ligs = [Chem.MolFromMol2Block(lig_block) for lig_block in lig_blocks]
				self.graphsl = self._mol_to_graph()
			elif ligs.endswith(".sdf"):
				lig_blocks = self._sdf_split(ligs)	
				self.ligs = [Chem.MolFromMolBlock(lig_block) for lig_block in lig_blocks]
				self.graphsl = self._mol_to_graph()
			else:
				try:	
					self.graphsl,_ = load_graphs(ligs)
				except:
					raise ValueError('Only the ligands with .sdf or .mol2 or a file to genrate DGLGraphs will be supported')
		
		if ids is None:
			if self.ligs is not None:
				self.idsx = ["%s-%s"%(self.get_ligname(lig),i) for i, lig in enumerate(self.ligs)]
			else:
				self.idsx = ["lig%s"%i for i in range(len(self.graphsl))]
		else:
			self.idsx = ids

		self.ids, self.graphsl = zip(*filter(lambda x: x[1] != None, zip(self.idsx, self.graphsl)))
		self.ids = list(self.ids)
		self.graphsl = list(self.graphsl)
		assert len(self.ids) == len(self.graphsl)
		if self.pocketdir is not None:
			shutil.rmtree(self.pocketdir)
		
	def __getitem__(self, idx): 
		""" Get graph and label by index
	
        Parameters
        ----------
        idx : int
            Item index
	
		Returns
        -------
        (dgl.DGLGraph, Tensor)
        """
		return self.ids[idx], self.graphsl[idx], self.graphp
	
	def __len__(self):
		"""Number of graphs in the dataset"""
		return len(self.ids)	
		
	def _mol2_split(self, infile):
		contents = open(infile, 'r').read()
		return ["@<TRIPOS>MOLECULE\n" + c for c in contents.split("@<TRIPOS>MOLECULE\n")[1:]]
	
	def _sdf_split(self, infile):
		contents = open(infile, 'r').read()
		return [c + "$$$$\n" for c in contents.split("$$$$\n")[:-1]]
	
	def _mol_to_graph0(self, lig):
		try:
			gx = mol_to_graph(lig, explicit_H=self.explicit_H, use_chirality=self.use_chirality)
		except:
			print("failed to scoring for {} and {}".format(self.graphp, lig))
			return None
		return gx

	def _mol_to_graph(self):
		if self.parallel:
			return Parallel(n_jobs=-1, backend="threading")(delayed(self._mol_to_graph0)(lig) for lig in self.ligs)
		else:
			graphs = []
			for lig in self.ligs:
				graphs.append(self._mol_to_graph0(lig))
			return graphs
	
	def get_ligname(self, m):
		if m is None:
			return None
		else:
			if m.HasProp("_Name"):
				return m.GetProp("_Name")
			else:
				return None
	

	
