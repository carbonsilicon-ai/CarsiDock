"""
using prody to extract the pocket
"""
import os, re
import prody as pr
from openbabel import openbabel as ob
#import subprocess
#os.environ["BABEL_LIBDIR"] = "/home/shenchao/.conda/envs/my2/lib/openbabel/3.1.0"

def write_file(output_file, outline):
	buffer = open(output_file, 'w')
	buffer.write(outline)
	buffer.close()


def lig_rename(infile, outfile):
	##some peptides may impede the generation of pocket, so rename the ligname first.
	lines = open(infile, 'r').readlines()
	newlines = []
	for line in lines:		
		if re.search(r'^HETATM|^ATOM', line):
			newlines.append(line[:17] + "LIG" + line[20:])
		else:
			newlines.append(line)		
	write_file(outfile, ''.join(newlines))	


def check_mol(infile, outfile):
	# Some metals may have the same ID as ligand, thus making ligand included in the pocket.
	os.system("cat %s | sed '/LIG/d' > %s"%(infile, outfile))
	
	

def extract_pocket(protpath, 
					ligpath, 
					cutoff=5.0, 
					protname=None,
					ligname=None,
					workdir='.'):
	"""
		protpath: the path of protein file (.pdb).
		ligpath: the path of ligand file (.sdf|.mol2|.pdb).
		cutoff: the distance range within the ligand to determine the pocket.
		protname: the name of the protein.
		ligname: the name of the ligand.
		workdir: working directory.
	"""
	if protname is None:
		protname = os.path.basename(protpath).split('.')[0]
	if ligname is None:
		ligname = os.path.basename(ligpath).split('.')[0]
	obConversion = ob.OBConversion()
	obConversion.SetInAndOutFormats(ligpath.split('.')[-1], "pdb")
	
	if not re.search(r'.pdb$', ligpath):
		# convert ligand to pdb
		ligand = ob.OBMol()
		obConversion.ReadFile(ligand, ligpath)
		obConversion.WriteFile(ligand, "%s/%s.pdb"%(workdir, ligname))
	
	xprot = pr.parsePDB(protpath)
	#xlig = pr.parsePDB("%s/%s.pdb"%(workdir, ligname))
	
	#if (xlig.getResnames() == xlig.getResnames()[0]).all():
	#	lresname = xlig.getResnames()[0]
	#else:
	lig_rename("%s/%s.pdb"%(workdir, ligname), "%s/%s2.pdb"%(workdir, ligname))
	os.remove("%s/%s.pdb"%(workdir, ligname))
	os.rename("%s/%s2.pdb"%(workdir, ligname), "%s/%s.pdb"%(workdir, ligname)) 
	xlig = pr.parsePDB("%s/%s.pdb"%(workdir, ligname))
	lresname = xlig.getResnames()[0]
	xcom = xlig + xprot
	
	# select ONLY atoms that belong to the protein
	ret = xcom.select(f'same residue as exwithin %s of resname %s'%(cutoff, lresname))
	
	pr.writePDB("%s/%s_pocket_%s_temp.pdb"%(workdir, protname, cutoff), ret)
	#ret = pr.parsePDB("%s/%s_pocket_%s.pdb"%(workdir, protname, cutoff))
	
	check_mol("%s/%s_pocket_%s_temp.pdb"%(workdir, protname, cutoff), "%s/%s_pocket_%s.pdb"%(workdir, protname, cutoff))
	os.remove("%s/%s_pocket_%s_temp.pdb"%(workdir, protname, cutoff))






