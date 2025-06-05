import os
from pathlib import Path
import sys
import numpy as np
import shutil
import glob
import pymesh
import Bio.PDB
from Bio.PDB import *
from rdkit import Chem
import warnings
warnings.filterwarnings("ignore")
from IPython.utils import io
from sklearn.neighbors import KDTree
from scipy.spatial import distance

from default_config.masif_opts import masif_opts
from compute_normal import compute_normal
from computeAPBS import computeAPBS
from computeCharges import computeCharges, assignChargesToNewMesh
from computeHydrophobicity import computeHydrophobicity
from computeMSMS import computeMSMS
from fixmesh import fix_mesh
from save_ply import save_ply
from mol2graph import *


def asl_to_pocket_selection(asl: str) -> str:
    """
    Take in a string like "((chain.name A ) AND (res.num 300,301)) OR ((chain.name B ) AND (res.num 351))"
    and return a str like "A_300,A_301,B_351"
    Borrowed from chai1/input_parser.py

    :param asl: ASL string
    :return: list of tuples
    """
    chain_asl_pattern = r'\(\(chain\.name (\w)\s*\) AND \(res\.num (\d+(?:,\d+)*\s*)\)\)'
    full_asl_pattern = f'{chain_asl_pattern}(?: OR {chain_asl_pattern})*'
    pocket_selection = []
    for match in re.finditer(full_asl_pattern, asl):
        chain, res_nums, _, _ = match.groups()
        for res_num in res_nums.split(','):
            pocket_selection.append(f"{chain}_{res_num}")
    return ",".join(pocket_selection)


def compute_inp_surface(
    target_filename, ligand_filename, out_dir, dist_threshold=10, pocket_asl=""
):
    try:
        out_dir = out_dir / target_filename.parent.name
        out_dir.mkdir(exist_ok=True)

        # sufix = '/' + os.path.splitext(target_filename)[0].split('/')[-1] + '_'+str(dist_threshold)+'A.pdb'

        # This is the file we ultimately want, can just skip everything if it already
        #  exists
        ply_out_path = out_dir / f"{target_filename.name.split('.')[0]}_{dist_threshold}A.ply"
        if ply_out_path:
            print(f"{ply_out_path} already computed", flush=True)
            return 0

        if ligand_filename.suffix == ".sdf":
            mol = Chem.SDMolSupplier(ligand_filename, sanitize=False)[0]
        elif ligand_filename.suffix == ".pdb":
            mol = Chem.MolFromPDBFile(ligand_filename, sanitize=False)
        g = mol_to_nx(mol)
        atomCoords = np.array([g.nodes[i]['pos'].tolist() for i in g.nodes])

        # Read protein and select aminino acids in the binding pocket
        parser = Bio.PDB.PDBParser(QUIET=True) # QUIET=True avoids comments on errors in the pdb.

        structures = parser.get_structure('target', target_filename)
        structure = structures[0] # 'structures' may contain several proteins in this case only one.

        atoms  = Bio.PDB.Selection.unfold_entities(structure, 'A')
        ns = Bio.PDB.NeighborSearch(atoms)

        close_residues= []
        for a in atomCoords:
            close_residues.extend(ns.search(a, dist_threshold, level='R'))
        close_residues = Bio.PDB.Selection.uniqueify(close_residues)

        class SelectNeighbors(Select):
            def accept_residue(self, residue):
                if residue in close_residues:
                    if all(a in [i.get_name() for i in residue.get_unpacked_list()] for a in ['N', 'CA', 'C', 'O']) or residue.resname=='HOH':
                        return True
                    else:
                        return False
                else:
                    return False

        pdbio = PDBIO()
        pdbio.set_structure(structure)
        pdbio.save(out_filename+sufix, SelectNeighbors())

        # Identify closes atom to the ligand
        structures = parser.get_structure('target', out_filename+sufix)
        structure = structures[0] # 'structures' may contain several proteins in this case only one.
        atoms  = Bio.PDB.Selection.unfold_entities(structure, 'A')

        #dist = [distance.euclidean(atomCoords.mean(axis=0), a.get_coord()) for a in atoms]
        #atom_idx = np.argmin(dist)
        #dist = [[distance.euclidean(ac, a.get_coord()) for ac in atomCoords] for a in atoms]
        #atom_idx = np.argsort(np.min(dist, axis=1))[0]

        # Compute MSMS of surface w/hydrogens,
        try:
            dist = [distance.euclidean(atomCoords.mean(axis=0), a.get_coord()) for a in atoms]
            atom_idx = np.argmin(dist)
            vertices1, faces1, normals1, names1, areas1 = computeMSMS(out_filename+sufix,\
                                                                    protonate=True, one_cavity=atom_idx)

            # Find the distance between every vertex in binding site surface and each atom in the ligand.
            kdt = KDTree(atomCoords)
            d, r = kdt.query(vertices1)
            assert(len(d) == len(vertices1))
            iface_v = np.where(d <= dist_threshold-5)[0]
            faces_to_keep = [idx for idx, face in enumerate(faces1) if all(v in iface_v  for v in face)]

            # Compute "charged" vertices
            if masif_opts['use_hbond']:
                vertex_hbond = computeCharges(input_filename, vertices1, names1)

            # For each surface residue, assign the hydrophobicity of its amino acid.
            if masif_opts['use_hphob']:
                vertex_hphobicity = computeHydrophobicity(names1)

            # If protonate = false, recompute MSMS of surface, but without hydrogens (set radius of hydrogens to 0).
            vertices2 = vertices1
            faces2 = faces1

            # Fix the mesh.
            mesh = pymesh.form_mesh(vertices2, faces2)
            mesh = pymesh.submesh(mesh, faces_to_keep, 0)
            with io.capture_output() as captured:
                regular_mesh = fix_mesh(mesh, masif_opts['mesh_res'])

        except:
            try:
                dist = [[distance.euclidean(ac, a.get_coord()) for ac in atomCoords] for a in atoms]
                atom_idx = np.argsort(np.min(dist, axis=1))[0]
                vertices1, faces1, normals1, names1, areas1 = computeMSMS(out_filename+sufix,\
                                                                        protonate=True, one_cavity=atom_idx)

                # Find the distance between every vertex in binding site surface and each atom in the ligand.
                kdt = KDTree(atomCoords)
                d, r = kdt.query(vertices1)
                assert(len(d) == len(vertices1))
                iface_v = np.where(d <= dist_threshold-5)[0]
                faces_to_keep = [idx for idx, face in enumerate(faces1) if all(v in iface_v  for v in face)]

                # Compute "charged" vertices
                if masif_opts['use_hbond']:
                    vertex_hbond = computeCharges(input_filename, vertices1, names1)

                # For each surface residue, assign the hydrophobicity of its amino acid.
                if masif_opts['use_hphob']:
                    vertex_hphobicity = computeHydrophobicity(names1)

                # If protonate = false, recompute MSMS of surface, but without hydrogens (set radius of hydrogens to 0).
                vertices2 = vertices1
                faces2 = faces1

                # Fix the mesh.
                mesh = pymesh.form_mesh(vertices2, faces2)
                mesh = pymesh.submesh(mesh, faces_to_keep, 0)
                with io.capture_output() as captured:
                    regular_mesh = fix_mesh(mesh, masif_opts['mesh_res'])

            except:
                vertices1, faces1, normals1, names1, areas1 = computeMSMS(out_filename+sufix,\
                                                                        protonate=True, one_cavity=None)

                # Find the distance between every vertex in binding site surface and each atom in the ligand.
                kdt = KDTree(atomCoords)
                d, r = kdt.query(vertices1)
                assert(len(d) == len(vertices1))
                iface_v = np.where(d <= dist_threshold-5)[0]
                faces_to_keep = [idx for idx, face in enumerate(faces1) if all(v in iface_v  for v in face)]

                # Compute "charged" vertices
                if masif_opts['use_hbond']:
                    vertex_hbond = computeCharges(input_filename, vertices1, names1)

                # For each surface residue, assign the hydrophobicity of its amino acid.
                if masif_opts['use_hphob']:
                    vertex_hphobicity = computeHydrophobicity(names1)

                # If protonate = false, recompute MSMS of surface, but without hydrogens (set radius of hydrogens to 0).
                vertices2 = vertices1
                faces2 = faces1

                # Fix the mesh.
                mesh = pymesh.form_mesh(vertices2, faces2)
                mesh = pymesh.submesh(mesh, faces_to_keep, 0)
                with io.capture_output() as captured:
                    regular_mesh = fix_mesh(mesh, masif_opts['mesh_res'])

        # Compute the normals
        vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)
        # Assign charges on new vertices based on charges of old vertices (nearest
        # neighbor)

        if masif_opts['use_hbond']:
            vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1,\
                vertex_hbond, masif_opts)

        if masif_opts['use_hphob']:
            vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1,\
                vertex_hphobicity, masif_opts)

        if masif_opts['use_apbs']:
            vertex_charges = computeAPBS(regular_mesh.vertices, out_filename+sufix, out_filename+"_temp")

        # Compute the principal curvature components for the shape index.
        regular_mesh.add_attribute("vertex_mean_curvature")
        H = regular_mesh.get_attribute("vertex_mean_curvature")
        regular_mesh.add_attribute("vertex_gaussian_curvature")
        K = regular_mesh.get_attribute("vertex_gaussian_curvature")
        elem = np.square(H) - K
        # In some cases this equation is less than zero, likely due to the method that computes the mean and gaussian curvature.
        # set to an epsilon.
        elem[elem<0] = 1e-8
        k1 = H + np.sqrt(elem)
        k2 = H - np.sqrt(elem)
        # Compute the shape index
        si = (k1+k2)/(k1-k2)
        si = np.arctan(si)*(2/np.pi)

        # Convert to ply and save.
        save_ply(out_filename+f"/{sufix.split('.')[0]}.ply", regular_mesh.vertices,\
                regular_mesh.faces, normals=vertex_normal, charges=vertex_charges,\
                normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity,\
                si=si)

        return 0
    except:
        return target_filename


if __name__ == "__main__":
    from joblib import delayed,Parallel
    # arguments
    from argparse import ArgumentParser, Namespace, FileType
    parser = ArgumentParser()
    parser.add_argument("--data_dir", required=True, type=Path,  help="")
    parser.add_argument("--out_dir", required=True, type=Path, help="")
    parser.add_argument("--pools", type=int, default=1, help="")
    parser.add_argument(
        "--pocket_asl",
        type=str,
        default="",
        help="ASL string specifying the pocket if there's no ligand to define it.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    # sys.path.append(args.out_dir)
    from tqdm import tqdm
    args_list = []
    # Loop through all child directories. Each one should be named after the protein
    #  inside, and should contain at minimum a protein PDB file (if pocket_asl is
    #  specified), or a protein PDB file and a ligand SDF file
    for protein_dir in tqdm(list(args.data_dir.iterdir())):
        if not protein_dir.isdir():
            continue

        target_name = protein_dir.name
        prot_path = protein_dir / f"{target_name}.pdb"
        lig_path = protein_dir / f"{target_name}_ligand.sdf"
        if not lig_path.exists():
            lig_path = protein_dir / f"{target_name}_ligand.mol2"
        if not lig_path.exists():
            print(
                f"No ligand found for {target_name}, using pocket_asl.", flush=True
            )
            lig_path = None
        args_list.append((prot_path, lig_path, args.pocket_asl))
    print(f"{len(args_list)} target structure found.", flush=True)
    results = Parallel(
        n_jobs = 30,backend = "multiprocessing"
    )(
        delayed(compute_inp_surface)(
            prot_path,
            lig_path,
            args.out_dir,
            dist_threshold=8,
            pocket_asl=pocket_asl,
        )
        for (prot_path, lig_path, pocket_asl) in tqdm(args_list)
    )

    # Delete all temp files generated in the process
    for f in args.out_dir.glob("*_temp*"):
        f.unlink()
    for f in args.out_dir.glob("*msms*"):
        f.unlink()
