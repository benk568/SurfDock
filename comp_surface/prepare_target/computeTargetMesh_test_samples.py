import os
from pathlib import Path
import re
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
    Adapted from chai1/input_parser.py

    :param asl: ASL string
    :return: list of tuples
    """
    chain_asl_pattern = r"\(\(chain\.name (\w)\s*\) AND \(res\.num (\d+(?:,\d+)*\s*)\)\)"
    full_asl_pattern = f"{chain_asl_pattern}(?: OR {chain_asl_pattern})*"
    # Dict mapping chain to list of residues
    pocket_selection = {}
    for match in re.finditer(full_asl_pattern, asl):
        chain, res_nums, _, _ = match.groups()
        for res_num in res_nums.split(","):
            try:
                pocket_selection[chain] += [int(res_num)]
            except KeyError:
                pocket_selection[chain] = [int(res_num)]

    # Can only have one binding pocket
    if len(pocket_selection) > 1:
        raise ValueError(
            f"Invalid ASL selection '{asl}', multiple chains is not supported."
        )
    if len(pocket_selection) == 0:
        raise ValueError(
            f"Invalid ASL selection '{asl}', need to specify at least one chain."
        )

    # Unpack dict into tuples of (chain_id, res_id_list) and return the first (only) one
    return tuple(*zip(pocket_selection.items()))[0]


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
        if ply_out_path.exists():
            print(f"{ply_out_path} already computed", flush=True)
            return 0

        # Read protein and select aminino acids in the binding pocket
        parser = Bio.PDB.PDBParser(QUIET=True) # QUIET=True avoids comments on errors in the pdb.

        structures = parser.get_structure("target", target_filename)
        structure = structures[0] # 'structures' may contain several proteins in this case only one.

        atoms  = Bio.PDB.Selection.unfold_entities(structure, "A")
        ns = Bio.PDB.NeighborSearch(atoms)

        if ligand_filename is None:
            # Convert passed ASL selection into a chain ID and residues
            asl_chain, asl_resids = asl_to_pocket_selection(pocket_asl)

            atom_coords = []
            for a in atoms:
                # Returns a lot of information we don't really need
                _, _, atom_chain, (_, resid, _), *_ = a.get_full_id()

                if (atom_chain == asl_chain) and (resid in asl_resids):
                    atom_coords += [a.coord]
            atom_coords = np.asarray(atom_coords)

        else:
            # Load ligand
            if ligand_filename.suffix == ".sdf":
                mol = Chem.SDMolSupplier(ligand_filename, sanitize=False)[0]
            elif ligand_filename.suffix == ".pdb":
                mol = Chem.MolFromPDBFile(ligand_filename, sanitize=False)
            else:
                raise ValueError(f"Unknown ligand format {ligand_filename.suffix}")

            # Convert loaded mol to graph
            g = mol_to_nx(mol)
            atom_coords = np.array([g.nodes[i]["pos"].tolist() for i in g.nodes])

        close_residues= []
        for a in atom_coords:
            close_residues.extend(ns.search(a, dist_threshold, level="R"))
        close_residues = Bio.PDB.Selection.uniqueify(close_residues)

        class SelectNeighbors(Select):
            def accept_residue(self, residue):
                if residue in close_residues:
                    # Make sure the residue has all backbone atoms (I think?)
                    return all(
                        [
                            a in [i.get_name() for i in residue.get_unpacked_list()]
                            for a in ["N", "CA", "C", "O"]
                        ]
                    ) or residue.resname=="HOH"
                else:
                    return False

        pdb_out_path = ply_out_path.with_suffix(".pdb")
        pdbio = PDBIO()
        pdbio.set_structure(structure)
        pdbio.save(pdb_out_path.open("w"), SelectNeighbors())

        # Identify closest atom to the ligand
        structure = parser.get_structure("target", pdb_out_path)
        # structure = structures["target"]
        atoms  = Bio.PDB.Selection.unfold_entities(structure, "A")
        print(len(atoms), "atoms", flush=True)

        #dist = [distance.euclidean(atom_coords.mean(axis=0), a.get_coord()) for a in atoms]
        #atom_idx = np.argmin(dist)
        #dist = [[distance.euclidean(ac, a.get_coord()) for ac in atom_coords] for a in atoms]
        #atom_idx = np.argsort(np.min(dist, axis=1))[0]

        # Compute MSMS of surface w/hydrogens,
        try:
            dist = [
                distance.euclidean(atom_coords.mean(axis=0), a.get_coord())
                for a in atoms
            ]
            atom_idx = np.argmin(dist)
            vertices1, faces1, normals1, names1, areas1 = computeMSMS(
                pdb_out_path, protonate=True, one_cavity=atom_idx
            )

            # Find the distance between every vertex in binding site surface and each atom in the ligand.
            kdt = KDTree(atom_coords)
            d, r = kdt.query(vertices1)
            assert(len(d) == len(vertices1))
            iface_v = np.where(d <= dist_threshold-5)[0]
            faces_to_keep = [idx for idx, face in enumerate(faces1) if all(v in iface_v  for v in face)]

            # Compute "charged" vertices
            if masif_opts["use_hbond"]:
                vertex_hbond = computeCharges(
                    str(target_filename.with_suffix("")), vertices1, names1
                )

            # For each surface residue, assign the hydrophobicity of its amino acid.
            if masif_opts["use_hphob"]:
                vertex_hphobicity = computeHydrophobicity(names1)

            # If protonate = false, recompute MSMS of surface, but without hydrogens (set radius of hydrogens to 0).
            vertices2 = vertices1
            faces2 = faces1

            # Fix the mesh.
            mesh = pymesh.form_mesh(vertices2, faces2)
            mesh = pymesh.submesh(mesh, faces_to_keep, 0)
            with io.capture_output() as captured:
                regular_mesh = fix_mesh(mesh, masif_opts["mesh_res"])

        except:
            try:
                dist = [
                    [distance.euclidean(ac, a.get_coord()) for ac in atom_coords]
                    for a in atoms
                ]
                atom_idx = np.argsort(np.min(dist, axis=1))[0]
                vertices1, faces1, normals1, names1, areas1 = computeMSMS(
                    pdb_out_path, protonate=True, one_cavity=atom_idx
                )

                # Find the distance between every vertex in binding site surface and each atom in the ligand.
                kdt = KDTree(atom_coords)
                d, r = kdt.query(vertices1)
                assert(len(d) == len(vertices1))
                iface_v = np.where(d <= dist_threshold-5)[0]
                faces_to_keep = [idx for idx, face in enumerate(faces1) if all(v in iface_v  for v in face)]

                # Compute "charged" vertices
                if masif_opts["use_hbond"]:
                    vertex_hbond = computeCharges(
                        str(target_filename.with_suffix("")), vertices1, names1
                    )

                # For each surface residue, assign the hydrophobicity of its amino acid.
                if masif_opts["use_hphob"]:
                    vertex_hphobicity = computeHydrophobicity(names1)

                # If protonate = false, recompute MSMS of surface, but without hydrogens (set radius of hydrogens to 0).
                vertices2 = vertices1
                faces2 = faces1

                # Fix the mesh.
                mesh = pymesh.form_mesh(vertices2, faces2)
                mesh = pymesh.submesh(mesh, faces_to_keep, 0)
                with io.capture_output() as captured:
                    regular_mesh = fix_mesh(mesh, masif_opts["mesh_res"])

            except:
                vertices1, faces1, normals1, names1, areas1 = computeMSMS(
                    pdb_out_path, protonate=True, one_cavity=None
                )

                # Find the distance between every vertex in binding site surface and each atom in the ligand.
                kdt = KDTree(atom_coords)
                d, r = kdt.query(vertices1)
                assert(len(d) == len(vertices1))
                iface_v = np.where(d <= dist_threshold-5)[0]
                faces_to_keep = [idx for idx, face in enumerate(faces1) if all(v in iface_v  for v in face)]

                # Compute "charged" vertices
                if masif_opts["use_hbond"]:
                    vertex_hbond = computeCharges(
                        str(target_filename.with_suffix("")), vertices1, names1
                    )

                # For each surface residue, assign the hydrophobicity of its amino acid.
                if masif_opts["use_hphob"]:
                    vertex_hphobicity = computeHydrophobicity(names1)

                # If protonate = false, recompute MSMS of surface, but without hydrogens (set radius of hydrogens to 0).
                vertices2 = vertices1
                faces2 = faces1

                # Fix the mesh.
                mesh = pymesh.form_mesh(vertices2, faces2)
                mesh = pymesh.submesh(mesh, faces_to_keep, 0)
                with io.capture_output() as captured:
                    regular_mesh = fix_mesh(mesh, masif_opts["mesh_res"])

        # Compute the normals
        vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)
        # Assign charges on new vertices based on charges of old vertices (nearest
        # neighbor)

        if masif_opts["use_hbond"]:
            vertex_hbond = assignChargesToNewMesh(
                regular_mesh.vertices, vertices1, vertex_hbond, masif_opts
            )

        if masif_opts["use_hphob"]:
            vertex_hphobicity = assignChargesToNewMesh(
                regular_mesh.vertices, vertices1, vertex_hphobicity, masif_opts
            )

        if masif_opts["use_apbs"]:
            # APBS expects us to be in the directory of the output files
            orig_dir = Path.cwd()
            os.chdir(pdb_out_path.parent)
            vertex_charges = computeAPBS(
                regular_mesh.vertices,
                pdb_out_path.name,
                f"{pdb_out_path.stem}_temp",
            )
            os.chdir(orig_dir)

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
        save_ply(
            str(ply_out_path),
            regular_mesh.vertices, regular_mesh.faces,
            normals=vertex_normal,
            charges=vertex_charges,
            normalize_charges=True,
            hbond=vertex_hbond,
            hphob=vertex_hphobicity,
            si=si
        )

        print(f"{target_filename.parent.name} succeeded", flush=True)
        return 0
    except Exception as e:
        print(f"{target_filename.parent.name} failed", flush=True)
        raise e


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
        if not protein_dir.is_dir():
            continue

        target_name = protein_dir.name
        prot_path = protein_dir / f"{target_name}.pdb"
        if not prot_path.exists():
            print(
                f"No protein found for {target_name}, skipping directory.", flush=True
            )
            continue

        lig_path = protein_dir / f"{target_name}_ligand.sdf"
        if not lig_path.exists():
            lig_path = protein_dir / f"{target_name}_ligand.mol2"
        if not lig_path.exists():
            print(
                f"No ligand found for {target_name}, using pocket_asl.", flush=True
            )
            lig_path = None
            if args.pocket_asl == "":
                raise ValueError(
                    "Need to specify --pocket_asl if no ligand is present."
                )
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
