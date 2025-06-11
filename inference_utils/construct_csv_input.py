from pathlib import Path

import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import os
from argparse import ArgumentParser, Namespace, FileType

parser = ArgumentParser()
parser.add_argument(
    "--data_dir",
    required=True,
    type=Path,
    help="Input directory with full structure PDB files.",
)
parser.add_argument(
    "--surface_out_dir",
    required=True,
    type=Path,
    help=(
        "Output directory from compute mesh script, containing pocket structures and "
        "meshes."
    ),
)
parser.add_argument(
    "--ligands_to_dock",
    required=True,
    type=Path,
    help="Path to SDF file or directory of SDF files containing ligands to dock.",
)
parser.add_argument("--output_csv_file", required=True, type=Path, help="")
args = parser.parse_args()

# Make sure input paths exist
if not args.data_dir.exists():
    raise ValueError(f"{args.data_dir} doesn't exist")
if not args.surface_out_dir.exists():
    raise ValueError(f"{args.surface_out_dir} doesn't exist")
if not args.ligands_to_dock.exists():
    raise ValueError(f"{args.ligands_to_dock} doesn't exist")

# Make output directory for CSV file if it doesn't exist already
args.output_csv_file.parent.mkdir(parents=True, exist_ok=True)

# Dict that creates an empty list for the key if it's not already in the dict
args_list = defaultdict(list)

# Get all the protein names (directories in top-level dir)
proteins = [prot.name for prot in args.surface_out_dir.iterdir() if prot.is_dir()]
for protein in tqdm(proteins):
    # Make sure expected input files exist
    target_filename = args.data_dir / protein / f"{protein}.pdb"
    if not target_filename.exists():
        print(f"{target_filename} not found, skipping.", flush=True)

    pocket = args.surface_out_dir / protein / f"{protein}_8A.pdb"
    if not pocket.exists():
        print(f"{pocket} not found, skipping.", flush=True)

    surface = args.surface_out_dir / protein / f"{protein}_8A.ply"
    if not surface.exists():
        print(f"{surface} not found, skipping.", flush=True)

    # Add paths to dict
    args_list["protein_path"].append(target_filename)
    args_list["pocket_path"].append(pocket)
    args_list["protein_surface"].append(surface)
    args_list["ref_ligand"].append("")
    args_list["ligand_path"].append(args.ligands_to_dock)

pd.DataFrame(args_list).to_csv(args.output_csv_file, index=False)
