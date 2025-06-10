# global_vars.py: Global variables used by MaSIF -- mainly pointing to environment variables of programs used by MaSIF.
# Pablo Gainza - LPDI STI EPFL 2018-2019
# Released under an Apache License 2.0

import os
epsilon = 1.0e-6
import sys

default_env_vars = {
   "MSMS_BIN": "~/SurfDock/comp_surface/tools/APBS-3.4.1.Linux/bin/msms",
   "PDB2PQR_BIN": "~/SurfDock/comp_surface/tools/pdb2pqr-linux-bin64-2.1.1/pdb2pqr",
    "APBS_BIN": "~/SurfDock/comp_surface/tools/APBS-3.4.1.Linux/bin/apbs",
    "MULTIVALUE_BIN": "~/SurfDock/comp_surface/tools/APBS-3.4.1.Linux/share/apbs/tools/bin/multivalue"
}

msms_bin = os.environ.get("MSMS_BIN", default_env_vars["MSMS_BIN"])
pdb2pqr_bin = os.environ.get("PDB2PQR_BIN", default_env_vars["PDB2PQR_BIN"])
apbs_bin = os.environ.get("APBS_BIN", default_env_vars["APBS_BIN"])
multivalue_bin = os.environ.get("MULTIVALUE_BIN", default_env_vars["MULTIVALUE_BIN"])
class NoSolutionError(Exception):
    pass
