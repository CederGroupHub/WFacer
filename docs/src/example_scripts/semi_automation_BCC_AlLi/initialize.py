"""An example to semi-automate with fireworks. Generate initial document."""

import json

from monty.json import jsanitize
from pymatgen.core import Lattice, Structure

from WFacer.jobs import initialize_document

# construct a BCC Al-Li structure
alli_prim = Structure(
    lattice=Lattice.cubic(3.75),
    species=[
        {"Al": 0.5, "Li": 0.5},
        {"Al": 0.5, "Li": 0.5},
    ],
    coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
)

# 333 supercell as objective.
# Loose relax, then tight, then static.
user_incar_settings_relax = {
    "ISMEAR": 1,
    "SIGMA": 0.2,
    "ENCUT": 300,
    "EDIFF": 1e-5,
    "EDIFFG": -0.02,
}
user_kpoints_settings_relax = {"reciprocal_density": 100}
user_incar_settings_tight = {
    "ISMEAR": 2,
    "SIGMA": 0.2,
    "ENCUT": 520,
    "EDIFF": 1e-5,
    "EDIFFG": -0.01,
}
user_kpoints_settings_tight = {"reciprocal_density": 400}
user_incar_settings_static = {"ENCUT": 680, "EDIFF": 1e-6, "EDIFFG": -0.01}
user_kpoints_settings_static = {"reciprocal_density": 800}

relax_kwargs = {
    "user_incar_settings": user_incar_settings_relax,
    "user_kpoints_settings": user_kpoints_settings_relax,
    "user_potcar_functional": "PBE_54",
}
tight_kwargs = {
    "user_incar_settings": user_incar_settings_tight,
    "user_kpoints_settings": user_kpoints_settings_tight,
    "user_potcar_functional": "PBE_54",
}
static_kwargs = {
    "user_incar_settings": user_incar_settings_static,
    "user_kpoints_settings": user_kpoints_settings_static,
    "user_potcar_functional": "PBE_54",
}

# Lasso, grid-search.
options = {
    "objective_num_sites": 54,
    "comp_enumeration_step": 3,
    "n_parallel": 8,
    "add_tight_relax": True,
    "relax_generator_kwargs": relax_kwargs,
    "tight_generator_kwargs": tight_kwargs,
    "static_generator_kwargs": static_kwargs,
    "cutoffs": {2: 9.0, 3: 8.0, 4: 5.0},
}

document = initialize_document(alli_prim, "alli_bcc_ce", options=options)
data = jsanitize(document, strict=True, enum_values=True)
with open("document.json", "w") as fout:
    json.dump(data, fout)
