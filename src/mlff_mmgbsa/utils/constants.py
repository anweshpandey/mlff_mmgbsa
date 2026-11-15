"""Physical constants and GB parameters for MMGBSA calculations."""

from typing import Dict, Literal
import numpy as np

# Physical constants
KCAL_PER_MOL_TO_KJ_PER_MOL = 4.184
ANGSTROM_TO_NM = 0.1
NA_AVOGADRO = 6.02214076e23

# GB model parameters (AMBER GBOBC variants)
# igb=2: OBC(I), igb=5: OBC(II), igb=7: Neck, igb=8: Neck2
GB_PARAMS: Dict[int, Dict[str, float]] = {
    2: {  # OBC(I) - Hawkins, Cramer, Truhlar J. Phys. Chem. B 1996, 100, 19824
        "alpha": 0.8,
        "beta": 0.0,
        "gamma": 2.909125,
        "name": "OBC(I)",
    },
    5: {  # OBC(II) - Onufriev, Bashford, Case Proteins 2000, 41, 361
        "alpha": 1.0,
        "beta": 0.8,
        "gamma": 4.85,
        "name": "OBC(II)",
    },
    7: {  # GB-Neck - Im et al. J. Comput. Chem. 2003, 24, 1691
        "alpha": 1.0,
        "beta": 0.8,
        "gamma": 4.85,
        "neck_scale": 0.73,
        "name": "GB-Neck",
    },
    8: {  # GB-Neck2 - Nguyen et al. J. Chem. Theory Comput. 2013, 9, 2020
        "alpha": 1.0,
        "beta": 0.8,
        "gamma": 4.85,
        "neck_scale": 0.73,
        "name": "GB-Neck2",
    },
}

# Nonpolar solvation parameters (LCPO-based approximation)
NONPOLAR_GAMMA = 0.00542  # kcal/mol/A^2 (surface tension)
NONPOLAR_BETA = 0.92      # kcal/mol (offset)

# Radii for common atom types (in Angstroms) - Bondi radii + modifications
ATOM_RADII: Dict[str, float] = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
    "F": 1.47,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "MG": 1.73,
    "CA": 1.99,
    "ZN": 1.39,
    "FE": 1.26,
    "MN": 1.61,
    "CU": 1.40,
}

# Default radius for unknown atoms
DEFAULT_RADIUS = 1.5

# Scaling factor for intrinsic radii (GBOBC)
RADIUS_SCALE = 1.0  # Can be adjusted per GB model

# Dielectric constants
EPSILON_IN = 1.0    # Internal dielectric
EPSILON_OUT = 78.5  # External dielectric (water)

# Temperature for entropy (if computed)
TEMPERATURE = 298.15  # K

# GB cutoff for pairwise interactions (Angstroms)
GB_CUTOFF = 999.0  # No cutoff by default


def get_atom_radius(element: str) -> float:
    """Get atomic radius for an element symbol."""
    return ATOM_RADII.get(element.upper(), DEFAULT_RADIUS)


def get_gb_params(igb: int) -> Dict[str, float]:
    """Get GB parameters for a given igb value."""
    if igb not in GB_PARAMS:
        raise ValueError(f"Unknown igb value: {igb}. Supported: {list(GB_PARAMS.keys())}")
    return GB_PARAMS[igb].copy()