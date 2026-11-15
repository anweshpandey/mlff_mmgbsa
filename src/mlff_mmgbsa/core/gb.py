"""GBOBC (Generalized Born with Onufriev-Bashford-Case) solvation implementation."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import torch
from scipy.spatial.distance import cdist
from mlff_mmgbsa.utils.constants import (
    get_gb_params,
    get_atom_radius,
    EPSILON_IN,
    EPSILON_OUT,
    NONPOLAR_GAMMA,
    NONPOLAR_BETA,
    KCAL_PER_MOL_TO_KJ_PER_MOL,
)


@dataclass
class GBResult:
    """Results from GB solvation calculation."""
    polar_solvation: float        # kcal/mol
    nonpolar_solvation: float     # kcal/mol
    total_solvation: float        # kcal/mol
    born_radii: np.ndarray        # Born radii per atom (Angstrom)
    pairwise_gb: Optional[np.ndarray] = None  # Pairwise GB matrix (n_atoms, n_atoms)


def compute_born_radii_hct(
    coords: np.ndarray,
    radii: np.ndarray,
    params: Dict[str, float],
    cutoff: float = 999.0,
) -> np.ndarray:
    """
    Compute Born radii using HCT (Hawkins-Cramer-Truhlar) approximation.
    
    This is the standard pairwise descreening approach used in AMBER's GBOBC.
    
    Args:
        coords: (n_atoms, 3) coordinates in Angstrom
        radii: (n_atoms,) intrinsic radii in Angstrom
        params: GB parameters dict with alpha, beta, gamma
        cutoff: Distance cutoff for pairwise interactions
    
    Returns:
        Born radii array (n_atoms,) in Angstrom
    """
    n_atoms = len(coords)
    if n_atoms == 1:
        return radii.copy()
    
    # Pairwise distances
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (n, n, 3)
    dist = np.sqrt(np.sum(diff**2, axis=2) + 1e-10)  # (n, n)
    
    # Mask self-interactions and beyond cutoff
    np.fill_diagonal(dist, np.inf)
    mask = dist < cutoff
    
    # HCT descreening: sum over j != i of (r_j^3 / (dist_ij^3 + r_j^3))
    # Vectorized computation
    r_j = radii[np.newaxis, :]  # (1, n)
    r_j3 = r_j**3
    
    # Avoid division by zero
    dist3 = dist**3
    dist3[~mask] = np.inf
    
    # Descreening sum for each atom i
    # S_i = sum_j (r_j^3 / (dist_ij^3 + r_j^3))
    with np.errstate(divide='ignore', invalid='ignore'):
        descreen = np.where(mask, r_j3 / (dist3 + r_j3), 0.0)
    descreen_sum = np.sum(descreen, axis=1)  # (n,)
    
    # Born radius: 1/rho_i = 1/r_i - alpha * S_i
    # rho_i = 1 / (1/r_i - alpha * S_i)
    alpha = params.get("alpha", 1.0)
    inv_radii = 1.0 / radii
    inv_born = inv_radii - alpha * descreen_sum
    
    # Ensure positive Born radii (clamp to small fraction of intrinsic)
    min_born = 0.1 * radii
    born_radii = 1.0 / np.maximum(inv_born, 1.0 / min_born)
    
    return born_radii


def compute_gb_pairwise(
    coords: np.ndarray,
    born_radii: np.ndarray,
    charges: np.ndarray,
    params: Dict[str, float],
    cutoff: float = 999.0,
) -> Tuple[float, np.ndarray]:
    """
    Compute pairwise GB interaction energy.
    
    E_GB = -0.5 * (1/eps_in - 1/eps_out) * sum_ij (q_i q_j / f_GB)
    
    where f_GB = sqrt(r_ij^2 + rho_i rho_j exp(-r_ij^2 / (4 rho_i rho_j)))
    
    Args:
        coords: (n, 3) coordinates
        born_radii: (n,) Born radii
        charges: (n,) partial charges
        params: GB parameters (gamma for OBC)
        cutoff: Distance cutoff
    
    Returns:
        (total_energy_kcal, pairwise_matrix)
    """
    n_atoms = len(coords)
    if n_atoms == 0:
        return 0.0, np.array([])
    
    # Pairwise distances
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2) + 1e-10)
    
    # Born radius products
    rho_i = born_radii[:, np.newaxis]
    rho_j = born_radii[np.newaxis, :]
    rho_prod = rho_i * rho_j
    
    # f_GB = sqrt(r_ij^2 + rho_i rho_j * exp(-r_ij^2 / (4 rho_i rho_j)))
    gamma = params.get("gamma", 1.0)
    arg = -dist**2 / (4.0 * rho_prod + 1e-10)
    arg = np.clip(arg, -50, 0)  # Prevent underflow
    f_gb = np.sqrt(dist**2 + rho_prod * np.exp(arg))
    
    # Charge products
    q_i = charges[:, np.newaxis]
    q_j = charges[np.newaxis, :]
    q_prod = q_i * q_j
    
    # Upper triangular only (i < j) to avoid double counting
    triu_indices = np.triu_indices(n_atoms, k=1)
    pairwise_energy = -0.5 * (1.0/EPSILON_IN - 1.0/EPSILON_OUT) * \
                      np.sum(q_prod[triu_indices] / f_gb[triu_indices])
    
    # Convert to kcal/mol (charges in elementary charge, distances in Angstrom)
    # 1/(4*pi*eps0) = 332.0637 kcal*Angstrom/mol/e^2
    COULOMB_CONST = 332.0637
    total_energy = pairwise_energy * COULOMB_CONST
    
    # Create full pairwise matrix for decomposition
    pairwise_matrix = -0.5 * (1.0/EPSILON_IN - 1.0/EPSILON_OUT) * \
                      q_prod / f_gb * COULOMB_CONST
    
    return total_energy, pairwise_matrix


def compute_sasa_lcpo(
    coords: np.ndarray,
    radii: np.ndarray,
    probe_radius: float = 1.4,
    n_points: int = 1000,
) -> float:
    """
    Approximate SASA using LCPO (Linear Combination of Pairwise Overlaps).
    This is a simplified version - full LCPO is more complex.
    
    For production, consider using `freesasa` or `mdtraj` for accurate SASA.
    """
    # Simplified: use pairwise overlap approximation
    # For now, use a rough estimate based on exposed surface
    n_atoms = len(coords)
    if n_atoms == 0:
        return 0.0
    
    # Simple approximation: SASA ~ sum 4*pi*r_i^2 * exposure_factor
    # Exposure factor estimated from neighbor count
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2) + 1e-10)
    np.fill_diagonal(dist, np.inf)
    
    # Count neighbors within (r_i + r_j + probe)
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :] + probe_radius
    buried = dist < r_sum
    neighbor_count = np.sum(buried, axis=1)
    
    # Rough exposure: fewer neighbors = more exposed
    max_neighbors = 12  # Typical max for close packing
    exposure = np.clip(1.0 - neighbor_count / max_neighbors, 0.1, 1.0)
    
    sasa = np.sum(4.0 * np.pi * radii**2 * exposure)
    return sasa


def compute_gb_solvation(
    coords: np.ndarray,
    elements: np.ndarray,
    charges: np.ndarray,
    igb: int = 8,
    compute_pairwise: bool = False,
) -> GBResult:
    """
    Full GB solvation energy calculation (polar + nonpolar).
    
    Args:
        coords: (n, 3) coordinates in Angstrom
        elements: (n,) element symbols
        charges: (n,) partial charges (in elementary charge units)
        igb: GB model (2, 5, 7, 8)
        compute_pairwise: Whether to return pairwise GB matrix
    
    Returns:
        GBResult with polar, nonpolar, total solvation energies
    """
    n_atoms = len(coords)
    if n_atoms == 0:
        return GBResult(
            polar_solvation=0.0,
            nonpolar_solvation=0.0,
            total_solvation=0.0,
            born_radii=np.array([]),
        )
    
    # Get intrinsic radii for each atom
    radii = np.array([get_atom_radius(el) for el in elements])
    
    # Get GB parameters
    params = get_gb_params(igb)
    
    # Compute Born radii
    born_radii = compute_born_radii_hct(coords, radii, params)
    
    # Compute polar solvation (GB)
    polar_energy, pairwise_matrix = compute_gb_pairwise(
        coords, born_radii, charges, params
    )
    
    # Compute nonpolar solvation (SASA-based)
    sasa = compute_sasa_lcpo(coords, radii)
    nonpolar_energy = NONPOLAR_GAMMA * sasa + NONPOLAR_BETA
    
    total = polar_energy + nonpolar_energy
    
    result = GBResult(
        polar_solvation=polar_energy,
        nonpolar_solvation=nonpolar_energy,
        total_solvation=total,
        born_radii=born_radii,
    )
    
    if compute_pairwise:
        result.pairwise_gb = pairwise_matrix
    
    return result


def compute_gb_solvation_torch(
    coords: torch.Tensor,
    elements: list,
    charges: torch.Tensor,
    igb: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch version for GPU acceleration and differentiability.
    
    Args:
        coords: (n, 3) tensor
        elements: list of element symbols
        charges: (n,) tensor
        igb: GB model
    
    Returns:
        (polar_energy, born_radii) as tensors
    """
    device = coords.device
    dtype = coords.dtype
    n_atoms = coords.shape[0]
    
    if n_atoms == 0:
        return torch.tensor(0.0, device=device, dtype=dtype), \
               torch.tensor([], device=device, dtype=dtype)
    
    # Radii
    radii_np = np.array([get_atom_radius(el) for el in elements])
    radii = torch.tensor(radii_np, device=device, dtype=dtype)
    
    params = get_gb_params(igb)
    alpha = params.get("alpha", 1.0)
    gamma = params.get("gamma", 1.0)
    
    # Pairwise distances
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # (n, n, 3)
    dist = torch.sqrt(torch.sum(diff**2, dim=2) + 1e-10)  # (n, n)
    dist.fill_diagonal_(float('inf'))
    
    # Born radii via HCT
    r_j = radii.unsqueeze(0)  # (1, n)
    r_j3 = r_j**3
    dist3 = dist**3
    
    descreen = r_j3 / (dist3 + r_j3)
    descreen.fill_diagonal_(0.0)
    descreen_sum = torch.sum(descreen, dim=1)  # (n,)
    
    inv_radii = 1.0 / radii
    inv_born = inv_radii - alpha * descreen_sum
    min_born = 0.1 * radii
    born_radii = 1.0 / torch.maximum(inv_born, 1.0 / min_born)
    
    # GB pairwise
    rho_i = born_radii.unsqueeze(1)
    rho_j = born_radii.unsqueeze(0)
    rho_prod = rho_i * rho_j
    
    arg = -dist**2 / (4.0 * rho_prod + 1e-10)
    arg = torch.clamp(arg, -50, 0)
    f_gb = torch.sqrt(dist**2 + rho_prod * torch.exp(arg))
    
    q_i = charges.unsqueeze(1)
    q_j = charges.unsqueeze(0)
    q_prod = q_i * q_j
    
    # Upper triangular
    triu_mask = torch.triu(torch.ones(n_atoms, n_atoms, device=device, dtype=torch.bool), diagonal=1)
    pairwise_energy = -0.5 * (1.0/EPSILON_IN - 1.0/EPSILON_OUT) * \
                      torch.sum(q_prod[triu_mask] / f_gb[triu_mask])
    
    COULOMB_CONST = 332.0637
    polar_energy = pairwise_energy * COULOMB_CONST
    
    return polar_energy, born_radii