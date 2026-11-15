"""Per-residue energy decomposition for MMGBSA."""

from dataclasses import dataclass
from typing import List, Dict, Optional, Literal
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from ase import Atoms
from mlff_mmgbsa.core.models import MACEModel
from mlff_mmgbsa.core.energy import compute_system_energy, SystemEnergy
from mlff_mmgbsa.core.snapshot import ProcessedSnapshot
from mlff_mmgbsa.utils.topology import (
    get_residue_atom_indices,
    SystemSelection,
    extract_subsystem,
)


@dataclass
class ResidueDecomposition:
    """Per-residue energy decomposition for one frame."""
    frame_id: int
    residue_ids: np.ndarray           # Residue indices
    residue_names: np.ndarray         # Residue names
    chain_ids: np.ndarray             # Chain IDs
    delta_g_contrib: np.ndarray       # Contribution to binding energy
    mlff_contrib: np.ndarray          # MLFF contribution
    gb_polar_contrib: np.ndarray      # GB polar contribution
    gb_nonpolar_contrib: np.ndarray   # GB nonpolar contribution
    # Per-system energies for each residue
    complex_mlff: np.ndarray
    protein_mlff: np.ndarray
    ligand_mlff: np.ndarray
    complex_gb_polar: np.ndarray
    protein_gb_polar: np.ndarray
    ligand_gb_polar: np.ndarray


@dataclass
class DecompositionResult:
    """Full decomposition results across frames."""
    per_frame: List[ResidueDecomposition]
    mean_contrib: np.ndarray          # Mean per-residue contribution
    std_contrib: np.ndarray           # Std per-residue contribution
    residue_ids: np.ndarray
    residue_names: np.ndarray
    chain_ids: np.ndarray


def compute_residue_decomposition_frame(
    model: MACEModel,
    snapshot: ProcessedSnapshot,
    igb: int = 8,
    method: Literal["alanine_scan", "interaction"] = "interaction",
    charge_method: str = "mace",
) -> ResidueDecomposition:
    """
    Compute per-residue decomposition for a single frame.
    
    Methods:
    - "alanine_scan": Mutate each residue to Ala, recompute (expensive, accurate)
    - "interaction": Pairwise interaction energy decomposition (approximate)
    
    For MLFF, alanine scanning requires structure modification which is complex.
    The interaction method uses energy decomposition of the complex.
    """
    selection = snapshot.selection
    
    # Get protein residue atom indices
    protein_residue_indices = get_residue_atom_indices(
        snapshot.complex_atoms, selection, "protein"
    )
    protein_residue_ids = np.unique(selection.protein_residue_ids)
    protein_residue_names = []
    protein_chain_ids = []
    
    for res_id in protein_residue_ids:
        mask = selection.protein_residue_ids == res_id
        protein_residue_names.append(selection.residue_names[selection.protein_indices][mask][0])
        protein_chain_ids.append(selection.chain_ids[selection.protein_indices][mask][0])
    
    n_residues = len(protein_residue_indices)
    
    if method == "alanine_scan":
        return _alanine_scan_decomposition(
            model, snapshot, protein_residue_indices, protein_residue_ids,
            protein_residue_names, protein_chain_ids, igb, charge_method
        )
    else:
        return _interaction_decomposition(
            model, snapshot, protein_residue_indices, protein_residue_ids,
            protein_residue_names, protein_chain_ids, igb, charge_method
        )


def _interaction_decomposition(
    model: MACEModel,
    snapshot: ProcessedSnapshot,
    protein_residue_indices: List[np.ndarray],
    residue_ids: np.ndarray,
    residue_names: List[str],
    chain_ids: List[str],
    igb: int,
    charge_method: str,
) -> ResidueDecomposition:
    """
    Interaction-based decomposition.
    
    Approximates per-residue contribution by computing interaction energy
    between each residue and the ligand in the complex.
    """
    n_residues = len(protein_residue_indices)
    ligand_indices = snapshot.selection.ligand_indices
    
    # Reference energies
    complex_energy = compute_system_energy(model, snapshot.complex_atoms, igb, charge_method)
    protein_energy = compute_system_energy(model, snapshot.protein_atoms, igb, charge_method)
    ligand_energy = compute_system_energy(model, snapshot.ligand_atoms, igb, charge_method)
    
    delta_g_bind = (complex_energy.total_energy 
                    - protein_energy.total_energy 
                    - ligand_energy.total_energy)
    
    # For each residue, compute interaction with ligand
    # This is approximate: E_int = E(residue+ligand) - E(residue) - E(ligand)
    # But we don't have isolated residue energies easily.
    
    # Alternative: Use GB pairwise decomposition + MLFF energy decomposition
    # For MLFF, we can use the fact that energy is sum of atomic contributions
    # (MACE gives per-atom energies via the message passing)
    
    # Simplified approach: distribute total binding energy by residue-ligand contacts
    # This is a placeholder - full implementation needs per-atom MLFF energies
    
    # For now, return zero contributions with correct structure
    delta_g_contrib = np.zeros(n_residues)
    mlff_contrib = np.zeros(n_residues)
    gb_polar_contrib = np.zeros(n_residues)
    gb_nonpolar_contrib = np.zeros(n_residues)
    
    # Per-system placeholders
    complex_mlff = np.zeros(n_residues)
    protein_mlff = np.zeros(n_residues)
    ligand_mlff = np.zeros(n_residues)
    complex_gb_polar = np.zeros(n_residues)
    protein_gb_polar = np.zeros(n_residues)
    ligand_gb_polar = np.zeros(n_residues)
    
    return ResidueDecomposition(
        frame_id=snapshot.frame_id,
        residue_ids=residue_ids,
        residue_names=np.array(residue_names),
        chain_ids=np.array(chain_ids),
        delta_g_contrib=delta_g_contrib,
        mlff_contrib=mlff_contrib,
        gb_polar_contrib=gb_polar_contrib,
        gb_nonpolar_contrib=gb_nonpolar_contrib,
        complex_mlff=complex_mlff,
        protein_mlff=protein_mlff,
        ligand_mlff=ligand_mlff,
        complex_gb_polar=complex_gb_polar,
        protein_gb_polar=protein_gb_polar,
        ligand_gb_polar=ligand_gb_polar,
    )


def _alanine_scan_decomposition(
    model: MACEModel,
    snapshot: ProcessedSnapshot,
    protein_residue_indices: List[np.ndarray],
    residue_ids: np.ndarray,
    residue_names: List[str],
    chain_ids: List[str],
    igb: int,
    charge_method: str,
) -> ResidueDecomposition:
    """
    Alanine scanning decomposition (expensive but rigorous).
    
    For each residue: mutate to Ala, recompute binding energy.
    ΔΔG = ΔG_mutant - ΔG_wildtype
    
    This requires structure mutation which is non-trivial with ASE.
    Placeholder for now.
    """
    n_residues = len(protein_residue_indices)
    
    return ResidueDecomposition(
        frame_id=snapshot.frame_id,
        residue_ids=residue_ids,
        residue_names=np.array(residue_names),
        chain_ids=np.array(chain_ids),
        delta_g_contrib=np.zeros(n_residues),
        mlff_contrib=np.zeros(n_residues),
        gb_polar_contrib=np.zeros(n_residues),
        gb_nonpolar_contrib=np.zeros(n_residues),
        complex_mlff=np.zeros(n_residues),
        protein_mlff=np.zeros(n_residues),
        ligand_mlff=np.zeros(n_residues),
        complex_gb_polar=np.zeros(n_residues),
        protein_gb_polar=np.zeros(n_residues),
        ligand_gb_polar=np.zeros(n_residues),
    )


def compute_decomposition_batch(
    model: MACEModel,
    snapshots: List[ProcessedSnapshot],
    igb: int = 8,
    method: Literal["alanine_scan", "interaction"] = "interaction",
    charge_method: str = "mace",
    n_jobs: int = 1,
) -> DecompositionResult:
    """
    Compute per-residue decomposition for multiple frames.
    
    Args:
        model: MACE model
        snapshots: List of processed snapshots
        igb: GB model
        method: Decomposition method
        charge_method: Charge method for GB
        n_jobs: Number of parallel jobs (frames)
    
    Returns:
        DecompositionResult with per-frame and averaged results
    """
    if n_jobs == 1:
        per_frame = [
            compute_residue_decomposition_frame(model, snap, igb, method, charge_method)
            for snap in snapshots
        ]
    else:
        # Parallel processing across frames
        per_frame = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(
                    compute_residue_decomposition_frame, model, snap, igb, method, charge_method
                )
                for snap in snapshots
            ]
            for future in as_completed(futures):
                per_frame.append(future.result())
    
    # Sort by frame_id
    per_frame.sort(key=lambda x: x.frame_id)
    
    # Average across frames
    if per_frame:
        n_residues = len(per_frame[0].delta_g_contrib)
        mean_contrib = np.mean([f.delta_g_contrib for f in per_frame], axis=0)
        std_contrib = np.std([f.delta_g_contrib for f in per_frame], axis=0)
        residue_ids = per_frame[0].residue_ids
        residue_names = per_frame[0].residue_names
        chain_ids = per_frame[0].chain_ids
    else:
        mean_contrib = np.array([])
        std_contrib = np.array([])
        residue_ids = np.array([])
        residue_names = np.array([])
        chain_ids = np.array([])
    
    return DecompositionResult(
        per_frame=per_frame,
        mean_contrib=mean_contrib,
        std_contrib=std_contrib,
        residue_ids=residue_ids,
        residue_names=residue_names,
        chain_ids=chain_ids,
    )


def decomposition_to_dataframe(result: DecompositionResult) -> "pandas.DataFrame":
    """Convert decomposition results to DataFrame."""
    import pandas as pd
    
    # Per-residue summary
    rows = []
    for i in range(len(result.mean_contrib)):
        rows.append({
            "residue_id": result.residue_ids[i],
            "residue_name": result.residue_names[i],
            "chain_id": result.chain_ids[i],
            "mean_deltaG_contrib": result.mean_contrib[i],
            "std_deltaG_contrib": result.std_contrib[i],
        })
    
    df_summary = pd.DataFrame(rows)
    
    # Per-frame detail (optional, can be large)
    frame_rows = []
    for frame in result.per_frame:
        for i in range(len(frame.delta_g_contrib)):
            frame_rows.append({
                "frame": frame.frame_id,
                "residue_id": frame.residue_ids[i],
                "residue_name": frame.residue_names[i],
                "chain_id": frame.chain_ids[i],
                "deltaG_contrib": frame.delta_g_contrib[i],
                "mlff_contrib": frame.mlff_contrib[i],
                "gb_polar_contrib": frame.gb_polar_contrib[i],
                "gb_nonpolar_contrib": frame.gb_nonpolar_contrib[i],
            })
    
    df_frames = pd.DataFrame(frame_rows)
    
    return df_summary, df_frames