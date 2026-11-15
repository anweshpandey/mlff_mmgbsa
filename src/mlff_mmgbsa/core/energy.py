"""MMGBSA energy computation combining MLFF and GB solvation."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import torch
from ase import Atoms
from mlff_mmgbsa.core.models import MACEModel, compute_energy_batch
from mlff_mmgbsa.core.gb import compute_gb_solvation, GBResult
from mlff_mmgbsa.core.snapshot import ProcessedSnapshot, SnapshotBatch, get_atom_charges
from mlff_mmgbsa.utils.constants import KCAL_PER_MOL_TO_KJ_PER_MOL, EV_TO_KCAL_PER_MOL


@dataclass
class SystemEnergy:
    """Energy components for a single system (complex/protein/ligand)."""
    mlff_energy_ev: float           # MLFF potential energy (eV)
    mlff_energy_kcal: float         # MLFF potential energy (kcal/mol)
    gb_polar: float                 # GB polar solvation (kcal/mol)
    gb_nonpolar: float              # GB nonpolar solvation (kcal/mol)
    gb_total: float                 # Total GB solvation (kcal/mol)
    total_energy: float             # Total: MLFF + GB (kcal/mol)
    gb_result: GBResult             # Full GB result


@dataclass
class MMGBSAResult:
    """MMGBSA binding energy result for a single frame."""
    frame_id: int
    complex_energy: SystemEnergy
    protein_energy: SystemEnergy
    ligand_energy: SystemEnergy
    delta_g_bind: float             # Binding free energy (kcal/mol)
    delta_g_components: dict        # Component breakdown


@dataclass
class MMGBSABatchResult:
    """MMGBSA results for a batch of frames."""
    results: List[MMGBSAResult]
    mean_delta_g: float
    std_delta_g: float
    mean_components: dict
    std_components: dict


EV_TO_KCAL = 23.0605  # 1 eV = 23.0605 kcal/mol


def compute_system_energy(
    model: MACEModel,
    atoms: Atoms,
    igb: int = 8,
    charge_method: str = "mace",
) -> SystemEnergy:
    """
    Compute total energy for a system: MLFF + GB solvation.
    
    Args:
        model: MACE model wrapper
        atoms: ASE Atoms object
        igb: GB model variant
        charge_method: Method for partial charges (for GB)
    
    Returns:
        SystemEnergy with all components
    """
    # MLFF energy (in eV)
    mlff_energy_ev = model.get_potential_energy(atoms)
    mlff_energy_kcal = mlff_energy_ev * EV_TO_KCAL
    
    # Get coordinates and elements
    coords = atoms.get_positions()
    elements = np.array(atoms.get_chemical_symbols())
    
    # Get partial charges for GB
    charges = get_atom_charges(atoms, method=charge_method)
    
    # GB solvation
    gb_result = compute_gb_solvation(coords, elements, charges, igb=igb)
    
    total = mlff_energy_kcal + gb_result.total_solvation
    
    return SystemEnergy(
        mlff_energy_ev=mlff_energy_ev,
        mlff_energy_kcal=mlff_energy_kcal,
        gb_polar=gb_result.polar_solvation,
        gb_nonpolar=gb_result.nonpolar_solvation,
        gb_total=gb_result.total_solvation,
        total_energy=total,
        gb_result=gb_result,
    )


def compute_mmgbsa_frame(
    model: MACEModel,
    snapshot: ProcessedSnapshot,
    igb: int = 8,
    charge_method: str = "mace",
) -> MMGBSAResult:
    """
    Compute MMGBSA binding energy for a single frame.
    
    ΔG_bind = G_complex - G_protein - G_ligand
    """
    # Compute energies for three systems
    complex_energy = compute_system_energy(
        model, snapshot.complex_atoms, igb, charge_method
    )
    protein_energy = compute_system_energy(
        model, snapshot.protein_atoms, igb, charge_method
    )
    ligand_energy = compute_system_energy(
        model, snapshot.ligand_atoms, igb, charge_method
    )
    
    # Binding energy
    delta_g = (complex_energy.total_energy 
               - protein_energy.total_energy 
               - ligand_energy.total_energy)
    
    # Component breakdown
    components = {
        "delta_mlff": (complex_energy.mlff_energy_kcal 
                       - protein_energy.mlff_energy_kcal 
                       - ligand_energy.mlff_energy_kcal),
        "delta_gb_polar": (complex_energy.gb_polar 
                           - protein_energy.gb_polar 
                           - ligand_energy.gb_polar),
        "delta_gb_nonpolar": (complex_energy.gb_nonpolar 
                              - protein_energy.gb_nonpolar 
                              - ligand_energy.gb_nonpolar),
        "delta_gb_total": (complex_energy.gb_total 
                           - protein_energy.gb_total 
                           - ligand_energy.gb_total),
    }
    
    return MMGBSAResult(
        frame_id=snapshot.frame_id,
        complex_energy=complex_energy,
        protein_energy=protein_energy,
        ligand_energy=ligand_energy,
        delta_g_bind=delta_g,
        delta_g_components=components,
    )


def compute_mmgbsa_batch(
    model: MACEModel,
    batch: SnapshotBatch,
    igb: int = 8,
    charge_method: str = "mace",
    show_progress: bool = True,
) -> MMGBSABatchResult:
    """
    Compute MMGBSA for a batch of snapshots.
    """
    results = []
    
    iterator = batch.snapshots
    if show_progress:
        from rich.progress import track
        iterator = track(batch.snapshots, description="Computing MMGBSA...")
    
    for snapshot in iterator:
        result = compute_mmgbsa_frame(model, snapshot, igb, charge_method)
        results.append(result)
    
    # Statistics
    delta_g_values = [r.delta_g_bind for r in results]
    mean_delta_g = np.mean(delta_g_values)
    std_delta_g = np.std(delta_g_values)
    
    # Component statistics
    component_names = list(results[0].delta_g_components.keys())
    mean_components = {}
    std_components = {}
    for comp in component_names:
        vals = [r.delta_g_components[comp] for r in results]
        mean_components[comp] = np.mean(vals)
        std_components[comp] = np.std(vals)
    
    return MMGBSABatchResult(
        results=results,
        mean_delta_g=mean_delta_g,
        std_delta_g=std_delta_g,
        mean_components=mean_components,
        std_components=std_components,
    )


def energy_to_dataframe(results: List[MMGBSAResult]) -> "pandas.DataFrame":
    """Convert results to pandas DataFrame for output."""
    import pandas as pd
    
    rows = []
    for r in results:
        row = {
            "frame": r.frame_id,
            "deltaG_bind": r.delta_g_bind,
            "complex_total": r.complex_energy.total_energy,
            "protein_total": r.protein_energy.total_energy,
            "ligand_total": r.ligand_energy.total_energy,
            "complex_mlff": r.complex_energy.mlff_energy_kcal,
            "protein_mlff": r.protein_energy.mlff_energy_kcal,
            "ligand_mlff": r.ligand_energy.mlff_energy_kcal,
            "complex_gb_polar": r.complex_energy.gb_polar,
            "protein_gb_polar": r.protein_energy.gb_polar,
            "ligand_gb_polar": r.ligand_energy.gb_polar,
            "complex_gb_nonpolar": r.complex_energy.gb_nonpolar,
            "protein_gb_nonpolar": r.protein_energy.gb_nonpolar,
            "ligand_gb_nonpolar": r.ligand_energy.gb_nonpolar,
        }
        # Add component deltas
        for k, v in r.delta_g_components.items():
            row[k] = v
        rows.append(row)
    
    return pd.DataFrame(rows)