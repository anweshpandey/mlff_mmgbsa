"""Snapshot processing for protein-ligand complexes."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from ase import Atoms
from mlff_mmgbsa.utils.topology import (
    parse_pdb_to_ase,
    create_system_selections,
    extract_subsystem,
    SystemSelection,
    identify_ligand,
)


@dataclass
class ProcessedSnapshot:
    """A single processed snapshot with three systems."""
    frame_id: int
    complex_atoms: Atoms
    protein_atoms: Atoms
    ligand_atoms: Atoms
    selection: SystemSelection
    ligand_resname: str
    ligand_chain: Optional[str]


@dataclass
class SnapshotBatch:
    """Batch of processed snapshots."""
    snapshots: List[ProcessedSnapshot]
    n_frames: int
    n_complex_atoms: int
    n_protein_atoms: int
    n_ligand_atoms: int


class SnapshotProcessor:
    """Processes PDB snapshots into complex/protein/ligand systems."""
    
    def __init__(
        self,
        ligand_resname: Optional[str] = None,
        ligand_chain: Optional[str] = None,
        add_hydrogens: bool = False,
    ):
        """
        Args:
            ligand_resname: Residue name of ligand (e.g., "LIG", "DRG")
            ligand_chain: Chain ID of ligand (e.g., "L", "B")
            add_hydrogens: Whether to add missing hydrogens (requires openbabel/pdbfixer)
        """
        self.ligand_resname = ligand_resname
        self.ligand_chain = ligand_chain
        self.add_hydrogens = add_hydrogens
    
    def process_pdb(self, pdb_path: str, frame_id: int = 0) -> ProcessedSnapshot:
        """Process a single PDB file into three systems."""
        # Parse PDB
        atoms = parse_pdb_to_ase(pdb_path)
        
        # Identify ligand
        ligand_mask, protein_mask = identify_ligand(
            atoms, self.ligand_resname, self.ligand_chain
        )
        
        # Determine ligand identifier for output
        residue_names = atoms.info.get("residue_names", np.array([]))
        chain_ids = atoms.info.get("chain_ids", np.array([]))
        
        if self.ligand_resname:
            ligand_id = self.ligand_resname
        elif self.ligand_chain:
            ligand_id = f"chain_{self.ligand_chain}"
        else:
            # Find first ligand residue name
            lig_indices = np.where(ligand_mask)[0]
            ligand_id = residue_names[lig_indices[0]] if len(lig_indices) > 0 else "LIG"
        
        ligand_chain_id = chain_ids[lig_indices[0]] if len(lig_indices) > 0 else None
        
        # Create selections
        selection = create_system_selections(atoms, self.ligand_resname, self.ligand_chain)
        
        # Extract three systems
        complex_atoms = atoms.copy()
        protein_atoms = extract_subsystem(atoms, selection.protein_indices)
        ligand_atoms = extract_subsystem(atoms, selection.ligand_indices)
        
        return ProcessedSnapshot(
            frame_id=frame_id,
            complex_atoms=complex_atoms,
            protein_atoms=protein_atoms,
            ligand_atoms=ligand_atoms,
            selection=selection,
            ligand_resname=ligand_id,
            ligand_chain=ligand_chain_id,
        )
    
    def process_directory(
        self,
        snapshots_dir: str,
        pattern: str = "*.pdb",
        max_frames: Optional[int] = None,
    ) -> SnapshotBatch:
        """Process all PDB files in a directory."""
        import glob
        from pathlib import Path
        
        pdb_files = sorted(glob.glob(str(Path(snapshots_dir) / pattern)))
        
        if max_frames is not None:
            pdb_files = pdb_files[:max_frames]
        
        snapshots = []
        for i, pdb_file in enumerate(pdb_files):
            try:
                snap = self.process_pdb(pdb_file, frame_id=i)
                snapshots.append(snap)
            except Exception as e:
                print(f"Warning: Failed to process {pdb_file}: {e}")
                continue
        
        if not snapshots:
            raise ValueError(f"No valid snapshots found in {snapshots_dir}")
        
        # Verify consistent atom counts
        n_complex = len(snapshots[0].complex_atoms)
        n_protein = len(snapshots[0].protein_atoms)
        n_ligand = len(snapshots[0].ligand_atoms)
        
        for snap in snapshots[1:]:
            if len(snap.complex_atoms) != n_complex:
                raise ValueError("Inconsistent atom counts across frames")
            if len(snap.protein_atoms) != n_protein:
                raise ValueError("Inconsistent protein atom counts across frames")
            if len(snap.ligand_atoms) != n_ligand:
                raise ValueError("Inconsistent ligand atom counts across frames")
        
        return SnapshotBatch(
            snapshots=snapshots,
            n_frames=len(snapshots),
            n_complex_atoms=n_complex,
            n_protein_atoms=n_protein,
            n_ligand_atoms=n_ligand,
        )
    
    def process_file_list(self, pdb_files: List[str]) -> SnapshotBatch:
        """Process a list of PDB files."""
        snapshots = []
        for i, pdb_file in enumerate(pdb_files):
            snap = self.process_pdb(pdb_file, frame_id=i)
            snapshots.append(snap)
        
        n_complex = len(snapshots[0].complex_atoms)
        n_protein = len(snapshots[0].protein_atoms)
        n_ligand = len(snapshots[0].ligand_atoms)
        
        return SnapshotBatch(
            snapshots=snapshots,
            n_frames=len(snapshots),
            n_complex_atoms=n_complex,
            n_protein_atoms=n_protein,
            n_ligand_atoms=n_ligand,
        )


def get_atom_charges(atoms: Atoms, method: str = "mace") -> np.ndarray:
    """
    Get partial charges for atoms.
    
    For MACE, charges are not explicitly needed as the model predicts energy directly.
    This is a placeholder for methods that need explicit charges (e.g., for GB).
    
    Args:
        atoms: ASE Atoms object
        method: "mace" (returns zeros - MACE doesn't need charges), 
                "gasteiger", "am1bcc", "zeros"
    
    Returns:
        Array of partial charges (n_atoms,)
    """
    n_atoms = len(atoms)
    
    if method == "mace":
        # MACE is charge-aware but doesn't require explicit charges as input
        # Return zeros as placeholder
        return np.zeros(n_atoms)
    
    elif method == "zeros":
        return np.zeros(n_atoms)
    
    elif method == "gasteiger":
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            # Convert ASE to RDKit mol - requires bond info
            # This is a stub - full implementation needs connectivity
            return np.zeros(n_atoms)
        except ImportError:
            print("RDKit not available, returning zeros")
            return np.zeros(n_atoms)
    
    else:
        return np.zeros(n_atoms)