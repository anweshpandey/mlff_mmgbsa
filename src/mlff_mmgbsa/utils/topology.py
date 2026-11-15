"""Topology utilities for protein-ligand system handling."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Set
import numpy as np
from ase import Atoms
import biotite.structure as struc
import biotite.structure.io.pdb as pdb


@dataclass
class SystemSelection:
    """Container for complex/protein/ligand atom selections."""
    complex_indices: np.ndarray      # All atoms
    protein_indices: np.ndarray      # Protein atoms only
    ligand_indices: np.ndarray       # Ligand atoms only
    protein_residue_ids: np.ndarray  # Residue index for each protein atom
    ligand_residue_ids: np.ndarray   # Residue index for each ligand atom
    residue_names: np.ndarray        # Residue name for each atom
    chain_ids: np.ndarray            # Chain ID for each atom
    atom_names: np.ndarray           # Atom name for each atom
    elements: np.ndarray             # Element symbol for each atom


def parse_pdb_to_ase(pdb_path: str) -> Atoms:
    """Parse PDB file to ASE Atoms object with metadata."""
    structure = pdb.PDBFile.read(pdb_path).get_structure()
    
    # Get coordinates
    coords = structure.coord
    
    # Get elements (fallback to atom name first character)
    elements = []
    for atom in structure:
        el = atom.element
        if el == "" or el == " ":
            # Infer from atom name
            name = atom.atom_name.strip()
            el = name[0].upper()
            if len(name) > 1 and name[1].islower():
                el = name[:2].capitalize()
        elements.append(el)
    
    atoms = Atoms(symbols=elements, positions=coords)
    
    # Store metadata as arrays
    atoms.info["residue_names"] = np.array([atom.res_name for atom in structure])
    atoms.info["residue_ids"] = np.array([atom.res_id for atom in structure])
    atoms.info["chain_ids"] = np.array([atom.chain_id for atom in structure])
    atoms.info["atom_names"] = np.array([atom.atom_name for atom in structure])
    atoms.info["hetero"] = np.array([atom.hetero for atom in structure])
    
    return atoms


def identify_ligand(
    atoms: Atoms,
    ligand_resname: Optional[str] = None,
    ligand_chain: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify ligand atoms based on residue name or chain.
    
    Returns:
        ligand_mask: Boolean mask for ligand atoms
        protein_mask: Boolean mask for protein atoms
    """
    residue_names = atoms.info.get("residue_names", np.array([]))
    chain_ids = atoms.info.get("chain_ids", np.array([]))
    hetero = atoms.info.get("hetero", np.array([]))
    
    n_atoms = len(atoms)
    ligand_mask = np.zeros(n_atoms, dtype=bool)
    protein_mask = np.zeros(n_atoms, dtype=bool)
    
    if ligand_resname is not None:
        # Match by residue name (case-insensitive)
        ligand_mask = np.char.upper(residue_names) == ligand_resname.upper()
    elif ligand_chain is not None:
        # Match by chain ID
        ligand_mask = chain_ids == ligand_chain
    else:
        # Default: HETATM records that are not water/ions
        water_resnames = {"HOH", "WAT", "H2O", "TIP3", "TIP4", "SOL"}
        ion_resnames = {"NA", "CL", "K", "MG", "CA", "ZN", "FE", "CU", "MN"}
        
        for i in range(n_atoms):
            res = residue_names[i]
            is_het = hetero[i] if i < len(hetero) else False
            if is_het and res not in water_resnames and res not in ion_resnames:
                ligand_mask[i] = True
    
    protein_mask = ~ligand_mask
    
    # Sanity check
    if not ligand_mask.any():
        raise ValueError("No ligand atoms identified. Check ligand_resname or ligand_chain.")
    if not protein_mask.any():
        raise ValueError("No protein atoms identified.")
    
    return ligand_mask, protein_mask


def create_system_selections(
    atoms: Atoms,
    ligand_resname: Optional[str] = None,
    ligand_chain: Optional[str] = None,
) -> SystemSelection:
    """Create SystemSelection with all three systems identified."""
    ligand_mask, protein_mask = identify_ligand(atoms, ligand_resname, ligand_chain)
    
    complex_indices = np.arange(len(atoms))
    protein_indices = np.where(protein_mask)[0]
    ligand_indices = np.where(ligand_mask)[0]
    
    residue_names = atoms.info.get("residue_names", np.array(["UNK"] * len(atoms)))
    residue_ids = atoms.info.get("residue_ids", np.array([0] * len(atoms)))
    chain_ids = atoms.info.get("chain_ids", np.array([" "] * len(atoms)))
    atom_names = atoms.info.get("atom_names", np.array([""] * len(atoms)))
    elements = np.array(atoms.get_chemical_symbols())
    
    # Create residue IDs for protein and ligand separately
    protein_residue_ids = residue_ids[protein_mask]
    ligand_residue_ids = residue_ids[ligand_mask]
    
    return SystemSelection(
        complex_indices=complex_indices,
        protein_indices=protein_indices,
        ligand_indices=ligand_indices,
        protein_residue_ids=protein_residue_ids,
        ligand_residue_ids=ligand_residue_ids,
        residue_names=residue_names,
        chain_ids=chain_ids,
        atom_names=atom_names,
        elements=elements,
    )


def extract_subsystem(atoms: Atoms, indices: np.ndarray) -> Atoms:
    """Extract a subset of atoms as a new ASE Atoms object."""
    sub_atoms = atoms[indices.tolist()].copy()
    # Preserve metadata for subset
    for key, value in atoms.info.items():
        if isinstance(value, np.ndarray) and len(value) == len(atoms):
            sub_atoms.info[key] = value[indices]
    return sub_atoms


def get_residue_atom_indices(
    atoms: Atoms,
    selection: SystemSelection,
    system: Literal["protein", "ligand"] = "protein",
) -> List[np.ndarray]:
    """Get atom indices grouped by residue for a given system."""
    if system == "protein":
        residue_ids = selection.protein_residue_ids
        base_indices = selection.protein_indices
    else:
        residue_ids = selection.ligand_residue_ids
        base_indices = selection.ligand_indices
    
    unique_residues = np.unique(residue_ids)
    residue_atom_indices = []
    
    for res_id in unique_residues:
        mask = residue_ids == res_id
        residue_atom_indices.append(base_indices[mask])
    
    return residue_atom_indices


def mutate_to_alanine(atoms: Atoms, residue_indices: np.ndarray) -> Atoms:
    """
    Mutate a residue to alanine by keeping only backbone atoms (N, CA, C, O).
    This is a simplified version - real implementation would need proper geometry.
    """
    # For now, return a copy - full alanine scanning requires structure rebuilding
    # This is a placeholder for the decomposition module
    return atoms.copy()