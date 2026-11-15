"""MACE model loading and inference utilities."""

from dataclasses import dataclass
from typing import Optional, List, Union, Literal
import torch
from mace.calculators import MACECalculator
from ase import Atoms
import logging

logger = logging.getLogger(__name__)


@dataclass
class MACEModel:
    """Wrapper for MACE calculator with metadata."""
    calculator: MACECalculator
    model_name: str
    device: torch.device
    dtype: torch.dtype
    
    def __call__(self, atoms: Atoms) -> dict:
        """Compute energy and forces for an ASE Atoms object."""
        return self.calculator.calculate(atoms)
    
    def get_potential_energy(self, atoms: Atoms) -> float:
        """Get potential energy in eV."""
        return self.calculator.get_potential_energy(atoms)
    
    def get_forces(self, atoms: Atoms) -> torch.Tensor:
        """Get forces in eV/Angstrom."""
        return self.calculator.get_forces(atoms)
    
    def batch_compute(self, atoms_list: List[Atoms]) -> List[dict]:
        """Compute energies for multiple structures (sequential for now)."""
        results = []
        for atoms in atoms_list:
            results.append(self(atoms))
        return results


def load_mace_model(
    model: Union[str, Literal["mace-mp-0", "mace-off", "mace-off23", "mace-off24"]] = "mace-mp-0",
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
    compile_mode: Optional[str] = None,
) -> MACEModel:
    """
    Load a MACE model from the hub or local file.
    
    Args:
        model: Model identifier - can be:
            - "mace-mp-0" (MP-0 foundation model)
            - "mace-off" / "mace-off23" (OFF 2023 model for organics/drugs)
            - "mace-off24" (OFF 2024 large model)
            - Path to a local .model file
        device: "cuda", "cpu", "mps", or torch.device. Auto-detects if None.
        dtype: torch.float32 (default) or torch.float64
        compile_mode: torch.compile mode ("default", "reduce-overhead", "max-autotune")
    
    Returns:
        MACEModel wrapper with calculator
    """
    # Determine device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    
    logger.info(f"Loading MACE model '{model}' on {device} with dtype {dtype}")
    
    # Handle model path/identifier
    if model in ("mace-mp-0", "mace-off", "mace-off23", "mace-off24"):
        # These are handled by mace-torch's model hub
        model_paths = {
            "mace-mp-0": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0/mace-mp-0.model",
            "mace-off": "https://github.com/ACEsuit/mace-off/releases/download/mace_off23/mace-off23.model",
            "mace-off23": "https://github.com/ACEsuit/mace-off/releases/download/mace_off23/mace-off23.model",
            "mace-off24": "https://github.com/ACEsuit/mace-off/releases/download/mace_off24/mace-off24.model",
        }
        model_path = model_paths[model]
    else:
        # Local file path
        model_path = model
    
    # Create calculator
    calculator = MACECalculator(
        model_paths=model_path,
        device=device,
        dtype=dtype,
        compile_mode=compile_mode,
    )
    
    return MACEModel(
        calculator=calculator,
        model_name=model,
        device=device,
        dtype=dtype,
    )


def compute_energy_batch(
    model: MACEModel,
    atoms_list: List[Atoms],
    show_progress: bool = True,
) -> List[float]:
    """
    Compute potential energies for a batch of structures.
    
    Returns:
        List of energies in eV
    """
    energies = []
    iterator = atoms_list
    if show_progress:
        from rich.progress import track
        iterator = track(atoms_list, description="Computing MLFF energies...")
    
    for atoms in iterator:
        energy = model.get_potential_energy(atoms)
        energies.append(energy)
    
    return energies