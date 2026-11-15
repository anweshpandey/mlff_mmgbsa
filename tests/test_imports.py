"""Test basic imports work correctly."""

def test_imports():
    """Test that all modules can be imported."""
    from mlff_mmgbsa import (
        load_mace_model,
        MACEModel,
        compute_mmgbsa_batch,
        MMGBSAResult,
        SnapshotProcessor,
        ProcessedSnapshot,
    )
    from mlff_mmgbsa.core.models import load_mace_model as _lm
    from mlff_mmgbsa.core.energy import compute_system_energy
    from mlff_mmgbsa.core.gb import compute_gb_solvation, GBResult
    from mlff_mmgbsa.core.snapshot import SnapshotProcessor as _sp
    from mlff_mmgbsa.core.decomposition import compute_decomposition_batch
    from mlff_mmgbsa.utils.constants import get_atom_radius, get_gb_params
    from mlff_mmgbsa.utils.topology import parse_pdb_to_ase, identify_ligand
    from mlff_mmgbsa.io.output import write_mmgbsa_results
    from mlff_mmgbsa.cli import app
    
    assert True  # If we get here, imports work


def test_constants():
    """Test constants are accessible."""
    from mlff_mmgbsa.utils.constants import (
        GB_PARAMS, ATOM_RADII, EPSILON_IN, EPSILON_OUT,
        NONPOLAR_GAMMA, NONPOLAR_BETA
    )
    
    assert 2 in GB_PARAMS
    assert 5 in GB_PARAMS
    assert 7 in GB_PARAMS
    assert 8 in GB_PARAMS
    assert "C" in ATOM_RADII
    assert "N" in ATOM_RADII
    assert "O" in ATOM_RADII
    assert EPSILON_IN == 1.0
    assert EPSILON_OUT == 78.5
    assert NONPOLAR_GAMMA > 0
    assert NONPOLAR_BETA > 0


def test_get_atom_radius():
    """Test atom radius lookup."""
    from mlff_mmgbsa.utils.constants import get_atom_radius
    
    assert get_atom_radius("C") == 1.70
    assert get_atom_radius("N") == 1.55
    assert get_atom_radius("O") == 1.52
    assert get_atom_radius("H") == 1.20
    assert get_atom_radius("UNKNOWN") == 1.5  # Default


def test_get_gb_params():
    """Test GB parameter retrieval."""
    from mlff_mmgbsa.utils.constants import get_gb_params
    
    params2 = get_gb_params(2)
    assert params2["alpha"] == 0.8
    assert params2["beta"] == 0.0
    assert params2["gamma"] == 2.909125
    
    params8 = get_gb_params(8)
    assert params8["alpha"] == 1.0
    assert params8["beta"] == 0.8
    assert params8["gamma"] == 4.85
    assert params8["neck_scale"] == 0.73
    
    try:
        get_gb_params(99)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    test_imports()
    test_constants()
    test_get_atom_radius()
    test_get_gb_params()
    print("All tests passed!")