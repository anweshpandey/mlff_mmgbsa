#!/usr/bin/env python
"""Example script demonstrating MLFF-MMGBSA usage."""

import tempfile
import os
from pathlib import Path

# Create a minimal test PDB for demonstration
TEST_PDB = """HEADER    TEST PROTEIN-LIGAND COMPLEX
ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00  0.00           N
ATOM      2  CA  ALA A   1      11.400  10.000  10.000  1.00  0.00           C
ATOM      3  C   ALA A   1      12.000  11.300  10.500  1.00  0.00           C
ATOM      4  O   ALA A   1      11.500  12.300  10.200  1.00  0.00           O
ATOM      5  CB  ALA A   1      11.900   9.000   9.000  1.00  0.00           C
ATOM      6  N   GLY A   2      13.200  11.200  11.000  1.00  0.00           N
ATOM      7  CA  GLY A   2      13.900  12.400  11.500  1.00  0.00           C
ATOM      8  C   GLY A   2      15.300  12.300  11.000  1.00  0.00           C
ATOM      9  O   GLY A   2      15.800  11.300  10.600  1.00  0.00           O
ATOM     10  N   LIG L   1      20.000  20.000  20.000  1.00  0.00           N
ATOM     11  CA  LIG L   1      21.000  20.000  20.000  1.00  0.00           C
ATOM     12  C   LIG L   1      22.000  21.000  20.000  1.00  0.00           C
ATOM     13  O   LIG L   1      22.500  21.500  19.200  1.00  0.00           O
TER
END
"""


def create_test_snapshots(n_frames: int = 3) -> Path:
    """Create test snapshot directory."""
    tmpdir = Path(tempfile.mkdtemp(prefix="mlff_mmgbsa_test_"))
    
    for i in range(n_frames):
        pdb_file = tmpdir / f"frame_{i:03d}.pdb"
        # Add small random perturbations to coordinates
        import numpy as np
        lines = TEST_PDB.strip().split('\n')
        perturbed = []
        for line in lines:
            if line.startswith("ATOM"):
                x = float(line[30:38]) + np.random.normal(0, 0.01)
                y = float(line[38:46]) + np.random.normal(0, 0.01)
                z = float(line[46:54]) + np.random.normal(0, 0.01)
                line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
            perturbed.append(line)
        pdb_file.write_text('\n'.join(perturbed) + '\n')
    
    return tmpdir


def main():
    """Run example."""
    print("=" * 60)
    print("MLFF-MMGBSA Example")
    print("=" * 60)
    
    # Create test data
    print("\n1. Creating test snapshots...")
    snapshots_dir = create_test_snapshots(3)
    print(f"   Created: {snapshots_dir}")
    
    # Show CLI commands
    print("\n2. Example CLI commands:")
    print(f"   mlff-mmgbsa compute -d {snapshots_dir} -r LIG -o results.csv")
    print(f"   mlff-mmgbsa decompose -d {snapshots_dir} -r LIG --n-jobs 2 -o decomp.csv")
    print(f"   mlff-mmgbsa both -d {snapshots_dir} -r LIG -o results")
    
    # Python API example
    print("\n3. Python API example:")
    print("""
   from mlff_mmgbsa import (
       load_mace_model,
       SnapshotProcessor,
       compute_mmgbsa_batch,
       compute_decomposition_batch,
   )
   
   model = load_mace_model("mace-mp-0", device="cuda")
   processor = SnapshotProcessor(ligand_resname="LIG")
   batch = processor.process_directory("./snapshots")
   
   result = compute_mmgbsa_batch(model, batch, igb=8)
   print(f"Mean ΔG_bind: {result.mean_delta_g:.2f} kcal/mol")
   
   decomp = compute_decomposition_batch(model, batch.snapshots, igb=8, n_jobs=4)
   """)
    
    # Cleanup
    import shutil
    shutil.rmtree(snapshots_dir)
    print("\n4. Test snapshots cleaned up.")
    
    print("\n" + "=" * 60)
    print("Run 'mlff-mmgbsa info' for more options")
    print("=" * 60)


if __name__ == "__main__":
    main()