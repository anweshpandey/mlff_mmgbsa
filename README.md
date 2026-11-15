# MLFF-MMGBSA

**MMGBSA binding energy calculator using Machine Learning Force Fields (MACE)**

Compute protein-ligand binding free energies and per-residue energy decomposition using MACE (Machine Learning Force Fields) instead of traditional molecular mechanics force fields, with GBOBC implicit solvation.

## Features

- **MLFF Backend**: MACE-MP-0, MACE-OFF, MACE-OFF24 models for accurate potential energies
- **Implicit Solvation**: GBOBC (igb=2,5,7,8) Generalized Born models
- **Binding Energy**: ΔG = G_complex - G_protein - G_ligand (single-trajectory)
- **Decomposition**: Per-residue energy decomposition
- **CLI & Python API**: Easy to use from command line or scripts
- **Batch Processing**: Efficient multi-frame computation with progress tracking

## Installation

```bash
# From source
git clone https://github.com/yourusername/mlff_mmgbsa
cd mlff_mmgbsa
pip install -e .[dev]  # Includes dev dependencies

# Or with GPU support
pip install -e .[gpu]
```

### Dependencies

- `mace-torch` ≥ 0.3.0 (MACE models)
- `torch` ≥ 2.0
- `ase` ≥ 3.22 (atomic structures)
- `biotite` ≥ 0.39 (PDB parsing)
- `numpy`, `pandas`, `scipy`
- `typer`, `rich` (CLI)

## Quick Start

### Prepare Snapshots

You need PDB files for each snapshot (frame) of your protein-ligand complex:
```
snapshots/
├── frame_001.pdb
├── frame_002.pdb
├── frame_003.pdb
└── ...
```

Each PDB should contain both protein and ligand with hydrogens.

### Compute Binding Energies

```bash
# Basic usage - auto-detect ligand from HETATM
mlff-mmgbsa compute --snapshots-dir ./snapshots --output results.csv

# Specify ligand by residue name
mlff-mmgbsa compute -d ./snapshots -r LIG -o results.csv

# Use MACE-OFF for drug-like ligands, GB-Neck2
mlff-mmgbsa compute -d ./snapshots -r LIG -m mace-off --igb 8 -o results.csv

# GPU acceleration
mlff-mmgbsa compute -d ./snapshots -r LIG --device cuda -o results.csv
```

### Per-Residue Decomposition

```bash
# Decomposition with parallel processing
mlff-mmgbsa decompose -d ./snapshots -r LIG --n-jobs 4 -o decomp.csv

# Both binding energy and decomposition
mlff-mmgbsa both -d ./snapshots -r LIG -o results
```

### Python API

```python
from mlff_mmgbsa import (
    load_mace_model,
    SnapshotProcessor,
    compute_mmgbsa_batch,
    compute_decomposition_batch,
)

# Load model
model = load_mace_model("mace-mp-0", device="cuda")

# Process snapshots
processor = SnapshotProcessor(ligand_resname="LIG")
batch = processor.process_directory("./snapshots")

# Compute MMGBSA
result = compute_mmgbsa_batch(model, batch, igb=8)
print(f"Mean ΔG_bind: {result.mean_delta_g:.2f} kcal/mol")

# Decomposition
decomp = compute_decomposition_batch(model, batch.snapshots, igb=8, n_jobs=4)
```

## Command Reference

### `mlff-mmgbsa compute`
Compute MMGBSA binding energies.

```bash
mlff-mmgbsa compute [OPTIONS]

Options:
  -d, --snapshots-dir PATH     Directory with PDB files
  -s, --snapshots PATH...      List of PDB files
  -r, --ligand-resname TEXT    Ligand residue name (e.g., LIG)
  -c, --ligand-chain TEXT      Ligand chain ID (e.g., L)
  -m, --model TEXT             MACE model (default: mace-mp-0)
  --igb INTEGER                GB model: 2, 5, 7, 8 (default: 8)
  --device TEXT                Device: auto, cuda, cpu, mps (default: auto)
  --dtype TEXT                 float32 or float64 (default: float32)
  -o, --output PATH            Output file (default: mmgbsa_results.csv)
  -f, --format TEXT            csv, json, excel (default: csv)
  --charge-method TEXT         mace, zeros (default: mace)
  --max-frames INTEGER         Limit number of frames
  --pattern TEXT               Glob pattern (default: *.pdb)
```

### `mlff-mmgbsa decompose`
Per-residue energy decomposition.

```bash
mlff-mmgbsa decompose [OPTIONS]

Options:
  (same as compute, plus:)
  --method TEXT                interaction, alanine_scan (default: interaction)
  -j, --n-jobs INTEGER         Parallel jobs (default: 1)
  --top-n INTEGER              Top residues to display (default: 10)
```

### `mlff-mmgbsa both`
Compute both binding energies and decomposition.

```bash
mlff-mmgbsa both [OPTIONS]
```

### `mlff-mmgbsa info`
Show available models and options.

```bash
mlff-mmgbsa info
```

## Output Format

### MMGBSA Results (CSV)
| Column | Description |
|--------|-------------|
| frame | Frame index |
| deltaG_bind | Binding free energy (kcal/mol) |
| complex_total | Complex total energy |
| protein_total | Protein total energy |
| ligand_total | Ligand total energy |
| complex_mlff | Complex MLFF energy |
| protein_mlff | Protein MLFF energy |
| ligand_mlff | Ligand MLFF energy |
| complex_gb_polar | Complex GB polar solvation |
| ... | ... |
| delta_mlff | ΔMLFF contribution |
| delta_gb_polar | ΔGB polar contribution |
| delta_gb_nonpolar | ΔGB nonpolar contribution |

### Decomposition Results
- `decomp_results.csv` — Per-residue summary (mean ± std across frames)
- `decomp_results_per_frame.csv` — Per-frame per-residue detail

## GB Models

| igb | Model | Reference |
|-----|-------|-----------|
| 2 | OBC(I) | Hawkins, Cramer, Truhlar JPCB 1996 |
| 5 | OBC(II) | Onufriev, Bashford, Case Proteins 2000 |
| 7 | GB-Neck | Im et al. JCC 2003 |
| 8 | GB-Neck2 | Nguyen et al. JCTC 2013 |

## MACE Models

| Model | Description | Best For |
|-------|-------------|----------|
| `mace-mp-0` | Foundation model (MP-0) | General purpose, proteins |
| `mace-off` / `mace-off23` | OFF 2023 | Drug-like molecules |
| `mace-off24` | OFF 2024 Large | Highest accuracy |

## Performance

Typical performance on NVIDIA A100 (50k atom complex):
- MLFF energy: ~50-200 ms/frame
- GB solvation: ~5-20 ms/frame (CPU)
- Total: ~100-300 ms/frame

Decomposition adds significant overhead (per-residue calculations).

## Citation

If you use this tool, please cite:
- MACE: [Batatia et al., 2022](https://arxiv.org/abs/2206.07697)
- GBOBC: [Onufriev et al., 2000](https://doi.org/10.1002/(SICI)1097-0134(20001101)41:3<361::AID-PROT10>3.0.CO;2-7)
- MMGBSA: [Kollman et al., 2000](https://doi.org/10.1002/(SICI)1097-0134(20001101)41:3<361::AID-PROT10>3.0.CO;2-7)

## License

MIT License