"""Command-line interface for MLFF-MMGBSA."""

import typer
from typing import Optional, List
from pathlib import Path
from rich.console import Console
import torch

from mlff_mmgbsa.core.models import load_mace_model, MACEModel
from mlff_mmgbsa.core.snapshot import SnapshotProcessor, SnapshotBatch
from mlff_mmgbsa.core.energy import compute_mmgbsa_batch, MMGBSABatchResult
from mlff_mmgbsa.core.decomposition import compute_decomposition_batch, DecompositionResult
from mlff_mmgbsa.io.output import (
    write_mmgbsa_results,
    write_decomposition_results,
    write_combined_results,
    print_summary,
    print_decomposition_summary,
)

app = typer.Typer(
    name="mlff-mmgbsa",
    help="MMGBSA binding energy calculator using Machine Learning Force Fields (MACE)",
    add_completion=False,
)
console = Console()


@app.command()
def compute(
    snapshots_dir: Optional[Path] = typer.Option(
        None, "--snapshots-dir", "-d", help="Directory containing PDB snapshot files"
    ),
    snapshot_files: Optional[List[Path]] = typer.Option(
        None, "--snapshots", "-s", help="List of PDB snapshot files"
    ),
    ligand_resname: Optional[str] = typer.Option(
        None, "--ligand-resname", "-r", help="Ligand residue name (e.g., LIG, DRG)"
    ),
    ligand_chain: Optional[str] = typer.Option(
        None, "--ligand-chain", "-c", help="Ligand chain ID (e.g., L, B)"
    ),
    model: str = typer.Option(
        "mace-mp-0", "--model", "-m", help="MACE model: mace-mp-0, mace-off, mace-off23, mace-off24, or path to .model file"
    ),
    igb: int = typer.Option(
        8, "--igb", help="GB model: 2=OBC(I), 5=OBC(II), 7=GB-Neck, 8=GB-Neck2"
    ),
    device: str = typer.Option(
        "auto", "--device", help="Device: auto, cuda, cpu, mps"
    ),
    dtype: str = typer.Option(
        "float32", "--dtype", help="Data type: float32 or float64"
    ),
    output: Path = typer.Option(
        "mmgbsa_results.csv", "--output", "-o", help="Output file path"
    ),
    format: str = typer.Option(
        "csv", "--format", "-f", help="Output format: csv, json, excel"
    ),
    charge_method: str = typer.Option(
        "mace", "--charge-method", help="Charge method for GB: mace, zeros"
    ),
    max_frames: Optional[int] = typer.Option(
        None, "--max-frames", help="Maximum number of frames to process"
    ),
    pattern: str = typer.Option(
        "*.pdb", "--pattern", help="Glob pattern for snapshot files in directory"
    ),
):
    """
    Compute MMGBSA binding energies for protein-ligand snapshots.
    
    Provide either --snapshots-dir or --snapshots (list of files).
    """
    # Validate inputs
    if snapshots_dir is None and snapshot_files is None:
        console.print("[red]Error: Must provide either --snapshots-dir or --snapshots[/red]")
        raise typer.Exit(1)
    
    if snapshots_dir is not None and snapshot_files is not None:
        console.print("[red]Error: Provide only one of --snapshots-dir or --snapshots[/red]")
        raise typer.Exit(1)
    
    if ligand_resname is None and ligand_chain is None:
        console.print("[yellow]Warning: No ligand identifier provided. Will try to auto-detect from HETATM records.[/yellow]")
    
    # Load model
    console.print(f"[cyan]Loading MACE model: {model}[/cyan]")
    device_arg = None if device == "auto" else device
    dtype_arg = getattr(torch, dtype)
    
    import torch
    model_wrapper = load_mace_model(model, device=device_arg, dtype=dtype_arg)
    console.print(f"[green]Model loaded on {model_wrapper.device}[/green]")
    
    # Process snapshots
    processor = SnapshotProcessor(
        ligand_resname=ligand_resname,
        ligand_chain=ligand_chain,
    )
    
    if snapshots_dir:
        console.print(f"[cyan]Processing snapshots from: {snapshots_dir}[/cyan]")
        batch = processor.process_directory(
            str(snapshots_dir), pattern=pattern, max_frames=max_frames
        )
    else:
        console.print(f"[cyan]Processing {len(snapshot_files)} snapshot files[/cyan]")
        batch = processor.process_file_list([str(f) for f in snapshot_files])
    
    console.print(f"[green]Loaded {batch.n_frames} frames[/green]")
    console.print(f"  Complex: {batch.n_complex_atoms} atoms")
    console.print(f"  Protein: {batch.n_protein_atoms} atoms")
    console.print(f"  Ligand: {batch.n_ligand_atoms} atoms")
    
    # Compute MMGBSA
    console.print(f"[cyan]Computing MMGBSA (igb={igb})...[/cyan]")
    result = compute_mmgbsa_batch(
        model_wrapper,
        batch,
        igb=igb,
        charge_method=charge_method,
        show_progress=True,
    )
    
    # Print summary
    print_summary(result)
    
    # Write output
    write_mmgbsa_results(result, str(output), format=format)
    console.print(f"[green]Results written to: {output}[/green]")


@app.command()
def decompose(
    snapshots_dir: Optional[Path] = typer.Option(
        None, "--snapshots-dir", "-d", help="Directory containing PDB snapshot files"
    ),
    snapshot_files: Optional[List[Path]] = typer.Option(
        None, "--snapshots", "-s", help="List of PDB snapshot files"
    ),
    ligand_resname: Optional[str] = typer.Option(
        None, "--ligand-resname", "-r", help="Ligand residue name"
    ),
    ligand_chain: Optional[str] = typer.Option(
        None, "--ligand-chain", "-c", help="Ligand chain ID"
    ),
    model: str = typer.Option(
        "mace-mp-0", "--model", "-m", help="MACE model"
    ),
    igb: int = typer.Option(
        8, "--igb", help="GB model: 2, 5, 7, 8"
    ),
    device: str = typer.Option(
        "auto", "--device", help="Device: auto, cuda, cpu, mps"
    ),
    dtype: str = typer.Option(
        "float32", "--dtype", help="Data type: float32 or float64"
    ),
    output: Path = typer.Option(
        "decomp_results.csv", "--output", "-o", help="Output file path"
    ),
    format: str = typer.Option(
        "csv", "--format", "-f", help="Output format: csv, json, excel"
    ),
    method: str = typer.Option(
        "interaction", "--method", help="Decomposition method: interaction, alanine_scan"
    ),
    charge_method: str = typer.Option(
        "mace", "--charge-method", help="Charge method for GB"
    ),
    n_jobs: int = typer.Option(
        1, "--n-jobs", "-j", help="Number of parallel jobs"
    ),
    max_frames: Optional[int] = typer.Option(
        None, "--max-frames", help="Maximum number of frames"
    ),
    pattern: str = typer.Option(
        "*.pdb", "--pattern", help="Glob pattern for snapshot files"
    ),
    top_n: int = typer.Option(
        10, "--top-n", help="Number of top residues to display"
    ),
):
    """
    Compute per-residue energy decomposition for protein-ligand binding.
    """
    # Validate inputs
    if snapshots_dir is None and snapshot_files is None:
        console.print("[red]Error: Must provide either --snapshots-dir or --snapshots[/red]")
        raise typer.Exit(1)
    
    # Load model
    console.print(f"[cyan]Loading MACE model: {model}[/cyan]")
    device_arg = None if device == "auto" else device
    dtype_arg = getattr(torch, dtype)
    
    import torch
    model_wrapper = load_mace_model(model, device=device_arg, dtype=dtype_arg)
    console.print(f"[green]Model loaded on {model_wrapper.device}[/green]")
    
    # Process snapshots
    processor = SnapshotProcessor(
        ligand_resname=ligand_resname,
        ligand_chain=ligand_chain,
    )
    
    if snapshots_dir:
        batch = processor.process_directory(
            str(snapshots_dir), pattern=pattern, max_frames=max_frames
        )
    else:
        batch = processor.process_file_list([str(f) for f in snapshot_files])
    
    console.print(f"[green]Loaded {batch.n_frames} frames[/green]")
    
    # Compute decomposition
    console.print(f"[cyan]Computing per-residue decomposition (method={method})...[/cyan]")
    decomp_result = compute_decomposition_batch(
        model_wrapper,
        batch.snapshots,
        igb=igb,
        method=method,
        charge_method=charge_method,
        n_jobs=n_jobs,
    )
    
    # Print summary
    print_decomposition_summary(decomp_result, top_n=top_n)
    
    # Write output
    write_decomposition_results(decomp_result, str(output), format=format)
    console.print(f"[green]Decomposition results written to: {output}[/green]")


@app.command()
def both(
    snapshots_dir: Optional[Path] = typer.Option(
        None, "--snapshots-dir", "-d", help="Directory containing PDB snapshot files"
    ),
    snapshot_files: Optional[List[Path]] = typer.Option(
        None, "--snapshots", "-s", help="List of PDB snapshot files"
    ),
    ligand_resname: Optional[str] = typer.Option(
        None, "--ligand-resname", "-r", help="Ligand residue name"
    ),
    ligand_chain: Optional[str] = typer.Option(
        None, "--ligand-chain", "-c", help="Ligand chain ID"
    ),
    model: str = typer.Option(
        "mace-mp-0", "--model", "-m", help="MACE model"
    ),
    igb: int = typer.Option(
        8, "--igb", help="GB model: 2, 5, 7, 8"
    ),
    device: str = typer.Option(
        "auto", "--device", help="Device: auto, cuda, cpu, mps"
    ),
    dtype: str = typer.Option(
        "float32", "--dtype", help="Data type: float32 or float64"
    ),
    output_prefix: Path = typer.Option(
        "results", "--output", "-o", help="Output file prefix"
    ),
    format: str = typer.Option(
        "csv", "--format", "-f", help="Output format: csv, json, excel"
    ),
    decomp_method: str = typer.Option(
        "interaction", "--decomp-method", help="Decomposition method"
    ),
    charge_method: str = typer.Option(
        "mace", "--charge-method", help="Charge method for GB"
    ),
    n_jobs: int = typer.Option(
        1, "--n-jobs", "-j", help="Number of parallel jobs for decomposition"
    ),
    max_frames: Optional[int] = typer.Option(
        None, "--max-frames", help="Maximum number of frames"
    ),
    pattern: str = typer.Option(
        "*.pdb", "--pattern", help="Glob pattern"
    ),
    top_n: int = typer.Option(
        10, "--top-n", help="Top residues to display"
    ),
):
    """
    Compute both MMGBSA binding energies and per-residue decomposition.
    """
    # Validate
    if snapshots_dir is None and snapshot_files is None:
        console.print("[red]Error: Must provide either --snapshots-dir or --snapshots[/red]")
        raise typer.Exit(1)
    
    # Load model
    console.print(f"[cyan]Loading MACE model: {model}[/cyan]")
    device_arg = None if device == "auto" else device
    dtype_arg = getattr(torch, dtype)
    
    import torch
    model_wrapper = load_mace_model(model, device=device_arg, dtype=dtype_arg)
    console.print(f"[green]Model loaded on {model_wrapper.device}[/green]")
    
    # Process snapshots
    processor = SnapshotProcessor(
        ligand_resname=ligand_resname,
        ligand_chain=ligand_chain,
    )
    
    if snapshots_dir:
        batch = processor.process_directory(
            str(snapshots_dir), pattern=pattern, max_frames=max_frames
        )
    else:
        batch = processor.process_file_list([str(f) for f in snapshot_files])
    
    console.print(f"[green]Loaded {batch.n_frames} frames[/green]")
    
    # Compute MMGBSA
    console.print(f"[cyan]Computing MMGBSA...[/cyan]")
    mmgbsa_result = compute_mmgbsa_batch(
        model_wrapper,
        batch,
        igb=igb,
        charge_method=charge_method,
        show_progress=True,
    )
    print_summary(mmgbsa_result)
    
    # Compute decomposition
    console.print(f"[cyan]Computing decomposition...[/cyan]")
    decomp_result = compute_decomposition_batch(
        model_wrapper,
        batch.snapshots,
        igb=igb,
        method=decomp_method,
        charge_method=charge_method,
        n_jobs=n_jobs,
    )
    print_decomposition_summary(decomp_result, top_n=top_n)
    
    # Write combined
    write_combined_results(mmgbsa_result, decomp_result, str(output_prefix), format)
    console.print(f"[green]All results written with prefix: {output_prefix}[/green]")


@app.command()
def info():
    """Show information about available models and GB options."""
    console.print("[bold]MLFF-MMGBSA Calculator[/bold]")
    console.print()
    console.print("[cyan]Available MACE Models:[/cyan]")
    console.print("  mace-mp-0    : Foundation model (general purpose, proteins + materials)")
    console.print("  mace-off     : OFF 2023 model (organic molecules, drug-like)")
    console.print("  mace-off23   : Same as mace-off")
    console.print("  mace-off24   : OFF 2024 large model (highest accuracy)")
    console.print("  <path>       : Local .model file path")
    console.print()
    console.print("[cyan]GB Models (igb):[/cyan]")
    console.print("  2  : GBOBC(I)     - Hawkins/Cramer/Truhlar 1996")
    console.print("  5  : GBOBC(II)    - Onufriev/Bashford/Case 2000")
    console.print("  7  : GB-Neck      - Im et al. 2003")
    console.print("  8  : GB-Neck2     - Nguyen et al. 2013 (recommended)")
    console.print()
    console.print("[cyan]Ligand Identification:[/cyan]")
    console.print("  --ligand-resname LIG  : By residue name")
    console.print("  --ligand-chain L      : By chain ID")
    console.print("  (none)                : Auto-detect from HETATM (excludes water/ions)")
    console.print()
    console.print("[cyan]Example Usage:[/cyan]")
    console.print("  mlff-mmgbsa compute -d ./snapshots -r LIG -m mace-mp-0 --igb 8 -o results.csv")
    console.print("  mlff-mmgbsa decompose -d ./snapshots -r LIG --n-jobs 4 -o decomp.csv")
    console.print("  mlff-mmgbsa both -d ./snapshots -r LIG -o results")


if __name__ == "__main__":
    app()