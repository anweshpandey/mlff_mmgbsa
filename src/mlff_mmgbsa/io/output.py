"""Output formatting for MMGBSA results."""

from typing import List, Optional
import pandas as pd
import json
from mlff_mmgbsa.core.energy import MMGBSAResult, MMGBSABatchResult, energy_to_dataframe
from mlff_mmgbsa.core.decomposition import DecompositionResult, decomposition_to_dataframe


def write_mmgbsa_results(
    result: MMGBSABatchResult,
    output_path: str,
    format: str = "csv",
    include_per_frame: bool = True,
) -> None:
    """
    Write MMGBSA results to file.
    
    Args:
        result: MMGBSABatchResult object
        output_path: Output file path
        format: "csv", "json", or "excel"
        include_per_frame: Whether to include per-frame data
    """
    df = energy_to_dataframe(result.results)
    
    if format == "csv":
        df.to_csv(output_path, index=False)
    elif format == "json":
        df.to_json(output_path, orient="records", indent=2)
    elif format == "excel":
        df.to_excel(output_path, index=False)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    # Also write summary statistics
    summary_path = output_path.replace(f".{format}", "_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"MMGBSA Binding Energy Summary\n")
        f.write(f"=" * 40 + "\n")
        f.write(f"Number of frames: {len(result.results)}\n")
        f.write(f"Mean ΔG_bind: {result.mean_delta_g:.3f} kcal/mol\n")
        f.write(f"Std ΔG_bind: {result.std_delta_g:.3f} kcal/mol\n")
        f.write(f"\nComponent breakdown (mean ± std):\n")
        for comp, mean in result.mean_components.items():
            std = result.std_components[comp]
            f.write(f"  {comp}: {mean:.3f} ± {std:.3f} kcal/mol\n")


def write_decomposition_results(
    result: DecompositionResult,
    output_path: str,
    format: str = "csv",
) -> None:
    """
    Write per-residue decomposition results.
    
    Creates two files:
    - {output_path}: Per-residue summary (mean ± std across frames)
    - {output_path}_per_frame.{format}: Per-frame per-residue detail
    """
    df_summary, df_frames = decomposition_to_dataframe(result)
    
    if format == "csv":
        df_summary.to_csv(output_path, index=False)
        df_frames.to_csv(output_path.replace(".csv", "_per_frame.csv"), index=False)
    elif format == "json":
        df_summary.to_json(output_path, orient="records", indent=2)
        df_frames.to_json(output_path.replace(".json", "_per_frame.json"), orient="records", indent=2)
    elif format == "excel":
        with pd.ExcelWriter(output_path) as writer:
            df_summary.to_excel(writer, sheet_name="Summary", index=False)
            df_frames.to_excel(writer, sheet_name="PerFrame", index=False)
    else:
        raise ValueError(f"Unknown format: {format}")


def write_combined_results(
    mmgbsa_result: MMGBSABatchResult,
    decomp_result: Optional[DecompositionResult],
    output_prefix: str,
    format: str = "csv",
) -> None:
    """Write both MMGBSA and decomposition results with common prefix."""
    write_mmgbsa_results(mmgbsa_result, f"{output_prefix}_mmgbsa.{format}", format)
    if decomp_result is not None:
        write_decomposition_results(decomp_result, f"{output_prefix}_decomp.{format}", format)


def print_summary(result: MMGBSABatchResult) -> None:
    """Print a nice summary to console."""
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    
    # Main table
    table = Table(title="MMGBSA Binding Energy Results")
    table.add_column("Component", style="cyan")
    table.add_column("Mean (kcal/mol)", justify="right", style="green")
    table.add_column("Std (kcal/mol)", justify="right", style="yellow")
    
    table.add_row("ΔG_bind (total)", f"{result.mean_delta_g:.3f}", f"{result.std_delta_g:.3f}")
    table.add_section()
    
    for comp, mean in result.mean_components.items():
        std = result.std_components[comp]
        table.add_row(f"  {comp}", f"{mean:.3f}", f"{std:.3f}")
    
    console.print(table)
    
    # Per-frame stats
    if len(result.results) > 1:
        delta_g_vals = [r.delta_g_bind for r in result.results]
        console.print(f"\nPer-frame ΔG range: [{min(delta_g_vals):.3f}, {max(delta_g_vals):.3f}] kcal/mol")


def print_decomposition_summary(result: DecompositionResult, top_n: int = 10) -> None:
    """Print top contributing residues."""
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    
    # Sort by absolute mean contribution
    contrib_abs = np.abs(result.mean_contrib)
    top_indices = np.argsort(contrib_abs)[::-1][:top_n]
    
    table = Table(title=f"Top {top_n} Residue Contributions to Binding")
    table.add_column("Residue", style="cyan")
    table.add_column("Chain", style="blue")
    table.add_column("Mean ΔG (kcal/mol)", justify="right", style="green")
    table.add_column("Std", justify="right", style="yellow")
    
    for idx in top_indices:
        res_name = result.residue_names[idx]
        res_id = result.residue_ids[idx]
        chain = result.chain_ids[idx]
        mean = result.mean_contrib[idx]
        std = result.std_contrib[idx]
        table.add_row(f"{res_name}{res_id}", chain, f"{mean:.3f}", f"{std:.3f}")
    
    console.print(table)


# Need numpy for abs sort
import numpy as np