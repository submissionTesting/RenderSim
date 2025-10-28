"""Utility to visualise operator pipelines as dependency graphs.

Generates visualizations for all pipelines including:
    • Existing inference pipelines (ICARUS, NeuRex, CICERO, GSCore, SRender)
    • New training pipelines (GSArch, GBU, Instant3D)

Graphs are saved as PNGs beside the script.
"""
# Builds pipelines and leverages their internal plotting utility.

from pipelines.neurex_pipeline import build_neurex_pipeline
from pipelines.gaurast_pipeline import build_gaurast_pipeline
from pipelines.gscore_pipeline import build_gscore_pipeline
from pipelines.srender_pipeline import build_srender_pipeline
from pipelines.icarus_pipeline import build_icarus_pipeline
from pipelines.cicero_pipeline import build_cicero_pipeline

# Import new training pipelines
from pipelines.gsarch_pipeline import build_gsarch_training_pipeline
from pipelines.gbu_pipeline import build_gbu_pipeline
from pipelines.instant3d_pipeline import build_instant3d_training_pipeline

from utils.operator_graph import FineOperatorGraph


def main():
    dim = (4, 64)  # example batch/sample dims

    print("Generating pipeline visualizations...")
    print("=" * 50)
    
    # ===== EXISTING INFERENCE PIPELINES =====
    print("\n--- Inference Pipelines ---")
    
    # Icarus pipeline
    print("Generating ICARUS pipeline graphs...")
    icarus_graph = build_icarus_pipeline((800*800, 192))
    icarus_graph.plot_graph("Icarus Pipeline (Coarse)", "icarus_pipeline_graph.png")
    icarus_graph.plot_fine_graph("Icarus Pipeline (Fine)", "icarus_pipeline_graph_fine.png")

    # Neurex pipeline
    print("Generating NeuRex pipeline graphs...")
    neurex_graph = build_neurex_pipeline(dim)
    neurex_graph.plot_graph("Neurex Pipeline (Coarse)", "neurex_pipeline_graph.png")
    neurex_graph.plot_fine_graph("Neurex Pipeline (Fine)", "neurex_pipeline_graph_fine.png")

    # Gaurast pipeline
    print("Generating Gaurast pipeline graphs...")
    gaurast_graph = build_gaurast_pipeline((400,400))
    gaurast_graph.plot_graph("Gaurast Pipeline (Coarse)", "gaurast_pipeline_graph.png")
    gaurast_graph.plot_fine_graph("Gaurast Pipeline (Fine)", "gaurast_pipeline_graph_fine.png")

    # GSCore pipeline
    print("Generating GSCore pipeline graphs...")
    gscore_graph = build_gscore_pipeline((400,400))
    gscore_graph.plot_graph("GSCore Pipeline (Coarse)", "gscore_pipeline_graph.png")
    gscore_graph.plot_fine_graph("GSCore Pipeline (Fine)", "gscore_pipeline_graph_fine.png")

    # S‑render pipeline
    print("Generating SRender pipeline graphs...")
    srender_graph = build_srender_pipeline(dim)
    srender_graph.plot_graph("Staged‑Render Pipeline (Coarse)", "srender_pipeline_graph.png")
    srender_graph.plot_fine_graph("Staged‑Render Pipeline (Fine)", "srender_pipeline_graph_fine.png")

    # CICERO pipeline
    print("Generating CICERO pipeline graphs...")
    cicero_graph = build_cicero_pipeline(dim)
    cicero_graph.plot_graph("CICERO Pipeline (Coarse)", "cicero_pipeline_graph.png")
    cicero_graph.plot_fine_graph("CICERO Pipeline (Fine)", "cicero_pipeline_graph_fine.png")
    
    # ===== NEW TRAINING PIPELINES =====
    print("\n--- Training Pipelines with Backward Passes ---")
    
    # GSArch training pipeline
    print("Generating GSArch training pipeline graphs...")
    gsarch_graph = build_gsarch_training_pipeline(dim)
    gsarch_graph.plot_graph("GSArch Training Pipeline (Coarse)", "gsarch_training_graph.png")
    gsarch_graph.plot_fine_graph("GSArch Training Pipeline (Fine)", "gsarch_training_graph_fine.png")
    
    # GBU pipeline
    print("Generating GBU pipeline graphs...")
    gbu_graph = build_gbu_pipeline(dim)
    gbu_graph.plot_graph("GBU Pipeline (Coarse)", "gbu_pipeline_graph.png")
    gbu_graph.plot_fine_graph("GBU Pipeline (Fine)", "gbu_pipeline_graph_fine.png")
    
    # Instant3D training pipeline
    print("Generating Instant3D training pipeline graphs...")
    instant3d_graph = build_instant3d_training_pipeline(dim)
    instant3d_graph.plot_graph("Instant3D Training Pipeline (Coarse)", "instant3d_training_graph.png")
    instant3d_graph.plot_fine_graph("Instant3D Training Pipeline (Fine)", "instant3d_training_graph_fine.png")
    
    print("\n" + "=" * 50)
    print("All pipeline visualizations generated successfully!")
    print("\nGenerated files:")
    print("  Inference pipelines: *_pipeline_graph.png, *_pipeline_graph_fine.png")
    print("  Training pipelines: *_training_graph.png, *_training_graph_fine.png")


if __name__ == "__main__":
    main() 