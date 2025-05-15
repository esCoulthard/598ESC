import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import re

def compute_shuffled_degrees(csv_file, hemisphere="left", trials=1000, 
                             xlim=None, ylim=None, bin_size=10,
                             individual_files=None, individual_labels=None,
                             condition="default"):
    """
    Loads an aggregated CSV file and computes:
      - Aggregated original weighted degrees,
      - Aggregated shuffled weighted degrees (by shuffling edge weights over trials),
      - If provided, the individual weighted degree distributions.
    Then it returns four lists plus individual histograms:
      - original_in_degs, original_out_degs,
      - all_in_degrees, all_out_degs,
      - individual_in_hist, individual_out_hist.
    """
    # Process aggregated CSV file.
    df = pd.read_csv(csv_file)
    df = df[df["source"] != df["target"]].copy()
    df = df[~(df["source"].str.contains("MOB", case=False) | 
              df["target"].str.contains("MOB", case=False))]
    
    if hemisphere.lower() == "left":
        df_hemi = df[df["source"].str.contains("left", case=False) & 
                     df["target"].str.contains("left", case=False)].copy()
    elif hemisphere.lower() == "right":
        df_hemi = df[df["source"].str.contains("right", case=False) & 
                     df["target"].str.contains("right", case=False)].copy()
    else:
        raise ValueError("Hemisphere must be 'left' or 'right'")
    
    if df_hemi.empty:
        print(f"No edges found for the {hemisphere} hemisphere.")
        return [], [], [], [], {}, {}
    
    # Build aggregated graph.
    G_original = nx.from_pandas_edgelist(df_hemi, source="source", target="target",
                                         edge_attr="weight", create_using=nx.DiGraph())
    original_in_degs = [G_original.in_degree(n, weight="weight") for n in G_original.nodes()]
    original_out_degs = [G_original.out_degree(n, weight="weight") for n in G_original.nodes()]
    
    # Compute shuffled aggregated degrees.
    all_in_degs = []
    all_out_degs = []
    for t in range(trials):
        df_shuffled = df_hemi.copy()
        weights = df_shuffled["weight"].values.copy()
        np.random.shuffle(weights)
        df_shuffled["weight"] = weights
        G_shuffled = nx.from_pandas_edgelist(df_shuffled, source="source", target="target",
                                             edge_attr="weight", create_using=nx.DiGraph())
        in_degs = [G_shuffled.in_degree(n, weight="weight") for n in G_shuffled.nodes()]
        out_degs = [G_shuffled.out_degree(n, weight="weight") for n in G_shuffled.nodes()]
        all_in_degs.extend(in_degs)
        all_out_degs.extend(out_degs)
    
    # Process individual CSV files if provided.
    individual_in_hist = {}
    individual_out_hist = {}
    if individual_files is not None and individual_labels is not None:
        if len(individual_files) != len(individual_labels):
            raise ValueError("Length of individual_files and individual_labels must match.")
        for file, label in zip(individual_files, individual_labels):
            df_ind = pd.read_csv(file)
            df_ind = df_ind[df_ind["source"] != df_ind["target"]].copy()
            df_ind = df_ind[~(df_ind["source"].str.contains("MOB", case=False) | 
                              df_ind["target"].str.contains("MOB", case=False))]
            if hemisphere.lower() == "left":
                df_ind = df_ind[df_ind["source"].str.contains("left", case=False) & 
                                df_ind["target"].str.contains("left", case=False)].copy()
            elif hemisphere.lower() == "right":
                df_ind = df_ind[df_ind["source"].str.contains("right", case=False) & 
                                df_ind["target"].str.contains("right", case=False)].copy()
            if df_ind.empty:
                continue
            G_ind = nx.from_pandas_edgelist(df_ind, source="source", target="target",
                                            edge_attr="weight", create_using=nx.DiGraph())
            individual_in_hist[label] = [G_ind.in_degree(n, weight="weight") for n in G_ind.nodes()]
            individual_out_hist[label] = [G_ind.out_degree(n, weight="weight") for n in G_ind.nodes()]
    
    return original_in_degs, original_out_degs, all_in_degs, all_out_degs, individual_in_hist, individual_out_hist

def plot_histograms_toggled(csv_file, hemisphere, trials, bin_size, xlim, ylim, condition,
                            individual_files=None, individual_labels=None,
                            degree_type="in"):
    """
    Plots a combined histogram for a given degree type ("in" or "out") for a given condition and hemisphere.
    It overlays:
      - Aggregated original (black),
      - Aggregated shuffled (red; counts normalized by trials),
      - Individual mouse histograms (if provided) using fixed colors.
    The plot is automatically saved as a PNG file.
    """
    orig_in, orig_out, all_in, all_out, indiv_in, indiv_out = compute_shuffled_degrees(
        csv_file, hemisphere=hemisphere, trials=trials, xlim=xlim, ylim=ylim, bin_size=bin_size,
        individual_files=individual_files, individual_labels=individual_labels, condition=condition)
    
    if degree_type.lower() == "in":
        aggregated = orig_in
        aggregated_shuffled = all_in
        individual_hist = indiv_in if (individual_files is not None and individual_labels is not None) else None
        degree_label = "In-Degree"
    else:
        aggregated = orig_out
        aggregated_shuffled = all_out
        individual_hist = indiv_out if (individual_files is not None and individual_labels is not None) else None
        degree_label = "Out-Degree"
    
    # Determine bins.
    combined = aggregated + aggregated_shuffled
    if individual_hist is not None:
        for vals in individual_hist.values():
            combined.extend(vals)
    if xlim is not None:
        bins = np.arange(xlim[0], xlim[1] + bin_size, bin_size)
    else:
        bins = np.arange(min(combined), max(combined) + bin_size, bin_size)
    
    plt.figure(figsize=(8,6))
    # Plot aggregated original (black)
    plt.hist(aggregated, bins=bins, weights=np.ones_like(aggregated),
             histtype="step", linewidth=3, color="black", label="Aggregated Original")
    # Plot aggregated shuffled (red)
    plt.hist(aggregated_shuffled, bins=bins, weights=np.ones_like(aggregated_shuffled)/trials,
             histtype="step", linewidth=3, color="red", label="Aggregated Shuffled")
    
    # Plot individual histograms (if provided)
    if individual_hist:
        # Use fixed color palette.
        palette = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 
                   'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
        for i, (label, values) in enumerate(individual_hist.items()):
            plt.hist(values, bins=bins, weights=np.ones_like(values),
                     histtype="step", linewidth=2, color=palette[i % len(palette)], label=f"Mouse {label}")
    
    plt.xlabel(f"Weighted {degree_label}")
    plt.ylabel("Frequency (Count; Shuffled normalized by trials)")
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    title = f"Weighted {degree_label}: {hemisphere.capitalize()} Hemisphere, {trials} Trials, Bin size: {bin_size}\nCondition: {condition}"
    plt.title(title)
    plt.legend(loc='upper left', bbox_to_anchor=(1,1))
    plt.grid(True)
    plt.tight_layout()
    filename = f"weighted_{degree_type}_degree_{condition}_{hemisphere}_trials{trials}_binsize{bin_size}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# ==========================================================
# Example usage:
# For the Awake condition (control) with individual files for iexp=30 to iexp=37.
control_csv = r"C:\Users\ebug2\OneDrive\Desktop\Source_Sink_analysis-20250312T054436Z-001\Source_Sink_analysis\Awake\edges_numActivation_thresh=1.csv"
control_files = [rf"C:\Users\ebug2\OneDrive\Desktop\Source_Sink_analysis-20250312T054436Z-001\Source_Sink_analysis\individualmice\Awake\iexp={i}\edges_numActivation_thresh=1.csv" for i in range(30,38)]
control_labels = [str(i) for i in range(30,38)]

# For the Iso condition with individual files for iexp=38 to iexp=43.
iso_csv = r"C:\Users\ebug2\OneDrive\Desktop\Source_Sink_analysis-20250312T054436Z-001\Source_Sink_analysis\Iso1\edges_numActivation_thresh=1.csv"
iso_files = [rf"C:\Users\ebug2\OneDrive\Desktop\Source_Sink_analysis-20250312T054436Z-001\Source_Sink_analysis\individualmice\Iso1\iexp={i}\edges_numActivation_thresh=1.csv" for i in range(38,44)]
iso_labels = [str(i) for i in range(38,44)]

# Set parameters.
trials = 1000
bin_size = 10
xlim = (0, 210)
ylim = (0, 17.5)

# Produce 8 plots:
# Control (Awake) Left Hemisphere
plot_histograms_toggled(control_csv, hemisphere="left", trials=trials, bin_size=bin_size,
                        xlim=xlim, ylim=ylim, condition="control", 
                        individual_files=control_files, individual_labels=control_labels, degree_type="in")
plot_histograms_toggled(control_csv, hemisphere="left", trials=trials, bin_size=bin_size,
                        xlim=xlim, ylim=ylim, condition="control", 
                        individual_files=control_files, individual_labels=control_labels, degree_type="out")

# Control (Awake) Right Hemisphere
plot_histograms_toggled(control_csv, hemisphere="right", trials=trials, bin_size=bin_size,
                        xlim=xlim, ylim=ylim, condition="control", 
                        individual_files=control_files, individual_labels=control_labels, degree_type="in")
plot_histograms_toggled(control_csv, hemisphere="right", trials=trials, bin_size=bin_size,
                        xlim=xlim, ylim=ylim, condition="control", 
                        individual_files=control_files, individual_labels=control_labels, degree_type="out")

# Iso Left Hemisphere
plot_histograms_toggled(iso_csv, hemisphere="left", trials=trials, bin_size=bin_size,
                        xlim=xlim, ylim=ylim, condition="iso", 
                        individual_files=iso_files, individual_labels=iso_labels, degree_type="in")
plot_histograms_toggled(iso_csv, hemisphere="left", trials=trials, bin_size=bin_size,
                        xlim=xlim, ylim=ylim, condition="iso", 
                        individual_files=iso_files, individual_labels=iso_labels, degree_type="out")

# Iso Right Hemisphere
plot_histograms_toggled(iso_csv, hemisphere="right", trials=trials, bin_size=bin_size,
                        xlim=xlim, ylim=ylim, condition="iso", 
                        individual_files=iso_files, individual_labels=iso_labels, degree_type="in")
plot_histograms_toggled(iso_csv, hemisphere="right", trials=trials, bin_size=bin_size,
                        xlim=xlim, ylim=ylim, condition="iso", 
                        individual_files=iso_files, individual_labels=iso_labels, degree_type="out")
