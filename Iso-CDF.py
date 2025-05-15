import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import re

# ---------------------------
# This computes the cumulative distribution function for the Isoflurane condition, and downloads/displays it for the in/out degrees, along with the left/right hemispheres, giving 4 outputs.
# ---------------------------
def compute_shuffled_degrees(csv_file, hemisphere="left", trials=1000, 
                             xlim=None, ylim=None, bin_size=10,
                             individual_files=None, individual_labels=None,
                             condition="default"):
    # Process aggregated CSV file.
    df = pd.read_csv(csv_file)
    df = df[df["source"] != df["target"]].copy()
    df = df[~(df["source"].str.contains("MOB", case=False) | 
              df["target"].str.contains("MOB", case=False))]
    df = df[~(df["source"].str.contains("SSs1", case=False) | 
              df["target"].str.contains("SSs1", case=False))]
    
    # Select hemisphere.
    if hemisphere.lower() == "left":
        df_hemi = df[df["source"].str.contains("left", case=False) & 
                     df["target"].str.contains("left", case=False)].copy()
    elif hemisphere.lower() == "right":
        df_hemi = df[df["source"].str.contains("right", case=False) & 
                     df["target"].str.contains("right", case=False)].copy()
    
    # Build aggregated original graph.
    G_original = nx.from_pandas_edgelist(df_hemi, source="source", target="target",
                                         edge_attr="weight", create_using=nx.DiGraph())
    nodes_list = list(G_original.nodes())
    original_in_degs = [G_original.in_degree(n, weight="weight") for n in nodes_list]
    original_out_degs = [G_original.out_degree(n, weight="weight") for n in nodes_list]
    

    average_shuffled_in = 1
    average_shuffled_out = 1
    
    # Process individual CSV files if provided.
    individual_in_hist = {}
    individual_out_hist = {}

    for file, label in zip(individual_files, individual_labels):
        df_ind = pd.read_csv(file)
        df_ind = df_ind[df_ind["source"] != df_ind["target"]].copy()
        df_ind = df_ind[~(df_ind["source"].str.contains("MOB", case=False) | 
                            df_ind["target"].str.contains("MOB", case=False))]
        df_ind = df_ind[~(df_ind["source"].str.contains("SSs1", case=False) | 
                            df_ind["target"].str.contains("SSs1", case=False))]
        if hemisphere.lower() == "left":
            df_ind = df_ind[df_ind["source"].str.contains("left", case=False) & 
                            df_ind["target"].str.contains("left", case=False)].copy()
        elif hemisphere.lower() == "right":
            df_ind = df_ind[df_ind["source"].str.contains("right", case=False) & 
                            df_ind["target"].str.contains("right", case=False)].copy()
        G_ind = nx.from_pandas_edgelist(df_ind, source="source", target="target",
                                        edge_attr="weight", create_using=nx.DiGraph())
        individual_in_hist[label] = [G_ind.in_degree(n, weight="weight") if n in G_ind else 0
                                        for n in nodes_list]
        individual_out_hist[label] = [G_ind.out_degree(n, weight="weight") if n in G_ind else 0
                                        for n in nodes_list]
    
    return original_in_degs, original_out_degs, average_shuffled_in, average_shuffled_out, individual_in_hist, individual_out_hist

def compute_empirical_ccdf(data):
    data = np.array(data)
    n = len(data)
    sorted_data = np.sort(data)
    ccdf = np.array([np.sum(data > x) for x in sorted_data]) / n
    new_x = np.insert(sorted_data, 0, 0)
    new_ccdf = np.insert(ccdf, 0, 1.0)
    return new_x, new_ccdf

def compute_empirical_cdf(data):
    x, ccdf = compute_empirical_ccdf(data)
    cdf = 1 - ccdf
    new_x = np.append(x, 100)
    new_cdf = np.append(cdf, 1.0)
    return new_x, new_cdf


def plot_combined_cdf(control_csv, iso_csv, hemisphere, trials, bin_size, xlim, ylim, condition,
                      individual_files_control=None, individual_labels_control=None,
                      individual_files_iso=None, individual_labels_iso=None,
                      degree_type="in", extra_iso2_file=None):
    # Compute control data.
    orig_in_ctrl, orig_out_ctrl, _, _, indiv_in_ctrl, indiv_out_ctrl = compute_shuffled_degrees(
        control_csv, hemisphere=hemisphere, trials=trials, bin_size=bin_size,
        individual_files=individual_files_control, individual_labels=individual_labels_control,
        condition=condition)
    
    # Compute iso data.
    orig_in_iso, orig_out_iso, _, _, indiv_in_iso, indiv_out_iso = compute_shuffled_degrees(
        iso_csv, hemisphere=hemisphere, trials=trials, bin_size=bin_size,
        individual_files=individual_files_iso, individual_labels=individual_labels_iso,
        condition=condition)
    
    if degree_type.lower() == "in":
        control_agg = orig_in_ctrl
        iso_agg = orig_in_iso
        control_indiv = indiv_in_ctrl
        iso_indiv = indiv_in_iso
        degree_label = "In-Degree"
    else:
        control_agg = orig_out_ctrl
        iso_agg = orig_out_iso
        control_indiv = indiv_out_ctrl
        iso_indiv = indiv_out_iso
        degree_label = "Out-Degree"
    
    plt.figure(figsize=(8,6))
    
    # Plot individual curves for control in blue.
    if control_indiv:
        for label, values in control_indiv.items():
            x_ind, cdf_ind = compute_empirical_cdf(values)
            plt.step(x_ind, cdf_ind, where='post', linewidth=0.55, alpha=0.5, color="blue")
    
    # Plot individual curves for iso in red.
    if iso_indiv:
        for label, values in iso_indiv.items():
            x_ind, cdf_ind = compute_empirical_cdf(values)
            plt.step(x_ind, cdf_ind, where='post', linewidth=0.55, alpha=0.5, color="red")
    
    # Compute aggregated empirical CDFs.
    x_ctrl, cdf_ctrl = compute_empirical_cdf(control_agg)
    x_iso, cdf_iso = compute_empirical_cdf(iso_agg)
    
    # Plot aggregated curves.
    plt.step(x_ctrl, cdf_ctrl, where='post', linewidth=2, color="blue", label="Control")
    plt.step(x_iso, cdf_iso, where='post', linewidth=2, color="red", label="Iso. 1%")
    
    # Plot high dose.
    df_extra = pd.read_csv(extra_iso2_file)
    df_extra = df_extra[df_extra["source"] != df_extra["target"]].copy()
    df_extra = df_extra[~(df_extra["source"].str.contains("MOB", case=False) | 
                            df_extra["target"].str.contains("MOB", case=False))]
    df_extra = df_extra[~(df_extra["source"].str.contains("SSs1", case=False) | 
                            df_extra["target"].str.contains("SSs1", case=False))]
    if hemisphere.lower() == "left":
        df_extra = df_extra[df_extra["source"].str.contains("left", case=False) & 
                            df_extra["target"].str.contains("left", case=False)].copy()
    elif hemisphere.lower() == "right":
        df_extra = df_extra[df_extra["source"].str.contains("right", case=False) & 
                            df_extra["target"].str.contains("right", case=False)].copy()
    G_extra = nx.from_pandas_edgelist(df_extra, source="source", target="target",
                                        edge_attr="weight", create_using=nx.DiGraph())
    extra_degs = []
    for n in G_extra.nodes():
        if degree_type.lower() == "in":
            extra_degs.append(G_extra.in_degree(n, weight="weight"))
        else:
            extra_degs.append(G_extra.out_degree(n, weight="weight"))
    x_extra, cdf_extra = compute_empirical_cdf(extra_degs)
    plt.step(x_extra, cdf_extra, where='post', linewidth=2, color="magenta", label="Iso. 2%")
    
    plt.xlabel(f"Weighted Degree", fontsize=35)
    plt.tick_params(axis='y', labelsize=35)
    plt.tick_params(axis='x', labelsize=35)
    plt.ylabel("% of Nodes", fontsize = 25)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    title = f"Weighted {degree_label} ({hemisphere.capitalize()} Hemisphere): Isoflurane"
    plt.title(title)
    plt.legend( fontsize=25)
    plt.grid(True)
    plt.tight_layout()
    filename = f"combined_weighted_{degree_type}_degree_{condition}_{hemisphere}_trials{trials}_empirical_cdf.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# ==========================================================

# Control condition.
control_csv = r'Source_Sink_analysis\new\Awake\edges_numActivation_thresh=1.csv'
control_files = [rf"Source_Sink_analysis\old2\individuals\Awake\iexp={i}\edges_numActivation_thresh=1.csv" for i in range(30,38)]
control_labels = [str(i) for i in range(30,38)]

# Iso condition.
iso_csv = r'Source_Sink_analysis\new\Iso1\edges_numActivation_thresh=1.csv'
iso_files = [rf"Source_Sink_analysis\old2\individuals\Iso1\iexp={i}\edges_numActivation_thresh=1.csv" for i in range(38,44)]
iso_labels = [str(i) for i in range(38,44)]

# Extra network for Iso2.
extra_iso2_file = r'Source_Sink_analysis\new\Iso2\edges_numActivation_thresh=1.csv'

# Set parameters.
trials = 100    
bin_size = 0
xlim = (-0.01, 65)
ylim = (0, 1.01)
condition = "Control vs Isoflurane"

# Produce combined plots:
# In-Degree Left Hemisphere.
plot_combined_cdf(control_csv, iso_csv, hemisphere="left", trials=trials, bin_size=bin_size,
                  xlim=xlim, ylim=ylim, condition=condition,
                  individual_files_control=control_files, individual_labels_control=control_labels,
                  individual_files_iso=iso_files, individual_labels_iso=iso_labels,
                  degree_type="in", extra_iso2_file=extra_iso2_file)

# Out-Degree Left Hemisphere.
plot_combined_cdf(control_csv, iso_csv, hemisphere="left", trials=trials, bin_size=bin_size,
                  xlim=xlim, ylim=ylim, condition=condition,
                  individual_files_control=control_files, individual_labels_control=control_labels,
                  individual_files_iso=iso_files, individual_labels_iso=iso_labels,
                  degree_type="out", extra_iso2_file=extra_iso2_file)

# In-Degree Right Hemisphere.
plot_combined_cdf(control_csv, iso_csv, hemisphere="right", trials=trials, bin_size=bin_size,
                  xlim=xlim, ylim=ylim, condition=condition,
                  individual_files_control=control_files, individual_labels_control=control_labels,
                  individual_files_iso=iso_files, individual_labels_iso=iso_labels,
                  degree_type="in", extra_iso2_file=extra_iso2_file)

# Out-Degree Right Hemisphere.
plot_combined_cdf(control_csv, iso_csv, hemisphere="right", trials=trials, bin_size=bin_size,
                  xlim=xlim, ylim=ylim, condition=condition,
                  individual_files_control=control_files, individual_labels_control=control_labels,
                  individual_files_iso=iso_files, individual_labels_iso=iso_labels,
                  degree_type="out", extra_iso2_file=extra_iso2_file)
