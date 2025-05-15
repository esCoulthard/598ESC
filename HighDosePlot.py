import math
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import random
import re

# ---------------------------
def compute_metrics(G):
    for source, target, wdata in G.edges(data=True):
        wdata["inv_weight"] = 1 / wdata["weight"]
    largest_scc = max(nx.strongly_connected_components(G), key=len)
    G_scc = G.subgraph(largest_scc).copy()
    avg_sp = nx.average_shortest_path_length(G_scc, weight="inv_weight")
    betweenness = nx.betweenness_centrality(G_scc, weight="inv_weight")
    avg_bet = np.mean(list(betweenness.values()))
    avg_clust = nx.average_clustering(G_scc, weight="weight")
    return {
        "Average Shortest Path Length": round(avg_sp, 4),
        "Betweenness Centrality": round(avg_bet, 4),
        "Clustering Coefficient": round(avg_clust, 4)
    }

def analyze_graph_metrics_both_cases(csv):
    edges = pd.read_csv(csv)
    edges = edges[edges["source"] != edges["target"]].copy()
    edges = edges[~(edges["source"].str.contains("MOB", case=False) | 
                    edges["target"].str.contains("MOB", case=False))]
    edges = edges[~(edges["source"].str.contains("SSs", case=False) | 
                    edges["target"].str.contains("SSs", case=False))]
    left_edges = edges[edges["source"].str.contains("left", case=False) & 
                        edges["target"].str.contains("left", case=False)].copy()
    right_edges = edges[edges["source"].str.contains("right", case=False) & 
                         edges["target"].str.contains("right", case=False)].copy()
    left_metric = compute_metrics(nx.from_pandas_edgelist(left_edges, "source", "target", edge_attr="weight", create_using=nx.DiGraph()))
    right_metric = compute_metrics(nx.from_pandas_edgelist(right_edges, "source", "target", edge_attr="weight", create_using=nx.DiGraph()))
    return {"Left Hemisphere": left_metric, "Right Hemisphere": right_metric}

# ---------------------------
def plot_metric_low_high(metric_name, metrics_dict, title_prefix="High dose plot", ylim=None):
    conditions = ["Control", "Isoflurane", "Ketamine", "Pentobarbital"]
    base_positions = {cond: i for i, cond in enumerate(conditions)}
    offsets = np.array([-0.15, 0.15])
    drug_colors = {
         "Control": "steelblue",
         "Isoflurane": "forestgreen",
         "Ketamine": "darkorange",
         "Pentobarbital": "purple"
    }

    fig, ax = plt.subplots(figsize=(10,7))
    
    for cond in conditions:
        base = base_positions[cond]
        try:
            left_val = metrics_dict[cond]["Left Hemisphere"].get(metric_name, 0)
        except:
            left_val = 0
        try:
            right_val = metrics_dict[cond]["Right Hemisphere"].get(metric_name, 0)
        except:
            right_val = 0
        ax.bar(base + offsets[0], left_val, 0.3, color=drug_colors[cond], alpha=1.0, label=cond if cond not in ax.get_legend_handles_labels()[1] else "")
        ax.bar(base + offsets[1], right_val, 0.3, color=drug_colors[cond], alpha=0.8, label=cond if cond not in ax.get_legend_handles_labels()[1] else "")
    
    ax.set_xticks(list(base_positions.values()))
    ax.set_xticklabels(["Control", "Iso.", "Ket.", "Pent."], rotation=15, fontsize=35)
    ax.tick_params(axis='y', labelsize=35)
    ax.set_ylabel(f"{metric_name}", fontsize = 25)
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.tight_layout()
    filename = f"{title_prefix}_{metric_name}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------

files = {
    "Control": r"Source_Sink_analysis\new\Awake\edges_numActivation_thresh=1.csv",
    "Isoflurane": r"Source_Sink_analysis\new\Iso2\edges_numActivation_thresh=1.csv",
    "Ketamine": r"Source_Sink_analysis\new\ket100\edges_numActivation_thresh=1.csv",
    "Pentobarbital": r"Source_Sink_analysis\new\Pent80_30\edges_numActivation_thresh=1.csv"
}


aggregated_metrics = {}
for cond, fpath in files.items():
    aggregated_metrics[cond] = analyze_graph_metrics_both_cases(fpath)["Left Hemisphere"], analyze_graph_metrics_both_cases(fpath)["Right Hemisphere"]
    aggregated_metrics[cond] = {"Left Hemisphere": analyze_graph_metrics_both_cases(fpath)["Left Hemisphere"],
                                "Right Hemisphere": analyze_graph_metrics_both_cases(fpath)["Right Hemisphere"]}



ylim_dict = {
    "Average Shortest Path Length": (0, 10),
    "Betweenness Centrality": (0, 0.25),
    "Clustering Coefficient": (0, 0.06)
}


metrics_to_plot = ["Average Shortest Path Length", "Betweenness Centrality", "Clustering Coefficient"]
for metric in metrics_to_plot:
    ylim = ylim_dict.get(metric, None)
    plot_metric_low_high(metric, aggregated_metrics, title_prefix="High Dose", ylim=ylim)
