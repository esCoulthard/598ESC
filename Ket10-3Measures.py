import math
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import random
import re

# ---------------------------
# This should give you the Average Shortest Path Length, Betweenness Centrality, and Clustering Coefficient for the chosen condition.
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
        "Shortest Path Length": round(avg_sp, 4),
        "Betweenness Centrality": round(avg_bet, 4),
        "Clustering Coefficient": round(avg_clust, 4)
    }

def compute_metrics_repeated(G, trials=1000):
    trial_sps = []
    trial_bet = []
    trial_clust = []
    all_weights = {}
    largest_scc = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(largest_scc).copy()
    for u, v, d in G.edges(data=True):
        d["inv_weight"] = 1 / d["weight"]
    for _ in range(trials):
        current_weights = [G[u][v]["weight"] for u, v in G.edges()]
        np.random.shuffle(current_weights)
        for (u, v), new_weight in zip(G.edges(), current_weights):
            G[u][v]["weight"] = new_weight
        for u, v, d in G.edges(data=True):
            d["inv_weight"] = 1 / d["weight"]
        sp = nx.average_shortest_path_length(G, weight="inv_weight")
        trial_sps.append(sp)
        betw = nx.betweenness_centrality(G, weight="inv_weight")
        trial_bet.append(np.mean(list(betw.values())))
        clust = nx.average_clustering(G, weight="weight")
        trial_clust.append(clust)
        weights_dict = nx.get_edge_attributes(G, "weight")
        for edge, weight in weights_dict.items():
            all_weights.setdefault(edge, []).append(weight)
    avg_sp = np.median(trial_sps)
    max_sp = np.max(trial_sps)
    lower_err_sp = avg_sp - np.percentile(trial_sps, 25)
    upper_err_sp = np.percentile(trial_sps, 75) - avg_sp
    avg_bet = np.mean(trial_bet)
    max_bet = np.max(trial_bet)
    lower_err_bet = avg_bet - np.percentile(trial_bet, 25)
    upper_err_bet = np.percentile(trial_bet, 75) - avg_bet
    avg_clust = np.mean(trial_clust)
    max_clust = np.max(trial_clust)
    lower_err_clust = avg_clust - np.percentile(trial_clust, 25)
    upper_err_clust = np.percentile(trial_clust, 75) - avg_clust
    metrics = {
        "Shortest Path Length": {
            "average": round(avg_sp, 4),
            "max": round(max_sp, 4),
            "err": (round(lower_err_sp, 4), round(upper_err_sp, 4))
        },
        "Betweenness Centrality": {
            "average": round(avg_bet, 4),
            "max": round(max_bet, 4),
            "err": (round(lower_err_bet, 4), round(upper_err_bet, 4))
        },
        "Clustering Coefficient": {
            "average": round(avg_clust, 4),
            "max": round(max_clust, 4),
            "err": (round(lower_err_clust, 4), round(upper_err_clust, 4))
        }
    }
    return metrics, all_weights, trial_sps

def analyze_graph_metrics_both_cases(csv):
    edges = pd.read_csv(csv)
    edges = edges[edges["source"] != edges["target"]].copy()
    edges = edges[~(edges["source"].str.contains("MOB", case=False) | edges["target"].str.contains("MOB", case=False))]
    edges = edges[~(edges["source"].str.contains("SSs", case=False) | edges["target"].str.contains("SSs", case=False))]
    left_edges = edges[edges["source"].str.contains("left", case=False) & 
                        edges["target"].str.contains("left", case=False)].copy()
    right_edges = edges[edges["source"].str.contains("right", case=False) & 
                         edges["target"].str.contains("right", case=False)].copy()
    avg_weight_left = left_edges["weight"].mean()
    avg_weight_right = right_edges["weight"].mean()
    left_edges_avg = left_edges.copy()
    right_edges_avg = right_edges.copy()
    left_edges_avg["weight"] = avg_weight_left
    right_edges_avg["weight"] = avg_weight_right
    G_left = nx.from_pandas_edgelist(left_edges, "source", "target", edge_attr="weight", create_using=nx.DiGraph())
    G_right = nx.from_pandas_edgelist(right_edges, "source", "target", edge_attr="weight", create_using=nx.DiGraph())
    G_left_avg = nx.from_pandas_edgelist(left_edges_avg, "source", "target", edge_attr="weight", create_using=nx.DiGraph())
    G_right_avg = nx.from_pandas_edgelist(right_edges_avg, "source", "target", edge_attr="weight", create_using=nx.DiGraph())
    left_original = compute_metrics(G_left)
    right_original = compute_metrics(G_right)
    left_avg = compute_metrics(G_left_avg)
    right_avg = compute_metrics(G_right_avg)
    left_scrambled, _, _ = compute_metrics_repeated(G_left.copy(), trials=100)
    right_scrambled, _, _ = compute_metrics_repeated(G_right.copy(), trials=100)
    return {
        "Original Weights": {"Left Hemisphere": left_original, "Right Hemisphere": right_original},
        "Averaged Weights": {"Left Hemisphere": left_avg, "Right Hemisphere": right_avg},
        "Scrambled Weights": {"Left Hemisphere": left_scrambled, "Right Hemisphere": right_scrambled}
    }

def compute_individual_metrics(file_list):
    result = {
        "Left Hemisphere": {"Shortest Path Length": [], "Betweenness Centrality": [], "Clustering Coefficient": []},
        "Right Hemisphere": {"Shortest Path Length": [], "Betweenness Centrality": [], "Clustering Coefficient": []}
    }
    labels = {"Left Hemisphere": [], "Right Hemisphere": []}
    for file in file_list:
        match = re.search(r'iexp=(\d+)', file)
        label = match.group(1)
        edges = pd.read_csv(file)
        edges = edges[edges["source"] != edges["target"]].copy()
        edges = edges[~(edges["source"].str.contains("MOB", case=False) | 
                        edges["target"].str.contains("MOB", case=False))]
        edges = edges[~(edges["source"].str.contains("SSs", case=False) | 
                        edges["target"].str.contains("SSs", case=False))]
        left_edges = edges[edges["source"].str.contains("left", case=False) & 
                            edges["target"].str.contains("left", case=False)].copy()
        right_edges = edges[edges["source"].str.contains("right", case=False) & 
                             edges["target"].str.contains("right", case=False)].copy()
        if not left_edges.empty:
            G_left = nx.from_pandas_edgelist(left_edges, "source", "target", edge_attr="weight", create_using=nx.DiGraph())
            if G_left.number_of_edges() > 0:
                left_metrics = compute_metrics(G_left.copy())
                for metric in left_metrics:
                    result["Left Hemisphere"][metric].append(left_metrics[metric])
                labels["Left Hemisphere"].append(label)
        if not right_edges.empty:
            G_right = nx.from_pandas_edgelist(right_edges, "source", "target", edge_attr="weight", create_using=nx.DiGraph())
            if G_right.number_of_edges() > 0:
                right_metrics = compute_metrics(G_right.copy())
                for metric in right_metrics:
                    result["Right Hemisphere"][metric].append(right_metrics[metric])
                labels["Right Hemisphere"].append(label)
    return result, labels

def compute_average_network_with_zeros(file_list):
    left_dfs = []
    right_dfs = []
    left_edges_set = set()
    right_edges_set = set()
    for file in file_list:
        df = pd.read_csv(file)
        df = df[df["source"] != df["target"]].copy()
        df = df[~(df["source"].str.contains("MOB", case=False) | df["target"].str.contains("MOB", case=False))]
        df = df[~(df["source"].str.contains("SSs", case=False) | df["target"].str.contains("SSs", case=False))]
        df_left = df[df["source"].str.contains("left", case=False) & df["target"].str.contains("left", case=False)]
        for _, row in df_left.iterrows():
            left_edges_set.add((row["source"], row["target"]))
        df_right = df[df["source"].str.contains("right", case=False) & df["target"].str.contains("right", case=False)]
        for _, row in df_right.iterrows():
            right_edges_set.add((row["source"], row["target"]))
    left_edges_full = pd.DataFrame(list(left_edges_set), columns=["source", "target"])
    right_edges_full = pd.DataFrame(list(right_edges_set), columns=["source", "target"])
    for file in file_list:
        df = pd.read_csv(file)
        df = df[df["source"] != df["target"]].copy()
        df = df[~(df["source"].str.contains("MOB", case=False) | df["target"].str.contains("MOB", case=False))]
        df = df[~(df["source"].str.contains("SSs", case=False) | df["target"].str.contains("SSs", case=False))]
        df_left = df[df["source"].str.contains("left", case=False) & df["target"].str.contains("left", case=False)]
        df_left_complete = left_edges_full.merge(df_left[["source", "target", "weight"]], on=["source", "target"], how="left")
        df_left_complete["weight"] = df_left_complete["weight"].fillna(0)
        df_left_complete["file"] = file
        left_dfs.append(df_left_complete)
        df_right = df[df["source"].str.contains("right", case=False) & df["target"].str.contains("right", case=False)]
        df_right_complete = right_edges_full.merge(df_right[["source", "target", "weight"]], on=["source", "target"], how="left")
        df_right_complete["weight"] = df_right_complete["weight"].fillna(0)
        df_right_complete["file"] = file
        right_dfs.append(df_right_complete)
    all_left = pd.concat(left_dfs, ignore_index=True)
    all_right = pd.concat(right_dfs, ignore_index=True)
    avg_left = all_left.groupby(["source", "target"], as_index=False)["weight"].mean()
    avg_right = all_right.groupby(["source", "target"], as_index=False)["weight"].mean()
    G_left = nx.from_pandas_edgelist(avg_left, "source", "target", edge_attr="weight", create_using=nx.DiGraph())
    G_right = nx.from_pandas_edgelist(avg_right, "source", "target", edge_attr="weight", create_using=nx.DiGraph())
    return G_left, G_right

# ---------------------------
def plot_metric_merged(metric_name, metrics_dict, title_prefix="Control", 
                       individual_metrics=None, individual_labels=None, 
                       legend_outside=False, ylim=None):
    if metric_name == "Clustering Coefficient":
        methods = ["Original Weights", "Scrambled Weights"]
        base_positions = {"Original Weights": 0, "Scrambled Weights": 1}
        fixed_offset = 0.5
    else:
        methods = ["Original Weights", "Averaged Weights", "Scrambled Weights"]
        base_positions = {"Original Weights": 0, "Averaged Weights": 1, "Scrambled Weights": 2}
        fixed_offset = 0.4

    offsets = np.array([-0.15, 0.15])
    method_colors = {"Original Weights": "black", "Averaged Weights": "gray", "Scrambled Weights": "red"}

    fig, ax = plt.subplots(figsize=(8,5))
    legend_handles = {}

    for method in methods:
        base = base_positions[method]
        for i, hemi in enumerate(["Left Hemisphere", "Right Hemisphere"]):
            try:
                metric_val = metrics_dict[method][hemi].get(metric_name, 0)
            except:
                metric_val = 0
            if isinstance(metric_val, dict):
                val = metric_val.get("average", 0)
                err = metric_val.get("err", (0, 0))
            else:
                val = metric_val
                err = (0, 0)
            x_pos = base + offsets[i]
            alpha_val = 1.0 if i == 0 else 0.8
            label = method if method not in legend_handles else None
            if method == "Scrambled Weights":
                bar = ax.bar(x_pos, val, 0.3, color=method_colors[method], alpha=alpha_val,
                             label=label, yerr=np.array(err)[:, None], capsize=5)
            else:
                bar = ax.bar(x_pos, val, 0.3, color=method_colors[method], alpha=alpha_val,
                             label=label)
            if label is not None:
                legend_handles[method] = bar


    base_indiv = base_positions["Scrambled Weights"]
    x_indiv_fixed = base_indiv + fixed_offset


    ax.set_xticks(list(base_positions.values()))
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xticklabels(methods, rotation=15, fontsize=15)
    ax.set_title(f"Ketamine 10mg/kg - {metric_name}", fontsize=15)
    ax.set_ylabel("Metric Value")
    if ylim is not None:
        ax.set_ylim(ylim)
    handles, labels_ = ax.get_legend_handles_labels()
    unique = {}
    for h, l in zip(handles, labels_):
        if l not in unique:
            unique[l] = h

    
    filename = f"{title_prefix}_{metric_name}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

def print_and_plot_combined_metrics_merged(file1, file2, control_individual_files, iso_individual_files, 
                                           control_individual, control_labels, iso_individual, iso_labels, ylim_dict=None):
    iso_metrics = analyze_graph_metrics_both_cases(file2)
    iso_G_left, iso_G_right = compute_average_network_with_zeros(iso_individual_files)
    iso_combined = {
        "Left Hemisphere": compute_metrics(iso_G_left),
        "Right Hemisphere": compute_metrics(iso_G_right)
    }
    iso_metrics["Combined Average"] = iso_combined
    print("--- Pentobarbital 12.5 (Original Weights) ---")
    print(f"Left Hemisphere:  {iso_metrics['Original Weights']['Left Hemisphere']}")
    print(f"Right Hemisphere: {iso_metrics['Original Weights']['Right Hemisphere']}\n")
    metrics_to_plot = ["Shortest Path Length", "Betweenness Centrality", "Clustering Coefficient"]
    for metric in metrics_to_plot:
        ylims = ylim_dict.get(metric) if ylim_dict and metric in ylim_dict else None
        plot_metric_merged(metric, iso_metrics, title_prefix="Pentobarbital 12.5", 
                           individual_metrics=iso_individual, individual_labels=iso_labels, 
                           legend_outside=True, ylim=ylims)


# ---------------------------
file1 = r'Source_Sink_analysis\new\Awake\edges_numActivation_thresh=1.csv'
file2 = r'Source_Sink_analysis\new\Ket10\edges_numActivation_thresh=1.csv'

control_individual_files = [
    rf"Source_Sink_analysis\old2\individuals\Awake\iexp={i}\edges_numActivation_thresh=1.csv"
    for i in range(30, 38)
]
iso_individual_files = [
    rf"Source_Sink_analysis\old2\individuals\Iso1\iexp={i}\edges_numActivation_thresh=1.csv"
    for i in range(38, 44)
]

control_individual, control_labels = compute_individual_metrics(control_individual_files)
iso_individual, iso_labels = compute_individual_metrics(iso_individual_files)

ylim_dict = {   
    "Shortest Path Length": (0, 17),
    "Betweenness Centrality": (0, 0.25),
    "Clustering Coefficient": (0, 0.05)
}

print_and_plot_combined_metrics_merged(file1, file2, control_individual_files, iso_individual_files, 
                                       control_individual=control_individual, control_labels=control_labels,
                                       iso_individual=iso_individual, iso_labels=iso_labels, ylim_dict=ylim_dict)
