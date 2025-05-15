import math
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import random

def plot_weight_adjusted_networks(nodes_file, edges_file, mask_file, left_source_scale,
                                  right_source_scale, left_global_scale, right_global_scale,
                                  arrow_size, title, selfloops):


    nodes = pd.read_csv(nodes_file)
    nodes = nodes[~nodes["names"].str.contains("MOB", case=False)]
    nodes = nodes[~nodes["names"].str.contains("SSs", case=False)]
                  
    edges = pd.read_csv(edges_file)
    edges = edges[edges["source"] != edges["target"]].copy()
    edges = edges[~(edges["source"].str.contains("MOB", case=False) | 
                    edges["target"].str.contains("MOB", case=False))]
    edges = edges[~(edges["source"].str.contains("SSs", case=False) | 
                    edges["target"].str.contains("SSs", case=False))]
    nodes = nodes[(nodes['posx'] != 0) | (nodes['posy'] != 0)]
    nodes['posy'] -= 4   # Aligns nodes better
    edgesf = edges.groupby(["source"]).sum(numeric_only=True).reset_index()
    activdict = pd.Series(edgesf['weight'].values, index=edgesf['source']).to_dict()
    edges['activations'] = edges['source'].map(activdict)
    pdf = edges[edges["source"] == edges["target"]].copy()


    weight_threshold = np.percentile(edges['weight'], 50) 
    filtered_edges = edges[edges['weight'] >= weight_threshold].copy()

    left_nodes = nodes[nodes['names'].str.contains('left', case=False)]
    right_nodes = nodes[nodes['names'].str.contains('right', case=False)]
    left_edges_global_adj = filtered_edges[filtered_edges['source'].isin(left_nodes['names']) & filtered_edges['target'].isin(left_nodes['names'])].copy()
    left_edges_global_adj['weight_global_adjusted'] = left_edges_global_adj['weight']/left_edges_global_adj['weight'].sum()
    right_edges_global_adj = filtered_edges[filtered_edges['source'].isin(right_nodes['names']) & filtered_edges['target'].isin(right_nodes['names'])].copy()
    right_edges_global_adj['weight_global_adjusted'] = right_edges_global_adj['weight']/right_edges_global_adj['weight'].sum()
    G_left_global_adj = nx.from_pandas_edgelist(left_edges_global_adj, source='source', target='target', edge_attr='weight_global_adjusted', create_using=nx.DiGraph())
    G_right_global_adj = nx.from_pandas_edgelist(right_edges_global_adj, source='source', target='target', edge_attr='weight_global_adjusted', create_using=nx.DiGraph())

    pos = {row['names']: (row['posx'], row['posy']) for _, row in nodes.iterrows()}

    data = pd.read_csv(mask_file, skiprows=1)
    data_array = data.to_numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(data_array, cmap="PuOr")
    
    nx.draw_networkx_nodes(G_left_global_adj, pos=pos, node_color="green", nodelist=left_nodes['names'], node_size=75)
    nx.draw_networkx_nodes(G_right_global_adj, pos=pos, node_color="green", nodelist=right_nodes['names'], node_size=75)
    left_edge_weights_global = [G_left_global_adj[i][j]['weight_global_adjusted'] * left_global_scale for i, j in G_left_global_adj.edges()]
    nx.draw_networkx_edges(G_left_global_adj, pos=pos, edge_color='k', width=left_edge_weights_global, arrows=True, arrowsize=arrow_size, connectionstyle='arc3,rad=0.1')
    right_edge_weights_global = [G_right_global_adj[i][j]['weight_global_adjusted'] * right_global_scale for i, j in G_right_global_adj.edges()]
    nx.draw_networkx_edges(G_right_global_adj, pos=pos, edge_color='k', width=right_edge_weights_global, arrows=True, arrowsize=arrow_size, connectionstyle='arc3,rad=0.1')
    # plt.title(title)
    plt.savefig(title, bbox_inches = "tight", transparent = 1)

    plt.show()


nodes = r'Source_Sink_analysis/new/Awake/nodes.csv'
edges = r"Source_Sink_analysis\new\Awake\edges_numActivation_thresh=1.csv"
mask = r"Source_Sink_analysis\mask.csv"
plot_weight_adjusted_networks(
    nodes,
    edges,
    mask,
    left_source_scale=5, right_source_scale=5,
    left_global_scale=30, right_global_scale=30,
    arrow_size=20, title = "Awake .", selfloops = 0 
)