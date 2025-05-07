#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 11:11:46 2024

@author: Manli Luo

network_calculate_plot
"""
import numpy as np
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300

mpl.rcParams['agg.path.chunksize'] = 10000
# mpl.use('TkAgg')
# mpl.use('Agg')
import matplotlib
# mpl.use('agg')
import os.path as op
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as scio
import seaborn as sns
import source_function
from source_function import annotation_plot_
import networkx as nx
import os
from test_statistics import statistical_analysis, convert_pvalue_to_asterisks
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

matplotlib.rcParams['svg.fonttype'] = 'none'

cm = 1 / 2.54


def mst_network(connect):
    """Create maximum spanning tree from fc matrix

    :param connect: functional connectivity
    :return: MST measures: [Diameter, Eccentricity, Leaf number, Tree hierarchy]
    """

    # create graph from adj matrix
    G = nx.from_numpy_array(connect)

    # set node color
    r, g, b = [0.9, 0.6, 0.5], [0.83, 0.83, 0.83], [30 / 255, 144 / 255, 255 / 255]
    node_color = []
    edge_color = []
    for i in range(29):
        if i in [0, 3, 5, 8, 10, 13, 15, 17, 19, 22, 24, 27]:
            G.nodes[i]['loc'] = 'l'
            node_color.append(r)
        elif i in [1, 4, 6, 9, 11, 14, 16, 18, 20, 23, 25, 28]:
            G.nodes[i]['loc'] = 'r'
            node_color.append(b)
        else:
            G.nodes[i]['loc'] = 'm'
            node_color.append(g)

    # create minimum spanning tree
    MST = nx.maximum_spanning_tree(G, weight='weight')

    '''
    # set edge color
    for e in list(MST.edges):
        loc1 = MST.nodes[e[0]]['loc']
        loc2 = MST.nodes[e[1]]['loc']
        if (loc1 == 'l' and loc2 == 'r') or (loc1 == 'r' and loc2 == 'l'):
            edge_color.append(g)
        elif (loc1 == 'l' and loc2 == 'l') or (loc1 == 'l' and loc2 == 'm') or (loc1 == 'm' and loc2 == 'l'):
            edge_color.append(r)
        elif (loc1 == 'r' and loc2 == 'r') or (loc1 == 'r' and loc2 == 'm') or (loc1 == 'm' and loc2 == 'r'):
            edge_color.append(b)
        else:
            edge_color.append(g)
    '''

    # MST measure
    diameter = nx.diameter(MST)  # 直径 Returns the diameter of the graph G
    ecc = nx.eccentricity(MST)  # eccentricity，Returns the eccentricity of nodes in G.偏心率
    ecc_mean = sum(ecc.values()) / len(ecc)
    leaf_nodes = [x for x in MST.nodes() if MST.degree(x) == 1]
    bet_cen = nx.betweenness_centrality(MST)  # Compute the shortest-path betweenness centrality for nodes
    tree_hie = len(leaf_nodes) / (2 * 28 * max(bet_cen.values()))

    return np.array([diameter, ecc_mean, len(leaf_nodes), tree_hie])


def load_mat_file(filepath):
    data = scio.loadmat(filepath)
    imcoh = abs(data['imcoh'])
    for w in range(imcoh.shape[0]):
        imcoh[w, w] = np.min(imcoh)
    return imcoh


def process_folders3(name, folder, stage_path):
    files = {file: os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.mat')}
    results = []
    for file_name in files:
        data = load_mat_file(files[file_name])
        result = mst_network(data)
        results.append(result)

    results_array = np.array(results)
    output_file_path = op.join(stage_path, f"{name}_value")
    np.save(output_file_path, results_array)


def feature_calculation(base_path, save_path, stages):
    '''
    Calculate connectivity features (imaginary coherence) for specified sleep stages.

    This function processes EEG data to compute connectivity measures between different
    brain regions for each specified sleep stage.

    Parameters:
    -----------
    base_path : str
        Base directory containing the input data organized by sleep stage
    save_path : str
        Directory where calculated connectivity features will be saved
    stages : list
        List of sleep stages to process (e.g., ['N2', 'N3', 'REM'])

    The function:
    1. Creates output directories for each sleep stage
    2. Processes each subject's data folder within each stage
    3. Calculates connectivity features using imaginary coherence (imcoh)
    '''

    # Process each specified sleep stage
    for stage in stages:
        # Set up paths for current stage
        file_path = op.join(base_path, stage)  # Input path for current stage
        stage_path = op.join(save_path, stage)  # Output path for current stage

        # Create output directory if it doesn't exist
        source_function.mkdir(stage_path)

        # Get all subject folders for current stage
        # Only include directories, ignoring any files
        folders = {
            name: op.join(file_path, name)
            for name in os.listdir(file_path)
            if op.isdir(op.join(file_path, name))
        }

        # Process each subject's data folder
        for folder_name, folder_path in folders.items():
            # Calculate connectivity features for this subject
            process_folders3(folder_name, folder_path, stage_path)


def network_test_plot(data_, name_list, savename, y_label, ax=None, color=None, color1=None):
    '''network统计分析,对于eeg数据'''
    dict_ = {'stages': [], 'Value': []}
    for stage in range(len(name_list)):
        for v in range(len(data_[0])):
            dict_['stages'].append(name_list[stage])
            dict_['Value'].append(data_[stage][v])

    df = pd.DataFrame(dict_, columns=['stages', 'Value'])
    ### plot
    if color == None:
        color = {0: '#B8DBB3',  # Wake
                 1: '#EAB883',  # N1
                 2: '#A8CBDF',  # N2
                 3: '#8074C8',  # N3
                 4: '#F5EBAE',  # REM
                 }
    if color1 == None:
        color1 = {0: '#cfe6cb',
                  1: '#f0cda8',
                  2: '#c7deea',
                  3: '#9b91d4',
                  4: '#f9f4d2'
                  }
    color_pal = sns.color_palette([color[3], color[2], color[4]])

    color_pal1 = sns.color_palette([color[3], color[2], color[4]])
    if ax == None:
        fig, ax = plt.subplots(figsize=(3, 2.5))

    p = statistical_analysis(data_[0], data_[1], data_[2])
    annotations = [convert_pvalue_to_asterisks(i) for i in p]
    pairs = [
        ('N2', 'N3'), ('N3', 'REM'), ('N2', 'REM')]
    annotation_plot_(df, 'stages', 'Value', annotations, color_pal, color_pal1, name_list,
                     ax, pairs, y_label, types='box', box_alpha=0.3)
    ax.set_ylim(np.min(data_) - np.min(data_) * 0.2, np.max(data_) + np.max(data_) * 0.1)
    # print('y_label',y_label)

    # plt.savefig(savename,bbox_inches = 'tight')


def network_test_plot(data_, name_list, savename, y_label, ax=None, color=None, color1=None):
    '''
    Perform statistical analysis and visualization of EEG network features across sleep stages.

    This function creates box plots with statistical annotations to compare network
    connectivity measures between different sleep stages.

    Parameters:
    -----------
    data_ : list of arrays
        List containing arrays of network values for each sleep stage
    name_list : list
        Names of the sleep stages being compared (e.g., ['N2','N3','REM'])
    savename : str
        Path to save the output figure (commented out in current implementation)
    y_label : str
        Label for the y-axis (network metric being visualized)
    ax : matplotlib axis, optional
        Axis object to plot on (creates new figure if None)
    color : dict, optional
        Primary color palette for box plot elements
    color1 : dict, optional
        Secondary color palette for plot elements

    The function:
    1. Organizes data into a pandas DataFrame for plotting
    2. Sets default color palettes if none provided
    3. Performs statistical comparisons between stages
    4. Creates annotated box plots showing network measure distributions
    5. Adjusts plot aesthetics and scaling
    '''

    # Prepare data in dictionary format for DataFrame creation
    dict_ = {'stages': [], 'Value': []}
    for stage in range(len(name_list)):
        for v in range(len(data_[0])):
            dict_['stages'].append(name_list[stage])
            dict_['Value'].append(data_[stage][v])

    # Create DataFrame from the organized data
    df = pd.DataFrame(dict_, columns=['stages', 'Value'])

    ### Set default color palettes if not provided ###
    # Primary colors for box plot elements
    if color is None:
        color = {0: '#B8DBB3',  # Wake (light green)
                 1: '#EAB883',  # N1 (light orange)
                 2: '#A8CBDF',  # N2 (light blue)
                 3: '#8074C8',  # N3 (purple)
                 4: '#F5EBAE',  # REM (light yellow)
                 }

    # Secondary colors for plot elements
    if color1 is None:
        color1 = {0: '#cfe6cb',  # Lighter green
                  1: '#f0cda8',  # Lighter orange
                  2: '#c7deea',  # Lighter blue
                  3: '#9b91d4',  # Lighter purple
                  4: '#f9f4d2'  # Lighter yellow
                  }

    # Create color palettes for the 3 stages being compared (N2, N3, REM)
    color_pal = sns.color_palette([color[3], color[2], color[4]])  # N3, N2, REM
    color_pal1 = sns.color_palette([color[3], color[2], color[4]])

    # Create new figure if no axis provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 2.5))

    # Perform statistical analysis between groups
    p = statistical_analysis(data_[0], data_[1], data_[2])

    # Convert p-values to asterisk annotations
    annotations = [convert_pvalue_to_asterisks(i) for i in p]

    # Define which group pairs to compare
    pairs = [('N2', 'N3'), ('N3', 'REM'), ('N2', 'REM')]

    # Create the annotated plot
    annotation_plot_(
        df,
        'stages',
        'Value',
        annotations,
        color_pal,
        color_pal1,
        name_list,
        ax,
        pairs,
        y_label,
        types='box',
        box_alpha=0.3  # Transparency for boxes
    )

    # Adjust y-axis limits with padding
    y_min = np.min(data_)
    y_max = np.max(data_)
    ax.set_ylim(y_min - y_min * 0.2,  # 20% padding below
                y_max + y_max * 0.1)  # 10% padding above

    # Note: Figure saving is currently commented out
    # plt.savefig(savename, bbox_inches='tight')



def con_graph_plot(con, node_color, title):
    '''Draw the graph G '''

    G = nx.from_numpy_array(con)
    MST = nx.maximum_spanning_tree(G, weight='weight')
    # adj = nx.to_numpy_array(MST)
    # adj[adj > 0] = 1-adj[adj > 0]
    # MST_new = nx.from_numpy_array(adj)
    plt.figure(figsize=(6, 6))
    nx.draw_networkx(MST, pos=nx.kamada_kawai_layout(MST), node_color=node_color)
    plt.title(title)
    plt.axis("off")


def inter_brain_distrubution(MST_adj):
    '''
    Calculate the distribution of Minimum Spanning Tree (MST) connections across brain hemispheres
    for different sleep stages.

    This function computes:
    1. Intra-hemispheric connections within the left hemisphere
    2. Intra-hemispheric connections within the right hemisphere
    3. Inter-hemispheric connections between left and right hemispheres

    Parameters:
    -----------
    MST_adj : numpy.ndarray
        Adjacency matrix representing the Minimum Spanning Tree connections
        between brain regions (labels)

    Returns:
    --------
    tuple: (inter_l, inter_r, cross_lr)
        inter_l: Number of connections within left hemisphere
        inter_r: Number of connections within right hemisphere
        cross_lr: Number of connections between hemispheres
    '''

    # Fetch the fsaverage brain template for source localization
    from mne.datasets import fetch_fsaverage
    import mne
    fs_dir = fetch_fsaverage(verbose=True)
    subjects_dir = op.dirname(fs_dir)

    # Load anatomical labels from the Desikan-Killiany atlas
    subject = "fsaverage"
    labels_names = mne.read_labels_from_annot(
        subject,
        parc='aparc',  # Use Desikan-Killiany parcellation
        subjects_dir=subjects_dir
    )

    # Filter out any 'unknown' labels
    label_names = [i for i in labels_names if 'unknown' not in i.name]

    # Initialize counters for different connection types
    inter_l = 0  # Left hemisphere intra-connections
    inter_r = 0  # Right hemisphere intra-connections
    cross_lr = 0  # Cross-hemisphere connections

    # Find indices of all connections in the MST (where adjacency > 0)
    idx = np.where(MST_adj > 0)

    # Classify each connection
    for i in range(len(idx[0])):
        hemi_i = idx[0][i]  # Source region index
        hemi_j = idx[1][i]  # Target region index

        # Check if both regions are in left hemisphere
        if (label_names[hemi_i].hemi == 'lh' and
            label_names[hemi_i].hemi == label_names[hemi_j].hemi):
            inter_l += 1

        # Check if both regions are in right hemisphere
        elif (label_names[hemi_i].hemi == 'rh' and
              label_names[hemi_i].hemi == label_names[hemi_j].hemi):
            inter_r += 1

        # Otherwise it's a cross-hemisphere connection
        else:
            cross_lr += 1

    return inter_l, inter_r, cross_lr


def network_logic_plot(data, y_label, savename, ax=None):
    """
    Plot network logic analysis with linear regression and confidence intervals.

    Args:
        data (array-like): Input data to be analyzed (should be convertible to 66 values)
        y_label (str): Label for the Y-axis
        savename (str): Filename to save the plot
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. If None, creates new figure.

    Returns:
        matplotlib.axes.Axes: The axes object with the plot
    """

    ### Prepare data paired with stages
    # Convert input to numpy array and reshape to 66 values (3 stages × 22 samples)
    data = np.array(data)
    x = np.arange(3)  # Create stage values [0, 1, 2]
    x = x.repeat(22)  # Repeat each stage 22 times [0,0,...,1,1,...,2,2,...]
    y = data.reshape(66)  # Flatten data to match x dimensions

    ### Perform linear regression
    X = sm.add_constant(x)  # Add constant term for intercept (statsmodels requirement)
    model = sm.OLS(y, X).fit()  # Ordinary Least Squares regression

    # Extract statistical values
    p = model.f_pvalue  # p-value for the model F-test
    t = model.tvalues[1]  # t-value for the slope coefficient
    r = model.rsquared  # R-squared value

    # Calculate confidence intervals
    # Returns: std_dev, lower_CI, upper_CI
    _, confidence_interval_lower, confidence_interval_upper = wls_prediction_std(model)

    # Get fitted values (regression line)
    y_fitted = model.fittedvalues

    ### Create plot if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 2.5))

    ### Plot regression line
    ax.plot(x, y_fitted, c='k', zorder=3, lw=1)  # Black line with high z-order

    # [Rest of your plotting code would go here...]
    # ax.scatter() for data points
    # ax.fill_between() for confidence intervals
    # ax.set() for labels/titles

    return ax





