import numpy as np
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['agg.path.chunksize'] = 10000
import mne
import os.path as op
import yasa
import matplotlib.pyplot as plt
from lspopt import spectrogram_lspopt
from mne.time_frequency import csd_morlet
from mne.beamformer import make_dics, apply_dics_csd
from mne.datasets import fetch_fsaverage
from mne.viz import circular_layout
from mne_connectivity import spectral_connectivity_epochs, spectral_connectivity_time
from mne_connectivity.viz import plot_connectivity_circle
import os
from nilearn import plotting
from mne.beamformer import apply_lcmv, make_lcmv
import seaborn as sns
from statannotations.Annotator import Annotator
import matplotlib
from test_statistics import convert_pvalue_to_asterisks

import pandas as pd

cm = 1 / 2.54

fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)
# The files live in:
subject = "fsaverage"
trans = "fsaverage"  # MNE has a built-in fsaverage transformation
src = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
bem = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_rgba(value):
    alpha = 1
    rgba_color = (value[0] / 255, value[1] / 255, value[2] / 255, alpha)
    return rgba_color


def _gen_dsmp(evoked, inv, method, lambda2):
    ####source estimate
    """
       Generate distributed source model solution from evoked data.

       Performs minimum norm estimation (MNE) source localization to reconstruct
       brain activity from sensor-level measurements.

       Args:
           evoked (mne.Evoked): Evoked response data (sensor-level)
           inv (mne.minimum_norm.InverseOperator): Inverse operator
           method (str): Inverse method ('dSPM', 'sLORETA', 'MNE')
           lambda2 (float): Regularization parameter

       Returns:
           mne.SourceEstimate: Source time course estimates

       Notes:
           - Uses pre-computed inverse operator for efficient computation
           - Supports multiple inverse methods for different statistical properties
           - Returned source estimate can be visualized or further analyzed
    """
    # 设置源空间并进行正向计算
    stc = mne.minimum_norm.apply_inverse(
        evoked,
        inv,
        lambda2,
        method=method,
        pick_ori=None,
        return_residual=False,
        verbose=True,
    )  ###evoked计算
    return stc


def brain_plot(stc, subjects_dir, message, mean=True, hemi="split", views=["lat", "med"],
               surface="pial", image='normal', lims=None, annotation=False,
               barin_ann="aparc_sub", colorbar=False):
    '''
    Visualize source-estimated activity on cortical surfaces.

    Creates publication-quality brain plots showing:
    - Source localization results
    - Optional anatomical parcellations
    - Customizable viewing angles and surfaces

    Args:
        stc (mne.SourceEstimate): Source time course data
        subjects_dir (str): Path to FreeSurfer subjects directory
        message (str): Text annotation to display on plot
        mean (bool): Whether to plot mean activation (default: True)
        hemi (str): Hemisphere selection ('split', 'lh', 'rh', 'both'; default: 'split')
        views (list/str): Viewing angles (['lat','med'], 'axial', etc.; default: ['lat','med'])
        surface (str): Surface type ('pial', 'white', 'inflated'; default: 'pial')
        image (str): Plot style ('normal' or other future options)
        lims (list): Color scale limits [min, mid, max] (default: auto-scaled)
        annotation (bool): Whether to show anatomical parcellation (default: False)
        barin_ann (str): Parcellation scheme ('aparc_sub', 'HCPMMP1', etc.; default: 'aparc_sub')
        colorbar (bool): Whether to show colorbar (default: False)

    Returns:
        mne.viz.Brain: Brain visualization object
    '''

    if mean:
        stc_mean = stc.mean()
    else:
        stc_mean = stc
    vertno_max, time_max = stc_mean.get_peak(hemi="lh")
    if lims == None:
        vmin = np.min(stc_mean.data)
        vmax = np.max(stc_mean.data)
        vmed = np.median(stc_mean.data)
        # lims = [vmed, format(vmed+(vmed-vmin)/2, '.2f'), format(vmax, '.2f')]
        lims = [vmed, vmed + (vmax - vmed) / 2, vmax]

    else:
        lims = lims
    if image == 'normal':
        surfer_kwargs = dict(
            hemi=hemi,
            subjects_dir=subjects_dir,
            surface=surface,
            clim=dict(kind="value", lims=lims),
            views=views,  ###["lat", "med"]对应“split”;"axial"对应"both"
            initial_time=time_max,
            time_unit="s",
            size=(800, 600),
            smoothing_steps=10,
            background='white',
            alpha=1,
            # colormap='viridis',
            colorbar=colorbar
        )
        brain = stc_mean.plot(**surfer_kwargs)
        # brain.add_foci(
        #     vertno_max,
        #     coords_as_verts=True,
        #     hemi="lh",
        #     color="blue",
        #     scale_factor=0.6,
        #     alpha=0.5,
        # )
        brain.add_text(
            0.1, 0.9, message, font_size=10, color='k'
        )
        if annotation == True:
            brain.add_annotation(barin_ann)  ###分脑区不同方法："HCPMMP1"，"aparc.a2009s","HCPMMP1_combined","aparc_sub"

    return brain


# # You can save a brain movie with:
# # brain.save_movie(time_dilation=20, tmin=0.05, tmax=0.16, framerate=10,
# #                  interpolation='linear', time_viewer=True)
# brain.save_image('./sleep_boruikang/mx_0512_1/figure/wake_both2.png')


def source_con(evoked_all, subject, subjects_dir, inv, band=[0.5, 4], method="dSPM",
               snr=3.0, parc='aparc', con_methods='pli'):
    ''' Perform source-space connectivity analysis between cortical parcels.

    Computes functional connectivity between brain regions using:
    1. Source localization of sensor-level data
    2. Parcellation of cortical surfaces
    3. Spectral connectivity metrics

    Args:
        evoked_all (mne.Evoked): Sensor-level evoked data (multiple epochs)
        subject (str): FreeSurfer subject ID
        subjects_dir (str): Path to FreeSurfer subjects directory
        inv (mne.minimum_norm.InverseOperator): Source inversion operator
        band (list): Frequency band of interest [fmin, fmax] in Hz (default: [0.5, 4])
        method (str): Inverse method ('dSPM', 'MNE', 'sLORETA'; default: 'dSPM')
        snr (float): Signal-to-noise ratio for inverse calculation (default: 3.0)
        parc (str): Parcellation scheme ('aparc' or 'aparc.a2009s'; default: 'aparc')
        con_methods (str): Connectivity method ('pli', 'wpli', 'coh'; default: 'pli')

    Returns:
        ndarray: Connectivity matrix [n_labels × n_labels] for specified frequency band'''
    src = inv['src']
    lambda2 = 1.0 / snr ** 2
    stcs = mne.minimum_norm.apply_inverse_epochs(evoked_all, inv, lambda2, method, return_generator=True)  ###epochs计算
    # Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
    labels = mne.read_labels_from_annot(subject, parc=parc, subjects_dir=subjects_dir)  ###
    labels = [i for i in labels if 'unknown' not in i.name]
    label_colors = [label.color for label in labels]
    # Average the source estimates within each label using sign-flips to reduce
    # signal cancellations, also here we return a generator
    label_ts = mne.extract_label_time_course(stcs, labels, src, mode='auto', return_generator=True)
    fmin = band[0]
    fmax = band[1]
    sfreq = evoked_all.info['sfreq']
    con = spectral_connectivity_epochs(
        label_ts, method=con_methods, mode='multitaper', sfreq=sfreq, fmin=fmin,
        fmax=fmax, faverage=True, mt_adaptive=True, n_jobs=1)
    data = con.get_data(output='dense')[:, :, 0]
    return data


def source_con_plot(con, parc='aparc'):
    labels = mne.read_labels_from_annot(subject, parc=parc, subjects_dir=subjects_dir)  ###
    labels = [i for i in labels if 'unknown' not in i.name]
    label_colors = [label.color for label in labels]
    # con is a 3D array, get the connectivity for the first (and only) freq. band
    # for each method
    # con_res = dict()
    # for method, c in zip(con_methods, con):
    #     con_res[method] = c.get_data(output='dense')[:, :, 0]
    con_res = con.get_data()[:, :, 0]
    # First, we reorder the labels based on their location in the left hemi
    label_names = [label.name for label in labels]

    lh_labels = [name for name in label_names if name.endswith('lh')]
    # Get the y-location of the label
    label_ypos = list()
    for name in lh_labels:
        idx = label_names.index(name)
        ypos = np.mean(labels[idx].pos[:, 1])
        label_ypos.append(ypos)
    # Reorder the labels based on their location
    lh_labels = [label for (yp, label) in sorted(zip(label_ypos, lh_labels))]
    # For the right hemi
    rh_labels = [label[:-2] + 'rh' for label in lh_labels]
    # Save the plot order and create a circular layout
    node_order = list()
    node_order.extend(lh_labels[::-1])  # reverse the order
    node_order.extend(rh_labels)

    node_angles = circular_layout(label_names, node_order, start_pos=90,
                                  group_boundaries=[0, len(label_names) / 2])
    # Plot the graph using node colors from the FreeSurfer parcellation. We only
    # show the 300 strongest connections.
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='black',
                           subplot_kw=dict(polar=True))
    plot_connectivity_circle(con_res, label_names, n_lines=300,
                             node_angles=node_angles, node_colors=label_colors,
                             title='Wake '
                                   'Condition (PLI)', ax=ax, fontsize_names=6)
    fig.tight_layout()



def con_fuction(raw_eeg, band=(0.5, 1.25), method='pli', stage_name='N2'):
    fmin, fmax = band[0], band[1]
    sfreq = raw_eeg.info['sfreq']  # the sampling frequency
    tmin = 0.0  # exclude the baseline period
    if stage_name == None:
        epochs = raw_eeg
    else:
        epochs = raw_eeg[stage_name]

    freqs = np.linspace(band[0], band[1], 4)
    if method == 'pli' or method == 'plv':
        con = spectral_connectivity_time(
            epochs, method=method, mode='multitaper', freqs=freqs, fmin=fmin, fmax=fmax,
            faverage=True, sm_times=tmin, n_jobs=1)
        con_data = con.get_data(output='dense')[:, :, :, 0]
        con_data = np.mean(con_data, axis=0)
    else:
        con = spectral_connectivity_epochs(
            epochs, method=method, mode='multitaper', sfreq=sfreq, fmin=fmin, fmax=fmax,
            faverage=True, tmin=tmin, mt_adaptive=False, n_jobs=1)
        con_data = con.get_data(output='dense')[:, :, 0]

    return con_data


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径


def connection_matrix_plot(con, ch_names, title, savename, vmin, vmax):
    '''EEG connectivity plot'''
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 15))
    plotting.plot_matrix(
        con, labels=ch_names, colorbar=True, axes=ax, vmax=vmax, vmin=vmin, cmap="YlGn",
    )
    plt.savefig(savename + title + '.png', dpi=600, bbox_inches='tight')
    plt.close()


def connection_brain_plot(con, title, savename, vmax, thre=90, edge_threshold='90%'):
    '''EEG connectivity plot'''

    filename = './ANT_data/loc.xlsx'
    sheet_name = 'sheet1'
    df = pd.read_excel(filename, sheet_name=sheet_name)
    ch_names = df['name'].values
    loc = df[['x = r sinθ cosφ', 'y = r sinθ sinφ', 'z = r cosθ']].values * 800
    color = df['color'].values
    cmap = mpl.colors.ListedColormap(color)
    color_dict = {}
    for rsn, c in zip(ch_names, cmap.colors.tolist()):
        color_dict[rsn] = rgb_rgba(hex_to_rgb(c))
    node_color = []
    for nw in ch_names:
        node_color.append(color_dict[nw])
    node_size = 8
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    plotting.plot_connectome(con, loc, figure=fig, axes=ax, edge_threshold=edge_threshold, node_color=node_color,
                             node_size=node_size,
                             display_mode="z", edge_vmin=np.percentile(con, thre), edge_vmax=vmax, colorbar=True,
                             edge_cmap="YlGn")
    fig.axes[-1].set_yticks([np.percentile(con, thre), vmax],
                            ['{:.3f}'.format(tick) for tick in [np.percentile(con, thre), vmax]])
    # plt.show()
    plt.savefig(savename + title + '.svg', dpi=600, bbox_inches='tight')
    plt.close()


def connection_circle_plot(con, ch_names, savename, n_lines=100, vmin=None, vmax=None):
    '''EEG connectivity plot'''
    ####定义plot_connectivity_circle节点角度
    ch_names_change = ['Fpz', 'Fp1', 'AF3', 'AF7', 'F1', 'F3', 'F5', 'F7', 'FCz', 'FC1', 'FC3', 'FC5',
                       'FT7', 'C1', 'C3', 'C5', 'CP1', 'CP3', 'CP5', 'T7', 'TP7', 'M1', 'Pz', 'P1', 'P3', 'P5',
                       'P7', 'PO3', 'PO5', 'PO7', 'O1', 'Oz', 'O2', 'PO8', 'PO6', 'PO4', 'POz', 'P8', 'P6', 'P4',
                       'P2', 'TP8', 'T8', 'M2', 'CP6', 'CP4', 'CP2', 'C6', 'C4', 'C2', 'Cz', 'FT8', 'FC6', 'FC4', 'FC2',
                       'AF8', 'AF4', 'F8', 'F6', 'F4', 'F2', 'Fz', 'Fp2']
    node_angles = circular_layout(node_names=ch_names, node_order=ch_names_change, start_pos=90,
                                  group_sep=5, group_boundaries=[0, 13, 19, 22, 31, 41, 44, 51])
    color = ['#FAF0D7', '#FFD9C0', '#8CC0DE', '#CCEEBC', '#B5C99A', '#862B0D', '#FFF9C9', '#FFC95F']
    node_color_change = [color[7]] * 13 + [color[1]] * 6 + [color[2]] * 3 + [color[4]] * 9 + [color[4]] * 10 + [
        color[2]] * 3 + [color[1]] * 7 + [color[7]] * 12
    node_colors = []
    for i in range(len(ch_names)):
        index = np.where(np.array(ch_names_change) == ch_names[i])[0]
        node_colors.append(node_color_change[index[0]])

    ####plot subject's circle connectivity,n_lines=100
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    # plot_sensors_connectivity(raw_eeg.info,pli)
    if vmin:
        plot_connectivity_circle(con, node_names=ch_names, fig=fig, ax=ax, n_lines=n_lines, facecolor='white',
                                 textcolor='black', padding=0.5, vmin=vmin, vmax=vmax,
                                 colorbar_pos=(0.75, 0.5), colormap="YlGn",
                                 colorbar_size=0.8, node_angles=node_angles, node_colors=node_colors,
                                 node_edgecolor='White', show=False)
    else:
        plot_connectivity_circle(con, node_names=ch_names, fig=fig, ax=ax, n_lines=n_lines, facecolor='white',
                                 textcolor='black', padding=0.5,
                                 colorbar_pos=(0.75, 0.5), colormap="YlGn",
                                 colorbar_size=0.8, node_angles=node_angles, node_colors=node_colors,
                                 node_edgecolor='White', show=False)

    # # plt.show()
    plt.savefig(savename, dpi=600, bbox_inches='tight')
    plt.close()


def annotation_plot_(data, data_x, data_y, annotations, color_pal, color_pal1,
                     name_list, ax, pairs, y_label, cut=2, types='violin', loc='outside', box_alpha=0.1):
    '''data:三组统计数据，dict
       data_x,data_y:'stage','Value'
       name_list:三组数据名称,list
       color:小提琴图边缘颜色，list or dict
       color1:小提琴图点的颜色，list or dict
       pairs:统计分析对，例[('N2', 'N3'),('N2', 'REM'),('REM', 'N3')]
       annotations = [convert_pvalue_to_asterisks(p)]
       types:'violin' or 'box'
    '''

    ##plot N3
    # fig = plt.figure(figsize=(3.5, 2.5))
    if types == 'violin':
        sns.swarmplot(x=data_x, y=data_y, data=data, size=3, palette=color_pal1, ax=ax, alpha=0.8)
        # sns.stripplot(x=data_x, y=data_y, data=data, size=3, palette=color_pal1, ax=ax, alpha=0.8)
        # for dots in ax[0].collections:
        #     facecolors = dots.get_facecolors()
        #     dots.set_edgecolors(facecolors.copy())
        # dots.set_facecolors('none')
        # dots.set_linewidth(1)

        sns.violinplot(data=data, x=data_x, y=data_y, palette=color_pal, cut=cut, ax=ax, inner='box', scale='width')
        # ax = sns.boxplot(data=dict, x='stage', y='Value', palette=color_pal,width=0.3,**PROPS)
        colors = []
        # for i, box in enumerate(ax.artists):
        #     box.set_edgecolor(colors[-1])
        #     box.set_facecolor('none')
        for collection in ax.collections:
            if isinstance(collection, matplotlib.collections.PolyCollection):
                colors.append(collection.get_facecolor())
                collection.set_edgecolor(colors[-1])
                collection.set_facecolor('none')
        if len(ax.lines) == 2 * len(colors):  # suppose inner=='box'
            for lin1, lin2, c in zip(ax.lines[::2], ax.lines[1::2], colors):
                lin1.set_color(c)
                lin2.set_color(c)


    elif types == 'box':
        sns.stripplot(x=data_x, y=data_y, data=data, size=3, palette=color_pal1, ax=ax, alpha=0.8)
        # sns.stripplot(x=data_x, y=data_y, data=data, size=3, color='brown',ax=ax, alpha=0.8)
        # sns.barplot(data=df, x ='stages',y ='Value', ax=ax,linewidth=2,edgecolor=color_pal,facecolor=(1, 1, 1, 0))

        sns.boxplot(data=data, x=data_x, y=data_y, ax=ax, width=0.5, linewidth=1, fliersize=0, palette=color_pal)
        # sns.boxplot(data=data, x=data_x, y=data_y, ax=ax, width=0.5, linewidth=1, fliersize=0,color='k')
        box_patches = [patch for patch in ax.patches if type(patch) == matplotlib.patches.PathPatch]
        if len(box_patches) == 0:  # in matplotlib older than 3.5, the boxes are stored in ax2.artists
            box_patches = ax.artists
        num_patches = len(box_patches)
        lines_per_boxplot = len(ax.lines) // num_patches
        for i, patch in enumerate(box_patches):
            # Set the linecolor on the patch to the facecolor, and set the facecolor to None
            col = patch.get_facecolor()
            patch.set_edgecolor(col)
            patch.set_alpha(box_alpha)
            patch.set_facecolor('None')
            # Each box has associated Line2D objects (to make the whiskers, fliers, etc.)
            # Loop over them here, and use the same color as above
            for line in ax.lines[i * lines_per_boxplot: (i + 1) * lines_per_boxplot]:
                line.set_color(col)
                line.set_mfc(col)  # facecolor of fliers
                line.set_mec(col)  # edgecolor of fliers
                line.set_alpha(box_alpha)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel(y_label)
    ax.set_xlabel('')
    order = name_list
    annotator = Annotator(ax, data=data, x=data_x, y=data_y, pairs=pairs, order=order)
    # start = {'***':0.001, '**': 0.01,'*':0.05, 'ns':1}
    # annotator.configure(test='Mann-Whitney', hide_non_significant=True, loc='outside')
    if annotations == None:
        annotator.configure(
            test='t-test_ind', pvalue_format={'text_format': 'star'},
            comparisons_correction="Bonferroni",
            line_offset=0, line_height=0, line_width=0.8,
            hide_non_significant=False, loc=loc
        )
        annotator.apply_and_annotate()
    else:
        annotator.configure(
            # test='t-test', pvalue_format={'text_format': 'star'},
            # comparisons_correction="Bonferroni",
            line_offset=0, line_height=0, line_width=0.8,
            hide_non_significant=False, loc=loc
        )

        print(annotations)
        annotator.annotate_custom_annotations(annotations)
    #### PValueFormat调整start,在 annotator的configure模块中搜索ymax_in_range_x1_x2 + offset，可以调整间距
    # annotator.annotate_custom_annotations([str(p) for p in ['***']])
    # annotator.apply_and_annotate
    # annotator.annotate()
    # ax, test_results = annotator.annotate()
    # ax.tick_params(labelsize=12)
    # plt.show()


def stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."


def box_plot(data, labels, ax, ylable=None, title=None, color=None):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if color == None:
        color = ['#e94958',  # Wake
                 '#7dc474',  # N1
                 '#40b7ad',  # N2
                 '#413d7b',  # N3
                 '#f8ad3c'  # REM
                 ]
    bpdict = ax.boxplot(data, labels=labels, showmeans=True, patch_artist=color,
                        flierprops={'marker': 'o', 'markerfacecolor': 'white', 'color': 'black'},
                        # 设置异常值属性，点的形状、填充色和边框色
                        meanprops={'marker': '*', 'color': 'black'},  # 设置均值点的属性，点的形状、填充色
                        medianprops={'linestyle': '-', 'color': 'black'})  # 设置中位数线的属性，线的类型和颜色)

    for i in range(len(data)):
        y = data[i]
        x = np.random.normal(i + 1, 0.04, size=len(y))

        ax.scatter(x, y, s=0.8, c=color[i], alpha=0.5)
    if ylable:
        ax.set_ylabel(ylable, size=14)
    if title:
        ax.set_title(title, size=16)

def brain_map_plot(data, savename, title, label=None, vmin=None, vmax=None, ch_names=None, mask=None, cmap=None,
                   cbarlabels=False):
    filename = "./data/sub_dwj_2023-11-08_13-07-58.fif"
    raw = mne.read_epochs(filename)
    if ch_names == None:
        ch_names = ['Fp1', 'Fpz', 'Fp2', 'F1', 'F3', 'F5', 'F7', 'Fz', 'F2', 'F4', 'F6', 'F8', 'AF3', 'AF7',
                    'AF4', 'AF8', 'FC1', 'FC3', 'FC5', 'FCz', 'FC2', 'FC4', 'FC6', 'FT7', 'FT8', 'C1', 'C3',
                    'C5', 'Cz', 'C2', 'C4', 'C6', 'CP1', 'CP3', 'CP5', 'CP2', 'CP4', 'CP6', 'T7', 'T8',
                    'TP7', 'TP8', 'P1', 'P3', 'P5', 'P7', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO3', 'PO5', 'PO7',
                    'POz', 'PO4', 'PO6', 'PO8', 'O1', 'Oz', 'O2']
    raw.pick(ch_names)
    # 创建 Montage 对象
    montage = mne.channels.make_standard_montage('standard_1005')
    # 调整 Montage 的位置和放缩大小
    raw.set_montage(montage)  ####读取info
    montage = raw.get_montage()
    for i in range(len(montage.dig)):
        montage.dig[i]['r'] = montage.dig[i]['r'] + np.array((0, -0.026, 0))
        montage.dig[i]['r'][0] = montage.dig[i]['r'][0] * 1.2
    raw.set_montage(montage)
    if vmin == None:
        vmin = np.min(data)
    if vmax == None:
        vmax = np.max(data)
    mask_params = dict(marker='o', markerfacecolor=None, markeredgecolor='k',
                       linewidth=0, markersize=4)
    fig, ax1 = plt.subplots(figsize=(6, 4), ncols=1)

    im, cm = mne.viz.plot_topomap(np.array(data[0:len(data) - 2]), raw.info, vlim=(vmin, vmax), sensors=True, axes=ax1,
                                  show=False,
                                  mask=mask, mask_params=mask_params, cmap=cmap)

    ax_x_start = 0.85
    ax_x_width = 0.04
    ax_y_start = 0.01
    ax_y_height = 0.9
    cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    ####设置y轴参数
    if cbarlabels == True:
        cbar = plt.colorbar(im, cax=cbar_ax, ticks=[vmin, 0, vmax])
        cbar.ax.set_yticklabels(['-' + r'${\pi}$' + '/2', '0', r'${\pi}$' + '/2'])
    else:
        cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=12)
    ax1.set_title(title, size=24)
    plt.title(label, size=14)
    plt.savefig(savename, dpi=500, bbox_inches='tight')
    plt.close()