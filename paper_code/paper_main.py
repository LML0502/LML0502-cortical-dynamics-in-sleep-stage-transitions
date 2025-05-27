'''
@author: Manli Luo
sleep stage transition main code
'''
import plot_figure
import glob
import mne
import source_function
import so_spindle_counts
import os
import network
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from so_spindle_counts import N2_3_R_num, Signal
import matplotlib as mpl
mpl.use('TkAgg')
mpl.rcParams['agg.path.chunksize'] = 10000

if __name__ == '__main__':
    cm = 1 / 2.54  # figsize*cm centimeters in inches
    # ========== FIGURE 1a: SLEEP RAW EEG DATA ==========
    # Plot raw data from multiple channels
    fig, ax = plt.subplots(1, 2, figsize=(20 * cm, 6 * cm), width_ratios=[2, 7])

    # Plot brain EEG topography on left subplot
    plot_figure.brain_eeg_plot(fig=fig, ax=ax[0])

    # Define custom color palette for raw data plot
    color1 = ['#934B43', '#5f97d2', '#BB9727', '#C497B2', '#9dc3e7', '#929491']

    # Plot raw EEG data on right subplot with custom colors
    file_name = "/mnt/D/sleep transition/data/sub_lxx_2023-12-07_13-06-35.set"
    plot_figure.raw_plot(file_name,color=color1, ax=ax[1])  # Plot raw data

    # Adjust subplot spacing (instead of tight_layout)
    plt.subplots_adjust(wspace=0.6)  # Horizontal space between subplots
    # Optional: Save figure as png
    # plt.savefig('../figures/fig1_a.png')

    # ========== FIGURE 1B: SLEEP STAGE HYPNOGRAM WITH RAW SIGNALS ==========
    # Define custom color palette for sleep stages
    color = ['#e94958',  # Wake
             '#7dc474',  # N1
             '#40b7ad',  # N2
             '#413d7b',  # N3
             '#f8ad3c'  # REM
             ]

    # Generate composite plot with hypnogram, spectrogram and raw signals
    filename_f = "/mnt/D/sleep transition/data/sub_tt_2023-11-09_13-01-50.set"
    filename_y = '/mnt/D/sleep transition/data/ica_so_pkl/'
    plot_figure.raw_eog_stage_plot(filename_f,filename_y,color=color)
    # Save figure
    figsave = "../figures/fig_1_b.png"
    # plt.savefig(figsave, dpi=500, bbox_inches='tight')

    # ========== FIGURE 1C: CHANNEL ACTIVITY BY SLEEP STAGE ==========
    # Load preprocessed EEG epochs data
    file_name = '/mnt/D/sleep transition/data/epoch_eeg_ica/sub_dwj_2023-11-08_13-07-58.fif'
    epochs = mne.read_epochs(file_name, preload=True)
    epochs.filter(0.5, 45)  # Bandpass filter (0.5-45 Hz)

    # Load corresponding raw EEG data for reference
    file_name = "/mnt/D/sleep transition/data/sub_dwj_2023-11-08_13-07-58.set"
    raw = mne.io.read_raw_eeglab(file_name, preload=True)
    raw.filter(0.5, 45)  # Apply same filtering

    # Create 5-panel figure (one per sleep stage)
    fig, ax = plt.subplots(1, 5, figsize=(15 * cm, 4 * cm))

    # Plot representative channel activity for each stage:
    plot_figure.stages_channels_plot(epochs, raw, 'wake', 5, ax[0],
                                     color=color[0], label='Wake', start=1)
    plot_figure.stages_channels_plot(epochs, raw, 'N1', 1, ax[1],
                                     color=color[1], label='N1', start=1)
    plot_figure.stages_channels_plot(epochs, raw, 'N2', 26, ax[2],
                                     color=color[2], label='N2', start=0.5)
    plot_figure.stages_channels_plot(epochs, raw, 'N3', 36, ax[3],
                                     color=color[3], label='N3', start=0)
    plot_figure.stages_channels_plot(epochs, raw, 'REM', 16, ax[4],
                                     color=color[4], label='REM', start=0)

    plt.tight_layout()
    # plt.savefig('../figures/fig1_c.png')

    # ========== FIGURE 1D: SLEEP STAGE DURATION PERCENTAGE ==========
    # Calculate and plot the time spent in each stage
    # File path pattern for EEG data files in FIF format
    file_name = '/mnt/D/sleep transition/data/epoch_eeg_ica/*.fif'
    plot_figure.sleep_time_state(file_name,color)
    plt.savefig('../figures/fig1_d.png')

    # ========== FIGURE 1E: SLEEP STAGE TRANSITION MATRIX ==========
    # Initialize transition matrices
    transition_matrix_y = np.zeros((5, 5))  # Count matrix
    transition_matrix_label = np.zeros((5, 5))  # Not used?

    # Process all subjects' data to build transition statistics
    file = '/mnt/D/sleep transition/data/ica_so_pkl/*.pkl'
    for filename in glob.glob(file):
        print(f"Processing {filename}")
        self = Signal(load_pickle=True, filename=filename)

        # Count transitions between stages
        for i in range(len(self.y) - 1):
            if self.y[i] != self.y[i + 1]:  # Only count actual transitions
                transition_matrix_y[self.y[i], self.y[i + 1]] += 1

    # Convert counts to probabilities
    transition_probabilities_y = transition_matrix_y / np.sum(transition_matrix_y,
                                                              axis=1,
                                                              keepdims=True)
    # Visualize the transition matrix
    plot_figure.transition_matrix_plot(transition_probabilities_y)
    # plt.savefig(f'../figures/fig1_e.png', bbox_inches='tight', dpi=600)


    # ========== FIGURE 2A-C: SOURCE ESTIMATION OF SLEEP STAGES ==========
    # Main execution block for sleep stage source analysis and visualization

    # Define paths for input data and output results
    data_path = '/mnt/D/sleep transition/data/epoch_eeg_ica/*.fif'  # Path pattern to find preprocessed EEG files in FIF format
    save_path = '/mnt/D/sleep transition/data/stc_dsmp/'  # Directory to save source localization results

    # 1. Perform source estimation for each sleep stage
    # This function will:
    # - Process each subject's EEG data
    # - Compute source estimates using dSPM method
    # - Save source time courses (STCs) for each sleep stage
    # plot_figure.source_stage(data_path, save_path)


    # 2. Generate group-level visualizations of source activity
    # This function will:
    # - Load all individual STC files
    # - Compute average source activation per sleep stage
    # - Create and save brain plots showing activation patterns
    plot_figure.source_stage_plot(save_path)

    # Note: The analysis pipeline consists of two main steps:
    # 1. Source estimation (individual level)
    # 2. Visualization and group averaging
    # Results will be saved in the specified directory structure:
    # ./data/stc_dsmp/
    #   ├── [stage_name]/          # Individual subject STCs per stage
    #   ├── [stage_name]/mean/     # Mean STCs across subjects
    #   └── figures/               # Visualization images

    # ========== FIGURE 2D-F: THE CORRELATION COEFFICIENTS OF STAGES' SOURCE ==========
    fig, ax = plt.subplots(3, 1, figsize=(5 * cm, 15 * cm))
    color_pal = [color[2], color[3], color[4]]
    color1_pal = [color[2], color[3], color[4]]
    plot_figure.source_coffe_via_stage(save_path, ax, color_pal, color1_pal)
    # plt.subplots_adjust(hspace =0.6)
    plt.tight_layout()
    plt.savefig('../figures/fig2_d-f.png',bbox_inches = 'tight')

    # ========== FIGURE 3: THE CORRELATION COEFFICIENTS OF STAGES' SOURCE ==========
    fig = plt.figure(figsize=(12*cm,8*cm))
    heights = [3, 2]
    width = [3, 3]
    gs = fig.add_gridspec(2, 1,wspace=0.5, hspace=0.28,height_ratios=heights)
    gs0 = gs[0].subgridspec(1, 2,wspace=0.5)
    gs1 = gs[1].subgridspec(1, 2,wspace=0.5)
    # gs2 = gs[2].subgridspec(1, 2,wspace=0.5, hspace=0.4)
    ax0 = [0,0]
    ax1 = [0,0]
    # ax2 = [0,0]
    for i in range(2):
        ax0[i] = fig.add_subplot(gs0[0, i])
    for i in range(2):
        ax1[i] = fig.add_subplot(gs1[0, i])
    # for i in range(2):
    #     ax2[i] = fig.add_subplot(gs2[0, i])

    plot_figure.transition_time(data_path,ax0,color)
    plot_figure.source_coffe_via_time(ax1,start=-5,end=5)
    fig.tight_layout()
    plt.savefig('../figures/fig3.png',dpi=500,bbox_inches = 'tight')


    # ========== FIGURE 4: THE DYNAMICS IN SLEEP TRANSITIONS ==========
    plot_figure.N2_stage_source(data_path,save_path)


    # ========== FIGURE 5: THE CORRELATION COEFFICIENTS OF STAGES' SOURCE ==========
    file = '/mnt/D/sleep transition/data/ica_so_pkl/*.pkl'
    # count_sub_n_sp, count_sub_r_sp, count_sub_r_win_sp, count_sub_n_win_sp = N2_3_R_num(type='sp', win=20)
    count_sub_n_so, count_sub_r_so, count_sub_r_win_so, count_sub_n_win_so = N2_3_R_num(type='so', win=20)
    # count_sub_n_co, count_sub_r_co, count_sub_r_win_co, count_sub_n_win_co = N2_3_R_num(type='co', win=20)
    fig, ax1 = plt.subplots(figsize=(4.5, 4))
    so_spindle_counts.density_win_plot(count_sub_r_win_so, count_sub_n_win_so, ax1, ybottom=-0.1, ytop=8, type='sp')
    plt.savefig('../figures/fig5_c.png')
    fig, ax2 = plt.subplots(figsize=(4.5, 4))
    so_spindle_counts.density_test_plot(count_sub_r_so, count_sub_n_so, ax2, type='sp', color=color1)
    plt.savefig('../figures/fig5_d.png')


    # ========== FIGURE 7: THE BRAIN NETWORK CONNECTIVITY ==========
    ### source connectivity plot
    base_path = '/mnt/D/sleep transition/data/stc_dsmp_imcoh'
    file = './data/stc_dsmp_imcoh/network_value'
    bands = ['[0.5, 4]']
    ylabels = ['Diameter', 'Eccentricity', 'Leaf number', 'Tree hierarchy']
    name_list = ['N3', 'N2', 'REM']
    # feature_calculation(base_path,file,bands)##计算network
    # fig,ax = plt.subplots(4,2,figsize=(11*cm,20*cm),width_ratios=[1.5,1])
    fig, ax = plt.subplots(2, 2, figsize=(10 * cm, 10 * cm))
    for band in bands:
        path = op.join(file, band)
        data = []
        for j in os.listdir(path):
            values = np.load(op.join(path, j))
            data.append(values)
        for i in range(4):
            data_ = [data[1][:, i], data[0][:, i], data[2][:, i]]
            base_name = '/mnt/D/sleep transition/data/stc_dsmp_imcoh/network_value/figure_n2_n3_rem/'
            source_function.mkdir(base_name)
            figname = base_name + band + ylabels[i] + '.png'
            if i == 0:
                ax_sub = ax[0, 0]
            if i == 1:
                ax_sub = ax[0, 1]
            if i == 2:
                ax_sub = ax[1, 0]
            if i == 3:
                ax_sub = ax[1, 1]
            network.network_test_plot(data_,name_list, figname, ylabels[i], ax=ax_sub, color=color, color1=color1)  # plot 小提琴图
            base_name = '/mnt/D/sleep transition/data/stc_dsmp_imcoh/network_value/figure_n2_n3_rem_logic/'
            source_function.mkdir(base_name)
            savename = base_name + band + ylabels[i] + '.png'
            network.network_logic_plot(data_, ylabels[i], savename, ax=ax_sub)  # plot 回归结果
    fig.tight_layout()
    plt.savefig('../figures/fig7.png')