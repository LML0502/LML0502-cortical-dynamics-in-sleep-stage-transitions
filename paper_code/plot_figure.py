'''
@author: Manli Luo
sleep stage transition main figure plot
'''
import glob
import mne
import so_spindle_counts
import matplotlib
import source_function
import scipy
import seaborn as sns
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from source_function import rgb_rgba,hex_to_rgb
from test_statistics import statistical_analysis,convert_pvalue_to_asterisks
from nilearn import plotting
from lspopt import spectrogram_lspopt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mne.datasets import fetch_fsaverage
from scipy import stats

cm = 1 / 2.54  # figsize*cm centimeters in inches

def brain_eeg_plot(fig=None, ax=None):
    if fig == None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    filename = './loc.xlsx'
    sheet_name = 'sheet2'
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
    con = np.zeros((65, 65))
    node_size = 8
    plotting.plot_connectome(con, loc, figure=fig, axes=ax, node_color=node_color,
                             node_size=node_size, annotate=False,
                             display_mode="z")
    ax.text(x=0.1, y=0.9, s='L', fontsize=10)
    ax.text(x=0.95, y=0.9, s='R', fontsize=10)


def raw_plot(file_name,color=None, ax=None):
    '''plot raw data'''
    raw_filtre = mne.io.read_raw_eeglab(file_name, preload=True)
    raw_filtre.filter(0.5, 45)
    win = 20 * 500
    data_EOG = np.squeeze(raw_filtre.get_data(picks='EOG') * 1000000)
    eeg_channels = ['Fp1', 'Fpz', 'Fp2', 'F1', 'F3', 'F5', 'F7', 'Fz', 'F2', 'F4', 'F6', 'F8', 'AF3', 'AF7',
                    'AF4', 'AF8', 'FC1', 'FC3', 'FC5', 'FCz', 'FC2', 'FC4', 'FC6', 'FT7', 'FT8', 'C1', 'C3',
                    'C5', 'Cz', 'C2', 'C4', 'C6', 'CP1', 'CP3', 'CP5', 'CP2', 'CP4', 'CP6', 'T7', 'T8',
                    'TP7', 'TP8', 'P1', 'P3', 'P5', 'P7', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO3', 'PO5', 'PO7',
                    'POz', 'PO4', 'PO6', 'PO8', 'O1', 'Oz', 'O2', 'M1', 'M2']
    data = (raw_filtre.get_data(picks=eeg_channels) * 1000000)[:, win * 100:101 * win]
    t = np.linspace(0, 20, len(data[0, :]))
    if color == None:
        color = ['#f66733', '#522d80', '#d4c99e', '#685c53', '#a25016', '#562e19', '#566127']
    if ax == None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9 * cm, 7 * cm))
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False), ax.spines['left'].set_visible(False)
    ax.tick_params(left=False)
    ax.set_xticks([])
    ax.margins(x=0)
    ax.plot(t, data[1, :] + 650, c=color[0], linewidth=0.6)
    ax.plot(t, data[4, :] + 550, c=color[0], linewidth=0.6)
    ax.plot(t, data[28, :] + 450, c=color[1], linewidth=0.6)
    ax.plot(t, data[32, :] + 350, c=color[1], linewidth=0.6)
    ax.plot(t, data[38, :] + 250, c=color[2], linewidth=0.6)
    ax.plot(t, data[46, :] + 150, c=color[3], linewidth=0.6)
    ax.plot(t, data[59, :] + 50, c=color[3], linewidth=0.6)
    ax.plot(t, data[60, :] - 50, c=color[4], linewidth=0.6)
    ax.plot(t, data_EOG[win * 100:101 * win] - 150, c=color[5], linewidth=0.6)
    ax.plot([19.2, 20.2], [-200, -200], c='k', linewidth=1)
    ax.text(x=19.2, y=-280, s='1 s', fontsize=8)
    ax.plot([20.2, 20.2], [-200, -150], c='k', linewidth=1)
    ax.text(x=20.5, y=-200, s='50 uV', fontsize=8, rotation='vertical')
    ax.set_yticks([-150, -50, 50, 150, 250, 350, 450, 550, 650],
                  [r'EOG', r'M1', r'Oz', r'Pz', r'T7', r'CP1', r'Cz', r'F3', r'Fpz'], fontsize=10)




def raw_eog_stage_plot(filename_f,filename_y,color=None):
    """Plot sleep stage hypnogram, F3 spectrogram, and EOG/EEG raw signals."""

    # Process each matching file (though only one path is provided)
        # Load raw EEG data
    raw_filtre = mne.io.read_raw_eeglab(filename_f, preload=True)
    s = op.basename(filename_f)

    # Load precomputed sleep stages from pickle
    filename = filename_y + s[:-4] + '.pkl'
    self = so_spindle_counts.Signal(load_pickle=True, filename=filename)
    y = np.array(self.y).T  # Sleep stage labels

    # Extract and scale channel data
    data_F3 = np.squeeze(raw_filtre.get_data(picks='F3') * 1000000)  # F3 EEG (μV)
    data_EOG = np.squeeze(raw_filtre.get_data(picks='EOG') * 1000000)  # EOG (μV)

    # Compute spectrogram for F3 channel
    f, t2, Sxx = spectrogram_lspopt(data_F3, fs=500, nperseg=500 * 30, noverlap=500 * 15)
    good_freqs = np.logical_and(f >= 0.3, f <= 40)  # Filter frequency range
    Sx = Sxx[good_freqs, :]
    f = f[good_freqs]
    t2 = t2 / 3600  # Convert seconds to hours
    Sx = np.log(Sx)  # Log-transform power

    # Create figure with custom size (16cm width, 6cm height)
    fig = plt.figure(figsize=(16 * cm, 6 * cm))

    # Hypnogram color settings
    stage = ['Wake', 'N1', 'N2', 'N3', 'REM']
    height = {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5}
    if color == None:
        color = {0: '#B8DBB3',  # Wake (green)
                 1: '#EAB883',  # N1 (orange)
                 2: '#A8CBDF',  # N2 (blue)
                 3: '#8074C8',  # N3 (purple)
                 4: '#F5EBAE',  # REM (yellow)
                 }

    # ========== Plot 1: Hypnogram ==========
    ax1 = plt.axes([0.1, 0.8, 0.78, 0.05])  # [left, bottom, width, height]
    t = np.linspace(0, len(y) * 30 / 3600, len(y))
    # Draw colored bars for each sleep stage
    for i in range(5):
        ax1.fill_between(x=t, y1=height[i], y2=0, where=y == i,
                         interpolate=False, step='post',
                         edgecolor=color[i], fc=color[i], linewidth=1.25)
        ax1.fill_between(x=t, y1=height[i], y2=0, where=y == i,
                     interpolate=False, step='pre',
                     edgecolor=color[i], fc=color[i], linewidth=1.25)

    # Create and position legend
    import matplotlib.patches as mpatches
    legends = [mpatches.Patch(color=color[i], label=stage[i]) for i in range(5)]
    legend = ax1.legend(handles=legends, framealpha=0,
                        bbox_to_anchor=(0.5, 2), loc='center', ncol=5,
                        bbox_transform=ax1.transAxes,
                        handlelength=0.8, markerscale=1,
                        handletextpad=0.3, columnspacing=0.99)

    # Format hypnogram axis
    ax1.set_xlim(0, t.max())
    ax1.set_ylim(0, 0.5)
    for spine in ['top', 'right', 'bottom', 'left']:
        ax1.spines[spine].set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)

    # ========== Plot 2: Spectrogram ==========
    ax2 = plt.axes([0.1, 0.4, 0.8, 0.35])

    # Plot spectrogram with dynamic range scaling
    trimperc = 1  # Percentile trimming for color scaling
    vmin, vmax = np.percentile(Sx, [0 + trimperc, 100 - trimperc])
    im = ax2.pcolormesh(t2, f, Sx, vmin=vmin, vmax=vmax, cmap='RdBu_r')

    # Add colorbar
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='1%', pad=0.05)
    cb = fig.colorbar(im, cax=cax, orientation='vertical',
                      ticks=[int(vmin), int(vmax)])
    cb.ax.tick_params(labelsize=8)

    # Format spectrogram axis
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.set_xticks([])
    ax2.tick_params(labelsize=8)
    ax2.set_xlim([0, np.max(t2)])
    ax2.set_ylabel('Frequency (Hz)', loc='center')
    ax2.set_yticks([0, 20, 40])
    ax2.text(x=np.max(t2) + 0.08, y=0.2,
             s='Power (dB)', fontsize=9, rotation='vertical')

    # ========== Plot 3: F3 EEG Channel ==========
    ax3 = plt.axes([0.1, 0.21, 0.82, 0.15])
    t3 = np.linspace(0, len(data_F3) / 500 / 3600, len(data_F3))

    # Plot raw EEG trace
    ax3.plot(t3, data_F3, c='k', linewidth=0.5)

    # Format EEG axis
    for spine in ['top', 'right', 'bottom', 'left']:
        ax3.spines[spine].set_visible(False)
    ax3.set_ylabel('F3', rotation='horizontal')
    ax3.yaxis.set_label_coords(-.05, .3)
    ax3.set_xlim([0, np.max(t3) + 0.1])
    ax3.set_xticks([])
    ax3.set_yticks([])

    # ========== Plot 4: EOG Channel ==========
    ax4 = plt.axes([0.1, 0.08, 0.82, 0.15])

    # Plot raw EOG trace
    ax4.plot(t3, data_EOG, c='k', linewidth=0.5)

    # Add scale bars
    ax4.plot([np.max(t3) + 0.04, np.max(t3) + 0.04],
             [np.min(data_EOG), np.min(data_EOG) + 300],
             c='k', lw=0.8, clip_on=False)
    ax4.text(x=np.max(t3) + 0.06, y=np.min(data_EOG) + 100,
             s='300 μV', fontsize=8, rotation='vertical')

    ax4.plot([np.max(t3) + 0.04 - 0.1, np.max(t3) + 0.04],
             [np.min(data_EOG), np.min(data_EOG)],
             c='k', lw=0.8, clip_on=False)
    ax4.text(x=np.max(t3) - 0.08, y=np.min(data_EOG) - 260,
             s='0.1 h', fontsize=8)

    # Format EOG axis
    for spine in ['top', 'right', 'bottom', 'left']:
        ax4.spines[spine].set_visible(False)
    ax4.set_ylabel('EOG', rotation='horizontal')
    ax4.yaxis.set_label_coords(-.05, .2)
    ax4.set_xlim([0, np.max(t3) + 0.1])
    ax4.set_xticks([])
    ax4.set_yticks([])




def stages_channels_plot(epochs, raw_filtre, stages, num, ax, color, label, start):
    # epochs.filter(0.5, 45)
    #### 10s data
    win = 10 * 500
    end = start + 1
    # data = (epochs[stages][num].get_data() * 1000000)[0, :, int(start*win):int(end* win)]
    event = epochs[stages][num].events
    eeg_channels = ['Fp1', 'Fpz', 'Fp2', 'F1', 'F3', 'F5', 'F7', 'Fz', 'F2', 'F4', 'F6', 'F8', 'AF3', 'AF7',
                    'AF4', 'AF8', 'FC1', 'FC3', 'FC5', 'FCz', 'FC2', 'FC4', 'FC6', 'FT7', 'FT8', 'C1', 'C3',
                    'C5', 'Cz', 'C2', 'C4', 'C6', 'CP1', 'CP3', 'CP5', 'CP2', 'CP4', 'CP6', 'T7', 'T8',
                    'TP7', 'TP8', 'P1', 'P3', 'P5', 'P7', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO3', 'PO5', 'PO7',
                    'POz', 'PO4', 'PO6', 'PO8', 'O1', 'Oz', 'O2', 'M1', 'M2']
    data = (raw_filtre.get_data(picks=eeg_channels) * 1000000)[:,
           int(event[0][0] + start * win):int(event[0][0] + end * win)]

    # raw_filtre.filter(0.5, 45)
    data_EOG = np.squeeze(raw_filtre.get_data(picks='EOG') * 1000000)

    t = np.linspace(0, 5, data.shape[1])
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False), ax.spines['left'].set_visible(False)
    # ax.tick_params(left=False)
    ax.margins(x=0)

    ax.plot(t, data[1, :] + 650, c=color, linewidth=0.6)
    ax.plot(t, data[4, :] + 550, c=color, linewidth=0.6)
    ax.plot(t, data[28, :] + 450, c=color, linewidth=0.6)
    ax.plot(t, data[32, :] + 350, c=color, linewidth=0.6)
    ax.plot(t, data[38, :] + 250, c=color, linewidth=0.6)
    ax.plot(t, data[46, :] + 150, c=color, linewidth=0.6)
    ax.plot(t, data[59, :] + 50, c=color, linewidth=0.6)
    ax.plot(t, data_EOG[int(event[0][0] + start * win):int(event[0][0] + end * win)] - 90, c='#929491', linewidth=0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_ylim([-200,850])
    ax.set_title(label, color=color)
    # ax.text(x=-1, y=220, s=label, color = color,rotation = 'vertical')
    if label == 'REM':
        ax.plot([4.2, 5.2], [-180, -180], c='k', linewidth=1)
        ax.text(x=4.2, y=-300, s='1 s', fontsize=8)
        ax.plot([5.2, 5.2], [-180, -80], c='k', linewidth=1)
        ax.text(x=5.5, y=-180, s='100 uV', fontsize=8, rotation='vertical')


def sleep_time_state(file_name, color):
    '''Calculate total sleep time data and distribution across different sleep stages'''

    # Initialize lists to store counts for each sleep stage
    w = []  # Wake
    n1 = []  # N1 sleep stage
    n2 = []  # N2 sleep stage
    n3 = []  # N3 sleep stage
    r = []  # REM sleep
    total = []  # Total epochs (minus 6 for some adjustment)

    # Process each EEG data file
    for filename in glob.glob(file_name):
        s = op.basename(filename)  # Get base filename
        raw_eeg = mne.read_epochs(filename)  # Read EEG data using MNE

        # Extract epochs for each sleep stage
        epoch_w = raw_eeg['wake']
        epoch_n1 = raw_eeg['N1']
        epoch_n2 = raw_eeg['N2']
        epoch_n3 = raw_eeg['N3']
        epoch_r = raw_eeg['REM']

        # Store counts of epochs for each stage
        w.append(len(epoch_w))
        n1.append(len(epoch_n1))
        n2.append(len(epoch_n2))
        n3.append(len(epoch_n3))
        r.append(len(epoch_r))
        total.append(len(raw_eeg) - 6)  # Total epochs with adjustment

    # Convert lists to numpy arrays for easier calculation
    w = np.array(w)
    n1 = np.array(n1)
    n2 = np.array(n2)
    n3 = np.array(n3)
    r = np.array(r)
    total = np.array(total)

    # Calculate percentage of time spent in each sleep stage
    data = [w / total, n1 / total, n2 / total, n3 / total, r / total]
    name_list = ['Wake', 'N1', 'N2', 'N3', 'REM']  # Sleep stage labels

    ### Create seaborn plot ###
    # Prepare data in dictionary format for plotting
    dict_ = {'stages': [], 'Value': []}
    for stage in range(len(name_list)):
        for v in range(len(data[stage])):
            dict_['stages'].append(name_list[stage])
            dict_['Value'].append(data[stage][v])

    # Set up plot colors and dataframe
    color_pal = sns.color_palette(color)  # Color palette based on input
    df = pd.DataFrame(dict_, columns=['stages', 'Value'])  # Create DataFrame

    # Initialize figure with specific size (converted from cm)
    fig, ax = plt.subplots(figsize=(6 * cm, 6 * cm))

    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Create strip plot (shows individual data points)
    sns.stripplot(x='stages', y='Value', data=df, size=3, palette=color_pal,
                  ax=ax, alpha=0.5)

    # Create box plot (shows distribution statistics)
    sns.boxplot(data=df, x='stages', y='Value', ax=ax, width=0.5,
                linewidth=1, fliersize=0, palette=color_pal)

    # Customize box plot appearance
    # Get all box plot patches (matplotlib 3.5+)
    box_patches = [patch for patch in ax.patches if type(patch) == matplotlib.patches.PathPatch]
    if len(box_patches) == 0:  # For older matplotlib versions
        box_patches = ax.artists

    num_patches = len(box_patches)
    lines_per_boxplot = len(ax.lines) // num_patches

    # Style each box plot element
    for i, patch in enumerate(box_patches):
        # Make boxes transparent with colored edges
        col = patch.get_facecolor()
        patch.set_edgecolor(col)
        patch.set_facecolor('None')

        # Style all related lines (whiskers, etc.) to match
        for line in ax.lines[i * lines_per_boxplot: (i + 1) * lines_per_boxplot]:
            line.set_color(col)
            line.set_mfc(col)  # Fliers face color
            line.set_mec(col)  # Fliers edge color

    # Final plot labels
    ax.set_xlabel('')  # No x-axis label
    ax.set_ylabel('Percentage (%)')  # Y-axis label

def transition_matrix_plot(c_m):
    classes = ['Wake', 'N1', 'N2', 'N3', 'REM']
    fig, ax = plt.subplots(figsize=(6*cm, 6*cm))
    ax.grid(False)
    im = ax.imshow(c_m, interpolation="nearest", cmap='YlGnBu')
    # We want to show all ticks...
    plt.xticks([0, 1, 2, 3, 4], classes)
    plt.yticks([0, 1, 2, 3, 4], classes)
    plt.ylabel('Stage t')
    plt.xlabel('Stage t+1')
    # Loop over data dimensions and create text annotations.
    fmt = ".2f"
    thresh = c_m.max() / 2.0
    for i in range(c_m.shape[0]):
        for j in range(c_m.shape[1]):
            ax.text(j, i, format(c_m[i, j], fmt), ha="center", va="center",
                    color="white" if c_m[i, j] > thresh else "black",fontsize=9)
    plt.tight_layout()


#  ========== Source estimation ==========
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)
# The files live in:
subject = "fsaverage"
trans = "fsaverage"  # MNE has a built-in fsaverage transformation
src = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
bem = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")

dict_stage = {
              0: 'N2',
              1: 'N3',
              2: 'REM'}


def source_stage(data_path, save_path):
    '''
    Perform source estimation for different sleep stages and save the results.

    Parameters:
    -----------
    data_path : str
        Path pattern to find EEG epoch files
    save_path : str
        Directory to save source estimation results
    '''

    # Define frequency band of interest (delta band for sleep)
    band = (0.5, 4)  # Frequency range in Hz

    # Path to source space file (ico-5 resolution)
    src = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")

    # Process each EEG file matching the data_path pattern
    for filename in glob.glob(data_path):
        # Load EEG epochs data
        raw_eeg = mne.read_epochs(filename)

        # Apply bandpass filter (6-24 Hz) for source localization
        raw_eeg.filter(6, 24)

        # Set standard 10-20 electrode montage
        montage = mne.channels.make_standard_montage("standard_1020")
        raw_eeg.set_montage(montage, match_case=False)

        # Set average reference (required for inverse modeling)
        raw_eeg.set_eeg_reference(projection=True)

        # Compute noise covariance matrix
        noise_cov_reg = mne.compute_covariance(raw_eeg)

        # Create forward solution (head model)
        fwd = mne.make_forward_solution(
            raw_eeg.info,
            trans=trans,  # Coordinate transformation
            src=src,  # Source space
            bem=bem,  # Boundary element model
            verbose=True
        )

        # Create inverse operator
        inv = mne.minimum_norm.make_inverse_operator(
            raw_eeg.info,
            fwd,
            noise_cov_reg,
            verbose=True
        )

        # Set parameters for source estimation
        method = "dSPM"  # Dynamic Statistical Parametric Mapping
        snr = 1.0  # Signal-to-noise ratio (1 for epochs, 3 for evoked)
        lambda2 = 1.0 / snr ** 2  # Regularization parameter

        # Process each sleep stage
        for i in range(len(dict_stage)):
            stage = dict_stage[i]  # Current sleep stage (e.g., 'N2', 'REM')
            epochs = raw_eeg[stage]  # Extract epochs for this stage

            # Compute evoked response by averaging epochs
            evoked = epochs.average()

            # Compute source time courses (STC) using dSPM
            stc = source_function._gen_dsmp(evoked, inv, method, lambda2)

            # Prepare save directory and filename
            s = op.basename(filename)
            source_function.mkdir(save_path + stage + '/')
            savename = save_path + stage + '/' + s[:-4] + '_' + stage

            # Save source estimate
            stc.save(savename, overwrite=True)


def source_stage_plot(stc_path):
    '''
    Plot and save averaged source estimates for each sleep stage.

    Parameters:
    -----------
    stc_path : str
        Path containing source time course (.stc) files organized by sleep stage
    '''

    # Process each sleep stage defined in dict_stage
    for i in range(len(dict_stage)):
        stage = dict_stage[i]  # Current sleep stage (e.g., 'N1', 'REM')
        path = stc_path + stage + '/*.stc'  # Pattern to find STC files

        # Initialize variables for averaging
        data_all = 0  # Will accumulate source data
        count = 0  # Count of processed files

        # Process each right hemisphere STC file
        for j in glob.glob(path):
            if j.endswith('-rh.stc'):  # Only process right hemisphere files
                # Load source estimate
                stc = mne.read_source_estimate(j, subject="fsaverage")
                stc_template = stc.copy()
                data = stc.data  # Get source activation data

                # Accumulate data for averaging
                data_all = data_all + data
                count += 1

                # Optional: Uncomment to plot individual subject STCs
                # stc_brain = source_function.brain_plot(stc, subjects_dir, message,
                #                                   hemi="split",
                #                                   views=["lat", "med"],
                #                                   colorbar=True)
                # s = op.basename(j)
                # source_function.mkdir('./path/to/save/individual/plots/' + stage + '/')
                # stc_brain.save_image('./path/to/save/individual/plots/' + stage + '/'+s[:-12]+'.png')

        # Calculate mean activation across subjects
        stc_mean = stc_template  # Use last loaded STC as template
        message = None  # Optional message for plot
        mean = data_all / count  # Compute mean activation

        # Apply mean data to template STC
        stc_mean.data = mean

        # Create directories for saving results
        source_function.mkdir(stc_path + stage + '/mean/')
        figsave_path = stc_path + 'figures/'
        source_function.mkdir(figsave_path)

        # Save mean source estimate
        stc_mean.save(stc_path + stage + '/mean/stc_mean', overwrite=True)

        # Generate brain plot of mean activation
        stc_mean_brain = source_function.brain_plot(
            stc_mean,
            subjects_dir,
            message,
            hemi="split",  # Show both hemispheres
            views=["lat", "med"],  # Lateral and medial views
            colorbar=False  # Omit colorbar for cleaner plot
        )

        # Save brain plot image
        stc_mean_brain.save_image(figsave_path + stage + '.png')


def source_coffe_via_stage(file_path, ax, color=None, color1=None):
    '''
    Analyze and visualize correlation patterns between sleep stages (N2, N3, REM)
    using source space data.

    Parameters:
    -----------
    file_path : str
        Path containing source estimate files organized by sleep stage
    ax : matplotlib axes
        Array of axes objects for plotting (3 axes expected)
    color : dict, optional
        Primary color palette for plotting
    color1 : dict, optional
        Secondary color palette for plotting

    This function:
    1. Loads mean source estimates for N2, N3, and REM stages
    2. Computes Pearson correlations between individual patterns and group means
    3. Performs statistical comparisons between stages
    4. Generates violin plots showing correlation distributions
    '''

    # Load mean source estimates for each stage (left hemisphere only)
    path_N3_mean = file_path + 'N3/mean/stc_mean-lh.stc'
    path_N2_mean = file_path + 'N2/mean/stc_mean-lh.stc'
    path_rem_mean = file_path + 'REM/mean/stc_mean-lh.stc'

    # Read and process N3 mean source estimate
    stc_N3_mean = mne.read_source_estimate(path_N3_mean, subject="fsaverage")
    stc_N3_mean = stc_N3_mean.mean()  # Temporal averaging
    data_N3_mean = stc_N3_mean.data  # Extract source activation data

    # Read and process N2 mean source estimate
    stc_N2_mean = mne.read_source_estimate(path_N2_mean, subject="fsaverage")
    stc_N2_mean = stc_N2_mean.mean()
    data_N2_mean = stc_N2_mean.data

    # Read and process REM mean source estimate
    stc_rem_mean = mne.read_source_estimate(path_rem_mean, subject="fsaverage")
    stc_rem_mean = stc_rem_mean.mean()
    data_rem_mean = stc_rem_mean.data

    # Clean up memory
    del stc_N3_mean, stc_rem_mean, stc_N2_mean

    # Initialize correlation coefficient containers
    # REM correlations
    coef_rem_r = []  # REM individual vs REM mean
    coef_rem_n2 = []  # REM individual vs N2 mean
    coef_rem_n3 = []  # REM individual vs N3 mean

    # N3 correlations
    coef_n3_r = []  # N3 individual vs REM mean
    coef_n3_n2 = []  # N3 individual vs N2 mean
    coef_n3_n3 = []  # N3 individual vs N3 mean

    # N2 correlations
    coef_n2_r = []  # N2 individual vs REM mean
    coef_n2_n2 = []  # N2 individual vs N2 mean
    coef_n2_n3 = []  # N2 individual vs N3 mean

    name_list = ['N2', 'N3', 'REM']  # Stage labels

    # Process each individual REM source file
    path_rem = file_path + 'REM/*.stc'
    for filename in glob.glob(path_rem):
        if filename.endswith('-lh.stc'):  # Only process left hemisphere
            s = op.basename(filename)
            file = s[:-11]  # Extract base filename

            # Process individual REM data
            stc_rem = mne.read_source_estimate(filename, subject="fsaverage")
            stc_rem = stc_rem.mean()
            data_rem = stc_rem.data

            # Compute correlations with mean patterns
            cc_rem_r = scipy.stats.pearsonr(np.squeeze(data_rem_mean), np.squeeze(data_rem)).statistic
            cc_n3_r = scipy.stats.pearsonr(np.squeeze(data_N3_mean), np.squeeze(data_rem)).statistic
            cc_n2_r = scipy.stats.pearsonr(np.squeeze(data_N2_mean), np.squeeze(data_rem)).statistic

            # Load and process corresponding N3 data for this subject
            stc_n3 = mne.read_source_estimate(file_path + 'N3/' + file + '_N3-lh.stc',
                                              subject="fsaverage")
            stc_n3 = stc_n3.mean()
            data_n3 = stc_n3.data

            # Compute correlations for N3
            cc_rem_n3 = scipy.stats.pearsonr(np.squeeze(data_rem_mean), np.squeeze(data_n3)).statistic
            cc_n3_n3 = scipy.stats.pearsonr(np.squeeze(data_N3_mean), np.squeeze(data_n3)).statistic
            cc_n2_n3 = scipy.stats.pearsonr(np.squeeze(data_N2_mean), np.squeeze(data_n3)).statistic

            # Load and process corresponding N2 data for this subject
            stc_n2 = mne.read_source_estimate(file_path + 'N2/' + file + '_N2-lh.stc',
                                              subject="fsaverage")
            stc_n2 = stc_n2.mean()
            data_n2 = stc_n2.data

            # Compute correlations for N2
            cc_rem_n2 = scipy.stats.pearsonr(np.squeeze(data_rem_mean), np.squeeze(data_n2)).statistic
            cc_n3_n2 = scipy.stats.pearsonr(np.squeeze(data_N3_mean), np.squeeze(data_n2)).statistic
            cc_n2_n2 = scipy.stats.pearsonr(np.squeeze(data_N2_mean), np.squeeze(data_n2)).statistic

            # Store all correlation coefficients
            coef_rem_r.append(cc_rem_r)
            coef_rem_n2.append(cc_rem_n2)
            coef_rem_n3.append(cc_rem_n3)

            coef_n3_r.append(cc_n3_r)
            coef_n3_n2.append(cc_n3_n2)
            coef_n3_n3.append(cc_n3_n3)

            coef_n2_r.append(cc_n2_r)
            coef_n2_n2.append(cc_n2_n2)
            coef_n2_n3.append(cc_n2_n3)

    # Prepare data for plotting - REM correlations
    dict_rem = {'stage': [], 'Value': []}
    data_rem = [np.array(coef_rem_n2), np.array(coef_rem_n3), np.array(coef_rem_r)]
    for stage in range(len(name_list)):
        for v in range(len(data_rem[0])):
            dict_rem['stage'].append(name_list[stage])
            dict_rem['Value'].append(data_rem[stage][v])

    # Prepare data for plotting - N3 correlations
    dict_n3 = {'stage': [], 'Value': []}
    data_n3 = [np.array(coef_n3_n2), np.array(coef_n3_n3), np.array(coef_n3_r)]
    for stage in range(len(name_list)):
        for v in range(len(data_n3[0])):
            dict_n3['stage'].append(name_list[stage])
            dict_n3['Value'].append(data_n3[stage][v])

    # Prepare data for plotting - N2 correlations
    dict_n2 = {'stage': [], 'Value': []}
    data_n2 = [np.array(coef_n2_n2), np.array(coef_n2_n3), np.array(coef_n2_r)]
    for stage in range(len(name_list)):
        for v in range(len(data_n2[0])):
            dict_n2['stage'].append(name_list[stage])
            dict_n2['Value'].append(data_n2[stage][v])

    # Set default color palettes if not provided
    if color is None:
        color = {0: '#7db1cf',  # Light blue for N2
                 1: '#8074C8',  # Purple for N3
                 2: '#ebd860',  # Yellow for REM
                 }
    if color1 is None:
        color1 = {0: '#3e81a8',  # Darker blue for N2
                  1: '#5848b7',  # Darker purple for N3
                  2: '#ccb319',  # Darker yellow for REM
                  }

    # Create color palettes
    color_pal = sns.color_palette([color[0], color[1], color[2]])
    color_pal1 = sns.color_palette([color1[0], color1[1], color1[2]])

    # Define pairs for statistical comparisons
    pairs = [('N2', 'N3'), ('N2', 'REM'), ('REM', 'N3')]

    # Plot N3 correlations
    df = pd.DataFrame(dict_n3, columns=['stage', 'Value'])
    y_label = 'Source$^{N3}$ Coef'  # Label indicating N3 reference pattern
    ylim = [-1, 1.5]  # Consistent y-axis limits
    ax[1].set_ylim(ylim)

    # Perform statistical tests and get p-values
    p = statistical_analysis(coef_n3_n2, coef_n3_n3, coef_n3_r)
    annotations = [convert_pvalue_to_asterisks(i) for i in p]

    # Generate violin plot with statistical annotations
    source_function.annotation_plot_(df, 'stage', 'Value', annotations, color_pal,
                                  color_pal1, name_list, ax[1], pairs, y_label,
                                  cut=2, types='violin', loc='outside')

    # Plot N2 correlations
    df = pd.DataFrame(dict_n2, columns=['stage', 'Value'])
    y_label = 'Source$^{N2}$ Coef'
    ax[0].set_ylim(ylim)
    p = statistical_analysis(coef_n2_n2, coef_n2_n3, coef_n2_r)
    annotations = [convert_pvalue_to_asterisks(i) for i in p]
    source_function.annotation_plot_(df, 'stage', 'Value', annotations, color_pal,
                                  color_pal1, name_list, ax[0], pairs, y_label,
                                  cut=2, types='violin', loc='outside')

    # Plot REM correlations
    df = pd.DataFrame(dict_rem, columns=['stage', 'Value'])
    y_label = 'Source$^{REM}$ Coef'
    ax[2].set_ylim(ylim)
    p = statistical_analysis(coef_rem_n2, coef_rem_n3, coef_rem_r)
    annotations = [convert_pvalue_to_asterisks(i) for i in p]
    source_function.annotation_plot_(df, 'stage', 'Value', annotations, color_pal,
                                  color_pal1, name_list, ax[2], pairs, y_label,
                                  cut=2, types='violin', loc='outside')


def index_N2_N3_REM(num):
    """
    Identify N2 segments that precede N3 or REM sleep stages and return their indices.

    This function analyzes a sleep stage sequence to find:
    1. N2 segments that transition to N3
    2. N2 segments that transition to REM

    Parameters:
    -----------
    num : list
        Sequence of sleep stages where:
        - '2' represents N2
        - '3' represents N3
        - '4' represents REM
        Other numbers may represent other stages (Wake, N1, etc.)

    Returns:
    --------
    tuple: (n2_to_n3, n2_to_rem)
        n2_to_n3: List of N2-N3 transition segments as [N2_start, N2_end, N3_start, N3_end]
        n2_to_rem: List of N2-REM transition segments as [N2_start, N2_end, REM_start, REM_end]
        All indices are in 30-second epoch units
    """

    # Initialize sequence segmentation
    # Each segment will be stored as [start_index, end_index, stage_label]
    seq = [[0, -1, str(num[0])]]  # Start with first stage

    # Segment the continuous stage sequence into distinct blocks
    for i, (s1, s2) in enumerate(zip(num[0:-1], num[1:])):
        if s1 != s2:  # When stage changes
            # Finalize previous segment
            seq[-1][1] = i  # Set end index of current segment
            # Start new segment
            seq.append([i + 1, -1, str(num[i + 1])])

    # Finalize the last segment
    seq[-1][1] = i + 1
    seq[-1][2] = str(num[i + 1])

    # Initialize containers for identified transitions
    n2_to_n3 = []  # Will store N2→N3 transitions
    n2_to_rem = []  # Will store N2→REM transitions

    # Analyze consecutive segment pairs for transitions
    for (l1, r1, s1), (l2, r2, s2) in zip(seq[0:-1], seq[1:]):
        # Only consider N2 segments that are at least 2 epochs (60s) long
        if s1 == '2' and (r1 - l1 + 1) > 1:
            # Only consider following segments that are at least 2 epochs (60s) long
            if (r2 - l2 + 1) > 1:
                if s2 == '3':  # N2→N3 transition
                    n2_to_n3.append([l1, r1, l2, r2])
                elif s2 == '4':  # N2→REM transition
                    n2_to_rem.append([l1, r1, l2, r2])

    return n2_to_n3, n2_to_rem


def transition_time(data_path, ax, color=None):
    """
    Analyze and visualize sleep stage transitions from N2 to N3 and N2 to REM.

    Parameters:
    -----------
    data_path : str
        Path pattern to find EEG epoch files
    ax : matplotlib axes
        Array of axes objects for plotting (2 axes expected)
    color : list, optional
        Custom color palette for sleep stage visualization

    This function:
    1. Identifies N2→N3 and N2→REM transitions in sleep data
    2. Extracts time windows around these transitions
    3. Creates heatmap visualizations of the transitions
    """

    # Initialize containers for transition data
    stage_num_n3 = []  # Will store N2→N3 transition sequences
    stage_num_r = []  # Will store N2→REM transition sequences

    # Process each EEG file
    for filename in glob.glob(data_path):
        s = op.basename(filename)
        raw_eeg = mne.read_epochs(filename)

        # Get sleep stage labels (events[:,2] contains the stage codes)
        num = raw_eeg.events[:, 2].T

        # Identify N2→N3 and N2→REM transitions
        idx_n2_3, idx_n2_r = index_N2_N3_REM(num)

        # Process N2→N3 transitions
        if len(idx_n2_3) == 1:  # If exactly one transition found
            # Extract 22 epochs (11 min) window around transition point
            # -10 epochs = 5 min before, +12 epochs = 6 min after
            idx_n2_3_win = np.arange(idx_n2_3[0][1] - 10, idx_n2_3[0][1] + 12)
            num_n2_3 = num[idx_n2_3_win]
            stage_num_n3.append(num_n2_3)

        elif len(idx_n2_3) == 2:  # If two transitions found
            # Process first transition
            idx_n2_3_win_1 = np.arange(idx_n2_3[0][1] - 10, idx_n2_3[0][1] + 12)
            num_n2_3_1 = num[idx_n2_3_win_1]

            # Process second transition
            idx_n2_3_win_2 = np.arange(idx_n2_3[1][1] - 10, idx_n2_3[1][1] + 12)
            num_n2_3_2 = num[idx_n2_3_win_2]

            # Store both transitions
            stage_num_n3.append(num_n2_3_1)
            stage_num_n3.append(num_n2_3_2)

        # Process N2→REM transitions
        if len(idx_n2_r) == 1:  # If exactly one transition found
            idx_n2_r_win = np.arange(idx_n2_r[0][1] - 10, idx_n2_r[0][1] + 12)
            num_n2_r = num[idx_n2_r_win]
            stage_num_r.append(num_n2_r)

    # Convert to numpy arrays for plotting
    stage_num_n3 = np.array(stage_num_n3)
    stage_num_r = np.array(stage_num_r)

    # Set default color palette if not provided
    if color is None:
        color = ['#B8DBB3',  # Wake (light green)
                 '#EAB883',  # N1 (light orange)
                 '#A8CBDF',  # N2 (light blue)
                 '#8074C8',  # N3 (purple)
                 '#F5EBAE',  # REM (light yellow)
                 '#F5EBAE']  # REM duplicate for colormap

    # Create colormap for sleep stages
    cmp = mpl.colors.ListedColormap(color)

    # Create time axis (in minutes)
    t = np.linspace(-5, 5, stage_num_n3.shape[1])  # -5 to +5 minutes

    # Create N2→N3 transition heatmap
    tran = np.arange(stage_num_n3.shape[0])  # Transition numbers
    bounds = [0, 1, 2, 3, 4, 5]  # Boundaries for stage colors
    norm = mpl.colors.BoundaryNorm(bounds, cmp.N, clip=True)

    # Plot N2→N3 transitions
    im = ax[0].pcolormesh(t, tran, stage_num_n3, cmap=cmp, norm=norm)
    ax[0].set_xlabel('Time (min)')
    ax[0].set_ylabel('N2 to N3 transition #')
    ax[0].set_yticks([0, 5, 10, 15, 20])
    ax[0].set_xlim(-5, 5)  # Show 5 min before/after transition
    ax[0].set_position([0.1, 0.6, 0.3, 0.4])  # Adjust subplot position

    # Plot N2→REM transitions
    tran = np.arange(stage_num_r.shape[0])  # Transition numbers
    im = ax[1].pcolormesh(t, tran, stage_num_r, cmap=cmp, norm=norm)
    ax[1].set_xlabel('Time (min)')
    ax[1].set_ylabel('N2 to REM transition #')
    ax[1].set_yticks([0, 4, 8, 12])
    ax[1].set_xlim(-5, 5)
    ax[1].set_position([0.6, 0.6, 0.31, 0.4])  # Adjust subplot position

    # Add colorbar to REM plot
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig = ax[1].get_figure()
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5])  # Center ticks in color blocks
    cbar.set_ticklabels(['Wake', 'N1', 'N2', 'N3', 'REM'])


def source_coffe_via_time(ax, start=-5, end=5):
    '''
    Calculate and visualize the temporal evolution of source space correlations
    during N2→N3 and N2→REM sleep stage transitions.

    Parameters:
    -----------
    ax : matplotlib axes
        Array of axes objects for plotting (2 axes expected)
    start : int, optional
        Start time in minutes relative to transition (default: -5)
    end : int, optional
        End time in minutes relative to transition (default: 5)

    This function:
    1. Loads mean source estimates for N3 and REM stages
    2. For each subject, computes source estimates around transitions
    3. Calculates Pearson correlations with mean N3/REM patterns
    4. Plots temporal evolution of correlation coefficients
    '''

    # Load mean source estimates for N3 and REM stages (left hemisphere)
    path_N3 = './data/stc_dsmp/N3/mean/stc_mean-lh.stc'
    stc_N3 = mne.read_source_estimate(path_N3, subject="fsaverage")
    stc_N3 = stc_N3.mean()  # Temporal averaging
    data_N3 = stc_N3.data  # Source activation data

    path_rem = './data/stc_dsmp/REM/mean/stc_mean-lh.stc'
    stc_rem = mne.read_source_estimate(path_rem, subject="fsaverage")
    stc_rem = stc_rem.mean()
    data_rem = stc_rem.data

    del stc_N3, stc_rem  # Free memory

    # Initialize containers for correlation coefficients
    coeff_n3 = []  # Will store N3 correlations
    coeff_rem = []  # Will store REM correlations

    # Process each subject's EEG data
    data_path = './data/epoch_eeg_ica/*.fif'
    for filename in glob.glob(data_path):
        s = op.basename(filename)
        raw_eeg = mne.read_epochs(filename)

        # Set up EEG montage and reference
        montage = mne.channels.make_standard_montage("standard_1020")
        raw_eeg.set_montage(montage, match_case=False)
        raw_eeg.set_eeg_reference(projection=True)  # Needed for source modeling

        # Compute noise covariance
        noise_cov_reg = mne.compute_covariance(raw_eeg)

        # Create forward and inverse operators
        fwd = mne.make_forward_solution(raw_eeg.info, trans=trans, src=src, bem=bem, verbose=True)
        inv = mne.minimum_norm.make_inverse_operator(raw_eeg.info, fwd, noise_cov_reg, verbose=True)

        # Set parameters for source estimation
        method = "dSPM"  # Dynamic Statistical Parametric Mapping
        snr = 3.0  # Signal-to-noise ratio (3 for evoked responses)
        lambda2 = 1.0 / snr ** 2  # Regularization parameter

        # Get sleep stage labels and identify transitions
        num = raw_eeg.events[:, 2].T
        idx_n2_3, idx_n2_r = index_N2_N3_REM(num)

        # Initialize subject-specific correlation containers
        coeff_sub_n3 = []
        coeff_sub_rem = []

        # Analyze time points from start to end minutes
        for n in range(int(start * 60 / 30), int(end * 60 / 30)):
            # Convert minutes to epoch indices (30s epochs)
            print('==========' + str(n) + '==========')

            # Process N2→N3 transitions
            if len(idx_n2_3) == 1:  # Single transition case
                # Extract 1-epoch window at time point n
                idx_n2_3_win = np.arange(idx_n2_3[0][1] + n, idx_n2_3[0][1] + n + 1)
                epoch_N2_3 = raw_eeg[idx_n2_3_win]

                # Compute source estimate
                evoked_N2_3 = epoch_N2_3.average()
                stc_dsmp_N2_3 = source_function._gen_dsmp(evoked_N2_3, inv, method, lambda2)
                stc_dsmp_N2_3 = stc_dsmp_N2_3.mean()
                data_n2_3 = stc_dsmp_N2_3.data

                # Calculate correlation with mean N3 pattern
                cc_n3 = scipy.stats.pearsonr(np.squeeze(data_N3), np.squeeze(data_n2_3)).statistic
                coeff_sub_n3.append(cc_n3)
                del stc_dsmp_N2_3, data_n2_3

            elif len(idx_n2_3) == 2:  # Two transitions case
                # Process first transition
                idx_n2_3_win_1 = np.arange(idx_n2_3[0][1] + n, idx_n2_3[0][1] + n + 1)
                epoch_N2_3_1 = raw_eeg[idx_n2_3_win_1]
                evoked_N2_3_1 = epoch_N2_3_1.average()
                stc_dsmp_N2_3_1 = source_function._gen_dsmp(evoked_N2_3_1, inv, method, lambda2)
                stc_dsmp_N2_3_1 = stc_dsmp_N2_3_1.mean()
                data_n2_3_1 = stc_dsmp_N2_3_1.data
                cc_n3_1 = scipy.stats.pearsonr(np.squeeze(data_N3), np.squeeze(data_n2_3_1)).statistic
                del stc_dsmp_N2_3_1, data_n2_3_1

                # Process second transition
                idx_n2_3_win_2 = np.arange(idx_n2_3[1][1] + n, idx_n2_3[1][1] + n + 1)
                epoch_N2_3_2 = raw_eeg[idx_n2_3_win_2]
                evoked_N2_3_2 = epoch_N2_3_2.average()
                stc_dsmp_N2_3_2 = source_function._gen_dsmp(evoked_N2_3_2, inv, method, lambda2)
                stc_dsmp_N2_3_2 = stc_dsmp_N2_3_2.mean()
                data_n2_3_2 = stc_dsmp_N2_3_2.data
                cc_n3_2 = scipy.stats.pearsonr(np.squeeze(data_N3), np.squeeze(data_n2_3_2)).statistic
                del stc_dsmp_N2_3_2, data_n2_3_2

                # Store average of two transitions
                coeff_sub_n3.append((cc_n3_1 + cc_n3_2) / 2)

            # Process N2→REM transitions
            if len(idx_n2_r) == 1:  # Single transition case
                idx_n2_r_win = np.arange(idx_n2_r[0][1] + n, idx_n2_r[0][1] + n + 1)
                epoch_N2_R = raw_eeg[idx_n2_r_win]
                evoked_N2_R = epoch_N2_R.average()
                stc_dsmp_N2_R = source_function._gen_dsmp(evoked_N2_R, inv, method, lambda2)
                stc_dsmp_N2_R = stc_dsmp_N2_R.mean()
                data_n2_rem = stc_dsmp_N2_R.data
                cc_rem = scipy.stats.pearsonr(np.squeeze(data_rem), np.squeeze(data_n2_rem)).statistic
                coeff_sub_rem.append(cc_rem)
                del stc_dsmp_N2_R, data_n2_rem

        # Store subject results if available
        if len(coeff_sub_n3) != 0:
            coeff_n3.append(coeff_sub_n3)
        if len(coeff_sub_rem) != 0:
            coeff_rem.append(coeff_sub_rem)

    # Convert to numpy arrays
    coeff_n3 = np.array(coeff_n3)
    coeff_rem = np.array(coeff_rem)

    # Calculate group statistics
    mean_r, std_r = np.mean(coeff_rem, axis=0), scipy.stats.sem(coeff_rem, axis=0)
    mean_n, std_n = np.mean(coeff_n3, axis=0), scipy.stats.sem(coeff_n3, axis=0)

    # Create time axis (minutes)
    t = np.linspace(-10, 5, len(mean_n))

    # Plot N3 correlation time course
    for i in range(coeff_n3.shape[0]):  # Individual subject traces
        ax[0].plot(t, coeff_n3[i, :], linewidth=0.5, c='darkgray', alpha=0.6)

    # Format N3 plot
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].plot(t, mean_n, '#3F6561', linewidth=1.5, label='N3')  # Mean trace
    ax[0].fill_between(t, mean_n - std_n, mean_n + std_n, alpha=1, color='#c5d1d2')  # SEM
    ax[0].set_xlabel('Time (min)')
    ax[0].margins(x=0)
    ax[0].set_ylim([-0.6, 0.8])
    ax[0].set_yticks([-0.5, 0, 0.5])
    ax[0].set_ylabel('Source$^{N3}$ Coef')

    # Plot REM correlation time course
    for i in range(coeff_rem.shape[0]):  # Individual subject traces
        ax[1].plot(t, coeff_rem[i, :], linewidth=0.5, c='darkgray', alpha=0.4)

    # Format REM plot
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].plot(t, mean_r, '#7bc4c5', linewidth=1.5, label='REM')  # Mean trace
    ax[1].fill_between(t, mean_r - std_r, mean_r + std_r, alpha=1, color='#d9eeee')  # SEM
    ax[1].set_xlabel('Time (min)')
    ax[1].margins(x=0)
    ax[1].set_ylim([-0.6, 0.8])
    ax[1].set_yticks([-0.5, 0, 0.5])
    ax[1].set_ylabel('Source$^{REM}$ Coef')

    # Adjust subplot positions
    ax[0].set_position([0.1, 0.15, 0.3, 0.32])
    ax[1].set_position([0.6, 0.15, 0.3, 0.32])


def N2_stage_source(data_path, save_path):
    '''
    Calculate and visualize source localization results for N2 sleep stage transitions
    to N3 or REM, including time periods before and after the transition.

    Parameters:
    -----------
    data_path : str
        Path pattern to find EEG epoch files
    save_path : str
        Directory to save source localization results and figures

    This function:
    1. Analyzes source activity at specific time points (-3min, -1min, 0min, +1min)
       relative to N2→N3 and N2→REM transitions
    2. Computes average source estimates across subjects
    3. Generates brain plots of the source activations
    '''

    # Define time points to analyze (in 30-second epoch units)
    # -6 epochs = -3 minutes, -2 epochs = -1 minute, etc.
    for t in [-6, -2, 0, 2]:
        # Initialize accumulators for source data
        data_n2_3 = 0  # For N2→N3 transitions
        count_n2_3 = 0  # Count of N2→N3 transitions
        data_n2_r = 0  # For N2→REM transitions
        count_n2_r = 0  # Count of N2→REM transitions

        # Process each subject's EEG data
        for filename in glob.glob(data_path):
            s = op.basename(filename)
            raw_eeg = mne.read_epochs(filename)

            # Set up EEG montage and reference
            montage = mne.channels.make_standard_montage("standard_1020")
            raw_eeg.set_montage(montage, match_case=False)
            raw_eeg.set_eeg_reference(projection=True)  # Needed for source modeling

            # Compute noise covariance
            noise_cov_reg = mne.compute_covariance(raw_eeg)

            # Create forward and inverse operators
            fwd = mne.make_forward_solution(raw_eeg.info, trans=trans, src=src, bem=bem, verbose=True)
            inv = mne.minimum_norm.make_inverse_operator(raw_eeg.info, fwd, noise_cov_reg, verbose=True)

            # Set parameters for source estimation
            method = "dSPM"  # Dynamic Statistical Parametric Mapping
            snr = 3.0  # Signal-to-noise ratio (3 for evoked responses)
            lambda2 = 1.0 / snr ** 2  # Regularization parameter

            # Get sleep stage labels and identify transitions
            num = raw_eeg.events[:, 2].T
            idx_n2_3, idx_n2_r = index_N2_N3_REM(num)

            # Process N2→N3 transitions
            if len(idx_n2_3) == 1:  # Single transition case
                # Extract 1-minute window (2 epochs) around time point t
                idx_n2_3_win = np.arange(idx_n2_3[0][1] + t, idx_n2_3[0][1] + t + 2)
                epoch_N2_3 = raw_eeg[idx_n2_3_win]

                # Compute source estimate
                evoked_N2_3 = epoch_N2_3.average()
                stc_dsmp_N2_3 = source_function._gen_dsmp(evoked_N2_3, inv, method, lambda2)

                # Accumulate source data
                data_n2_3 = data_n2_3 + stc_dsmp_N2_3.data
                count_n2_3 = count_n2_3 + 1

            elif len(idx_n2_3) == 2:  # Two transitions case
                # Process first transition
                idx_n2_3_win_1 = np.arange(idx_n2_3[0][1] + t, idx_n2_3[0][1] + t + 2)
                epoch_N2_3_1 = raw_eeg[idx_n2_3_win_1]
                evoked_N2_3_1 = epoch_N2_3_1.average()
                stc_dsmp_N2_3_1 = source_function._gen_dsmp(evoked_N2_3_1, inv, method, lambda2)
                data_n2_3 = data_n2_3 + stc_dsmp_N2_3_1.data
                count_n2_3 = count_n2_3 + 1

                # Process second transition
                idx_n2_3_win_2 = np.arange(idx_n2_3[1][1] + t, idx_n2_3[1][1] + t + 2)
                epoch_N2_3_2 = raw_eeg[idx_n2_3_win_2]
                evoked_N2_3_2 = epoch_N2_3_2.average()
                stc_dsmp_N2_3_2 = source_function._gen_dsmp(evoked_N2_3_2, inv, method, lambda2)
                data_n2_3 = data_n2_3 + stc_dsmp_N2_3_2.data
                count_n2_3 = count_n2_3 + 1

            # Process N2→REM transitions
            if len(idx_n2_r) == 1:  # Single transition case
                idx_n2_r_win = np.arange(idx_n2_r[0][1] + t, idx_n2_r[0][1] + t + 2)
                epoch_N2_R = raw_eeg[idx_n2_r_win]
                evoked_N2_R = epoch_N2_R.average()
                stc_dsmp_N2_R = source_function._gen_dsmp(evoked_N2_R, inv, method, lambda2)
                data_n2_r = data_n2_r + stc_dsmp_N2_R.data
                count_n2_r = count_n2_r + 1

        # Calculate mean source estimates across subjects
        if count_n2_3 > 0:
            data_mean_n = data_n2_3 / count_n2_3
            stc_mean_n = stc_dsmp_N2_3  # Use last stc as template
            stc_mean_n.data = data_mean_n

            # Save N2→N3 brain plot
            savename_n = save_path + 'figures/' + str(t) + '_mean_n.png'
            stc_mean_brain_n = source_function.brain_plot(
                stc_mean_n,
                subjects_dir,
                message=None,
                hemi='split',  # Show both hemispheres
                views=["lat", "med"],  # Lateral and medial views
                colorbar=False
            )
            stc_mean_brain_n.save_image(savename_n)

        if count_n2_r > 0:
            data_mean_r = data_n2_r / count_n2_r
            stc_mean_r = stc_dsmp_N2_R  # Use last stc as template
            stc_mean_r.data = data_mean_r

            # Save N2→REM brain plot
            savename_r = save_path + 'figures/' + str(t) + '_mean_r.png'
            stc_mean_brain_r = source_function.brain_plot(
                stc_mean_r,
                subjects_dir,
                message=None,
                hemi='split',
                views=["lat", "med"],
                colorbar=False
            )
            stc_mean_brain_r.save_image(savename_r)
