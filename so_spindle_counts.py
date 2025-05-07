import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import source_function
import glob
import pickle
import plot_figure
import scipy
import pandas as pd
import seaborn as sns
from itertools import chain
from test_statistics import samples_test
from test_statistics import convert_pvalue_to_asterisks
from scipy import signal
from lspopt import spectrogram_lspopt

matplotlib.rcParams['svg.fonttype'] = 'none'
cm = 1/2.54


def signal_preprocessing(signals, l_freq, h_freq,sample_rate):
    l, h = 2 * l_freq / sample_rate, 2 * h_freq / sample_rate
    b, a = signal.butter(2, [l, h], 'bandpass')
    signals = signal.filtfilt(b, a, signals)
    return signals

def SO_thre(data,sample_rate):
    maxpeak_idx = []
    minpeak_idx = []
    s = np.diff(np.sign(data))
    indx_down = np.where(s < 0)[0]  # positive to negative
    indx_up = np.where(s > 0)[0]+1  # negative to positive
        # # Make sure that the first zero crossing is from + to -
    if indx_up[0] > indx_down[0]:
       indx_up = indx_up[:-1]
       indx_down = indx_down[1:]
        # Measure the length of all intervals of postive to negative zero, crossings(t) is measured
    t = np.diff(indx_up) / sample_rate
    ev = []
    for i, t_i in enumerate(t):
        if 0.8 <= t_i <= 2:  # 0.8秒至2秒
            ev.append(i)

    for ev_i in ev:
        interval = data[indx_up[ev_i]: indx_up[ev_i + 1]]
        minpeak, maxpeak = np.min(interval), np.max(interval)
        minpeak_idx.append(minpeak)
        maxpeak_idx.append(maxpeak)
    maxpeak_idx = np.array(maxpeak_idx)
    minpeak_idx = np.array(minpeak_idx)
    thre_pp = np.percentile(maxpeak_idx[:] - minpeak_idx[:],75)
    thre_ma = np.percentile(maxpeak_idx,75)
    return thre_pp,thre_ma


def SO_thre_all_channel(data, sample_rate):
    thre_pp, thre_ma = 0, 0
    for ch in range(data.shape[0]):
        ss = list(chain.from_iterable(data[ch, :, :]))
        ss = np.array(ss) * 1000000
        so_data = signal_preprocessing(ss, l_freq=0.5, h_freq=1.25, sample_rate=sample_rate)
        thre_p, thre_m = SO_thre(so_data, sample_rate)
        thre_pp, thre_ma = thre_pp + thre_p, thre_ma + thre_m
    thre_pp, thre_ma = thre_pp / ch, thre_ma / ch
    print(f'  the threshold of SO for  channel: {thre_pp:.3f}')
    print(f'  the ma of SO for  channel: {thre_ma:.3f}')
    return thre_pp, thre_ma


def get_SO(data, thre, thre_m, sample_rate):
    '''
    Detect slow oscillations (SOs) in EEG data and analyze their characteristics.

    This function identifies SO events based on:
    1. Zero-crossings (negative to positive and vice versa)
    2. Peak-to-peak amplitude thresholds
    3. Duration criteria (0.8-2 seconds)

    Parameters:
    -----------
    data : numpy.ndarray
        EEG time series data
    thre : float
        Minimum peak-to-peak amplitude threshold (in mV/uV)
    thre_m : float
        Minimum peak amplitude threshold (in mV/uV)
    sample_rate : int
        Sampling rate of the data (in Hz)

    Returns:
    --------
    tuple: (group_SO_idx, up_duration, down_duration)
        group_SO_idx: List of [start, end] indices for each detected SO
        up_duration: List of up-state durations for each SO (in seconds)
        down_duration: List of down-state durations for each SO (in seconds)
    '''

    # Convert data to binary representation (1 for positive, -1 for negative)
    so_data = []
    for i in range(len(data)):
        if data[i] >= 0:
            so_data.append(1)
        else:
            so_data.append(-1)

    # Find zero crossings
    s = np.diff(so_data)  # Derivative of binary signal
    indx_down = np.where(s < 0)[0]  # Positive-to-negative zero crossings (down)
    indx_up = np.where(s > 0)[0] + 1  # Negative-to-positive zero crossings (up)

    # Ensure we start with a down crossing and end with up crossing
    if indx_up[0] < indx_down[0]:
        indx_up = indx_up[1:]
        indx_down = indx_down[:-1]

    # Initialize containers
    maxpeak_idx = []  # For storing peak maxima info
    minpeak_idx = []  # For storing peak minima info
    ev = []  # For storing candidate SO events
    group_SO_idx = []  # For storing validated SO events

    # Detect candidate SO events between zero crossings
    if len(indx_up) < len(indx_down):
        # Case where we have fewer up crossings
        for index in range(len(indx_up) - 1):
            # Find minimum between down crossings (trough)
            min1 = np.min(data[indx_down[index]:indx_down[index + 1]])
            min1_pos = np.where(data[indx_down[index]:indx_down[index + 1]] == min1)[0] + indx_down[index]

            # Find minimum between up crossings (next trough)
            min2 = np.min(data[indx_up[index]:indx_up[index + 1]])
            min2_pos = np.where(data[indx_up[index]:indx_up[index + 1]] == min2)[0] + indx_up[index]

            # Check if duration between troughs is 0.8-2 seconds
            if ((min2_pos[0] - min1_pos) / sample_rate >= 0.8 and (min2_pos[0] - min1_pos) / sample_rate <= 2):
                ev.append([min1_pos[0], min2_pos[0]])
    else:
        # Case where we have fewer down crossings
        for index in range(len(indx_down) - 1):
            min1 = np.min(data[indx_down[index]:indx_down[index + 1]])
            min1_pos = np.where(data[indx_down[index]:indx_down[index + 1]] == min1)[0] + indx_down[index]

            min2 = np.min(data[indx_up[index]:indx_up[index + 1]])
            min2_pos = np.where(data[indx_up[index]:indx_up[index + 1]] == min2)[0] + indx_up[index]

            if ((min2_pos[0] - min1_pos) / sample_rate >= 0.8 and (min2_pos[0] - min1_pos) / sample_rate <= 2):
                ev.append([min1_pos[0], min2_pos[0]])

    # Extract peak information for candidate events
    for i in range(len(ev)):
        interval = data[ev[i][0]:ev[i][1]].tolist()
        minpeak = data[ev[i][0]]  # Trough amplitude
        maxpeak = np.max(interval)  # Peak amplitude
        max_idx = interval.index(maxpeak)  # Peak position

        minpeak_idx.append([ev[i][0], minpeak])  # Store trough info
        maxpeak_idx.append([max_idx + ev[i][0] - 1, maxpeak])  # Store peak info

    maxpeak_idx = np.array(maxpeak_idx)
    minpeak_idx = np.array(minpeak_idx)

    # Validate SOs based on amplitude thresholds
    for j in range(len(ev)):
        # Check peak-to-peak amplitude and peak amplitude thresholds
        if (((maxpeak_idx[j, 1] - minpeak_idx[j, 1]) >= thre)) and (maxpeak_idx[j, 1] >= thre_m):
            group_SO_idx.append([ev[j][0], ev[j][1]])  # Store valid SO indices

    # Calculate up and down state durations for each SO
    up_duration = []
    down_duration = []
    for e in range(len(group_SO_idx)):
        segment = data[group_SO_idx[e][0]:group_SO_idx[e][1]]

        # Convert segment to binary representation
        sdata = []
        for s in range(len(segment)):
            if segment[s] >= 0:
                sdata.append(1)
            else:
                sdata.append(-1)

        # Find zero crossings within the SO
        ss = np.diff(sdata)
        in_down = np.where(ss < 0)[0]  # Down crossings

        # Calculate durations
        up_du = (in_down + 1) / sample_rate  # Up state duration (positive phase)
        down_du = (len(segment) - 1 - in_down) / sample_rate  # Down state duration (negative phase)

        up_duration.append(up_du)
        down_duration.append(down_du)

    return group_SO_idx, up_duration, down_duration

def stft_power(data, sf, window=2, step=0.2, band=(1, 30)):
    # Safety check
    data = np.asarray(data)
    assert step <= window
    step = 1 / sf if step == 0 else step
    # Define STFT parameters
    nperseg = int(window * sf)
    noverlap = int(nperseg - (step * sf))

    # Compute STFT and remove the last epoch
    f, t, Sxx = spectrogram_lspopt(data, sf, nperseg=nperseg, noverlap=noverlap, detrend=False)
    idx = np.logical_and(f >= band[0], f <= band[1])
    f = f[idx]
    # Compute power and interpolate
    Sxx = np.square(np.abs(Sxx[idx,:]))
    return f, t, Sxx


def spindle_thre_all_channel(data, sample_rate):
    up_thresh = []
    low_thresh = []
    for ch in range(data.shape[0]):
        ss = list(chain.from_iterable(data[ch, :, :]))
        ss = np.array(ss) * 1000000
        sp_data = signal_preprocessing(ss, l_freq=12, h_freq=16, sample_rate=sample_rate)

        uth_f, lth_f = spindle_thre_channel(sp_data, sample_rate)
        up_thresh.append(uth_f)
        low_thresh.append(lth_f)
    upper_thresh, lower_thresh = np.mean(np.array(up_thresh)), np.mean(np.array(low_thresh))
    print(f'  the threshold of upper: {upper_thresh:.3f}')
    print(f'  the threshold of lower: {lower_thresh:.3f}')
    return upper_thresh, lower_thresh

def spindle_thre_channel(data,sample_rate):
    smoothing_size = 0.35
    smoothing = np.ones(int(smoothing_size * sample_rate)) / int(smoothing_size * sample_rate)
    data_hilbert = signal.hilbert(signal.detrend(data))
    data_hilampl = np.abs(data_hilbert)  # envelope
    sig_filt_env_smooth = np.convolve(data_hilampl, smoothing, mode='same')
    upper_thresh = np.percentile(sig_filt_env_smooth, 90)
    lower_thresh = np.percentile(sig_filt_env_smooth, 70)
    return upper_thresh, lower_thresh

def merge_spindle(spindle_group,sample_rate):
    # Merge

    for i in range(0, len(spindle_group) - 1):
        if spindle_group[i + 1][0] - spindle_group[i][1] < 0.1 * sample_rate:    #gap=100ms
            if spindle_group[i + 1][1] - spindle_group[i][0] < 2 * sample_rate:
                spindle_group[i][1] = spindle_group[i + 1][1]
                spindle_group[i + 1] = [0, 0]
    spindle_group = list(filter(([0, 0]).__ne__, spindle_group))
    return spindle_group

def get_spindle_Hilbert(data, raw_data, upper_thresh, lower_thresh, sample_rate, band=(11, 16)):
    '''
    Detect sleep spindles using Hilbert transform and amplitude thresholds.

    This function identifies spindle events based on:
    1. Band-pass filtered signal envelope (using Hilbert transform)
    2. Upper and lower amplitude thresholds
    3. Duration criteria (0.5-2 seconds)
    4. Spectral power validation in spindle frequency band

    Parameters:
    -----------
    data : numpy.ndarray
        Band-pass filtered EEG data in spindle frequency range
    raw_data : numpy.ndarray
        Original unfiltered EEG data for spectral validation
    upper_thresh : float
        Upper amplitude threshold for spindle detection
    lower_thresh : float
        Lower amplitude threshold for spindle detection
    sample_rate : int
        Sampling rate of the data (in Hz)
    band : tuple, optional
        Frequency range for spindles (default: 11-16 Hz)

    Returns:
    --------
    list:
        List of [start, end] sample indices for each detected spindle
    '''

    ### Hilbert Transform for Envelope Calculation ###

    # Set smoothing window size (0.35 seconds)
    smoothing_size = 0.35
    smoothing = np.ones(int(smoothing_size * sample_rate)) / int(smoothing_size * sample_rate)

    # Calculate analytic signal using Hilbert transform
    data_hilbert = signal.hilbert(signal.detrend(data))

    # Get amplitude envelope
    data_hilampl = np.abs(data_hilbert)

    # Smooth the envelope
    sig_filt_env_smooth = np.convolve(data_hilampl, smoothing, mode='same')

    ### Threshold Detection ###

    # Find points below lower and upper thresholds
    lower_cond_met = np.where(sig_filt_env_smooth < lower_thresh)
    time_lower = lower_cond_met[0]  # Time points below lower threshold

    upper_cond_met = np.where(sig_filt_env_smooth < upper_thresh)
    time_upper = upper_cond_met[0]  # Time points below upper threshold

    # Initialize containers
    potential_spindles_s = []  # Potential spindle start times
    potential_spindles_e = []  # Potential spindle end times
    upper = []  # Segments above upper threshold

    ### Identify Candidate Spindles ###

    # Find segments where envelope stays above upper threshold for ≥0.25s
    for j in range(len(time_upper) - 1):
        alpha_min = time_upper[j]
        alpha_max = time_upper[j + 1]
        if (alpha_max - alpha_min) > 1:  # Non-consecutive points
            if ((alpha_max - alpha_min + 1) / sample_rate >= 0.25):
                upper.append([alpha_min, alpha_max])

    # Find segments where envelope stays above lower threshold
    lower = []
    for i in range(len(time_lower) - 1):
        delta_min = time_lower[i]
        delta_max = time_lower[i + 1]
        if (delta_max - delta_min > 1):  # Non-consecutive points
            lower.append([delta_min, delta_max])

    # Find segments where upper threshold region is bounded by lower threshold region
    for e in range(len(upper)):
        for s in range(len(lower)):
            if (lower[s][0] < upper[e][0]) and (lower[s][1] > upper[e][1]):
                potential_spindles_s.append(lower[s][0])
                potential_spindles_e.append(lower[s][1])

    # Get unique spindle boundaries
    potential_spindles_s = np.unique(np.array(potential_spindles_s))
    potential_spindles_e = np.unique(np.array(potential_spindles_e))

    ### Apply Duration Criteria (0.5-2 seconds) ###
    final_spindles = []
    for i in range(len(potential_spindles_s)):
        time_diff = (potential_spindles_e[i] - potential_spindles_s[i] + 1) / sample_rate
        if ((time_diff >= 0.5) and (time_diff <= 2.0)):
            start = potential_spindles_s[i]
            end = potential_spindles_e[i]
            final_spindles.append((start, end))

    spindle_group_potential = list(map(list, final_spindles))
    spindle_group_final = []

    ### Spectral Power Validation ###
    for c in range(len(spindle_group_potential)):
        # Extract spindle segment with 2s buffer on each side
        sp = data[int(spindle_group_potential[c][0]):int(spindle_group_potential[c][1])]
        start = int(spindle_group_potential[c][0]) - sample_rate * 2
        end = int(spindle_group_potential[c][1]) + sample_rate * 2

        # Handle edge cases
        if start < 0:
            start = 0
        if end > len(raw_data):
            end = len(raw_data)

        # Compute spectrogram around spindle
        f, t, Sxx = stft_power(raw_data[start:end], sample_rate,
                               window=0.5, step=0.004, band=band)

        # Compare spindle power to baseline (first and last 1s)
        Sxx_left = np.where(t >= 1)[0]
        Sxx_right = np.where(t >= (np.max(t) - 1))[0]
        Sl = np.mean(Sxx[:, 0:int(Sxx_left[0])])  # Baseline before
        SR = np.mean(Sxx[:, int(Sxx_right[0]):len(t) - 1])  # Baseline after
        S_sp = np.mean(Sxx[:, int(Sxx_left[0]):int(Sxx_right[0])])  # Spindle power

        # Only keep if spindle power > baseline power
        if S_sp > Sl and S_sp > SR:
            spindle_group_final.append(spindle_group_potential[c])

    return spindle_group_final
def get_coupling(spindle,spindle_pos, so_pos):
    '''
    Function: get_coupling
    Input:
        spindle, so: filtered signal
        spindle_pos, so_pos: spindle/SO index
    Output:
        coupling: coupling index
    '''

    coupling = []

    for i in range(len(spindle_pos)):        # peak in SO
        pp = np.max(spindle[int(spindle_pos[i][0]):int(spindle_pos[i][1])])
        pp_pos = int(np.where(spindle[int(spindle_pos[i][0]):int(spindle_pos[i][1])] == pp)[0])+int(spindle_pos[i][0])
        for j in range(len(so_pos)):
            if pp_pos >= so_pos[j][0] and pp_pos <= so_pos[j][1]:
                start = min(so_pos[j][0], spindle_pos[i][0])
                end = max(so_pos[j][1], spindle_pos[i][1])
                coupling.append([start, end])

    return coupling

class Signal:
    def __init__(self, data_path=None, load_pickle=None, filename=None):
        if load_pickle:
            f = open(filename, 'rb')
            self.__dict__.update(pickle.load(f))
            print("Read successfully！")
            return
        data_path = open(data_path, "rb")
        data = pickle.load(data_path)
        self.x = data['x']
        self.y = data['y']
        self.sample_rate = data['fs']
        self.ch_names = data['ch_names']

        self.so_data = {}
        self.spindle_data_f = {}

        self.coupling_group_f = {}  # 每个通道中每个慢波对应的时间段
        self.spindle_group_f = {}  # 每个通道中每个纺锤波对应的时间段
        self.so_group = {}  # 每个通道中每个慢波-纺锤波耦合对应的时间段

        self.ss = {}
        self.computer_data(self.x, self.y)

    def computer_data(self, data, y):
        th, th_m = SO_thre_all_channel(data, self.sample_rate)
        uth_f, lth_f = spindle_thre_all_channel(data, self.sample_rate)
        for ch in range(data.shape[0]):

            ss = list(chain.from_iterable(data[ch, :, :]))  ####全部数据
            self.ss[ch] = np.array(ss) * 1000000

            print(ch)
            self.so_data[ch] = signal_preprocessing(self.ss[ch], l_freq=0.5, h_freq=1.25,sample_rate=self.sample_rate)
            self.spindle_data_f[ch] = signal_preprocessing(self.ss[ch], l_freq=12, h_freq=16,sample_rate=self.sample_rate)  # # fast Spindle


            # so
            # th, th_m = SO_thre(self.so_data[ch], self.sample_rate)  # 阈值
            self.so_group[ch], self.so_up_t[ch], self.so_down_t[ch] = get_SO(self.so_data[ch], th, th_m,
                                                                             self.sample_rate)

            # Spindle

            spindle_group_f = get_spindle_Hilbert(self.spindle_data_f[ch], ss, uth_f, lth_f, self.sample_rate,
                                                  band=(12, 16))
            self.spindle_group_f[ch] = merge_spindle(spindle_group_f, self.sample_rate)

            self.coupling_group_f[ch] = get_coupling(self.spindle_data_f[ch],
                                                     self.spindle_group_f[ch],
                                                     self.so_group[ch])


    def save(self, savename):
        f = open(savename, 'wb')
        pickle.dump(self.__dict__, f)
        print('Saved successfully!')


def time_histogram(data, time, sample, win=30, step=0):
    '''Calculate the number of events within sliding time windows.

    Args:
        data (array): Input signal data (not directly used in current implementation)
        time (array): 2D array of event timings where each row contains [start, end] times
        sample (int): Sampling rate (samples per second)
        win (int): Window size in seconds (default: 30)
        step (int): Step size between windows in seconds (default: 0 = non-overlapping)

    Returns:
        list: Count of events occurring within each time window

    Note:
        - When step=0, uses non-overlapping windows (default behavior)
        - The 'data' parameter is currently unused but kept for potential future use
    '''
    l = 0  # Initialize window start position (in samples)
    time = np.array(time)  # Ensure input is numpy array
    count = []  # Store event counts per window

    # Slide window through the entire data length
    while l <= len(data):
        time_segment = 0  # Counter for events in current window

        # Check all events for current window
        for t in range(len(time)):
            # If event start time falls within current window
            if time[t][0] < l + win * sample and time[t][0] >= l:
                time_segment += 1

        count.append(time_segment)

        # Move window by step size (convert seconds to samples)
        if step == 0:
            l += win * sample  # Non-overlapping windows
        else:
            l += step * sample  # Sliding window with specified step

    print('Time histogram calculation completed')
    return count


def N2_3_R_num(file,type='sp', win=10):
    '''sp:spindle
       so:SO
       co:so-spindle coupling'''

    count_sub_r = []
    count_sub_n = []
    count_sub_r_win = []
    count_sub_n_win = []
    for z in range(len(glob.glob(file))):
        filename = glob.glob(file)[z]
        print(z, filename)
        self = Signal(load_pickle=True, filename=filename)
        num = self.y
        count_30 = []
        count_5 = []

        if type == 'sp':
            data = self.spindle_data_f
            group = self.spindle_group_f

        elif type == 'so':
            data = self.so_data
            group = self.so_group

        elif type == 'co':
            data = self.so_data
            group = self.coupling_group_f

        for ch in range(len(data)):
            count = time_histogram(data[ch], group[ch],
                                   self.sample_rate, win=30, step=30)
            count_win = time_histogram(data[ch], group[ch],
                                       self.sample_rate, win=10, step=1)
            print(len(count), len(count_win))

            count_30.append(count)
            count_5.append(count_win)

        idx_n2_3, idx_n2_r = plot_figure.index_N2_N3_REM(num)

        if len(idx_n2_3) == 0:
            N2_3_num_mean = np.zeros(63)
        elif len(idx_n2_3) == 1:
            index = np.arange(idx_n2_3[0][0], idx_n2_3[0][1] + 1)
            N2_3_num = np.array(count_30)[:, index]

            index_win = np.arange((idx_n2_3[0][1] - win) * 30 - 1, (idx_n2_3[0][1] + 10) * 30)
            N2_3_win = np.array(count_5)[:, index_win]
            N2_3_num_mean = np.mean(N2_3_num, axis=1)
            if N2_3_win.shape[1] >= 900:
                count_sub_n_win.append(N2_3_win[:, -900:])
        else:
            index_1 = np.arange(idx_n2_3[0][0], idx_n2_3[0][1] + 1)
            N2_3_num_1 = np.array(count_30)[:, index_1]
            index_2 = np.arange(idx_n2_3[1][0], idx_n2_3[1][1] + 1)
            N2_3_num_2 = np.array(count_30)[:, index_2]
            N2_3_num_mean = (np.mean(N2_3_num_1, axis=1) + np.mean(N2_3_num_2, axis=1)) / 2

            index_win_1 = np.arange((idx_n2_3[0][1] - win) * 30 - 1, (idx_n2_3[0][1] + 10) * 30)
            N2_3_win_1 = np.array(count_5)[:, index_win_1]
            index_win_2 = np.arange((idx_n2_3[1][1] - win) * 30 - 1, (idx_n2_3[1][1] + 10) * 30)
            N2_3_win_2 = np.array(count_5)[:, index_win_2]
            if N2_3_win_1.shape[1] >= 900:
                count_sub_n_win.append(N2_3_win_1[:, -900:])
            if N2_3_win_2.shape[1] >= 900:
                count_sub_n_win.append(N2_3_win_2[:, -900:])
        # N2_3_win_mean = np.mean(N2_3_win, axis=1)

        if len(idx_n2_r) == 0:
            N2_R_num_mean = np.zeros(63)
        else:
            index = np.arange(idx_n2_r[0][0], idx_n2_r[0][1] + 1)
            N2_R_num = np.array(count_30)[:, index]
            index_win = np.arange((idx_n2_r[0][0] - win) * 30 - 1, (idx_n2_r[0][1] + 10) * 30)
            N2_R_win = np.array(count_5)[:, index_win]
            N2_R_num_mean = np.mean(N2_R_num, axis=1)
            # N2_R_win_mean = np.mean(N2_R_win,axis=1)
            if N2_R_win.shape[1] >= 900:
                count_sub_r_win.append(N2_R_win[:, -900:])
        print(len(count_sub_n_win), len(count_sub_r_win))

        count_sub_r.append(N2_R_num_mean)
        count_sub_n.append(N2_3_num_mean)

    count_sub_r = np.array(count_sub_r) / (30 / 60)  #####density (./s)
    count_sub_n = np.array(count_sub_n) / (30 / 60)
    count_sub_r_win = np.array(count_sub_r_win) / (10 / 60)
    count_sub_n_win = np.array(count_sub_n_win) / (10 / 60)
    return count_sub_n, count_sub_r, count_sub_r_win, count_sub_n_win


def density_win_plot(count_sub_r_win, count_sub_n_win, ax, ybottom=None, ytop=None, type='so'):
    """
    Plot temporal dynamics of sleep events around N2 transitions (5 minutes before/after).

    Visualizes how event densities change during:
    - N2 to N3 transitions (dark teal)
    - N2 to REM transitions (light teal)
    with shaded error bands representing standard error.

    Args:
        count_sub_r_win (ndarray): REM transition densities [subjects × channels × timepoints]
        count_sub_n_win (ndarray): N2/N3 transition densities [subjects × channels × timepoints]
        ax (matplotlib.axes): Axes object to plot on
        ybottom (float, optional): Minimum y-axis value
        ytop (float, optional): Maximum y-axis value
        type (str): Event type - 'so' (slow oscillations), 'sp' (spindles), 'co' (couplings)

    Returns:
        None (modifies ax in place)
    """

    # Set y-axis label based on event type
    if type == 'sp':
        y_label = 'Spindle\n density (./min)'
    elif type == 'so':
        y_label = 'SO\n density (./min)'
    elif type == 'co':
        y_label = 'SO-spindle coupling\n density (./min)'

    # Calculate statistics for REM transitions:
    # 1. Mean across channels for each subject
    # 2. Then mean across subjects
    # 3. Standard error of the mean (SEM) across subjects
    mean_r = np.mean(np.mean(count_sub_r_win, axis=0), axis=0)
    std_r = scipy.stats.sem(np.mean(count_sub_r_win, axis=1), axis=0)

    # Calculate statistics for N2/N3 transitions (same process)
    mean_n = np.mean(np.mean(count_sub_n_win, axis=0), axis=0)
    std_n = scipy.stats.sem(np.mean(count_sub_n_win, axis=1), axis=0)

    # Create time axis (-5 to +5 minutes around transition)
    t = np.linspace(-5, 5, len(mean_n))

    # Format plot aesthetics
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Plot N2-N3 transition data
    ax.plot(t, mean_n, '#3f6561', linewidth=1.5, label='N2 to N3')
    ax.fill_between(t, mean_n - std_n, mean_n + std_n,
                    alpha=0.8, color='#c5d1d2')

    # Plot N2-REM transition data
    ax.plot(t, mean_r, '#7bc4c5', linewidth=1.5, label='N2 to REM')
    ax.fill_between(t, mean_r - std_r, mean_r + std_r,
                    alpha=0.8, color='#d9eeee')

    # Configure axes
    ax.set_ylim(ybottom, ytop)
    ax.set_yticks([0, ytop])  # Show only min/max y-values
    ax.set_xticks([])  # Hide x-ticks by default

    # Add labels
    ax.set_ylabel(y_label)
    ax.margins(x=0)  # Remove horizontal margins

    # Special formatting per event type
    if type == 'so':
        # Position legend above plot
        ax.legend(loc='center', bbox_to_anchor=(0.5, 1.1), frameon=False)
    if type == 'co':
        # Show time markers for coupling plots
        ax.set_xlabel('Time (min)')
        ax.set_xticks([-5, 0, 5])


def density_test_plot(count_sub_r, count_sub_n, ax, type='so', color=None):
    """Plot and statistically compare event densities between N2-N3 and N2-REM transitions.

    Creates boxplots with statistical annotations to compare:
    - Mean event densities during N2-to-N3 transitions vs
    - Mean event densities during N2-to-REM transitions

    Args:
        count_sub_r (array): Density values for REM transitions (subjects × channels)
        count_sub_n (array): Density values for N2/N3 transitions (same shape as count_sub_r)
        ax (matplotlib.axes): Axes object to draw the plot
        type (str): Event type - 'so' (slow oscillations), 'sp' (spindles), or 'co' (couplings)
        color (list): Optional custom colors for the boxplots [N2N3_color, N2REM_color]

    Returns:
        None (modifies the input ax directly)
    """

    # Set labels and titles based on event type
    global y_label
    if type == 'sp':
        y_label = 'Spindle\n density (./min)'
        title = 'N2_Spindle_numbers'
    elif type == 'so':
        y_label = 'SO\n density (./min)'
        title = 'N2_SO_numbers'
    elif type == 'co':
        y_label = 'SO-spindle coupling\n density (./min)'
        title = 'N2_CO_numbers'

    # Calculate mean density across channels for each subject
    count_mean_r = np.mean(count_sub_r, axis=1)  # Mean across channels (REM)
    count_mean_n = np.mean(count_sub_n, axis=1)  # Mean across channels (N2/N3)

    # Prepare data for seaborn plotting
    data_dict = {'stages': [], 'Value': []}
    name_list = ['N2$_{N3}$', 'N2$_{REM}$']
    data = [count_mean_n, count_mean_r]

    # Reshape data into long format
    for stage in range(len(name_list)):
        for v in range(len(data[0])):
            data_dict['stages'].append(name_list[stage])
            data_dict['Value'].append(data[stage][v])

    # Set color palette
    if color is None:
        color_pal = sns.color_palette('muted', 2)  # Default seaborn colors
    else:
        color_pal = sns.color_palette([color[0], color[1]])  # Custom colors

    # Create DataFrame for plotting
    df = pd.DataFrame(data_dict, columns=['stages', 'Value'])

    # Define comparison pairs for statistical testing
    pairs = [('N2$_{N3}$', 'N2$_{REM}$')]
    order = ['N2$_{N3}$', 'N2$_{REM}$']  # Plot order

    # Perform statistical test (assuming samples_test returns t-statistic and p-value)
    t, p = samples_test(count_mean_n, count_mean_r, alpha=0.05)
    annotations = [convert_pvalue_to_asterisks(p)]  # Convert p-value to stars

    # Generate the plot using custom annotation function
    source_function.annotation_plot_(
        df,
        x='stages',
        y='Value',
        annotations=annotations,
        color=color_pal,
        line_color=color_pal,
        name_list=name_list,
        ax=ax,
        pairs=pairs,
        y_label=y_label,
        types='box',
        box_alpha=0.8  # Boxplot transparency
    )