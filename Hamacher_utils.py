import numpy as np
from plotting_utils import plot_single_internal_representation
from scipy.signal import sosfiltfilt, butter

def apply_butter_LP_filter(spike_rate_matrix, binsize, cut_off_freq_Hz = 40, filter_order = 16 ):
    Fs = 1/binsize
    nyq_freq = Fs/2
    if len(spike_rate_matrix.shape) == 2:
        spike_rate_matrix_new = np.zeros(spike_rate_matrix.shape)
        for row in np.arange(spike_rate_matrix.shape[0]):
            signal_row = spike_rate_matrix[row,:]
            # b, a = butter(filter_order, cut_off_freq, 'lowpass', analog=False)
            # filtered_signal = filtfilt(b, a, signal_row, axis=0, padlen = None) #filtfilt to avoid phase delay
            sos = butter(filter_order, cut_off_freq_Hz, 'lp', fs=Fs, output='sos')
            filtered_signal = sosfiltfilt(sos, signal_row)
            # plotting
            # plt.figure()
            # plt.plot(signal_row, 'b')
            # plt.plot(filtered_signal, 'r')
            # plt.show()
            
            # # Check frequency content of filtered signal
            # from scipy import signal
            # f, t, Sxx = signal.spectrogram(filtered_signal, Fs)
            # plt.pcolormesh(t, f, Sxx, shading='gouraud')
            # plt.colorbar()
            # plt.xlabel('Time [s]')
            # plt.ylabel('Frequency [Hz]')
            # plt.title('Spectrogram of low pass filtered signal at: ' + str(cut_off_freq_Hz) + ' Hz')

            spike_rate_matrix_new[row, :] = filtered_signal
    else:
        sos = butter(filter_order, cut_off_freq_Hz, 'lp', fs=Fs, output='sos')
        spike_rate_matrix_new = sosfiltfilt(sos, spike_rate_matrix)
    
    return spike_rate_matrix_new


def compute_internal_representation(neurogram, 
                                    dt, 
                                    tau_in = 70e-3, # 70 ms
                                    tau_out = 70e-3, # 70 ms
                                    TP2_cut_off_Hz = 2000,
                                    TP2_filter_order = 1,
                                    band_type = 'critical',
                                    plot_IR=False,
                                    num_critical_bands =42):

    num_fibers, num_trials, num_samples = neurogram.get_data().shape

    t_unfiltered = np.arange(num_samples)*dt
    T_a = dt # sampling period
    Z = np.zeros(SR.shape)
    Y = np.zeros(SR.shape)
    IR = np.zeros(SR.shape)
    for n in range(num_remaining_crit_bands):
        Z_nk = 0
        for k in range(1,num_samples):
            SR_nk = SR[n,k]
            if SR_nk >=  Z_nk:
                c1 = np.exp(-T_a/tau_in)
                c2 = 1-c1
            elif SR_nk < Z_nk:
                c1 = np.exp(-T_a/tau_out)
                c2 = 0
            # Equation 8.6
            Z_nk = c1*Z_nk + c2*SR_nk 
            # Equation 8.5
            Y[n,k] = max(SR_nk, Z_nk)
            Z[n,k] = Z_nk
    # TP2
    IR = apply_butter_LP_filter(Y, T_a, cut_off_freq_Hz=TP2_cut_off_Hz, filter_order=TP2_filter_order)
    x=3
    if plot_IR:
        plot_single_internal_representation(IR, t_unfiltered, edge_frequency_critical_bands)