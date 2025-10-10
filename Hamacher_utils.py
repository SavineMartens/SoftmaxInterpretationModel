import numpy as np
from plotting_utils import plot_single_internal_representation
from scipy.signal import sosfiltfilt, butter, correlate
from utilities import *
from random import gauss
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os 
import yaml
import glob



def apply_butter_LP_filter(spike_rate_matrix, binsize, cut_off_freq_Hz = 40, filter_order = 16 ):
    Fs = 1/binsize
    
    if len(spike_rate_matrix.shape) == 2:
        spike_rate_matrix_new = np.zeros(spike_rate_matrix.shape)
        for row in np.arange(spike_rate_matrix.shape[0]):
            signal_row = spike_rate_matrix[row,:]
            sos = butter(filter_order, cut_off_freq_Hz, 'lp', fs=Fs, output='sos')
            filtered_signal = sosfiltfilt(sos, signal_row)
            # plotting
            # plt.figure()
            # plt.plot(signal_row, 'b')
            # plt.plot(filtered_signal, 'r')
            # plt.show()
            

            spike_rate_matrix_new[row, :] = filtered_signal
    else:
        sos = butter(filter_order, cut_off_freq_Hz, 'lp', fs=Fs, output='sos')
        spike_rate_matrix_new = sosfiltfilt(sos, spike_rate_matrix)
    
    return spike_rate_matrix_new


def select_critical_bands(spike_rate_matrix, fiber_frequencies, type='single', num_critical_bands = 42, number_of_fibers = 10):
    # taken from: https://www.sfu.ca/sonic-studio-webdav/handbook/Appendix_E.htmL changed edge 0 >20 according to https://en.wikipedia.org/wiki/Bark_scale
    if num_critical_bands == 24: # Bark scale
        centre_frequency_critical_bands =np.array([50, 150, 250, 350, 450, 570, 700, 840, 1000, 1170, 1370, 1600, 
                                                    1850, 2150, 2500, 2900, 3400, 4000, 4800, 5800, 7000, 8500,10500, 13500])
        edge_frequency_critical_bands = np.array([20, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 
                                        2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500])
    elif num_critical_bands == 42: # Hamacher
        edges_mm = np.arange(0.25, 32.5, 0.75)[::-1]
        edge_frequency_critical_bands = Greenwood_function_mm_to_f(edges_mm) # or should the max be max(edges_mm)
        centre_frequency_critical_bands = edge_frequency_critical_bands[:-1] + np.diff(edge_frequency_critical_bands)/2
    elif num_critical_bands == 85:
        edges_mm = np.arange(0.25, 32.5, 0.75/2)[::-1]
        edge_frequency_critical_bands = Greenwood_function_mm_to_f(edges_mm) # or should the max be max(edges_mm)
        centre_frequency_critical_bands = edge_frequency_critical_bands[:-1] + np.diff(edge_frequency_critical_bands)/2
    elif num_critical_bands == 171:
        edges_mm = np.arange(0.25, 32.5, 0.75/4)[::-1]
        edge_frequency_critical_bands = Greenwood_function_mm_to_f(edges_mm) # or should the max be max(edges_mm)
        centre_frequency_critical_bands = edge_frequency_critical_bands[:-1] + np.diff(edge_frequency_critical_bands)/2
    else:
        raise ValueError('Number of critical bands has to be either 24, 42, 85')
    start_critical_band = find_closest_index(centre_frequency_critical_bands, fiber_frequencies[0]) # Bruce's lowest possible frequency as input is 125    
    end_critical_band = find_closest_index(edge_frequency_critical_bands, fiber_frequencies[-1]) # was centre_frequency_critical_bands
    if num_critical_bands in [42, 85]:
        # end_critical_band -= 1 # centre frequency is otherwise exactly at edge Bruce matrix
        start_critical_band += 1
    list_frequency_idx = []
    new_fiber_frequencies_CF = []
    remaining_frequency_edges = edge_frequency_critical_bands[start_critical_band: end_critical_band+1]
    centre_remaining_frequency_bands = centre_frequency_critical_bands[start_critical_band: end_critical_band]
    if type == 'single':
        for i in range(start_critical_band, len(centre_frequency_critical_bands[start_critical_band:end_critical_band+1])):
            centre_frequency = centre_frequency_critical_bands[i]
            idx = (np.abs(fiber_frequencies - centre_frequency)).argmin()
            list_frequency_idx.append(idx)
            new_fiber_frequencies_CF.append(fiber_frequencies[idx])
        selected_spikes = spike_rate_matrix[list_frequency_idx,:]
    if type == 'band_x':
        new_matrix = np.zeros((len(centre_remaining_frequency_bands), spike_rate_matrix.shape[1]))
        for i in range(start_critical_band, len(centre_frequency_critical_bands[start_critical_band:end_critical_band+1])):
            centre_frequency = centre_frequency_critical_bands[i]
            idx = (np.abs(fiber_frequencies - centre_frequency)).argmin()
            list_frequency_idx.append(idx)
            new_fiber_frequencies_CF.append(fiber_frequencies[idx])
            new_matrix[i-start_critical_band,:] = np.sum(spike_rate_matrix[idx-number_of_fibers:idx+number_of_fibers,:], axis=0)
        selected_spikes = new_matrix
    if type == 'entire_band':
        new_matrix = np.zeros((len(centre_remaining_frequency_bands), spike_rate_matrix.shape[1]))
        for i in range(start_critical_band, len(centre_frequency_critical_bands[start_critical_band:end_critical_band+1])):
            low_frequency = edge_frequency_critical_bands[i]
            high_frequency = edge_frequency_critical_bands[i+1]
            low_idx = (np.abs(fiber_frequencies - low_frequency)).argmin()
            idx = (np.abs(fiber_frequencies - centre_frequency_critical_bands[i])).argmin()
            high_idx = (np.abs(fiber_frequencies - high_frequency)).argmin()
            list_frequency_idx.append(idx)
            new_fiber_frequencies_CF.append(fiber_frequencies[idx])
            new_matrix[i-start_critical_band,:] = np.sum(spike_rate_matrix[low_idx:high_idx,:], axis=0)
            # print(i, ':', high_idx-low_idx, 'fibers')
        selected_spikes = new_matrix
    x=3
    return selected_spikes, new_fiber_frequencies_CF, centre_remaining_frequency_bands, remaining_frequency_edges


def compute_internal_representation_from_object(neurogram, 
                                                fiber_frequencies,
                                                tau_in = 70e-3, # 70 ms
                                                tau_out = 70e-3, # 70 ms
                                                TP2_cut_off_Hz = 2000,
                                                TP2_filter_order = 1,
                                                band_type = 'critical',
                                                critical_band_type = 'entire_band', 
                                                plot_IR=False,
                                                num_critical_bands =42):

    dt = neurogram.bin_width
    num_fibers, num_trials, num_samples = neurogram.get_output().shape
    spike_matrix = neurogram.get_output().mean(axis=1) # average across trials
    # if still in 3 dimensions
    if len(spike_matrix.shape) == 3:
        spike_matrix = np.squeeze(spike_matrix)
    # to critical bands
    SR, _, _, edge_frequency_critical_bands = select_critical_bands(spike_matrix, fiber_frequencies, type=critical_band_type, num_critical_bands=num_critical_bands)

    # downsample
    Fs_neurogram = 1/dt
    Fs_Hamacher = 5e3 # kHz
    if Fs_Hamacher>Fs_neurogram:
        raise ValueError('new Fs must be smaller than 100 000 Hz')
    Fs_ratio = int(Fs_neurogram/Fs_Hamacher) # when new_Fs = 5000 --> same as Hamacher downsampling so same matrix as spike_matrix in matfile
    SR = SR[:, ::Fs_ratio]
    num_remaining_crit_bands, num_samples = SR.shape
    
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

    return IR


def compute_internal_representation_from_numpy(neurogram, 
                                               Fs_neurogram,
                                            fiber_frequencies,
                                            tau_in = 70e-3, # 70 ms
                                            tau_out = 70e-3, # 70 ms
                                            TP2_cut_off_Hz = 2000,
                                            TP2_filter_order = 1,
                                            band_type = 'critical',
                                            critical_band_type = 'entire_band', 
                                            plot_IR=False,
                                            num_critical_bands =42):

    dt = 1/Fs_neurogram
    num_fibers, num_trials, num_samples = neurogram.shape
    spike_matrix = neurogram.mean(axis=1) # average across trials
    # if still in 3 dimensions
    if len(spike_matrix.shape) == 3:
        spike_matrix = np.squeeze(spike_matrix)

    SR, _, _, edge_frequency_critical_bands = select_critical_bands(spike_matrix, fiber_frequencies, type=critical_band_type, num_critical_bands=num_critical_bands)

    # downsample
    Fs_Hamacher = 5e3 # kHz
    if Fs_Hamacher>Fs_neurogram:
        raise ValueError('new Fs must be smaller than 100 000 Hz')
    Fs_ratio = int(Fs_neurogram/Fs_Hamacher) # when new_Fs = 5000 --> same as Hamacher downsampling so same matrix as spike_matrix in matfile
    SR = SR[:, ::Fs_ratio]

    num_remaining_crit_bands, num_samples = SR.shape

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

    return IR


def  get_Hamacher_IR_from_numpy(fname,
                                fiber_IDs,
                                frequencies,
                                tau_in = 70e-3, # 70 ms
                                tau_out = 70e-3, # 70 ms
                                TP2_cut_off_Hz = 2000,
                                TP2_filter_order = 1,
                                band_type = 'critical',
                                plot_IR=False,
                                num_critical_bands =42,):
    spike_times = np.load(fname, allow_pickle=True) 
    # select partial fibers
    spike_times = spike_times[fiber_IDs,:]
    # print(spike_times.shape )
    [num_fibers, num_trials] = spike_times.shape 
    fname_clean = os.path.basename(os.path.realpath(fname)).replace('.npy','')
    output_dir = os.path.dirname(os.path.realpath(fname)) 
    time_stamp = fname_clean[:fname_clean.find('spike')]
    f_config = glob.glob(os.path.join(output_dir, time_stamp + "config_output*.yaml"))[0]
    with open(f_config, "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    start_bin = config['binsize']
    binsize = config['binsize']
    # spike_matrix = spike_rates*binsize
    try:
        PW = config['pulse_width']
    except:
        PW = 18e-6
    try:
        sound_duration = config['sound_duration']
    except:
        latest_list=[]
        for fiber in np.arange(spike_times.shape[0]):
            for trial in np.arange(num_trials):
                latest_list.append(np.max(spike_times[fiber][trial]))
        sound_duration =  round(np.max(latest_list), 2) #round(np.max(latest_list), int("{:e}".format(new_Fs)[-2:]))
        # print('Assuming sound duration is 1.0 s')
    # print(sound_duration)
    old_Fs = 1/PW
    new_Fs = 5000
    new_binsize = 1/new_Fs
    spike_matrix = calculate_bin_spikes(spike_times, binsize=new_binsize, sound_duration=sound_duration)
    # if unfiltered_type == 'Hamacher':
        # sigma_samples = 
        # spike_matrix = gaussian_filter1d
    _, _, _, _, _, _, _, _, fiber_frequencies, _ = load_mat_virtual_all_thresholds(config['virtual_thresholds_file'], nerve_model_type=3, state=1, array_type =2)

    for fiber in range(spike_matrix.shape[0]):
        spike_matrix[fiber,:] = discrete_gaussian_filter(spike_matrix[fiber,:], PW=18e-6)

    SR, _, _, edge_frequency_critical_bands = select_critical_bands(spike_matrix, frequencies, type='entire_band', num_critical_bands=num_critical_bands, number_of_fibers = num_fibers)
    num_remaining_crit_bands, num_samples = SR.shape
        
    t_unfiltered = np.arange(num_samples)*PW
    T_a = PW # sampling period
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
        plt.suptitle(config['sound_name'])

    return spike_matrix, IR #, t_unfiltered, edge_frequency_critical_bands


def get_Hamacher_NIR(IR, sigma):
    num_bands, num_samples = IR.shape
    # noise = np.zeros(num_samples)
    NIR = np.zeros(IR.shape)
    for c in range(num_bands):
        for s in range(num_samples):
            NIR[c,s] = IR[c,s] + gauss(mu=0, sigma=sigma) # NIR = X + N_w
    x=3
    return  NIR 

def iterate_3AFC_memory_softmax_correlation(IR_RT, IR_R, S, sigma_w, temperature, measure='pearson', n_iter=100, use_De=False, norm_bool=False, use_differences = True):
    num_remaining_crit_bands, _ = IR_R.shape
    probabilities = np.zeros((n_iter, 3))

    plot_X = False
    use_differences = True

    if plot_X:
        while is_prime(num_remaining_crit_bands):
            num_remaining_crit_bands += 1
        row_plot, column_plot = closestDivisors(num_remaining_crit_bands)
        fig, ax = plt.subplots(row_plot, column_plot, sharey=True, sharex=True)
        axes = ax.flatten()

    for i in range(n_iter):
        NIR_RT = get_Hamacher_NIR(IR_RT, sigma=sigma_w)
        # S = IR_RT - IR_R 
        X_RT = NIR_RT #- IR_R
        
        # create 2 alternatives
        NIR_R1 = get_Hamacher_NIR(IR_R, sigma=sigma_w)
        X_R1 = NIR_R1 #-IR_R
        NIR_R2 = get_Hamacher_NIR(IR_R, sigma=sigma_w)
        X_R2 = NIR_R2 #-IR_R

        score_matrix = np.zeros((num_remaining_crit_bands, 3))
        for c in range(num_remaining_crit_bands):
            S_k = S[c,:]
            # Eq. 9.11
            D_k = S_k.copy()
            E_D = sum(D_k**2)
            D_e_k = D_k/np.sqrt(E_D) # so this value does not work for 20 RPO vs 20 RPO without noise          
            if use_De:
                S_k = D_e_k

            if plot_X:
                if i == n_iter-1:
                    # plot comparison:
                    tranparency = 0.3
                    axes[c].plot(S_k, '--', label='S')
                    axes[c].plot(X_RT[c,:], '--', alpha=tranparency, label='X_RT') 
                    axes[c].plot(X_R1[c,:], '--', alpha=tranparency, label='X_R1') 
                    axes[c].plot(X_R2[c,:], '--', alpha=tranparency, label='X_R2')
                    axes[c].set_title('band ' + str(c+1))

            if measure == 'pearson':  # becomes all nan because of zero division with 20 RPO
                measure_RT = np.corrcoef(S_k, X_RT[c,:])[1,0]
                measure_R1 = np.corrcoef(S_k, X_R1[c,:])[1,0] 
                measure_R2 = np.corrcoef(S_k, X_R2[c,:])[1,0] 
                # if np.isnan(measure_RT):
                    # breakpoint()

            elif measure == 'xcorr': # becomes all zeroes with 20 RPO
                measure_RT = max(correlate(S_k, X_RT[c,:]))
                measure_R1 = max(correlate(S_k, X_R1[c,:]))
                measure_R2 =  max(correlate(S_k, X_R2[c,:]))

            elif measure == 'rsquared': # becomes all nan because of zero division with 20 RPO
                measure_RT = r2_score(S_k, X_RT[c,:])
                measure_R1 = r2_score(S_k, X_R1[c,:])
                measure_R2 =  r2_score(S_k, X_R2[c,:])

            if use_differences:
                # RT-R1
                diff_RTR1 = measure_RT - measure_R1
                # RT-R2
                diff_RTR2 = measure_RT - measure_R2
                # R1-R2
                diff_R1R2 = measure_R1 - measure_R2
                # no content in higher bands so replace correlation with 0 
                score_matrix[c, :] = np.nan_to_num([np.mean((diff_RTR1, diff_RTR2)), diff_R1R2, diff_R1R2], nan=0.0)
                # MAYBE I SHOULD JUST DELETE THESE ROWS
            else:
                # measures per critical band
                score_matrix[c, :] = np.nan_to_num([measure_RT, measure_R1, measure_R2], nan=0.0) # replace nan with 0 

        if norm_bool:
            # Subtract the maximum score for numerical stability 
            # This prevents overflow in the exponentiation step
            max_score = np.max(score_matrix, axis=1, keepdims=True) 
            score_matrix -= max_score 

        # Apply softmax formula
        expScores = np.exp(-score_matrix / temperature) #Using negative to invert the effect, lower MI -> higher score
        probabilities[i,:] = 1 - (np.nanmean(expScores / np.sum(expScores, axis=1, keepdims=True),axis=0))
        x=3

    if plot_X:
        # Put a legend to the right of the current axis
        font_size = 20
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fig.text(0.08, 0.35, 'Internal Representations (IR)', ha='center', rotation='vertical', fontsize=font_size)
        fig.text(0.5, 0.04, 'Time [s]', ha='center', fontsize=font_size)

    x=3
    return np.mean(probabilities[:,0])


def Hamacher_3AFC(IR_RT, IR_R, S, sigma_w, temperature, measure='pearson', n_iter=100, use_De=False, norm_bool=False, use_differences = True):
    num_remaining_crit_bands, _ = IR_R.shape
    probabilities = np.zeros((n_iter, 3))

    plot_X = False
    use_differences = True

    if plot_X:
        while is_prime(num_remaining_crit_bands):
            num_remaining_crit_bands += 1
        row_plot, column_plot = closestDivisors(num_remaining_crit_bands)
        fig, ax = plt.subplots(row_plot, column_plot, sharey=True, sharex=True)
        axes = ax.flatten()

    for i in range(n_iter):
        NIR_RT = get_Hamacher_NIR(IR_RT, sigma=sigma_w)
        # S = IR_RT - IR_R 
        X_RT = NIR_RT #- IR_R
        
        # create 2 alternatives
        NIR_R1 = get_Hamacher_NIR(IR_R, sigma=sigma_w)
        X_R1 = NIR_R1 #-IR_R
        NIR_R2 = get_Hamacher_NIR(IR_R, sigma=sigma_w)
        X_R2 = NIR_R2 #-IR_R

        score_matrix = np.zeros((num_remaining_crit_bands, 3))
        for c in range(num_remaining_crit_bands):
            S_k = S[c,:]
            # Eq. 9.11
            D_k = S_k.copy()
            E_D = sum(D_k**2)
            D_e_k = D_k/np.sqrt(E_D) # so this value does not work for 20 RPO vs 20 RPO without noise          
            if use_De:
                S_k = D_e_k

            if plot_X:
                if i == n_iter-1:
                    # plot comparison:
                    tranparency = 0.3
                    axes[c].plot(S_k, '--', label='S')
                    axes[c].plot(X_RT[c,:], '--', alpha=tranparency, label='X_RT') 
                    axes[c].plot(X_R1[c,:], '--', alpha=tranparency, label='X_R1') 
                    axes[c].plot(X_R2[c,:], '--', alpha=tranparency, label='X_R2')
                    axes[c].set_title('band ' + str(c+1))

            if measure == 'pearson':  # becomes all nan because of zero division with 20 RPO
                measure_RT = np.corrcoef(S_k, X_RT[c,:])[1,0]
                measure_R1 = np.corrcoef(S_k, X_R1[c,:])[1,0] 
                measure_R2 = np.corrcoef(S_k, X_R2[c,:])[1,0] 
                # if np.isnan(measure_RT):
                    # breakpoint()

            elif measure == 'xcorr': # becomes all zeroes with 20 RPO
                measure_RT = max(correlate(S_k, X_RT[c,:]))
                measure_R1 = max(correlate(S_k, X_R1[c,:]))
                measure_R2 =  max(correlate(S_k, X_R2[c,:]))

            elif measure == 'rsquared': # becomes all nan because of zero division with 20 RPO
                measure_RT = r2_score(S_k, X_RT[c,:])
                measure_R1 = r2_score(S_k, X_R1[c,:])
                measure_R2 =  r2_score(S_k, X_R2[c,:])

            if use_differences:
                # RT-R1
                diff_RTR1 = measure_RT - measure_R1
                # RT-R2
                diff_RTR2 = measure_RT - measure_R2
                # R1-R2
                diff_R1R2 = measure_R1 - measure_R2
                # no content in higher bands so replace correlation with 0 
                score_matrix[c, :] = np.nan_to_num([np.mean((diff_RTR1, diff_RTR2)), diff_R1R2, diff_R1R2], nan=0.0)
                # MAYBE I SHOULD JUST DELETE THESE ROWS
            else:
                # measures per critical band
                score_matrix[c, :] = np.nan_to_num([measure_RT, measure_R1, measure_R2], nan=0.0) # replace nan with 0 

        if norm_bool:
            # Subtract the maximum score for numerical stability 
            # This prevents overflow in the exponentiation step
            max_score = np.max(score_matrix, axis=1, keepdims=True) 
            score_matrix -= max_score 

        # Apply softmax formula
        expScores = np.exp(-score_matrix / temperature) #Using negative to invert the effect, lower MI -> higher score
        probabilities[i,:] = 1 - (np.nanmean(expScores / np.sum(expScores, axis=1, keepdims=True),axis=0))
        x=3

    if plot_X:
        # Put a legend to the right of the current axis
        font_size = 20
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fig.text(0.08, 0.35, 'Internal Representations (IR)', ha='center', rotation='vertical', fontsize=font_size)
        fig.text(0.5, 0.04, 'Time [s]', ha='center', fontsize=font_size)

    x=3
    return np.mean(probabilities[:,0])