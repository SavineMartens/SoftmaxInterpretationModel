import numpy as np 
import math
import datetime
from decimal import *
import platform
import scipy.signal
import scipy.io
from scipy.io import loadmat
import os
import sys
import datetime
import scipy
import random
import matplotlib.pyplot as plt
from pymatreader import read_mat

def closestDivisors(n):
    a = round(math.sqrt(n))
    while n%a > 0: a -= 1
    row = min(a,n//a)
    column = max(a,n//a)
    return row, column

def is_prime(n):
  for i in range(2,n):
    if (n%i) == 0:
      return False
  return True

def find_closest_index(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def Greenwood_function_mm_to_f(mm, max_Ln=35, A = 165.4, alpha = 2.1, k = 0.88):
    if hasattr(mm, "__len__"): # if vector
        f = []
        for m in mm:
            rel_mm = (max_Ln-m)/max_Ln
            f.append(A*(10**(alpha*rel_mm)-k))
    else: # if scalar
        rel_mm = (max_Ln-mm)/max_Ln
        f = A*(10**(alpha*rel_mm)-k)
    return f

def get_time_str(seconds=False):
    if seconds:
        now = datetime.datetime.now()
        sec = float("%d.%d" % (now.second, now.microsecond)) 
        sec = float(Decimal(sec).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
        sec = 'm' + str(sec) + 's'
        now = now.replace(second=0, microsecond=0) 
    else:
        now = datetime.datetime.now().replace(second=0, microsecond=0)
        sec = ''
    now = str(now).replace(':00','')
    now = now.replace(':','h')
    str_now = now.replace(' ', '_')
    str_now += sec
    return str_now

from scipy.optimize import curve_fit
def sigmoid(x, L ,x0, k, b):
    # L is responsible for scaling the output range from [0,1] to [0,L]
    # b adds bias to the output and changes its range from [0,L] to [b,L+b]
    # k is responsible for scaling the input, which remains in (-inf,inf)
    # x0 is the point in the middle of the Sigmoid, i.e. the point where Sigmoid should originally output the value 1/2 [since if x=x0, we get 1/(1+exp(0)) = 1/2].
    b = max(33, b)
    L = min(100-b, L)
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)

def fit_sigmoid(xdata, ydata):
            # L       x0            k  b 
    p0 =    [100-33, np.median(xdata), 1, 33] #[max(ydata), np.median(xdata), 1, min(ydata)] # this is an mandatory initial guess
    # bounds = ((33),(100))
    popt, pcov = curve_fit(sigmoid, xdata, ydata, p0, method='dogbox', maxfev=1e6)
    y = sigmoid(xdata, *popt)
    return y

def bounded_sigmoid(x, y, x0, k):
    b = max(33, b)
    L = min(100-b, L)
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return y

def fit_bounded_sigmoid(xdata, ydata):
    p0 = [np.median(xdata), 1]
    popt, pcov = curve_fit(bounded_sigmoid, xdata, ydata, p0, method='dogbox', maxfev=1e6)
    y = bounded_sigmoid(xdata, *popt)
    return y

def calculate_bin_spikes(spike_times, binsize, sound_duration):
    '''Takes spikes as input with trials still separated and gives spike rates as vector'''
    num_fibers, num_trials_with_spikes = spike_times.shape
    spike_matrix = np.zeros((num_fibers, int(np.ceil(sound_duration/binsize))+1))
    # print('shape:', spike_times.shape)
    if num_trials_with_spikes >1:
        for fiber in range(num_fibers):
            spike_times_vector = []
            for trial in range(num_trials_with_spikes):
                spike_times_vector.extend(spike_times[fiber][trial])
            spikes_idx = np.floor(np.asarray(spike_times_vector)/binsize).astype(int)
            for idx in spikes_idx:
                spike_matrix[fiber,idx] += 1
        return spike_matrix
    else:
        for fiber in np.arange(num_fibers):
            # print(fiber)
            spike_times_vector = []
            for trial in np.arange(num_trials_with_spikes):
                if len(spike_times[trial])!=0:
                    spike_times_vector.extend(spike_times[trial])
                else:
                    print('No spikes in trial for this fiber')
                    breakpoint()
            spike_times_vector = np.array(spike_times_vector)
            spikes_idx = np.floor(spike_times_vector/binsize).astype(int)
            if len(np.unique(spikes_idx)) != len(np.squeeze(spikes_idx)):
                # print('More spikes in one bin')
                vals, counts = np.unique(spikes_idx, return_counts=True)
                for val_i, val in enumerate(vals):
                    spike_matrix[fiber,val] = counts[val_i]
            else:
                spike_matrix[fiber,spikes_idx] = 1

        return spike_matrix
    
def discrete_gaussian_filter(signal, PW, sigma_c=1e-3):
    # Ts = 1/Fs # [/s]
    Ts = PW
    sigma_k = np.ceil(sigma_c/Ts) # [samples]
    N_g = 6*sigma_k  #  highly scientific
    # alpha = (N_g-1)/(2*sigma_k) # https://nl.mathworks.com/help/signal/ref/gausswin.html
    window = (1/(np.sqrt(2*np.pi)*sigma_k))* scipy.signal.windows.gaussian(N_g, sigma_k) 
    filtered_signal = np.convolve(window, signal, 'same')
    return filtered_signal

def load_mat_virtual_all_thresholds(matfile, nerve_model_type=3, state=1, array_type =2):
    # if platform.system() == 'Linux':
    #     dir_file = '/exports/kno-shark/users/Savine/python/temporal/data/'
    # else:    
    dir_file = './data'
    matfile = os.path.join(dir_file, matfile)
    
    # for now only received nerve model 3(?) and array type MS
    c = nerve_model_type-1 # cochlear model, nerve_model_type, CM3 is most average according to Randy
    a = array_type-1 # 1 is HiFocus 1J in lateral position, 2 is HiFocus MS in mid-scalar position, Most of Jacob's files are with MS
    m = state-1 # 1: healthy fibres, 2: fibres with shortened periferal ending, 3: fibres without dendrites
    PW = float(matfile[matfile.find('Morphologies ')+len('Morphologies '):matfile.rfind('us ')])*1e-6 # [s]
    mat = read_mat(matfile)
    # [16x1] T-level of electrode e (monopolair stimulation) in [mA] ALREADY FROM APICAL TO BASAL!
    T_levels = mat['Df120']['T'][m]*1e-3 # [mA] --> [A]
    # [16x1] T-level of electrode e (monopolair stimulation)
    M_levels = mat['Df120']['M'][m]*1e-3 # [mA] --> [A]
    # [15x9x3200]=[ep,n,f] thresholds for fibre f stimulated with electrode pair ep and alpha(n)
    # same unit as output of AB's hilbert function (log2 units)
    TI_env_log2 = mat['Df120']['TI_env_log2'][m] # --> if I want to use output of the hilbert function
    TI_env_log2 = np.nan_to_num(TI_env_log2, nan=1000) # NaNs mean the threshold was higher than the current range
    # [15x9x3200]=[ep,n,f] thresholds for fibre f stimulated with electrode pair ep and alpha(n)
    # Current on apical electrode [mA]
    TIa = mat['Df120']['TIa'][m] *1e-3 # turn to [A]
    TIa = np.nan_to_num(TIa, nan=1000) # NaNs mean the threshold was higher than the current range
    TIb = mat['Df120']['TIb'][m] *1e-3 # turn to [A]
    TIb = np.nan_to_num(TIb, nan=1000)
    # Ln needs to be reversed 
    Ln = np.flipud(mat['Df120']['Ln'][m]) # 3200 fibres
    Le = mat['Df120']['Le'][m] # 16 electrodes
    Fn = np.flip(mat['Df120']['Fn'][m])*1e3 # [Hz] 3200 fibers
    Fe = np.flip(mat['Df120']['Fe'][m])*1e3 # [Hz] 3200 fibers
    # x=3

    return T_levels, M_levels, TI_env_log2, TIa, TIb, Ln, Le, PW, Fn, Fe