import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from Hamacher_utils import *

# parms
scaling_factor_sigma = 0.2
temperature = 20

dir_to_loop = './MP/NH/IR/*.npy'

# get IR
R_name = glob.glob(os.path.join(dir_to_loop, '*masker_reference91_65.npy'))[0]
RT_max_name = glob.glob(os.path.join(dir_to_loop, '*masker_reference91_65dB_probe_65dB.py'))[0]

IR_R = np.load(R_name)
IR_RT_max = np.load(RT_max_name)

sigma_w =  np.std(IR_R)*scaling_factor_sigma

# S memory
S = IR_RT_max - IR_R


# loop RT
files = glob.glob(dir_to_loop)
for file in files:
    IR_RT = np.load(file)

    dB = int(file[file.index('probe_') + len('probe_'): file.index('dB.py')])
    iterate_3AFC_memory_softmax_correlation(IR_RT, IR_R, S, sigma_w, temperature, measure='pearson', n_iter=100, use_De=False, norm_bool=False, use_differences = True)
    
