import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from Hamacher_utils import *
import pandas as pd
from utilities import *

# parms
scaling_factor_sigma = 0.2
temperature = 20

dir_to_loop = './MP/NH/IR/*.npy'

# get IR
R_name = glob.glob(os.path.join(dir_to_loop, '*masker_reference91_65_*.npy'))[0]
RT_max_name = glob.glob(os.path.join(dir_to_loop, '*masker_reference91_65dB_probe_65dB*.py'))[0]

IR_R = np.load(R_name)
IR_RT_max = np.load(RT_max_name)

sigma_w =  np.std(IR_R)*scaling_factor_sigma

# S memory
S = IR_RT_max - IR_R

dB_list = []

# loop RT
files = glob.glob(dir_to_loop)

percentage_correct_correlation_matrix = np.zeros(len(files))

for f, file in enumerate(files):
    IR_RT = np.load(file)

    dB = int(file[file.index('probe_') + len('probe_'): file.index('dB.py')]) - 65
    percentage_based_on_correlation_correct = iterate_3AFC_memory_softmax_correlation(IR_RT, IR_R, S, sigma_w, temperature, measure='pearson', n_iter=100, use_De=False, norm_bool=False, use_differences = True)
    percentage_correct_correlation_matrix[f] = percentage_based_on_correlation_correct

y_list = percentage_correct_correlation_matrix*100
# plot psychometric curve
plt.figure(figsize=(8, 8))
plt.scatter(x=dB_list, y=y_list)

# fit sigmoid
sorted_x = np.sort(dB_list)
sorted_y = y_list[np.array(dB_list).argsort()]
try:
    y_sig = fit_sigmoid(sorted_x, sorted_y)
    plt.plot(sorted_x, y_sig)
except:
    print('Could not find psychometric fit')
# saving data to pandas
dict_pd = dict(zip(sorted_x, sorted_y))
dict_pd.update({"3AFCtype": 'memory_soft',
                "fit" : y_sig if 'y_sig' in locals() else 'no_fit',
                "temperature": temperature,
                "sigma_SF": scaling_factor_sigma})
dict_pd = pd.DataFrame(dict_pd.items())
dict_pd = dict_pd.transpose()
dict_pd.columns = dict_pd.iloc[0]
dict_pd = dict_pd.drop(dict_pd.index[[0]])

save_dir_figure = './output/MP/NH/figures/'
save_dir_results = './output/MP/NH/results/'

for folder in [save_dir_figure, save_dir_results]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# plt.savefig(save_dir_figure + '/3AFC_memory_soft_sigmaSF_' + str(scaling_factor_sigma)+ '_temp_' + str(temperature) + '.png')            
# np.save(save_dir_results + '/3AFC_memory_soft_sigmaSF_' + str(scaling_factor_sigma) + '_temp_' + str(temperature) + '.npy', dict)