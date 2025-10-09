import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from Hamacher_utils import *
import pandas as pd
from utilities import *

# To do
# [ ] check if RT max in memory causes not to reach 100% accuracy
# [ ] plot IR!!!
# [ ] create EH files
# [ ] run softmax with EH

# parms
scaling_factor_sigma = 0.2
temperature = 20

dir_to_loop = './AM/NH/IR/'

# get IR
R_name = glob.glob(os.path.join(dir_to_loop, '*masker_reference91_65_*.npy'))[0]
RT_max_name = glob.glob(os.path.join(dir_to_loop, '*masker_reference91_65dB_probe_65dB*.npy'))[0]

IR_R = np.load(R_name)
IR_RT_max = np.load(RT_max_name)
# S memory
S = IR_RT_max - IR_R

files = glob.glob(dir_to_loop + '*.npy')
files.remove(R_name)

scaling_factor_sigma_list = np.arange(0.2, 2.2, 0.2) 
temperature_list = [0.00002, 0.0002, 0.002, 0.02, 0.2, 2]

save_dir_figure = './output/MP/NH/figures/'
save_dir_results = './output/MP/NH/results/'


for folder in [save_dir_figure, save_dir_results]:
    if not os.path.exists(folder):
        os.makedirs(folder)

for scaling_factor_sigma in scaling_factor_sigma_list:
    for temperature in temperature_list:
        print(f'scaling factor sigma: {scaling_factor_sigma}, temperature: {temperature}')
        sigma_w =  np.std(IR_R)*scaling_factor_sigma
        dB_list = []
        percentage_correct_memory_matrix = np.zeros(len(files))
        percentage_correct_old_matrix = np.zeros(len(files))

        # loop RT
        for f, file in enumerate(files):
            print(file)
            IR_RT = np.load(file)
            dB = int(file[file.index('probe_') + len('probe_'): file.index('dB_IR')]) - 65
            dB_list.append(dB)
            percentage_correct_memory = iterate_3AFC_memory_softmax_correlation(IR_RT, IR_R, S, sigma_w, temperature, measure='pearson', n_iter=100, use_De=False, norm_bool=False, use_differences = True)           
            percentage_correct_memory_matrix[f] = percentage_correct_memory

            percentage_correct_old = iterate_3AFC_memory_softmax_correlation(IR_RT, IR_R, IR_RT - IR_R, sigma_w, temperature, measure='pearson', n_iter=100, use_De=False, norm_bool=False, use_differences = True)           
            percentage_correct_old_matrix[f] = percentage_correct_old

        y_list_memory = percentage_correct_memory_matrix*100
        y_list_old = percentage_correct_old_matrix*100
        # plot psychometric curve
        single_run = plt.figure(figsize=(8, 8))
        plt.scatter(x=dB_list, y=y_list_memory, label='memory S', color='blue')
        plt.scatter(x=dB_list, y=y_list_old, label='old S', color='red')
        plt.ylim((30, 100))
        plt.xlim((min(dB_list)-1, max(dB_list)+1))
        plt.xlabel('dB', fontsize=20)
        plt.ylabel('Percentage correct [%]', fontsize=20)
        plt.title('3AFC - masker probe, sigma_SF: ' + str(scaling_factor_sigma) + ', temp: ' + str(temperature), fontsize=20)
        plt.legend()

        # fit sigmoid
        sorted_x = np.sort(dB_list)
        sorted_y_memory = y_list_memory[np.array(dB_list).argsort()]
        sorted_y_old = y_list_old[np.array(dB_list).argsort()]
        try:
            y_sig_memory = fit_sigmoid(sorted_x, sorted_y_memory)
            plt.plot(sorted_x, y_sig_memory, color='blue')
            y_sig_old = fit_sigmoid(sorted_x, sorted_y_old)
            plt.plot(sorted_x, y_sig_old, color='red')
        except:
            print('Could not find psychometric fit')
        # saving data to pandas
        dict_pd = dict(zip(sorted_x, sorted_y_memory))
        dict_pd.update({"sorted_old": sorted_y_old,
                        "fit_memory" : y_sig_memory if 'y_sig' in locals() else 'no_fit',
                        "fit_old" : y_sig_old if 'y_sig' in locals() else 'no_fit',
                        "temperature": temperature,
                        "sigma_SF": scaling_factor_sigma})
        dict_pd = pd.DataFrame(dict_pd.items())
        dict_pd = dict_pd.transpose()
        dict_pd.columns = dict_pd.iloc[0]
        dict_pd = dict_pd.drop(dict_pd.index[[0]])

        single_run.savefig(save_dir_figure + '/3AFC_memory_soft_sigmaSF_' + str(scaling_factor_sigma)+ '_temp_' + str(temperature) + '.png')            
        np.save(save_dir_results + '/3AFC_memory_soft_sigmaSF_' + str(scaling_factor_sigma) + '_temp_' + str(temperature) + '.npy', dict)

        # both in one fig
        collected = plt.figure()
        plt.subplot(2,1,1)
        plt.scatter(x=dB_list, y=y_list_memory, label=f'T: {temperature}, sigma: {scaling_factor_sigma}')
        try:
            plt.plot(sorted_x, y_sig_memory, color='blue')  
        except:
            print('No fit')
        plt.title('Memory softmax')
        plt.legend()
        plt.xlabel('dB re Masker')
        plt.ylabel('Percentage correct [%]')
        plt.ylim((30, 100))
        plt.xlim((min(dB_list)-1, max(dB_list)+1))
        plt.subplot(2,1,2)
        plt.scatter(x=dB_list, y=y_list_old, label=f'T: {temperature}, sigma: {scaling_factor_sigma}')
        try:
            plt.plot(sorted_x, y_sig_old, color='red')
        except:
            print('No fit')
        plt.legend()
        plt.title('Old softmax')
        plt.xlabel('dB re Masker')
        plt.ylabel('Percentage correct [%]')
        plt.ylim((30, 100))
        plt.xlim((min(dB_list)-1, max(dB_list)+1))


collected.savefig(save_dir_figure + '/3AFC_collected_sigmaSF_' + str(scaling_factor_sigma_list)+ '_temp_' + str(temperature_list) + '.png')            
plt.show()