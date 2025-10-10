import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from Hamacher_utils import *
import pandas as pd
import argparse
import matplotlib as mpl
from utilities import *
import platform

# To do
# [ ] check if RT max in memory causes not to reach 100% accuracy
# [ ] plot IR!!!
# [ ] also create OG Hamacher


if platform.system() == 'Linux':
    plt.switch_backend('agg')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-test', type=str, default='AM', help='AM or MP')
    parser.add_argument('-hearing', type=str, default='NH', help='NH or EH')
    args = parser.parse_args()
    test = args.test
    hearing = args.hearing

    if test == 'AM':
        dir_to_loop = './AM/' + hearing + '/IR/'
        save_dir_figure = './output/AM/' + hearing + '/figures/'
        save_dir_results = './output/AM/' + hearing + '/results/'
        if hearing == 'NH':
            wildcard_R = '*unmodulated*reference91*.npy'
            wildcard_RT_max = '*modulated*reference91*_0dB*.npy'
            wildcard_dB_start = '91_'
            wildcard_dB_end = 'dB_IR'
        if hearing == 'EH':
            wildcard_R = '*unmodulated*reference1*.npy'
            wildcard_RT_max = '*modulated*reference1*_0dB*.npy'
            wildcard_dB_start = 'reference1_'
            wildcard_dB_end = 'dB_relscale'
    if test == 'MP':
        dir_to_loop = './MP/' + hearing + '/IR/'
        save_dir_figure = './output/MP/' + hearing + '/figures/'
        save_dir_results = './output/MP/' + hearing + '/results/'
        if hearing == 'NH':
            wildcard_R = '*masker_reference91_65_*.npy'
            wildcard_RT_max = '*masker_reference91_65dB_probe_65dB*.npy'
            wildcard_dB_start = 'probe_'
            wildcard_dB_end = 'dB_IR'
        if hearing == 'EH':
            wildcard_R = '*masker_reference1_rel*.npy'
            wildcard_RT_max = '*masker_reference1_*probe_0*.npy'
            wildcard_dB_start = 'probe_'
            wildcard_dB_end = 'dB_relscale'
    
    if hearing == 'NH' and test == 'MP':
        dB_correction = -65
    else:
        dB_correction = 0

    # get R and RT_max
    R_name = glob.glob(os.path.join(dir_to_loop, wildcard_R))[0]
    RT_max_name = glob.glob(os.path.join(dir_to_loop, wildcard_RT_max))[0]
    IR_R = np.load(R_name)
    IR_RT_max = np.load(RT_max_name)
    # S memory
    # S = IR_RT_max - IR_R                                                                    
    S = IR_RT_max - IR_R
    files = glob.glob(dir_to_loop + '*.npy')
    files.remove(R_name)    
    scaling_factor_sigma_list = np.arange(0.2, 2.2, 0.2)
    temperature_list = [0.00002, 0.0002, 0.002, 0.02, 0.2, 2]

    # create color map
    color_map = plt.get_cmap('viridis', len(temperature_list))
    custom_palette = [mpl.colors.rgb2hex(color_map(i)) for i in range(color_map.N)]

    # create folders
    for folder in [save_dir_figure, save_dir_results]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    # iterate over parameters
    for scaling_factor_sigma in scaling_factor_sigma_list:
        sigma_w =  np.std(IR_R)*scaling_factor_sigma
        # both in one fig
        collected = plt.figure(f'Collected with {scaling_factor_sigma}', figsize=(16, 8))

        for temperature in temperature_list:
            print(f'scaling factor sigma: {scaling_factor_sigma}, temperature: {temperature}')
            
            dB_list = []
            percentage_correct_memory_matrix = np.zeros(len(files))
            percentage_correct_old_matrix = np.zeros(len(files))

            # loop RT
            for f, file in enumerate(files):
                print(file)
                IR_RT = np.load(file)
                dB = int(file[file.index(wildcard_dB_start) + len(wildcard_dB_start): file.index(wildcard_dB_end)]) + dB_correction
                dB_list.append(dB)
                percentage_correct_memory = iterate_3AFC_memory_softmax_correlation(IR_RT, IR_R, S, sigma_w, temperature, measure='pearson', n_iter=100, use_De=False, norm_bool=False, use_differences = True)           
                percentage_correct_memory_matrix[f] = percentage_correct_memory
                percentage_correct_old = iterate_3AFC_memory_softmax_correlation(IR_RT, IR_R, IR_RT - IR_R, sigma_w, temperature, measure='pearson', n_iter=100, use_De=False, norm_bool=False, use_differences = True)           
                percentage_correct_old_matrix[f] = percentage_correct_old

            y_list_memory = percentage_correct_memory_matrix*100
            y_list_old = percentage_correct_old_matrix*100

            # plot psychometric curve
            single_run = plt.figure(figsize=(8, 8))
            plt.scatter(dB_list, y_list_memory, label='Memory', color='blue')
            plt.scatter(dB_list, y_list_old, label='Old', color='red')
            plt.ylim((30, 100))
            plt.xlim((min(dB_list)-1, max(dB_list)+1))
            plt.legend()
            plt.title(f'Scaling factor sigma: {scaling_factor_sigma}, temperature: {temperature}')
            plt.xlabel('dB re Masker')  
            plt.ylabel('Percentage correct [%]')

            # get fitted curve
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
            single_run.savefig(save_dir_figure + '/3AFC_sigmaSF_' + str(scaling_factor_sigma)+ '_temp_' + str(temperature) + '.png')            
            np.save(save_dir_results + '/3AFC_sigmaSF_' + str(scaling_factor_sigma) + '_temp_' + str(temperature) + '.npy', dict_pd)

            plt.figure(f'Collected with {scaling_factor_sigma}')
            plt.subplot(1,2,1)
            plt.scatter(x=dB_list, y=y_list_memory, label=f'T: {temperature}', color=custom_palette[temperature_list.index(temperature)])
            try:
                plt.plot(sorted_x, y_sig_memory, color=custom_palette[temperature_list.index(temperature)])  
            except:
                print('No fit')
            plt.subplot(1,2,2)
            plt.scatter(x=dB_list, y=y_list_old, label=f'T: {temperature}', color=custom_palette[temperature_list.index(temperature)])
            try:
                plt.plot(sorted_x, y_sig_old, color=custom_palette[temperature_list.index(temperature)])
            except:
                print('No fit')
        plt.figure(f'Collected with {scaling_factor_sigma}')
        plt.subplot(1,2,1)
        plt.title('Memory softmax')
        plt.legend(ncol=2)  
        plt.xlabel('dB re Masker')  
        plt.ylabel('Percentage correct [%]')    
        plt.ylim((30, 100))
        plt.xlim((min(dB_list)-1, max(dB_list)+1))  
        plt.subplot(1,2,2)
        plt.legend(ncol=2)
        plt.title('Old softmax')
        plt.xlabel('dB re Masker')  
        plt.ylabel('Percentage correct [%]')
        plt.ylim((30, 100))
        plt.xlim((min(dB_list)-1, max(dB_list)+1))
        plt.suptitle(f'Scaling factor sigma: {scaling_factor_sigma}')
        collected.savefig(save_dir_figure + '/3AFC_collected_sigmaSF_' + str(scaling_factor_sigma)+ '_temp_' + str(temperature_list) + '.png')


    plt.show()