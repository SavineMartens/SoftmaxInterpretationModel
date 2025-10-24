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
# [X] check if RT max in memory causes not to reach 100% accuracy --> caused by temperature
# [X] plot IR!!!
# [X] also create OG Hamacher
# [ ] implement Gumbel distributed noise 
# [ ] check if this is similar to e-softmax in this paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC5001502/pdf/nihms780191.pdf
# [X] check why it softmax won't go below 67%
# [X] check if using selected number of bands does allow to reach 100% --> does not seem to change that much
# [X] check why NH with MP has differences in loudness wrt R 
# [X] Try MP with other folder
# [X] figure as in paper with NH and EH side by side
# [X] include R in 3AFC
# [X] create seed42 folder for AM as well


if platform.system() == 'Linux':
    plt.switch_backend('agg')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-test', type=str, default='MP', help='AM or MP')
    parser.add_argument('-hearing', type=str, default='NH', help='NH or EH')
    parser.add_argument('-norm', default=False, action='store_true')
    args = parser.parse_args()
    test = args.test
    hearing = args.hearing
    # if on Windows
    if platform.system() == 'Windows':
        dir_to_loop = f'S:/python/SoftmaxInterpretationModel/{test}/{hearing}/IR/'
    else:
        dir_to_loop = f'./{test}/{hearing}/IR/'
    TP2_cut_off_Hz = 500
    num_fibers = 1903# 952

    NH_dB = 55

    if args.norm:
        norm_bool = True
    else:
        norm_bool = False
    print(f'Norm bool is set to {norm_bool}')

    save_dir_figure = f'./output/{test}/{hearing}/figures/'
    save_dir_results = f'./output/{test}/{hearing}/results/'


    if test == 'AM':
        if hearing == 'NH':
            wildcard_R = f'*unmodulated*reference91_{NH_dB}*'
            wildcard_RT_max = f'*modulated*reference91_{NH_dB}*_0dB*'
            wildcard_dB_start = f'91_{NH_dB}dB_'
            wildcard_dB_end = 'dB_IR'
            dir_to_loop += 'seed42/'
            save_dir_figure += f'seed42/{NH_dB}dB/'
            save_dir_results += f'seed42/{NH_dB}dB/'
        if hearing == 'EH':
            wildcard_R = f'*unmodulated*reference1*'
            wildcard_RT_max = f'*modulated*reference1*_0dB*'
            wildcard_dB_start = 'reference1_'
            wildcard_dB_end = 'dB_relscale'
    if test == 'MP':
        if hearing == 'NH':
            dir_to_loop += 'seed42/'
            save_dir_figure += f'seed42/{NH_dB}dB/'
            save_dir_results += f'seed42/{NH_dB}dB/'
            wildcard_R = f'*masker_reference91_{NH_dB}*'
            wildcard_RT_max = f'*masker_reference91_{NH_dB}dB_probe_{NH_dB}dB*'
            wildcard_dB_start = 'probe_'
            wildcard_dB_end = 'dB_IR'
        if hearing == 'EH':
            wildcard_R = f'*masker_reference1_rel*'
            wildcard_RT_max = f'*masker_reference1_*probe_0*'
            wildcard_dB_start = 'probe_'
            wildcard_dB_end = 'dB_relscale'
            
    

    num_bands = 24
    print(f'Running {test} for {hearing}')

    if hearing == 'NH' and test == 'MP':
        dB_correction = -1*NH_dB
    else:
        dB_correction = 0

    # get R and RT_max
    R_name = glob.glob(os.path.join(dir_to_loop, wildcard_R + f'*{num_fibers}CFs*{TP2_cut_off_Hz}Hz.npy'))[0]
    RT_max_name = glob.glob(os.path.join(dir_to_loop, wildcard_RT_max + f'*{num_fibers}CFs*{TP2_cut_off_Hz}Hz.npy'))[0]
    IR_R = np.load(R_name)
    IR_RT_max = np.load(RT_max_name)
    
    # remove empty bands
    IR_IR = remove_empty_bands(IR_R)
    IR_RT_max = remove_empty_bands(IR_RT_max)
    if IR_R.shape[0] != IR_RT_max.shape[0]:
        print('Warning: Different number of bands in R and RT_max')
        min_bands = min(IR_R.shape[0], IR_RT_max.shape[0])
        IR_R = IR_R[:min_bands, :]
        IR_RT_max = IR_RT_max[:min_bands, :]

    if NH_dB and hearing == 'NH':
        dB_sel = f'*91_{NH_dB}*' 
    else:
        dB_sel = '*'

    # S memory in softmax
    S = IR_RT_max - IR_R
    files = glob.glob(dir_to_loop + dB_sel + f'{num_fibers}CFs_{num_bands}bands*{TP2_cut_off_Hz}Hz.npy')

    if test == 'AM':
        scaling_factor_sigma_list = [0.02, 0.04, 0.06, 0.08, 0.2, 0.4, 0.6, 0.8]
        temperature_list =  [0.001, 0.003, 0.009, 0.027, 0.081, 0.243, 0.729, 2.187, 6.561]
    if test == 'MP':
        scaling_factor_sigma_list = [0.001, 0.003, 0.009, 0.027, 0.081, 0.243, 0.729, 2.187, 6.561]
        temperature_list =  [0.001, 0.003, 0.009, 0.027, 0.081, 0.243, 0.729, 2.187, 6.561]

    # create color map
    color_map_temperature = plt.get_cmap('viridis', len(temperature_list))
    custom_palette_temperature = [mpl.colors.rgb2hex(color_map_temperature(i)) for i in range(color_map_temperature.N)]
    color_map_scaling = plt.get_cmap('plasma', len(scaling_factor_sigma_list))
    custom_palette_scaling = [mpl.colors.rgb2hex(color_map_scaling(i)) for i in range(color_map_scaling.N)]

    # create folders
    for folder in [save_dir_figure, save_dir_results]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    # iterate over parameters
    for scaling_factor_sigma in scaling_factor_sigma_list:
        sigma_w =  np.std(IR_R)*scaling_factor_sigma
        # both in one fig
        collected = plt.figure(f'{hearing}: Collected with {scaling_factor_sigma}', figsize=(16, 8))

        dB_list = []
        percentage_correct_memory_matrix = np.zeros((len(files), len(temperature_list)))
        percentage_correct_old_softmax_matrix = np.zeros((len(files), len(temperature_list)))
        percentage_correct_Hamacher_matrix = np.zeros(len(files))
        percentage_correct_Hamacher_RTmax_matrix = np.zeros(len(files))

        plt.figure('Hamacher collected', figsize=(8, 8))

        # loop RT
        for f, file in enumerate(files):
            print(f'Processing file {f+1}/{len(files)}: {file}')
            IR_RT = np.load(file)
            IR_RT = remove_empty_bands(IR_RT)
            if IR_R.shape[0] != IR_RT.shape[0]:
                print('Warning: Different number of bands in R and RT_max')
                min_bands = min(IR_R.shape[0], IR_RT.shape[0])
                IR_R = IR_R[:min_bands, :]
                IR_RT = IR_RT[:min_bands, :]

            try:
                dB = int(file[file.index(wildcard_dB_start) + len(wildcard_dB_start): file.index(wildcard_dB_end)]) + dB_correction
            except: # when R 
                dB = -80
            dB_list.append(dB)

            # Hamacher
            percentage_correct_Hamacher = Hamacher_3AFC(IR_RT, IR_R,  IR_RT - IR_R, sigma_w, measure='pearson', n_iter=100, use_De=False)
            percentage_correct_Hamacher_matrix[f] = percentage_correct_Hamacher
            percentage_correct_Hamacher_RTmax = Hamacher_3AFC(IR_RT, IR_R,  IR_RT_max - IR_R, sigma_w, measure='pearson', n_iter=100, use_De=False)
            percentage_correct_Hamacher_RTmax_matrix[f] = percentage_correct_Hamacher_RTmax

            plt.figure('Hamacher collected')
            plt.scatter(x=dB, y=percentage_correct_Hamacher*100, color=custom_palette_scaling[scaling_factor_sigma_list.index(scaling_factor_sigma)])

            for t, temperature in enumerate(temperature_list):
                print(f'scaling factor sigma: {scaling_factor_sigma}, temperature: {temperature}')    
                # new version: sofmax with memory RT_max
                percentage_correct_memory = Softmax_memory_3AFC(IR_RT, IR_R, S, sigma_w, temperature, measure='pearson', n_iter=100, use_De=False, norm_bool=norm_bool)          
                percentage_correct_memory_matrix[f, t] = percentage_correct_memory
                # new version: softmax with old RT
                percentage_correct_old_softmax = Softmax_memory_3AFC(IR_RT, IR_R, IR_RT - IR_R, sigma_w, temperature, measure='pearson', n_iter=100, use_De=False, norm_bool=norm_bool)           
                percentage_correct_old_softmax_matrix[f, t] = percentage_correct_old_softmax
                
        y_list_memory = percentage_correct_memory_matrix*100
        y_list_old_softmax = percentage_correct_old_softmax_matrix*100

        for t, temperature in enumerate(temperature_list):
            # plot psychometric curve
            single_run = plt.figure(figsize=(8, 8))
            plt.scatter(dB_list, y_list_memory[:,t], label='softmax RT_max', color='blue')
            plt.scatter(dB_list, y_list_old_softmax[:,t], label='softmax RT', color='red')
            plt.scatter(dB_list, percentage_correct_Hamacher_matrix*100, label='Hamacher', color='green')
            plt.ylim((30, 101))
            plt.xlim((min(dB_list)-1, max(dB_list)+1))
            plt.legend()
            plt.title(f'{hearing}: Scaling factor sigma: {scaling_factor_sigma}, temperature: {temperature}')
            plt.xlabel('dB re Masker')  
            plt.ylabel('Percentage correct [%]')

            # get fitted curve
            sorted_x = np.sort(dB_list)
            sorted_y_memory = y_list_memory[np.array(dB_list).argsort(),t]
            sorted_y_old_softmax = y_list_old_softmax[np.array(dB_list).argsort(),t]
            sorted_y_Hamacher = percentage_correct_Hamacher_matrix[np.array(dB_list).argsort()]*100
            sorted_y_Hamacher_RTmax = percentage_correct_Hamacher_RTmax_matrix[np.array(dB_list).argsort()]*100
            try:
                y_sig_memory = fit_sigmoid(sorted_x, sorted_y_memory)
                plt.plot(sorted_x, y_sig_memory, color='blue')
                y_sig_old_softmax = fit_sigmoid(sorted_x, sorted_y_old_softmax)
                plt.plot(sorted_x, y_sig_old_softmax, color='red')
                y_sig_Hamacher = fit_sigmoid(sorted_x, sorted_y_Hamacher)
                plt.plot(sorted_x, y_sig_Hamacher, color='green')
                y_sig_Hamacher_RTmax = fit_sigmoid(sorted_x, sorted_y_Hamacher_RTmax)
                plt.plot(sorted_x, y_sig_Hamacher_RTmax, color='orange')
            except:
                print('Could not find psychometric fit')


            # saving data to dictionary
            data_dict = dict()
            data_dict.update({"dB_list": sorted_x,
                              "y_soft_RTmax": sorted_y_memory,
                              "y_soft_RT": sorted_y_old_softmax,
                              "y_Hamacher_RT": sorted_y_Hamacher,
                              "y_Hamacher_RTmax": sorted_y_Hamacher_RTmax,
                              "y_fit_soft_RTmax": y_sig_memory if 'y_sig_memory' in locals() else 'no_fit',
                              "y_fit_soft_RT" : y_sig_old_softmax if 'y_sig_old_softmax' in locals() else 'no_fit',
                              "y_fit_Hamacher_RT" : y_sig_Hamacher if 'y_sig_Hamacher' in locals() else 'no_fit',
                              "y_fit_Hamacher_RTmax" : y_sig_Hamacher_RTmax if 'y_sig_Hamacher_RTmax' in locals() else 'no_fit',
                              "temperature": temperature,
                              "sigma_SF": scaling_factor_sigma,
                              "norm_bool": norm_bool})

            single_run.savefig(save_dir_figure + '/3AFC_sigmaSF_' + str(scaling_factor_sigma)+ '_temp_' + str(temperature) +  '_norm_' + str(norm_bool) + '.png')            
            np.save(save_dir_results + '/3AFC_sigmaSF_' + str(scaling_factor_sigma) + '_temp_' + str(temperature) +  '_norm_' + str(norm_bool) + '.npy', data_dict)

            plt.figure(f'{hearing}: Collected with {scaling_factor_sigma}')
            plt.subplot(1,3,1)
            plt.title('Memory softmax with RT_max')
            plt.scatter(x=dB_list, y=y_list_memory[:,t], label=f'T: {temperature}', color=custom_palette_temperature[temperature_list.index(temperature)])
            try:
                plt.plot(sorted_x, y_sig_memory, color=custom_palette_temperature[temperature_list.index(temperature)])  
            except:
                print('No fit')
            plt.subplot(1,3,2)
            plt.title('Softmax with old RT')
            plt.scatter(x=dB_list, y=y_list_old_softmax[:,t], label=f'T: {temperature}', color=custom_palette_temperature[temperature_list.index(temperature)])
            try:
                plt.plot(sorted_x, y_sig_old_softmax, color=custom_palette_temperature[temperature_list.index(temperature)])
            except:
                print('No fit')
            if temperature == temperature_list[-1]:
                plt.suptitle(f'Scaling factor sigma: {scaling_factor_sigma}')
                plt.subplot(1,3,3)
                plt.scatter(x=dB_list, y=percentage_correct_Hamacher_matrix*100, label=f'RT Version (OG)', color=custom_palette_temperature[temperature_list.index(temperature)])
                plt.scatter(x=dB_list, y=percentage_correct_Hamacher_RTmax_matrix*100, label=f'RT_max version', color=custom_palette_temperature[temperature_list.index(temperature)], marker='x')
                try:
                    plt.plot(sorted_x, y_sig_Hamacher, color=custom_palette_temperature[temperature_list.index(temperature)])
                    plt.plot(sorted_x, y_sig_Hamacher_RTmax, color=custom_palette_temperature[temperature_list.index(temperature)], linestyle='--')
                except:
                    print('No fit')
                plt.title(f'Original Hamacher')
                for subplot in [1,2,3]:
                    plt.subplot(1,3,subplot)
                    plt.legend(ncol=2)  
                    plt.xlabel('dB re Masker')  
                    plt.ylabel('Percentage correct [%]')    
                    plt.ylim((30, 101))
                    plt.xlim((min(dB_list)-1, max(dB_list)+1))  

                plt.figure('Hamacher collected')
                plt.plot(sorted_x, y_sig_Hamacher, label=f'RT sigma: {scaling_factor_sigma}', color=custom_palette_scaling[scaling_factor_sigma_list.index(scaling_factor_sigma)])
                plt.plot(sorted_x, y_sig_Hamacher_RTmax, label=f'RTmax sigma: {scaling_factor_sigma}', color=custom_palette_temperature[temperature_list.index(temperature)], linestyle='--')
                plt.title('Hamacher collected')
                plt.xlabel('dB re Masker')
                plt.ylabel('Percentage correct [%]')
                plt.legend()
            
            collected.savefig(save_dir_figure + '/3AFC_collected_sigmaSF_' + str(scaling_factor_sigma)+ '_temp_' + str(temperature_list) +  '_norm_' + str(norm_bool) + '.png')


    plt.show()