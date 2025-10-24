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
# [X] run with greater range

if platform.system() == 'Linux':
    plt.switch_backend('agg')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-test', type=str, default='AM', help='AM or MP')
    parser.add_argument('-hearing', type=str, default='NH', help='NH or EH')
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

    save_dir_results = f'./output/{test}/{hearing}/results/'


    if test == 'AM':
        if hearing == 'NH':
            wildcard_R = f'*unmodulated*reference91*'
            wildcard_RT_max = f'*modulated*reference91*_0dB*'
            wildcard_dB_start = '91_'
            wildcard_dB_end = 'dB_IR'
            num_bands = 24
            dir_to_loop += 'seed42/'
            save_dir_results += 'seed42/'
        if hearing == 'EH':
            wildcard_R = f'*unmodulated*reference1*'
            wildcard_RT_max = f'*modulated*reference1*_0dB*'
            wildcard_dB_start = 'reference1_'
            wildcard_dB_end = 'dB_relscale'
            num_bands = 28
    if test == 'MP':
        if hearing == 'NH':
            dir_to_loop += 'seed42/'
            save_dir_results += 'seed42/'
            wildcard_R = f'*masker_reference91_65_*'
            wildcard_RT_max = f'*masker_reference91_65dB_probe_65dB*'
            wildcard_dB_start = 'probe_'
            wildcard_dB_end = 'dB_IR'
            num_bands = 24
        if hearing == 'EH':
            wildcard_R = f'*masker_reference1_rel*'
            wildcard_RT_max = f'*masker_reference1_*probe_0*'
            wildcard_dB_start = 'probe_'
            wildcard_dB_end = 'dB_relscale'
            num_bands = 24
    
    save_dir_results += 'Hamacher_only/'

    print(f'Running {test} for {hearing}')

    if hearing == 'NH' and test == 'MP':
        dB_correction = -65
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

    # S memory in softmax
    S = IR_RT_max - IR_R
    files = glob.glob(dir_to_loop + f'*{num_fibers}CFs_{num_bands}bands*{TP2_cut_off_Hz}Hz.npy')
    if test == 'AM':
        scaling_factor_sigma_list = [1.2, 1.6, 2, 4 ,6 ,8]
    if test == 'MP':
        scaling_factor_sigma_list = [0.001, 0.003, 0.009, 0.027, 0.081, 0.243, 0.729, 2.187, 6.561]


    # create color map
    color_map_scaling = plt.get_cmap('plasma', len(scaling_factor_sigma_list))
    custom_palette_scaling = [mpl.colors.rgb2hex(color_map_scaling(i)) for i in range(color_map_scaling.N)]

    # create folders
    if not os.path.exists(save_dir_results):
        os.makedirs(save_dir_results)
    
    # iterate over parameters
    for scaling_factor_sigma in scaling_factor_sigma_list:
        sigma_w =  np.std(IR_R)*scaling_factor_sigma
        # both in one fig
        collected = plt.figure(f'{hearing}: Collected with {scaling_factor_sigma}', figsize=(16, 8))

        dB_list = []
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
            percentage_correct_Hamacher_RTmax = Hamacher_3AFC(IR_RT_max, IR_R,  IR_RT_max - IR_R, sigma_w, measure='pearson', n_iter=100, use_De=False)
            percentage_correct_Hamacher_RTmax_matrix[f] = percentage_correct_Hamacher_RTmax

            plt.figure('Hamacher collected')
            plt.scatter(x=dB, y=percentage_correct_Hamacher*100, color=custom_palette_scaling[scaling_factor_sigma_list.index(scaling_factor_sigma)])

            # get fitted curve
            sorted_x = np.sort(dB_list)
            sorted_y_Hamacher = percentage_correct_Hamacher_matrix[np.array(dB_list).argsort()]*100
            sorted_y_Hamacher_RTmax = percentage_correct_Hamacher_RTmax_matrix[np.array(dB_list).argsort()]*100


            # saving data to dictionary
            data_dict = dict()
            data_dict.update({"dB_list": sorted_x,
                              "y_Hamacher_RT": sorted_y_Hamacher,
                              "y_Hamacher_RTmax": sorted_y_Hamacher_RTmax,
                              "sigma_SF": scaling_factor_sigma})
            np.save(save_dir_results + '/3AFC_sigmaSF_' + str(scaling_factor_sigma) + '.npy', data_dict)


    plt.show()