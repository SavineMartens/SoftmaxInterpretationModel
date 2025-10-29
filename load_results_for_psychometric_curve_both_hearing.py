import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import argparse
from utilities import fit_best_sigmoid
import matplotlib.transforms as mtransforms # labeling axes
import platform


# To do
# [X] create neurograms with smaller probe amplitudes for MP
# [X] run 3AFC & AMP with values from doc
# [X] als run with True normalization
# [X] improve fitting procedure 
# [X] more sigma for old Hamacher --> did not improve figure
# [X] fit without -80 dB?
# [X] label figs
# [X] NH with 55 dB
# [X] run all files again!!
# [X] figure to compare 55 and 65 dB NH?



def remove_R(data):
    data['y_Hamacher_RT'] = data['y_Hamacher_RT'][data['dB_list'] > -79]
    data['y_Hamacher_RTmax'] = data['y_Hamacher_RTmax'][data['dB_list'] > -79]
    try:
        data['y_soft_RTmax'] = data['y_soft_RTmax'][data['dB_list'] > -79]
        data['y_soft_RT'] = data['y_soft_RT'][data['dB_list'] > -79]
    except:
        pass
    data['dB_list'] = data['dB_list'][data['dB_list'] > -79]
    return data



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and plot results for psychometric curve")
    parser.add_argument("-test", type=str, default="MP", help="Test type (AM or FM)")
    parser.add_argument('-norm', default=False, action='store_true')
    parser.add_argument('-wo_R', default=False, action='store_true')
    parser.add_argument('-sigma', type=float, default=0.2, help="Sigma value for softmax")
    parser.add_argument('-temp', type=float, default=0.243, help="Temperature value for softmax")
    parser.add_argument('-NH_dB', type=int, default=65, help="Presentation level for NH")
    args = parser.parse_args()

    NH_dB = args.NH_dB

    fixed_sigma = args.sigma
    fixed_temp = args.temp

    test = args.test
    if args.norm:
        norm_bool = True
    else:
        norm_bool = False

    if args.wo_R: # exclude R or -80 dB point from fit
        wo_R = True
        wo_R_str = '_wo_R'
    else:
        wo_R = False
        wo_R_str = ''

    output_dir = f'./output/{test}/{NH_dB}dB/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # if on cluster different matplotlib backend
    if platform.system() == 'Linux':
        import matplotlib
        matplotlib.use('Agg')
        folder_results_NH = f'./output/{test}/NH/results/seed42/{NH_dB}dB/' 
        folder_results_EH = f'./output/{test}/EH/results/'
    else:# try:
        folder_results_NH = f'S:/python/SoftmaxInterpretationModel/output/{test}/NH/results/seed42/{NH_dB}dB/' 
        folder_results_EH = f'S:/python/SoftmaxInterpretationModel/output/{test}/EH/results/'
        if test == 'AM':
            fixed_sigma = 0.04
            fixed_temp = 0.081
        else:
            fixed_sigma = 0.009
            fixed_temp = 0.009


    all_files_NH = sorted(glob.glob(os.path.join(folder_results_NH, f'*norm*{norm_bool}.npy')))
    all_files_EH = sorted(glob.glob(os.path.join(folder_results_EH, f'*norm*{norm_bool}.npy')))

    # read values from filenames
    sigma_values_NH = []
    temp_values_NH = []
    sigma_values_EH = []
    temp_values_EH = []

    x_pos = -0.003
    y_pos = 0.07

    # desired values
    if test == 'AM':
        test_str = 'Amplitude Modulation'
        desired_sigma_values = [0.02, 0.04, 0.08, 0.2, 0.4, 0.8]
        desired_temp_values = [0.009, 0.027, 0.081, 0.162, 0.243, 0.486, 0.729, 2.187, 6.561] #[0.001, 0.003, 0.009, 0.027, 0.081, 0.243, 0.729, 2.187, 6.561]
        # fixed_sigma = 0.2 #0.08
        # fixed_temp = 0.243 #0.081
        x_label = 'Modulation Depth [dB]'
    elif test == 'MP':
        test_str = 'Masker Probe'
        desired_sigma_values = [0.001, 0.003, 0.009, 0.027, 0.081, 0.243, 0.729, 2.187, 6.561]
        desired_temp_values = [0.009, 0.027, 0.081, 0.162, 0.243, 0.486, 0.729, 2.187, 6.561] #[0.001, 0.003, 0.009, 0.027, 0.081, 0.243, 0.729, 2.187, 6.561]

        x_label = 'Probe dB re masker'

    for file in all_files_NH:
        base = os.path.basename(file)
        sigma_str = float(base[base.find('sigmaSF_') + len('sigmaSF_'): base.find('_temp')])
        temp_str = float(base[base.find('temp_') + len('temp_'): base.find('_norm')])
        sigma_values_NH.append(float(sigma_str))
        temp_values_NH.append(float(temp_str))
    for file in all_files_EH:
        base = os.path.basename(file)
        sigma_str = float(base[base.find('sigmaSF_') + len('sigmaSF_'): base.find('_temp')])
        temp_str = float(base[base.find('temp_') + len('temp_'): base.find('_norm')])
        sigma_values_EH.append(float(sigma_str))
        temp_values_EH.append(float(temp_str))
    sigma_values_NH = np.unique(np.array(sigma_values_NH))
    temp_values_NH = np.unique(np.array(temp_values_NH))
    sigma_values_EH = np.unique(np.array(sigma_values_EH))
    temp_values_EH = np.unique(np.array(temp_values_EH))
    print(f"Found sigma values for NH: {sigma_values_NH}")
    print(f"Found temperature values for NH: {temp_values_NH}")
    print(f"Found sigma values for EH: {sigma_values_EH}")
    print(f"Found temperature values for EH: {temp_values_EH}")

    # find overlapping sigma and temp values
    sigma_values = np.intersect1d(sigma_values_NH, sigma_values_EH)
    temp_values = np.intersect1d(temp_values_NH, temp_values_EH)

    desired_sigma_values = [s for s in desired_sigma_values if s in sigma_values]
    desired_temp_values = [t for t in desired_temp_values if t in temp_values]

    # create colormap
    cmap = plt.get_cmap('viridis', len(sigma_values))
    colors_sigma = [cmap(i) for i in range(cmap.N)]

    cmap = plt.get_cmap('plasma', len(temp_values))
    colors_temp = [cmap(i) for i in range(cmap.N)]

    # figure 4 subplots: softmax RTmax fixed temp, varying sigma, NH vs EH
    figSoftRTmax, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.076, bottom=0.07, top=0.914)
    # subplot 1: fixed sigma EH
    for temp in desired_temp_values:
        for file in all_files_EH:
            if f'temp_{temp}_' in file and f'sigmaSF_{fixed_sigma}_' in file:
                print(file)
                data_EH = np.load(file, allow_pickle=True).item()
                if wo_R: # remove -80 dB and y_val
                    data_EH = remove_R(data_EH)
                sigma = data_EH['sigma_SF']
                axes[0,0].scatter(data_EH['dB_list'], data_EH['y_soft_RTmax'], label=f'T={temp}', color=colors_temp[np.where(temp_values == temp)[0][0]])
                axes[0,0].plot(data_EH['dB_list'], fit_best_sigmoid(data_EH['dB_list'], data_EH['y_soft_RTmax']), color=colors_temp[np.where(temp_values == temp)[0][0]])
                axes[0,0].set_title(f'EH: fixed σ={fixed_sigma}', fontsize=14)

    # subplot 2: fixed temp EH
    for sigma in desired_sigma_values:
        for file in all_files_EH:
            if f'temp_{fixed_temp}_' in file and f'sigmaSF_{sigma}_' in file:
                print(file)
                data_EH = np.load(file, allow_pickle=True).item()
                if wo_R: # remove -80 dB and y_val
                    data_EH = remove_R(data_EH)
                sigma = data_EH['sigma_SF']
                axes[0,1].scatter(data_EH['dB_list'], data_EH['y_soft_RTmax'], label=f'σ={sigma}', color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
                axes[0,1].plot(data_EH['dB_list'], fit_best_sigmoid(data_EH['dB_list'], data_EH['y_soft_RTmax']), color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
                axes[0,1].set_title(f'EH: fixed T={fixed_temp}', fontsize=14)
    # subplot 3: fixed sigma NH
    for temp in desired_temp_values:
        for file in all_files_NH:
            if f'sigmaSF_{fixed_sigma}_' in file and f'temp_{temp}_' in file:
                print(file)
                data_NH = np.load(file, allow_pickle=True).item()
                if wo_R: # remove -80 dB and y_val
                    data_NH = remove_R(data_NH)
                temp = data_NH['temperature']
                axes[1,0].scatter(data_NH['dB_list'], data_NH['y_soft_RTmax'], label=f'T={temp}', color=colors_temp[np.where(temp_values == temp)[0][0]])
                axes[1,0].plot(data_NH['dB_list'], fit_best_sigmoid(data_NH['dB_list'], data_NH['y_soft_RTmax']), color=colors_temp[np.where(temp_values == temp)[0][0]])
                axes[1,0].set_title(f'NH: fixed σ={fixed_sigma}', fontsize=14)

    # subplot 4: fixed temp NH
    for sigma in desired_sigma_values:
        for file in all_files_NH:
            if f'sigmaSF_{sigma}_' in file and f'temp_{fixed_temp}_' in file:
                print(file)
                data_NH = np.load(file, allow_pickle=True).item()
                if wo_R: # remove -80 dB and y_val
                    data_NH = remove_R(data_NH)
                temp = data_NH['temperature']
                axes[1,1].scatter(data_NH['dB_list'], data_NH['y_soft_RTmax'], label=f'σ={sigma}', color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
                axes[1,1].plot(data_NH['dB_list'], fit_best_sigmoid(data_NH['dB_list'], data_NH['y_soft_RTmax']), color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
                axes[1,1].set_title(f'NH: fixed T={fixed_temp}', fontsize=14)
    trans = mtransforms.ScaledTranslation(10/72, -5/72, figSoftRTmax.dpi_scale_trans)
    letters = ['A', 'B', 'C', 'D']
    for a, ax in enumerate(axes.flatten()):
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel('Percentage correct [%]',fontsize=14)
        ax.set_ylim((25, 101))
        ax.set_xlim((min(data_NH['dB_list']), max(data_NH['dB_list'])))
        ax.legend(loc='upper left')
        ax.grid(color='lightgray')
        ax.text(x_pos, y_pos+0.02, letters[a], transform=ax.transAxes + trans,
            fontsize=16, verticalalignment='top', color='black')
    plt.suptitle(f'Softmax RT$_{{max}}$ Psychometric Curves - {test_str}', fontsize=16)
    figSoftRTmax.savefig(f'{output_dir}Softmax_RTmax_Psychometric_Curves_temp_{fixed_temp}_sigma_{fixed_sigma}_norm_{norm_bool}{wo_R_str}.png')


  # figure 4 subplots: softmax RT varying temp, fixed sigma, NH vs EH
    figSoftRT, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.076, bottom=0.07, top=0.914)
    # subplot 1: fixed sigma EH
    for temp in desired_temp_values:
        for file in all_files_EH:
            if f'temp_{temp}_' in file and f'sigmaSF_{fixed_sigma}_' in file:
                print(file)
                data_EH = np.load(file, allow_pickle=True).item()
                if wo_R: # remove -80 dB and y_val
                    data_EH = remove_R(data_EH)
                sigma = data_EH['sigma_SF']
                axes[0,0].scatter(data_EH['dB_list'], data_EH['y_soft_RT'], label=f'T={temp}', color=colors_temp[np.where(temp_values == temp)[0][0]])
                fit = fit_best_sigmoid(data_EH['dB_list'], data_EH['y_soft_RT'])
                axes[0,0].plot(data_EH['dB_list'], fit, color=colors_temp[np.where(temp_values == temp)[0][0]])
                axes[0,0].set_title(f'EH: fixed σ={fixed_sigma}', fontsize=14)

    # subplot 2: fixed temp EH
    for sigma in desired_sigma_values:
        for file in all_files_EH:
            if f'temp_{fixed_temp}_' in file and f'sigmaSF_{sigma}_' in file:
                print(file)
                data_EH = np.load(file, allow_pickle=True).item()
                if wo_R: # remove -80 dB and y_val
                    data_EH = remove_R(data_EH)
                sigma = data_EH['sigma_SF']
                axes[0,1].scatter(data_EH['dB_list'], data_EH['y_soft_RT'], label=f'σ={sigma}', color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
                axes[0,1].plot(data_EH['dB_list'], fit_best_sigmoid(data_EH['dB_list'], data_EH['y_soft_RT']), color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
                axes[0,1].set_title(f'EH: fixed T={fixed_temp}', fontsize=14)

    # subplot 3: fixed sigma NH
    for temp in desired_temp_values:
        for file in all_files_NH:
            if f'sigmaSF_{fixed_sigma}_' in file and f'temp_{temp}_' in file:
                print(file)
                data_NH = np.load(file, allow_pickle=True).item()
                if wo_R: # remove -80 dB and y_val
                    data_NH = remove_R(data_NH)
                temp = data_NH['temperature']
                axes[1,0].scatter(data_NH['dB_list'], data_NH['y_soft_RT'], label=f'T={temp}', color=colors_temp[np.where(temp_values == temp)[0][0]])
                axes[1,0].plot(data_NH['dB_list'], fit_best_sigmoid(data_NH['dB_list'], data_NH['y_soft_RT']), color=colors_temp[np.where(temp_values == temp)[0][0]])
                axes[1,0].set_title(f'NH: fixed σ={fixed_sigma}', fontsize=14)

    # subplot 4: fixed temp NH
    for sigma in desired_sigma_values:
        for file in all_files_NH:
            if f'sigmaSF_{sigma}_' in file and f'temp_{fixed_temp}_' in file:
                print(file)
                data_NH = np.load(file, allow_pickle=True).item()
                if wo_R: # remove -80 dB and y_val
                    data_NH = remove_R(data_NH)
                temp = data_NH['temperature']
                axes[1,1].scatter(data_NH['dB_list'], data_NH['y_soft_RT'], label=f'σ={sigma}', color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
                axes[1,1].plot(data_NH['dB_list'], fit_best_sigmoid(data_NH['dB_list'], data_NH['y_soft_RT']), color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
                axes[1,1].set_title(f'NH: fixed T={fixed_temp}', fontsize=14)

    trans = mtransforms.ScaledTranslation(10/72, -5/72, figSoftRT.dpi_scale_trans)
    for a, ax in enumerate(axes.flatten()):
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel('Percentage correct [%]',fontsize=14)
        ax.set_ylim((25, 101))
        ax.set_xlim((min(data_NH['dB_list']), max(data_NH['dB_list'])))
        ax.legend(loc='upper left')
        ax.grid(color='lightgray')
        ax.text(x_pos, y_pos+0.02, letters[a], transform=ax.transAxes + trans,
            fontsize=16, verticalalignment='top', color='black')
    plt.suptitle(f'Softmax RT Psychometric Curves - {test_str}', fontsize=16)
    figSoftRT.savefig(f'{output_dir}Softmax_RT_Psychometric_Curves_temp_{fixed_temp}_sigma_{fixed_sigma}_norm_{norm_bool}{wo_R_str}.png')


    # Hamacher RT: NH vs EH loop over sigma
    figHamRT, axes = plt.subplots(1, 2, figsize=(10, 6), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.13)
    axes = axes.flatten()
    # NH
    for sigma in desired_sigma_values:
        for file in all_files_NH:
            if f'sigmaSF_{sigma}_' in file and f'temp_{fixed_temp}_' in file:
                print(file)
                data_NH = np.load(file, allow_pickle=True).item()
                if wo_R: # remove -80 dB and y_val
                    data_NH = remove_R(data_NH)
                axes[0].scatter(data_NH['dB_list'], data_NH['y_Hamacher_RT'], label=f'σ={sigma}', color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
                axes[0].plot(data_NH['dB_list'], fit_best_sigmoid(data_NH['dB_list'], data_NH['y_Hamacher_RT']), color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
                axes[0].set_title(f'NH', fontsize=14)    
    # EH
        for file in all_files_EH:
            if f'sigmaSF_{sigma}_' in file and f'temp_{fixed_temp}_' in file:
                print(file)
                data_EH = np.load(file, allow_pickle=True).item()
                if wo_R: # remove -80 dB and y_val
                    data_EH = remove_R(data_EH)
                axes[1].scatter(data_EH['dB_list'], data_EH['y_Hamacher_RT'], label=f'σ={sigma}', color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
                axes[1].plot(data_EH['dB_list'], fit_best_sigmoid(data_EH['dB_list'], data_EH['y_Hamacher_RT']), color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
                axes[1].set_title(f'EH', fontsize=14)
    trans = mtransforms.ScaledTranslation(10/72, -5/72, figHamRT.dpi_scale_trans)
    for a, ax in enumerate(axes.flatten()):
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel('Percentage correct [%]',fontsize=14)
        ax.set_ylim((25, 101))
        ax.set_xlim((min(data_NH['dB_list']), max(data_NH['dB_list'])))
        ax.text(x_pos, y_pos, letters[a], transform=ax.transAxes + trans,
            fontsize=16, verticalalignment='top', color='black')
        ax.legend(loc='lower right')
        ax.grid(color='lightgray')
    plt.suptitle(f'Hamacher Psychometric Curves - {test_str}', fontsize=16)
    figHamRT.savefig(f'{output_dir}Hamacher_RT_Psychometric_Curves_sigma_{desired_sigma_values}{wo_R_str}.png')

    # Softmax comparing 4 figs: RT vs RTmax for fixed sigma and fixed temp
    figCompRTvsRTmax, axes = plt.subplots(1, 2, figsize=(10, 6), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.13, top=0.86)
    # NH
    for file in all_files_NH:
        if f'sigmaSF_{fixed_sigma}_temp_{fixed_temp}_' in file:
            print(file)
            data_NH = np.load(file, allow_pickle=True).item()
            if wo_R: # remove -80 dB and y_val
                data_NH = remove_R(data_NH)
            axes[0].scatter(data_NH['dB_list'], data_NH['y_soft_RT'], label='Softmax RT', color='blue')
            axes[0].plot(data_NH['dB_list'], fit_best_sigmoid(data_NH['dB_list'], data_NH['y_soft_RT']), color='blue')
            axes[0].scatter(data_NH['dB_list'], data_NH['y_soft_RTmax'], label='Softmax RT$_{max}$', color='orange')
            axes[0].plot(data_NH['dB_list'], fit_best_sigmoid(data_NH['dB_list'], data_NH['y_soft_RTmax']), color='orange')
            axes[0].set_title(f'NH', fontsize=14)
    # EH
    for file in all_files_EH:
        if f'sigmaSF_{fixed_sigma}_temp_{fixed_temp}_' in file:
            print(file)
            data_EH = np.load(file, allow_pickle=True).item()
            if wo_R: # remove -80 dB and y_val
                data_EH = remove_R(data_EH)
            axes[1].scatter(data_EH['dB_list'], data_EH['y_soft_RT'], label='Softmax RT', color='blue')
            axes[1].plot(data_EH['dB_list'], fit_best_sigmoid(data_EH['dB_list'], data_EH['y_soft_RT']), color='blue')
            axes[1].scatter(data_EH['dB_list'], data_EH['y_soft_RTmax'], label='Softmax RT$_{max}$', color='orange')
            axes[1].plot(data_EH['dB_list'], fit_best_sigmoid(data_EH['dB_list'], data_EH['y_soft_RTmax']), color='orange')
            axes[1].set_title(f'EH', fontsize=14)
    trans = mtransforms.ScaledTranslation(10/72, -5/72, figCompRTvsRTmax.dpi_scale_trans)
    for a, ax in enumerate(axes.flatten()):
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel('Percentage correct [%]',fontsize=14)
        ax.set_ylim((25, 101))
        ax.set_xlim((min(data_NH['dB_list']), max(data_NH['dB_list'])))
        ax.legend(loc='upper left')
        ax.grid(color='lightgray')
        ax.text(x_pos, y_pos, letters[a], transform=ax.transAxes + trans,
            fontsize=16, verticalalignment='top', color='black')
    plt.suptitle(f'Softmax Psychometric Curves - {test_str},\n comparing RT vs RT$_{{max}}$ with T={fixed_temp} and σ={fixed_sigma}', fontsize=16)
    figCompRTvsRTmax.savefig(f'{output_dir}Softmax_Comparison_RT_vs_RTmax_temp_{fixed_temp}_sigma_{fixed_sigma}_norm_{norm_bool}{wo_R_str}.png')

    # # comparing 55dB vs 65 dB NH for softmax RTmax
    # fig_compare_55_vs_65dB = plt.figure(figsize=(8,8))
    # # NH 55 dB
    # if platform.system() == 'Linux':
    #     folder_55dB = f'./output/{test}/NH/results/seed42/55dB/'  
    #     folder_65dB = f'./output/{test}/NH/results/seed42/65dB/'
    # else:
    #     folder_55dB = f'S:/python/SoftmaxInterpretationModel/output/{test}/NH/results/seed42/55dB/'  
    #     folder_65dB = f'S:/python/SoftmaxInterpretationModel/output/{test}/NH/results/seed42/65dB/'
    # all_files_NH_55dB = sorted(glob.glob(os.path.join(folder_55dB, f'*norm*{norm_bool}.npy')))
    # all_files_NH_65dB = sorted(glob.glob(os.path.join(folder_65dB, f'*norm*{norm_bool}.npy')))
    # for file in all_files_NH_55dB:
    #     if f'sigmaSF_{fixed_sigma}_temp_{fixed_temp}_' in file:
    #         print(file)
    #         data_NH_55dB = np.load(file, allow_pickle=True).item()
    #         if wo_R: # remove -80 dB and y_val
    #             data_NH_55dB = remove_R(data_NH_55dB)
    #         plt.scatter(data_NH_55dB['dB_list'], data_NH_55dB['y_soft_RTmax'], label='NH 55 dB', color='green')
    #         plt.plot(data_NH_55dB['dB_list'], fit_best_sigmoid(data_NH_55dB['dB_list'], data_NH_55dB['y_soft_RTmax']), color='green')
    #         # plt.set_title(f'NH 55 dB', fontsize=14)
    # # NH 65 dB
    # for file in all_files_NH_65dB:
    #     if f'sigmaSF_{fixed_sigma}_temp_{fixed_temp}_' in file:
    #         print(file)
    #         data_NH_65dB = np.load(file, allow_pickle=True).item()
    #         if wo_R: # remove -80 dB and y_val
    #             data_NH_65dB = remove_R(data_NH_65dB)
    #         plt.scatter(data_NH_65dB['dB_list'], data_NH_65dB['y_soft_RTmax'], label='NH 65 dB', color='blue')
    #         plt.plot(data_NH_65dB['dB_list'], fit_best_sigmoid(data_NH_65dB['dB_list'], data_NH_65dB['y_soft_RTmax']), color='blue')
    #         # plt.set_title(f'NH 65 dB', fontsize=14)
    # plt.xlabel(x_label, fontsize=14)
    # plt.ylabel('Percentage correct [%]',fontsize=14)
    # plt.ylim((25, 101))
    # plt.xlim((min(data_NH_55dB['dB_list']), max(data_NH_55dB['dB_list'])))
    # plt.legend(loc='upper left')
    # plt.grid(color='lightgray')
    # plt.title(f'Softmax RT$_{{max}}$ Psychometric Curves - {test_str},\n comparing NH 55 dB vs NH 65 dB with T={fixed_temp} and σ={fixed_sigma}', fontsize=16)
    # fig_compare_55_vs_65dB.savefig(f'{output_dir}Softmax_RTmax_Psychometric_Curves_compare_55dB_vs_65dB_temp_{fixed_temp}_sigma_{fixed_sigma}_norm_{norm_bool}{wo_R_str}.png')

    if test == 'AM':
        sigma_list_EH = np.array([0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.4, 0.6, 0.8])
        sigma_list_NH = np.array([0.14, 0.16, 0.18, 0.2, 0.4, 0.6, 0.8, 1.2, 1.6, int(2), int(4), int(6), int(8)])
        # create colormap
        cmap = plt.get_cmap('viridis', len(sigma_list_NH))
        colors_sigma = [cmap(i) for i in range(cmap.N)]
        folder_results_NH = f'./output/{test}/NH/results/seed42/65dB/Hamacher_only/' #f'S:/python/SoftmaxInterpretationModel/output/{test}/NH/results/seed42/Hamacher_only/'
        folder_results_EH = f'S:/python/SoftmaxInterpretationModel/output/{test}/EH/results/Hamacher_only/'
        all_files_NH_Hamacher_only = sorted(glob.glob(os.path.join(folder_results_NH, f'*.npy')))
        all_files_EH_Hamacher_only = sorted(glob.glob(os.path.join(folder_results_EH, f'*.npy')))
        # Hamacher RT: NH vs EH loop over sigma
        figHamRTfull, axes = plt.subplots(1, 2, figsize=(10, 6), sharex=True, sharey=True)
        axes = axes.flatten()
        # NH
        for s, sigma in enumerate(sigma_list_NH):
            if sigma > 1.8:
                sigma=int(sigma)
            print(sigma)
            for f, file in enumerate(all_files_NH_Hamacher_only):
                if f'sigmaSF_{sigma}.' in file:
                    print(file)
                    data_NH = np.load(file, allow_pickle=True).item()
                    if wo_R: # remove -80 dB and y_val
                        data_NH = remove_R(data_NH)
                    axes[0].scatter(data_NH['dB_list'], data_NH['y_Hamacher_RT'], label=f'σ={sigma}', color=colors_sigma[np.where(sigma_list_NH == sigma)[0][0]])
                    axes[0].plot(data_NH['dB_list'], fit_best_sigmoid(data_NH['dB_list'], data_NH['y_Hamacher_RT']), color=colors_sigma[np.where(sigma_list_NH == sigma)[0][0]])
                    axes[0].set_title(f'NH')    
        for sigma in sigma_list_EH:
            print(sigma)
        # EH
            for f, file in enumerate(all_files_EH_Hamacher_only):
                if f'sigmaSF_{sigma}.' in file:
                    print(file)
                    data_EH = np.load(file, allow_pickle=True).item()
                    if wo_R: # remove -80 dB and y_val
                        data_EH = remove_R(data_EH)
                    axes[1].scatter(data_EH['dB_list'], data_EH['y_Hamacher_RT'], label=f'σ={sigma}', color=colors_sigma[np.where(sigma_list_EH == sigma)[0][0]])
                    axes[1].plot(data_EH['dB_list'], fit_best_sigmoid(data_EH['dB_list'], data_EH['y_Hamacher_RT']), color=colors_sigma[np.where(sigma_list_EH == sigma)[0][0]])
                    axes[1].set_title(f'EH') 
        for ax in axes:
            ax.set_xlabel(x_label, fontsize=14)
            ax.set_ylabel('Percentage correct [%]',fontsize=14)
            ax.set_ylim((25, 101))
            ax.set_xlim((min(data_EH['dB_list']), max(data_EH['dB_list'])))
            ax.grid(color='lightgray')
            ax.legend() 
        plt.suptitle(f'Hamacher Psychometric Curves - {test_str}', fontsize=16)
        figHamRTfull.savefig(f'{output_dir}Hamacher_RT_Psychometric_Curves_sigma_NH{sigma_list_NH}_{NH_dB}dB_EH{sigma_list_EH}{wo_R_str}.png')

    # Hamacher figure RT vs RTmax for varying sigma --> ran it incorrectly, so commented out
    figHamRT_vs_RTmax, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    # NH subplot 1 en 2
    for sigma in desired_sigma_values:
        for file in all_files_NH:
            if f'sigmaSF_{sigma}_' in file and f'temp_{fixed_temp}_' in file:
                print(file)
                data_NH = np.load(file, allow_pickle=True).item()
                if wo_R: # remove -80 dB and y_val
                    data_NH = remove_R(data_NH)
                axes[0].scatter(data_NH['dB_list'], data_NH['y_Hamacher_RT'], label=f'σ={sigma}', color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
                axes[0].plot(data_NH['dB_list'], fit_best_sigmoid(data_NH['dB_list'], data_NH['y_Hamacher_RT']), color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
                axes[0].set_title(f'NH RT')
                axes[0].legend()
                axes[1].scatter(data_NH['dB_list'], data_NH['y_Hamacher_RTmax'], label=f'σ={sigma}', color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
                axes[1].plot(data_NH['dB_list'], fit_best_sigmoid(data_NH['dB_list'], data_NH['y_Hamacher_RTmax']), color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
                axes[1].set_title(f'NH RTmax')
                axes[1].legend()
    # EH subplot 3 en 4
        for file in all_files_EH:
            if f'sigmaSF_{sigma}_' in file and f'temp_{fixed_temp}_' in file:
                print(file)
                data_EH = np.load(file, allow_pickle=True).item()
                if wo_R: # remove -80 dB and y_val
                    data_EH = remove_R(data_EH)
                axes[2].scatter(data_EH['dB_list'], data_EH['y_Hamacher_RT'], label=f'σ={sigma}', color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
                axes[2].plot(data_EH['dB_list'], fit_best_sigmoid(data_EH['dB_list'], data_EH['y_Hamacher_RT']), color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
                axes[2].set_title(f'EH - RT')
                axes[3].scatter(data_EH['dB_list'], data_EH['y_Hamacher_RTmax'], label=f'σ={sigma}', color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
                axes[3].plot(data_EH['dB_list'], fit_best_sigmoid(data_EH['dB_list'], data_EH['y_Hamacher_RTmax']), color=colors_sigma[np.where(sigma_values == sigma)[0][0]])    
                axes[3].set_title(f'EH - RTmax')
    trans = mtransforms.ScaledTranslation(10/72, -5/72, figHamRT_vs_RTmax.dpi_scale_trans)
    for a, ax in enumerate(axes.flatten()):
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel('Percentage correct [%]',fontsize=14)
        ax.set_ylim((25, 101))
        ax.grid(color='lightgray')
        ax.set_xlim((min(data_NH['dB_list']), max(data_NH['dB_list'])))
        ax.legend()
        ax.text(x_pos, y_pos, letters[a], transform=ax.transAxes + trans,
            fontsize=16, verticalalignment='top', color='black')
    plt.suptitle(f'Hamacher Psychometric Curves - {test_str}, comparing RT vs RTmax', fontsize=16)
    figHamRT_vs_RTmax.savefig(f'{output_dir}Hamacher_RT_vs_RTmax_Psychometric_Curves_sigma{desired_sigma_values}{wo_R_str}.png')


    plt.show()