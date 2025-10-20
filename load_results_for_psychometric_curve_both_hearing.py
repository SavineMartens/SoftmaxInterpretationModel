import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import argparse

# To do
# [ ] create neurograms with smaller probe amplitudes for MP
# [ ] run 3AFC & AMP with values from doc
# [ ] als run with True normalization
# 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and plot results for psychometric curve")
    parser.add_argument("-test", type=str, default="MP", help="Test type (AM or FM)")
    parser.add_argument('-norm', default=False, action='store_true')
    args = parser.parse_args()

    test = args.test
    if args.norm:
        norm_bool = True
    else:
        norm_bool = False

    try:
        folder_results_NH = f'S:/python/SoftmaxInterpretationModel/output/{test}/NH/results/' 
        folder_results_EH = f'S:/python/SoftmaxInterpretationModel/output/{test}/EH/results/'
    except:
        print('Connection to cluster not working')
        folder_results_NH = f'./output/{test}/NH/results/'
        folder_results_EH = f'./output/{test}/EH/results/'

    all_files_NH = sorted(glob.glob(os.path.join(folder_results_NH, f'*norm*{norm_bool}.npy')))
    all_files_EH = sorted(glob.glob(os.path.join(folder_results_EH, f'*norm*{norm_bool}.npy')))

    # read values from filenames
    sigma_values_NH = []
    temp_values_NH = []
    sigma_values_EH = []
    temp_values_EH = []

    # desired values
    if test == 'AM':
        desired_sigma_values = [0.02, 0.04, 0.08, 0.2, 0.4, 0.8]
        desired_temp_values = [0.001, 0.003, 0.009, 0.027, 0.081, 0.243, 0.729, 2.187, 6.561]
        fixed_sigma = 0.08
        fixed_temp = 0.081
        x_label = 'Modulation Depth (dB)'
    elif test == 'MP':
        desired_sigma_values = [0.001, 0.003, 0.009, 0.027, 0.081, 0.243, 0.729, 2.187, 6.561]
        desired_temp_values = [0.001, 0.003, 0.009, 0.027, 0.081, 0.243, 0.729, 2.187, 6.561]
        fixed_sigma = 0.081
        fixed_temp = 0.027
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

    # desired_sigma_values = [s for s in desired_sigma_values if s in sigma_values]
    # desired_temp_values = [t for t in desired_temp_values if t in temp_values]

    # create colormap
    cmap = plt.get_cmap('viridis', len(sigma_values))
    colors_sigma = [cmap(i) for i in range(cmap.N)]

    cmap = plt.get_cmap('plasma', len(temp_values))
    colors_temp = [cmap(i) for i in range(cmap.N)]

    # figure 4 subplots: softmax RTmax fixed temp, varying sigma, NH vs EH
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    # subplot 1: fixed sigma EH
    for temp in desired_temp_values:
        for file in all_files_EH:
            if f'temp_{temp}_' in file and f'sigmaSF_{fixed_sigma}_' in file:
                print(file)
                data_EH = np.load(file, allow_pickle=True).item()
                sigma = data_EH['sigma_SF']
                axes[0,0].scatter(data_EH['dB_list'], data_EH['y_soft_RTmax'], label=f'EH, T={temp}', color=colors_temp[np.where(temp_values == temp)[0][0]])
                try:
                    axes[0,0].plot(data_EH['dB_list'], data_EH['y_fit_soft_RTmax'], color=colors_temp[np.where(temp_values == temp)[0][0]])
                except ValueError:
                    pass
                axes[0,0].set_title(f'NH, fixed sigma={fixed_sigma}')
                axes[0,0].legend()

    # subplot 2: fixed temp EH
    for sigma in desired_sigma_values:
        for file in all_files_EH:
            if f'temp_{fixed_temp}_' in file and f'sigmaSF_{sigma}_' in file:
                print(file)
                data_EH = np.load(file, allow_pickle=True).item()
                sigma = data_EH['sigma_SF']
                axes[0,1].scatter(data_EH['dB_list'], data_EH['y_soft_RTmax'], label=f'EH, sigma={sigma}', color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
                try:
                    axes[0,1].plot(data_EH['dB_list'], data_EH['y_fit_soft_RTmax'], color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
                except ValueError:
                    pass
                axes[0,1].set_title(f'EH, fixed sigma={fixed_temp}')
                axes[0,1].legend()
    # subplot 3: fixed sigma NH
    for temp in desired_temp_values:
        for file in all_files_NH:
            if f'sigmaSF_{fixed_sigma}_' in file and f'temp_{temp}_' in file:
                print(file)
                data_NH = np.load(file, allow_pickle=True).item()
                temp = data_NH['temperature']
                axes[1,0].scatter(data_NH['dB_list'], data_NH['y_soft_RTmax'], label=f'NH, temp={temp}', color=colors_temp[np.where(temp_values == temp)[0][0]])
                try:
                    axes[1,0].plot(data_NH['dB_list'], data_NH['y_fit_soft_RTmax'], color=colors_temp[np.where(temp_values == temp)[0][0]])
                except ValueError:
                    pass
                axes[1,0].set_title(f'NH, fixed sigma={fixed_sigma}')
                axes[1,0].legend()
    # subplot 4: fixed temp NH
    for sigma in desired_sigma_values:
        for file in all_files_NH:
            if f'sigmaSF_{sigma}_' in file and f'temp_{fixed_temp}_' in file:
                print(file)
                data_NH = np.load(file, allow_pickle=True).item()
                temp = data_NH['temperature']
                axes[1,1].scatter(data_NH['dB_list'], data_NH['y_soft_RTmax'], label=f'NH, σ={sigma}', color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
                try:
                    axes[1,1].plot(data_NH['dB_list'], data_NH['y_fit_soft_RTmax'], color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
                except ValueError:
                    pass
                axes[1,1].set_title(f'NH, fixed temp={fixed_temp}')
                axes[1,1].legend()

    for ax in axes.flatten():
        ax.set_xlabel(x_label)
        ax.set_ylabel('Percentage correct [%]')
        ax.set_ylim((30, 101))
    plt.suptitle(f'Softmax RTmax Psychometric Curves - {test}', fontsize=16)

    # for sigma in sigma_values:
    #     print(sigma)
    #     data_NH = np.load([f for f in all_files_NH if f'sigmaSF_{sigma}_' in f][0], allow_pickle=True).item()
    #     data_EH = np.load([f for f in all_files_EH if f'sigmaSF_{sigma}_' in f][0], allow_pickle=True).item()
    #     # temp = data_NH['temperature']
    #     # if temp not in temp_values:
    #         # continue
    #     # figure Hamacher RTmax
    #     fig_Hamacher_RTmax, axes = plt.subplots(1, 2, figsize=(10, 10), num=f'Hamacher RTmax - {test}')
    #     # fig_Hamacher_RTmax.canvas.set_window_title(f'Hamacher RTmax - {test}')
    #     axes = axes.flatten()
    #     axes[0].scatter(data_EH['dB_list'], data_EH['y_Hamacher_RTmax'], label=f'EH, σ={sigma}', color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
    #     try:
    #         axes[0].plot(data_EH['dB_list'], data_EH['y_fit_Hamacher_RTmax'], color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
    #     except ValueError:
    #         pass
    #     axes[0].set_xlabel('dB re Masker')
    #     axes[0].set_ylabel('Percentage correct [%]')
    #     axes[0].set_title('Hamacher: RT_{max} EH')
    #     axes[0].set_ylim((30, 101))
    #     axes[0].legend()
    #     axes[1].scatter(data_NH['dB_list'], data_NH['y_Hamacher_RT'], label=f'NH, σ={sigma}', color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
    #     try:
    #         axes[1].plot(data_NH['dB_list'], data_NH['y_fit_Hamacher_RT'], color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
    #     except ValueError:
    #         pass
    #     axes[1].set_xlabel('dB re Masker')
    #     axes[1].set_ylabel('Percentage correct [%]')
    #     axes[1].set_title('Hamacher: RT NH')
    #     axes[1].set_ylim((30, 101))
    #     axes[1].legend()
    #     fig_Hamacher_RTmax.suptitle(f'Hamacher RTmax- {test}', fontsize=16)
    #     # plt.tight_layout()

    #     # figure Hamacher RT
    #     fig_Hamacher_RT, axes = plt.subplots(1, 2, figsize=(10, 10), num=f'Hamacher RT - {test}')
    #     # fig_Hamacher_RT.canvas.set_window_title(f'Hamacher RT - {test}')
    #     axes = axes.flatten()
    #     axes[0].scatter(data_EH['dB_list'], data_EH['y_Hamacher_RT'], label=f'EH, σ={sigma}', color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
    #     try:
    #         axes[0].plot(data_EH['dB_list'], data_EH['y_fit_Hamacher_RT'], color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
    #     except ValueError:
    #         pass
    #     axes[0].set_xlabel('dB re Masker')
    #     axes[0].set_ylabel('Percentage correct [%]')
    #     axes[0].set_title('Hamacher: RT EH')
    #     axes[0].set_ylim((30, 101))
    #     axes[0].legend()
    #     axes[1].scatter(data_NH['dB_list'], data_NH['y_Hamacher_RT'], label=f'NH, σ={sigma}', color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
    #     try:
    #         axes[1].plot(data_NH['dB_list'], data_NH['y_fit_Hamacher_RT'], color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
    #     except ValueError:
    #         pass
    #     axes[1].set_xlabel('dB re Masker')
    #     axes[1].set_ylabel('Percentage correct [%]')
    #     axes[1].set_title('Hamacher: RT NH')
    #     axes[1].set_ylim((30, 101))
    #     axes[1].legend()
    #     fig_Hamacher_RT.suptitle(f'Hamacher RT- {test}', fontsize=16)
    #     # plt.tight_layout()

    #     # figure: Softmax RTmax, fixed sigma, varying temp
    #     # fig_Softmax_RTmax, axes = plt.subplots(1, 2, figsize=(10, 10), figname=f'Softmax RTmax - {test}, sigma={sigma}')
    #     # axes = axes.flatten()
    #     waitforbuttonpress = True


plt.show()