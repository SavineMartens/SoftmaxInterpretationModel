import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and plot results for psychometric curve")
    parser.add_argument("-hearing", type=str, default="EH", help="Hearing type (NH or EH)")
    parser.add_argument("-test", type=str, default="AM", help="Test type (AM or FM)")
    args = parser.parse_args()

    hearing = args.hearing
    test = args.test

    folder_results_NH = f'S:/python/SoftmaxInterpretationModel/output/{test}/NH/results/' #f'./output/{test}/{hearing}/results/'
    folder_results_EH = f'S:/python/SoftmaxInterpretationModel/output/{test}/EH/results/'

    all_files_NH = sorted(glob.glob(os.path.join(folder_results_NH, '*norm*.npy')))
    all_files_EH = sorted(glob.glob(os.path.join(folder_results_EH, '*norm*.npy')))

    # read values from filenames
    sigma_values_NH = []
    temp_values_NH = []
    sigma_values_EH = []
    temp_values_EH = []

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

    # create colormap
    cmap = plt.get_cmap('viridis', len(sigma_values))
    colors_sigma = [cmap(i) for i in range(cmap.N)]

    cmap = plt.get_cmap('plasma', len(temp_values))
    colors_temp = [cmap(i) for i in range(cmap.N)]

    for sigma in sigma_values:
        data_NH = np.load([f for f in all_files_NH if f'sigmaSF_{sigma}_' in f][0], allow_pickle=True).item()
        data_EH = np.load([f for f in all_files_EH if f'sigmaSF_{sigma}_' in f][0], allow_pickle=True).item()
        temp = data_NH['temperature']
        if temp not in temp_values:
            continue
        # figure Hamacher RTmax
        fig_Hamacher_RTmax, axes = plt.subplots(1, 2, figsize=(10, 10))
        axes = axes.flatten()
        axes[0].scatter(data_EH['dB_list'], data_EH['y_Hamacher_RTmax'], label=f'EH, σ={sigma}', color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
        axes[0].plot(data_EH['dB_list'], data_EH['y_fit_Hamacher_RTmax'], color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
        axes[0].set_xlabel('dB re Masker')
        axes[0].set_ylabel('Percentage correct [%]')
        axes[0].set_title('Hamacher: RT_{max} EH')
        axes[0].set_ylim((30, 101))
        axes[0].legend()
        axes[1].scatter(data_NH['dB_list'], data_NH['y_Hamacher_RT'], label=f'NH, σ={sigma}', color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
        axes[1].plot(data_NH['dB_list'], data_NH['y_fit_Hamacher_RT'], color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
        axes[1].set_xlabel('dB re Masker')
        axes[1].set_ylabel('Percentage correct [%]')
        axes[1].set_title('Hamacher: RT NH')
        axes[1].set_ylim((30, 101))
        axes[1].legend()
        fig_Hamacher_RTmax.suptitle(f'Hamacher RTmax- {test}', fontsize=16)
        plt.tight_layout()

        # figure Hamacher RT
        fig_Hamacher_RT, axes = plt.subplots(1, 2, figsize=(10, 10))
        axes = axes.flatten()
        axes[0].scatter(data_EH['dB_list'], data_EH['y_Hamacher_RT'], label=f'EH, σ={sigma}', color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
        axes[0].plot(data_EH['dB_list'], data_EH['y_fit_Hamacher_RT'], color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
        axes[0].set_xlabel('dB re Masker')
        axes[0].set_ylabel('Percentage correct [%]')
        axes[0].set_title('Hamacher: RT EH')
        axes[0].set_ylim((30, 101))
        axes[0].legend()
        axes[1].scatter(data_NH['dB_list'], data_NH['y_Hamacher_RT'], label=f'NH, σ={sigma}', color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
        axes[1].plot(data_NH['dB_list'], data_NH['y_fit_Hamacher_RT'], color=colors_sigma[np.where(sigma_values == sigma)[0][0]])
        axes[1].set_xlabel('dB re Masker')
        axes[1].set_ylabel('Percentage correct [%]')
        axes[1].set_title('Hamacher: RT NH')
        axes[1].set_ylim((30, 101))
        axes[1].legend()
        fig_Hamacher_RT.suptitle(f'Hamacher RT- {test}', fontsize=16)
        plt.tight_layout()

        # figure: Softmax RTmax, fixed sigma, varying temp
        fig_Softmax_RTmax, axes = plt.subplots(1, 2, figsize=(10, 10), figname=f'Softmax RTmax - {test}, sigma={sigma}')
        axes = axes.flatten()
        


plt.show()