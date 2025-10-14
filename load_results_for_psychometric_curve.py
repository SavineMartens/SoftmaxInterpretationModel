import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and plot results for psychometric curve")
    parser.add_argument("-hearing", type=str, default="NH", help="Hearing type (NH or EH)")
    parser.add_argument("-test", type=str, default="AM", help="Test type (AM or FM)")
    args = parser.parse_args()

    hearing = args.hearing
    test = args.test

    folder_results = f'./output/{test}/{hearing}/results/'

    all_files = sorted(glob.glob(os.path.join(folder_results, '*.npy')))

    # read values from filenames
    sigma_values = []
    temp_values = []
    for file in all_files:
        base = os.path.basename(file)
        sigma_str = float(base[base.find('sigmaSF_') + len('sigmaSF_'): base.find('_temp')])
        temp_str = float(base[base.find('temp_') + len('temp_'): base.find('.npy')])
        sigma_values.append(float(sigma_str))
        temp_values.append(float(temp_str))
    sigma_values = np.unique(np.array(sigma_values))
    temp_values = np.unique(np.array(temp_values))
    print(f"Found sigma values: {sigma_values}")
    print(f"Found temperature values: {temp_values}")

    # create colormap
    cmap = plt.get_cmap('viridis', len(sigma_values))
    colors = [cmap(i) for i in range(cmap.N)]

    Hamacher_sigma_list = []

    for file in all_files:
        data = np.load(file, allow_pickle=True).item()
        temp = data['temperature']
        sigma = data['sigma_SF']

        if sigma not in Hamacher_sigma_list:
            Hamacher_sigma_list.append(sigma)

            # figure Hamacher RTmax 
            plt.figure(f'Hamacher RTmax and RT - {hearing} - {test}', figsize=(10, 6))
            plt.subplot(1,2,1)
            plt.scatter(data['dB_list'], data['y_Hamacher_RTmax'], label=f'σ={sigma}', color=colors[np.where(sigma_values == sigma)[0][0]])
            # plt.plot(data['dB_list'], data['y_fit_Hamacher_RTmax'], color=colors[np.where(sigma_values == sigma)[0][0]])
            plt.xlabel('dB re Masker')
            plt.ylabel('Percentage correct [%]')
            plt.title('Hamacher: RTmax')
            plt.ylim((30, 101))
            plt.legend()
            plt.subplot(1,2,2)
            plt.scatter(data['dB_list'], data['y_Hamacher_RT'], label=f'σ={sigma}', color=colors[np.where(sigma_values == sigma)[0][0]])
            # plt.plot(data['dB_list'], data['y_fit_Hamacher_RT'], color=colors[np.where(sigma_values == sigma)[0][0]])
            plt.xlabel('dB re Masker')
            plt.title('Hamacher: RT')
            plt.ylim((30, 101))
            plt.legend()


        # # figure comparing Hamacher_RT and Hamacher_RTmax
        # plt.figure(f'Hamacher Comparison - {hearing} - {test}', figsize=(10, 6))
        # for file in all_files:
        #     data = np.load(file, allow_pickle=True).item()
        #     plt.plot(data['x'], data['y_Hamacher_RT'], label=f'RT')
        #     plt.plot(data['x'], data['y_Hamacher_RTmax'], label=f'Hamacher_RTmax - {file}', linestyle='--')
        # plt.xlabel('dB re Masker')
        # plt.ylabel('Percentage correct [%]')
        # plt.title(f'Hamacher Comparison: RT vs RT_max - {hearing} - {test}')
        # plt.legend()
        # plt.grid()
        # plt.show()

        # # figure comparing softmax_RT and softmax_RTmax
        # plt.figure(f'Softmax Comparison - {hearing} - {test}', figsize=(10, 6))
        # for file in all_files:
        #     data = np.load(file, allow_pickle=True).item()
        #     plt.plot(data['x'], data['y_softmax_RT'], label=f'Softmax_RT - {file}')
        #     plt.plot(data['x'], data['y_softmax_RTmax'], label=f'Softmax_RTmax - {file}', linestyle='--')
        # plt.xlabel('dB re Masker')
        # plt.ylabel('Percentage correct [%]')
        # plt.title(f'Softmax Comparison - {hearing} - {test}')
        # plt.legend()
        # plt.grid()
        # plt.show()

plt.show()