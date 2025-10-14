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

    plt.figure(f'Psychometric Curve - {hearing} - {test}: fixed sigma', figsize=(10, 6))
    for file in all_files:
        data = np.load(file, allow_pickle=True).item()


