import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from evaluation_metrics import get_FROC_fn
import pandas as pd


def plot_lesion_sizes():
    dl_info = pd.read_csv('../filtered_deeplesion_annotations/DL_info.csv')
    dl_info = dl_info.loc[((dl_info['Train_Val_Test'] == 1) & (dl_info['Possibly_noisy'] == 0))]
    # Filter by lesion size
    dl_info['Lesion_size_mm'] = 0
    for idx, row in dl_info.iterrows():
        lesion_diameter_long_px = eval(row['Lesion_diameters_Pixel_'].split(',')[0])
        lesion_diameter_short_px = eval(row['Lesion_diameters_Pixel_'].split(',')[1])
        pixel_spacing = eval(row['Spacing_mm_px_'].split(',')[0])
        # Average lesion size
        lesion_size_mm = (lesion_diameter_long_px + lesion_diameter_short_px)/2 * pixel_spacing
        dl_info.at[float(idx), 'Lesion_size_mm'] = lesion_size_mm
    
    num_sizes = []

    filtered_dl_info = dl_info.loc[(dl_info['Lesion_size_mm']) < 10]
    index = filtered_dl_info.index
    num_sizes.append(len(index))
    print(len(index))

    filtered_dl_info = dl_info.loc[((dl_info['Lesion_size_mm'] >= 10) & (dl_info['Lesion_size_mm'] <= 30))]
    index = filtered_dl_info.index
    num_sizes.append(len(index))
    print(len(index))

    filtered_dl_info = dl_info.loc[((dl_info['Lesion_size_mm'] > 30) & (dl_info['Lesion_size_mm'] <= 50))]
    index = filtered_dl_info.index
    num_sizes.append(len(index))
    print(len(index))

    filtered_dl_info = dl_info.loc[((dl_info['Lesion_size_mm'] > 50) & (dl_info['Lesion_size_mm'] <= 70))]
    index = filtered_dl_info.index
    num_sizes.append(len(index))
    print(len(index))

    filtered_dl_info = dl_info.loc[((dl_info['Lesion_size_mm'] > 70) & (dl_info['Lesion_size_mm'] <= 100))]
    index = filtered_dl_info.index
    num_sizes.append(len(index))
    print(len(index))

    filtered_dl_info = dl_info.loc[(dl_info['Lesion_size_mm']) > 100]
    index = filtered_dl_info.index
    num_sizes.append(len(index))

    
    lesion_sizes = ("< 10", "10 - 30", "30 - 50", "50 - 70", "70 - 100", "> 100")
    y_pos = np.arange(len(lesion_sizes))
 
    plt.bar(y_pos, num_sizes)

    plt.xticks(y_pos, lesion_sizes)
    plt.ylabel('Anzahl Läsionen')
    plt.title('Anzahl Läsionen mit verschiedenen Durchschnittsgrößen')

    plt.show()

def plot_lesion_types():
    dl_info = pd.read_csv('../filtered_deeplesion_annotations/DL_Info.csv')
    
    num_types = []
    dl_info = dl_info.loc[((dl_info['Train_Val_Test'] == 3) & (dl_info['Possibly_noisy'] == 0))]
    filtered_dl_info = dl_info.loc[(dl_info['Coarse_lesion_type'] == 1)]
    index = filtered_dl_info.index
    num_types.append(len(index))
    print(len(index))
    filtered_dl_info = dl_info.loc[(dl_info['Coarse_lesion_type'] == 2)]
    index = filtered_dl_info.index
    num_types.append(len(index))
    print(len(index))
    filtered_dl_info = dl_info.loc[(dl_info['Coarse_lesion_type'] == 3)]
    index = filtered_dl_info.index
    num_types.append(len(index))
    print(len(index))
    filtered_dl_info = dl_info.loc[(dl_info['Coarse_lesion_type'] == 4)]
    index = filtered_dl_info.index
    num_types.append(len(index))
    print(len(index))
    filtered_dl_info = dl_info.loc[(dl_info['Coarse_lesion_type'] == 5)]
    index = filtered_dl_info.index
    num_types.append(len(index))
    print(len(index))
    filtered_dl_info = dl_info.loc[(dl_info['Coarse_lesion_type'] == 6)]
    index = filtered_dl_info.index
    num_types.append(len(index))
    print(len(index))
    filtered_dl_info = dl_info.loc[(dl_info['Coarse_lesion_type'] == 7)]
    index = filtered_dl_info.index
    num_types.append(len(index))
    print(len(index))
    filtered_dl_info = dl_info.loc[(dl_info['Coarse_lesion_type'] == 8)]
    index = filtered_dl_info.index
    num_types.append(len(index))
    print(len(index))


    
    lesion_types = ("BN", "AB", "ME", "LV", "LU", "KD", "ST", "PV")
    y_pos = np.arange(len(lesion_types))
    plt.bar(y_pos, num_types, color="green")

    plt.xticks(y_pos, lesion_types)
    plt.ylabel('Anzahl Läsionen')
    plt.title('Anzahl Läsionen mit verschiedenen Typen')

    plt.show()


if __name__ == "__main__":
    plot_lesion_sizes()
    plot_lesion_types()
    




