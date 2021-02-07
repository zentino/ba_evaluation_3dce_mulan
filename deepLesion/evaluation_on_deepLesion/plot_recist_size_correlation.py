import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


if __name__ == "__main__":
    
    dl_info = pd.read_csv('../filtered_deeplesion_annotations/DL_info.csv')
    # Filter by dataset split
    dl_info = dl_info.loc[(dl_info['Possibly_noisy']) == 0]
    dl_info['Lesion_size_mm'] = 0
    for idx, row in dl_info.iterrows():
        lesion_diameter_long_px = eval(row['Lesion_diameters_Pixel_'].split(',')[0])
        lesion_diameter_short_px = eval(row['Lesion_diameters_Pixel_'].split(',')[1])
        pixel_spacing = eval(row['Spacing_mm_px_'].split(',')[0])
        # Average lesion size
        lesion_size_mm = (lesion_diameter_long_px + lesion_diameter_short_px)/2 * pixel_spacing
        dl_info.at[float(idx), 'Lesion_size_mm'] = lesion_size_mm

    dl_info_lt_10 = dl_info.loc[(dl_info['Lesion_size_mm'] < 10)]
    dl_info_gte_10_lte_30 = dl_info.loc[((dl_info['Lesion_size_mm'] <= 30) & (dl_info['Lesion_size_mm'] >= 10))]
    dl_info_gt_30_lte_50 = dl_info.loc[((dl_info['Lesion_size_mm'] > 30) & (dl_info['Lesion_size_mm'] <= 50))]
    dl_info_gt_50_lte_70 = dl_info.loc[((dl_info['Lesion_size_mm'] > 50) & (dl_info['Lesion_size_mm'] <= 70))]
    dl_info_gt_70_lte_100 = dl_info.loc[((dl_info['Lesion_size_mm'] > 70) & (dl_info['Lesion_size_mm'] <= 100))]
    dl_info_gt_100 = dl_info.loc[(dl_info['Lesion_size_mm'] > 100)]
    
    dl_infos = [dl_info_lt_10, dl_info_gte_10_lte_30, dl_info_gt_30_lte_50, dl_info_gt_50_lte_70, dl_info_gt_70_lte_100, dl_info_gt_100]
    lesion_sizes = ("< 10", "10 - 30", "30 - 50", "50 - 70", "70 - 100", "> 100")
    avg_ratios = []
    for dl_info in dl_infos:
        ratios = []
        for idx, row in dl_info.iterrows():
            lesion_diameter_long_px = eval(row['Lesion_diameters_Pixel_'].split(',')[0])
            lesion_diameter_short_px = eval(row['Lesion_diameters_Pixel_'].split(',')[1])
            pixel_spacing = eval(row['Spacing_mm_px_'].split(',')[0])
     
            lesion_diameter_long_mm = lesion_diameter_long_px * pixel_spacing
            lesion_diameter_short_mm = lesion_diameter_short_px * pixel_spacing

            ratio = 100/lesion_diameter_long_mm * lesion_diameter_short_mm
            ratios.append(ratio)
        avg_ratios.append(np.mean(ratios))
    
    y_pos = np.arange(len(lesion_sizes))
 
    plt.bar(y_pos, avg_ratios)

    plt.xticks(y_pos, lesion_sizes)
    plt.ylabel('Verhältnis kleiner und großer Durchmesser')
    plt.title('Verhältnis von kleinem und großem \nDurchmesser für verschiedene Läsionsgrößen')

    plt.show()