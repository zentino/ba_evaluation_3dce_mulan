import pandas as pd


if __name__ == "__main__":

    dl_info = pd.read_csv('DL_info.csv')
    # Filter by lesion type
    dl_info = dl_info.loc[(dl_info['Train_Val_Test']) == 3]
    filtered_dl_info = dl_info.loc[(dl_info['Coarse_lesion_type'] == 1)]
    filtered_dl_info.to_csv('DL_info_bone.csv', index=False)
    filtered_dl_info = dl_info.loc[(dl_info['Coarse_lesion_type'] == 2)]
    filtered_dl_info.to_csv('DL_info_abdomen.csv', index=False)
    filtered_dl_info = dl_info.loc[(dl_info['Coarse_lesion_type'] == 3)]
    filtered_dl_info.to_csv('DL_info_mediastinum.csv', index=False)
    filtered_dl_info = dl_info.loc[(dl_info['Coarse_lesion_type'] == 4)]
    filtered_dl_info.to_csv('DL_info_liver.csv', index=False)
    filtered_dl_info = dl_info.loc[(dl_info['Coarse_lesion_type'] == 5)]
    filtered_dl_info.to_csv('DL_info_lung.csv', index=False)
    filtered_dl_info = dl_info.loc[(dl_info['Coarse_lesion_type'] == 6)]
    filtered_dl_info.to_csv('DL_info_kidney.csv', index=False)
    filtered_dl_info = dl_info.loc[(dl_info['Coarse_lesion_type'] == 7)]
    filtered_dl_info.to_csv('DL_info_soft_tissue.csv', index=False)
    filtered_dl_info = dl_info.loc[(dl_info['Coarse_lesion_type'] == 8)]
    filtered_dl_info.to_csv('DL_info_pelvis.csv', index=False)

    # Filter by lesion size
    dl_info['Lesion_size_mm'] = 0
    print(dl_info.columns)
    for idx, row in dl_info.iterrows():
        lesion_diameter_long_px = eval(row['Lesion_diameters_Pixel_'].split(',')[0])
        lesion_diameter_short_px = eval(row['Lesion_diameters_Pixel_'].split(',')[1])
        pixel_spacing = eval(row['Spacing_mm_px_'].split(',')[0])
        # Average lesion size
        lesion_size_mm = (lesion_diameter_long_px + lesion_diameter_short_px)/2 * pixel_spacing
        dl_info.at[float(idx), 'Lesion_size_mm'] = lesion_size_mm

    filtered_dl_info = dl_info.loc[(dl_info['Lesion_size_mm'] < 10)]
    filtered_dl_info.to_csv('DL_info_size_lt_10.csv', index=False)
    filtered_dl_info = dl_info.loc[((dl_info['Lesion_size_mm'] <= 30) & (dl_info['Lesion_size_mm'] >= 10))]
    filtered_dl_info.to_csv('DL_info_size_gte_10_lte_30.csv', index=False)
    filtered_dl_info = dl_info.loc[((dl_info['Lesion_size_mm'] > 30) & (dl_info['Lesion_size_mm'] <= 50))]
    filtered_dl_info.to_csv('DL_info_size_gt_30_lte_50.csv', index=False)
    filtered_dl_info = dl_info.loc[((dl_info['Lesion_size_mm'] > 50) & (dl_info['Lesion_size_mm'] <= 70))]
    filtered_dl_info.to_csv('DL_info_size_gt_50_lte_70.csv', index=False)
    filtered_dl_info = dl_info.loc[((dl_info['Lesion_size_mm'] > 70) & (dl_info['Lesion_size_mm'] <= 100))]
    filtered_dl_info.to_csv('DL_info_size_gt_70_lte_100.csv', index=False)
    filtered_dl_info = dl_info.loc[(dl_info['Lesion_size_mm'] > 100)]
    filtered_dl_info.to_csv('DL_info_size_gt_100.csv', index=False)


