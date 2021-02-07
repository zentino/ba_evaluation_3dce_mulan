import pandas as pd


if __name__ == "__main__":

    # TCGA 350
    tcga_info = pd.read_csv('tcga_info_fake_recist_0.5_350.csv')
    # # Filter by lesion type
    filtered_tcga_info = tcga_info.loc[(tcga_info['anatomy'] == 'Ovarian')]
    filtered_tcga_info.to_csv('tcga_info_fake_recist_0.5_350_OV.csv', index=False)
    filtered_tcga_info = tcga_info.loc[(tcga_info['anatomy'] == 'Renal')]
    filtered_tcga_info.to_csv('tcga_info_fake_recist_0.5_350_KD.csv', index=False)
    filtered_tcga_info = tcga_info.loc[(tcga_info['anatomy'] == 'Liver')]
    filtered_tcga_info.to_csv('tcga_info_fake_recist_0.5_350_LV.csv', index=False)
    filtered_tcga_info = tcga_info.loc[(tcga_info['anatomy'] == 'Lung')]
    filtered_tcga_info.to_csv('tcga_info_fake_recist_0.5_350_LU.csv', index=False)
    # Filter by lesion size
    filtered_tcga_info = tcga_info.loc[(tcga_info['length'] < 10)]
    filtered_tcga_info.to_csv('tcga_info_fake_recist_0.5_350_size_lt_10.csv', index=False)
    filtered_tcga_info = tcga_info.loc[((tcga_info['length'] <= 30) & (tcga_info['length'] >= 10))]
    filtered_tcga_info.to_csv('tcga_info_fake_recist_0.5_350_size_gte_10_lte_30.csv', index=False)
    filtered_tcga_info = tcga_info.loc[((tcga_info['length'] > 30) & (tcga_info['length'] <= 50))]
    filtered_tcga_info.to_csv('tcga_info_fake_recist_0.5_350_size_gt_30_lte_50.csv', index=False)
    filtered_tcga_info = tcga_info.loc[((tcga_info['length'] > 50) & (tcga_info['length'] <= 70))]
    filtered_tcga_info.to_csv('tcga_info_fake_recist_0.5_350_size_gt_50_lte_70.csv', index=False)
    filtered_tcga_info = tcga_info.loc[((tcga_info['length'] > 70) & (tcga_info['length'] <= 100))]
    filtered_tcga_info.to_csv('tcga_info_fake_recist_0.5_350_size_gt_70_lte_100.csv', index=False)
    filtered_tcga_info = tcga_info.loc[(tcga_info['length'] > 100)]
    filtered_tcga_info.to_csv('tcga_info_fake_recist_0.5_350_size_gt_100.csv', index=False)

    # filtered_tcga_info = tcga_info.loc[((tcga_info['length'] > 100) & (tcga_info['anatomy'] == "Ovarian"))]
    # filtered_tcga_info.to_csv('ov_100.csv', index=False)
    
    # # TCGA 1007
    # tcga_info = pd.read_csv('tcga_info_fake_recist_0.5_1007.csv')
    # # Filter by lesion type
    # filtered_tcga_info = tcga_info.loc[(tcga_info['anatomy'] == 'Ovarian')]
    # filtered_tcga_info.to_csv('tcga_info_fake_recist_0.5_1007_OV.csv', index=False)
    # filtered_tcga_info = tcga_info.loc[(tcga_info['anatomy'] == 'Renal')]
    # filtered_tcga_info.to_csv('tcga_info_fake_recist_0.5_1007_KD.csv', index=False)
    # filtered_tcga_info = tcga_info.loc[(tcga_info['anatomy'] == 'Liver')]
    # filtered_tcga_info.to_csv('tcga_info_fake_recist_0.5_1007_LV.csv', index=False)
    # filtered_tcga_info = tcga_info.loc[(tcga_info['anatomy'] == 'Lung')]
    # filtered_tcga_info.to_csv('tcga_info_fake_recist_0.5_1007_LU.csv', index=False)
    # # Filter by lesion size
    # filtered_tcga_info = tcga_info.loc[(tcga_info['length'] < 10)]
    # filtered_tcga_info.to_csv('tcga_info_fake_recist_0.5_1007_size_lt_10.csv', index=False)
    # filtered_tcga_info = tcga_info.loc[((tcga_info['length'] <= 30) & (tcga_info['length'] >= 10))]
    # filtered_tcga_info.to_csv('tcga_info_fake_recist_0.5_1007_size_gte_10_lte_30.csv', index=False)
    # filtered_tcga_info = tcga_info.loc[((tcga_info['length'] > 30) & (tcga_info['length'] <= 50))]
    # filtered_tcga_info.to_csv('tcga_info_fake_recist_0.5_1007_size_gt_30_lte_50.csv', index=False)
    # filtered_tcga_info = tcga_info.loc[((tcga_info['length'] > 50) & (tcga_info['length'] <= 70))]
    # filtered_tcga_info.to_csv('tcga_info_fake_recist_0.5_1007_size_gt_50_lte_70.csv', index=False)
    # filtered_tcga_info = tcga_info.loc[((tcga_info['length'] > 70) & (tcga_info['length'] <= 100))]
    # filtered_tcga_info.to_csv('tcga_info_fake_recist_0.5_1007_size_gt_70_lte_100.csv', index=False)
    # filtered_tcga_info = tcga_info.loc[(tcga_info['length'] > 100)]
    # filtered_tcga_info.to_csv('tcga_info_fake_recist_0.5_1007_size_gt_100.csv', index=False)

