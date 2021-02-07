import pandas as pd
from scipy.io import loadmat
import evaluation_metrics as em
import matplotlib.pyplot as plt
import numpy as np

# On DeepLesion validation set
def sens_at_epochs_mulan():
    pass

if __name__ == "__main__":
    # Lesion types from DL_Info.csv
    lesion_types = {
        'BONE' : {'abb': 'BN', 'type': 1},
        'ABDOMEN' : {'abb': 'AB', 'type': 2},
        'MEDIASTINUM' : {'abb': 'ME', 'type': 3},
        'LIVER' : {'abb': 'LV', 'type': 4},
        'LUNG' : {'abb': 'LU', 'type': 5},
        'KIDNEY' : {'abb': 'KD', 'type': 6},
        'SOFT_TISSUE' : {'abb': 'ST', 'type': 7},
        'PELVIS' : {'abb': 'PV', 'type': 8}
    }

    # DeepLesion Metadata
    dl_info = pd.read_csv('../filtered_deeplesion_annotations/DL_info.csv')

    avgFP = [.5, 1, 2, 4, 8, 16]
    # IoU threshold
    iou_th = .5
    # Load proposals .mat files
    proposals_mulan_9_slices = loadmat('../proposal_files/mulan_proposal_files/proposals_all_lesions/MULAN_3_image_3_slice_test.mat')
    proposals_mulan_9_slices_lr_0002 = loadmat('../proposal_files/mulan_proposal_files/proposals_all_lesions/MULAN_3_image_3_slice_lr_0.002_test.mat')
    proposals_mulan_9_slices_lr_000283 = loadmat('../proposal_files/mulan_proposal_files/proposals_all_lesions/MULAN_3_image_3_slice_lr_0.00283_test.mat')
    proposals_mulan_15_slices_lr_0002 = loadmat('../proposal_files/mulan_proposal_files/proposals_all_lesions/MULAN_5_image_3_slice_lr_0.002_first_test.mat')

    # Sensitivity all
    print('--------Sensitivity all--------')
    sens = em.sens_at_FP(boxes_all=proposals_mulan_9_slices["boxes"][0], gts_all=proposals_mulan_9_slices["gts"][0], avgFP=avgFP, iou_th=iou_th)
    print(f'Sensitvity all of MULAN 9 slices BS=4:_____________{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')
    sens = em.sens_at_FP(boxes_all=proposals_mulan_9_slices_lr_0002["boxes"][0], gts_all=proposals_mulan_9_slices_lr_0002["gts"][0], avgFP=avgFP, iou_th=iou_th)
    print(f'Sensitvity all of MULAN 9 slices BS=4 lr=0.002:____{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')
    sens = em.sens_at_FP(boxes_all=proposals_mulan_9_slices_lr_000283["boxes"][0], gts_all=proposals_mulan_9_slices_lr_000283["gts"][0], avgFP=avgFP, iou_th=iou_th)
    print(f'Sensitvity all of MULAN 9 slices BS=4 lr=0.00283:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')
    sens = em.sens_at_FP(boxes_all=proposals_mulan_15_slices_lr_0002["boxes"][0], gts_all=proposals_mulan_15_slices_lr_0002["gts"][0], avgFP=avgFP, iou_th=iou_th)
    print(f'Sensitivity all of MULAN 15 slices BS=2 lr=0.002:{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    print()
    print('--------- Sensitivity lesion type MULAN 9 slices ---------')
    proposals_mulan_9_slices_bone = loadmat('../proposal_files/mulan_proposal_files/proposals_lesion_type_and_size/MULAN_3_image_3_slice_lr_0.00283_test_bone.mat')
    proposals_mulan_9_slices_abdomen = loadmat('../proposal_files/mulan_proposal_files/proposals_lesion_type_and_size/MULAN_3_image_3_slice_lr_0.00283_test_abdomen.mat')
    proposals_mulan_9_slices_mediastinum = loadmat('../proposal_files/mulan_proposal_files/proposals_lesion_type_and_size/MULAN_3_image_3_slice_lr_0.00283_test_mediastinum.mat')
    proposals_mulan_9_slices_liver = loadmat('../proposal_files/mulan_proposal_files/proposals_lesion_type_and_size/MULAN_3_image_3_slice_lr_0.00283_test_liver.mat')
    proposals_mulan_9_slices_lung = loadmat('../proposal_files/mulan_proposal_files/proposals_lesion_type_and_size/MULAN_3_image_3_slice_lr_0.00283_test_lung.mat')
    proposals_mulan_9_slices_kidney = loadmat('../proposal_files/mulan_proposal_files/proposals_lesion_type_and_size/MULAN_3_image_3_slice_lr_0.00283_test_kidney.mat')
    proposals_mulan_9_slices_soft_tissue = loadmat('../proposal_files/mulan_proposal_files/proposals_lesion_type_and_size/MULAN_3_image_3_slice_lr_0.00283_test_soft_tissue.mat')
    proposals_mulan_9_slices_pelvis = loadmat('../proposal_files/mulan_proposal_files/proposals_lesion_type_and_size/MULAN_3_image_3_slice_lr_0.00283_test_pelvis.mat')


    sens = em.sens_at_FP(boxes_all=proposals_mulan_9_slices_bone["boxes"][0], gts_all=proposals_mulan_9_slices_bone["gts"], avgFP=avgFP, iou_th=iou_th)
    print(f'Abount of BN lesions: {len(proposals_mulan_9_slices_bone["gts"])}')
    print(f'Sensitvity lesion type BN of MULAN 9 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_9_slices_abdomen["boxes"][0], gts_all=proposals_mulan_9_slices_abdomen["gts"][0], avgFP=avgFP, iou_th=iou_th)
    print(f'Abount of AB lesions: {len(proposals_mulan_9_slices_abdomen["gts"][0])}')
    print(f'Sensitvity lesion type AB of MULAN 9 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_9_slices_mediastinum["boxes"][0], gts_all=proposals_mulan_9_slices_mediastinum["gts"][0], avgFP=avgFP, iou_th=iou_th)
    print(f'Abount of ME lesions: {len(proposals_mulan_9_slices_mediastinum["gts"][0])}')
    print(f'Sensitvity lesion type ME of MULAN 9 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_9_slices_liver["boxes"][0], gts_all=proposals_mulan_9_slices_liver["gts"][0], avgFP=avgFP, iou_th=iou_th)
    print(f'Abount of LV lesions: {len(proposals_mulan_9_slices_liver["gts"][0])}')
    print(f'Sensitvity lesion type LV of MULAN 9 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_9_slices_lung["boxes"][0], gts_all=proposals_mulan_9_slices_lung["gts"][0], avgFP=avgFP, iou_th=iou_th)
    print(f'Abount of LU lesions: {len(proposals_mulan_9_slices_lung["gts"][0])}')
    print(f'Sensitvity lesion type LU of MULAN 9 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_9_slices_kidney["boxes"][0], gts_all=proposals_mulan_9_slices_kidney["gts"][0], avgFP=avgFP, iou_th=iou_th)
    print(f'Abount of KD lesions: {len(proposals_mulan_9_slices_kidney["gts"][0])}')
    print(f'Sensitvity lesion type KD of MULAN 9 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_9_slices_soft_tissue["boxes"][0], gts_all=proposals_mulan_9_slices_soft_tissue["gts"][0], avgFP=avgFP, iou_th=iou_th)
    print(f'Abount of ST lesions: {len(proposals_mulan_9_slices_soft_tissue["gts"][0])}')
    print(f'Sensitvity lesion type ST of MULAN 9 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_9_slices_pelvis["boxes"][0], gts_all=proposals_mulan_9_slices_pelvis["gts"][0], avgFP=avgFP, iou_th=iou_th)
    print(f'Abount of PV lesions: {len(proposals_mulan_9_slices_pelvis["gts"][0])}')
    print(f'Sensitvity lesion type PV of MULAN 9 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    print()
    print('--------- Sensitivity lesion size MULAN 9 slices---------')
    proposals_mulan_9_slices_size_lt_10 = loadmat('../proposal_files/mulan_proposal_files/proposals_lesion_type_and_size/MULAN_3_image_3_slice_lr_0.00283_test_size_lt_10.mat')
    proposals_mulan_9_slices_size_gte_10_lte_30 = loadmat('../proposal_files/mulan_proposal_files/proposals_lesion_type_and_size/MULAN_3_image_3_slice_lr_0.00283_test_size_gte_10_lte_30.mat')
    proposals_mulan_9_slices_size_gt_30_lte_50 = loadmat('../proposal_files/mulan_proposal_files/proposals_lesion_type_and_size/MULAN_3_image_3_slice_lr_0.00283_test_size_gt_30_lte_50.mat')
    proposals_mulan_9_slices_size_gt_50_lte_70 = loadmat('../proposal_files/mulan_proposal_files/proposals_lesion_type_and_size/MULAN_3_image_3_slice_lr_0.00283_test_size_gt_50_lte_70.mat')
    proposals_mulan_9_slices_size_gt_70_lte_100 = loadmat('../proposal_files/mulan_proposal_files/proposals_lesion_type_and_size/MULAN_3_image_3_slice_lr_0.00283_test_size_gt_70_lte_100.mat')
    proposals_mulan_9_slices_size_gt_100 = loadmat('../proposal_files/mulan_proposal_files/proposals_lesion_type_and_size/MULAN_3_image_3_slice_lr_0.00283_test_size_gt_100.mat')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_9_slices_size_lt_10["boxes"][0], gts_all=proposals_mulan_9_slices_size_lt_10["gts"][0], avgFP=avgFP, iou_th=iou_th)
    print(f'Abount of lesions size < 10: {len(proposals_mulan_9_slices_size_lt_10["gts"][0])}')
    print(f'Sensitvity lesions size < 10 of MULAN 9 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_9_slices_size_gte_10_lte_30["boxes"][0], gts_all=proposals_mulan_9_slices_size_gte_10_lte_30["gts"][0], avgFP=avgFP, iou_th=iou_th)
    print(f'Abount of lesions 10 <= size <= 30: {len(proposals_mulan_9_slices_size_gte_10_lte_30["gts"][0])}')
    print(f'Sensitvity lesions 10 <= size <= 30 of MULAN 9 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_9_slices_size_gt_30_lte_50["boxes"][0], gts_all=proposals_mulan_9_slices_size_gt_30_lte_50["gts"][0], avgFP=avgFP, iou_th=iou_th)
    print(f'Abount of lesions 30 < size <= 50: {len(proposals_mulan_9_slices_size_gt_30_lte_50["gts"][0])}')
    print(f'Sensitvity lesions 30 < size <= 50 of MULAN 9 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_9_slices_size_gt_50_lte_70["boxes"][0], gts_all=proposals_mulan_9_slices_size_gt_50_lte_70["gts"], avgFP=avgFP, iou_th=iou_th)
    print(f'Abount of lesions 50 < size <= 70: {len(proposals_mulan_9_slices_size_gt_50_lte_70["gts"])}')
    print(f'Sensitvity lesions 50 < size <= 70 of MULAN 9 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_9_slices_size_gt_70_lte_100["boxes"][0], gts_all=proposals_mulan_9_slices_size_gt_70_lte_100["gts"], avgFP=avgFP, iou_th=iou_th)
    print(f'Abount of lesions 70 < size <= 100: {len(proposals_mulan_9_slices_size_gt_70_lte_100["gts"])}')
    print(f'Sensitvity 70 < size <= 100 of MULAN 9 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_9_slices_size_gt_100["boxes"][0], gts_all=proposals_mulan_9_slices_size_gt_100["gts"], avgFP=avgFP, iou_th=iou_th)
    print(f'Abount of lesions size > 100: {len(proposals_mulan_9_slices_size_gt_100["gts"])}')
    print(f'Sensitvity lesions size > 100 of MULAN 9 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    print()
    print('--------- Sensitivity lesion type MULAN 15 slices---------')
    proposals_mulan_15_slices_bone = loadmat('../proposal_files/mulan_proposal_files/proposals_lesion_type_and_size/MULAN_5_image_3_slice_lr_0.002_first_test_bone.mat')
    proposals_mulan_15_slices_abdomen = loadmat('../proposal_files/mulan_proposal_files/proposals_lesion_type_and_size/MULAN_5_image_3_slice_lr_0.002_first_test_abdomen.mat')
    proposals_mulan_15_slices_mediastinum = loadmat('../proposal_files/mulan_proposal_files/proposals_lesion_type_and_size/MULAN_5_image_3_slice_lr_0.002_first_test_mediastinum.mat')
    proposals_mulan_15_slices_liver = loadmat('../proposal_files/mulan_proposal_files/proposals_lesion_type_and_size/MULAN_5_image_3_slice_lr_0.002_first_test_liver.mat')
    proposals_mulan_15_slices_lung = loadmat('../proposal_files/mulan_proposal_files/proposals_lesion_type_and_size/MULAN_5_image_3_slice_lr_0.002_first_test_lung.mat')
    proposals_mulan_15_slices_kidney = loadmat('../proposal_files/mulan_proposal_files/proposals_lesion_type_and_size/MULAN_5_image_3_slice_lr_0.002_first_test_kidney.mat')
    proposals_mulan_15_slices_soft_tissue = loadmat('../proposal_files/mulan_proposal_files/proposals_lesion_type_and_size/MULAN_5_image_3_slice_lr_0.002_first_test_soft_tissue.mat')
    proposals_mulan_15_slices_pelvis = loadmat('../proposal_files/mulan_proposal_files/proposals_lesion_type_and_size/MULAN_5_image_3_slice_lr_0.002_first_test_pelvis.mat')


    sens = em.sens_at_FP(boxes_all=proposals_mulan_15_slices_bone["boxes"][0], gts_all=proposals_mulan_15_slices_bone["gts"], avgFP=avgFP, iou_th=iou_th)
    print(f'Abount of BN lesions: {len(proposals_mulan_15_slices_bone["gts"])}')
    print(f'Sensitvity lesion type BN of MULAN 15 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_15_slices_abdomen["boxes"][0], gts_all=proposals_mulan_15_slices_abdomen["gts"][0], avgFP=avgFP, iou_th=iou_th)
    print(f'Abount of AB lesions: {len(proposals_mulan_15_slices_abdomen["gts"][0])}')
    print(f'Sensitvity lesion type AB of MULAN 15 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_15_slices_mediastinum["boxes"][0], gts_all=proposals_mulan_15_slices_mediastinum["gts"][0], avgFP=avgFP, iou_th=iou_th)
    print(f'Abount of ME lesions: {len(proposals_mulan_15_slices_mediastinum["gts"][0])}')
    print(f'Sensitvity lesion type ME of MULAN 15 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_15_slices_liver["boxes"][0], gts_all=proposals_mulan_15_slices_liver["gts"][0], avgFP=avgFP, iou_th=iou_th)
    print(f'Abount of LV lesions: {len(proposals_mulan_15_slices_liver["gts"][0])}')
    print(f'Sensitvity lesion type LV of MULAN 15 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_15_slices_lung["boxes"][0], gts_all=proposals_mulan_15_slices_lung["gts"][0], avgFP=avgFP, iou_th=iou_th)
    print(f'Abount of LU lesions: {len(proposals_mulan_15_slices_lung["gts"][0])}')
    print(f'Sensitvity lesion type LU of MULAN 15 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_15_slices_kidney["boxes"][0], gts_all=proposals_mulan_15_slices_kidney["gts"][0], avgFP=avgFP, iou_th=iou_th)
    print(f'Abount of KD lesions: {len(proposals_mulan_15_slices_kidney["gts"][0])}')
    print(f'Sensitvity lesion type KD of MULAN 15 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_15_slices_soft_tissue["boxes"][0], gts_all=proposals_mulan_15_slices_soft_tissue["gts"][0], avgFP=avgFP, iou_th=iou_th)
    print(f'Abount of ST lesions: {len(proposals_mulan_15_slices_soft_tissue["gts"][0])}')
    print(f'Sensitvity lesion type ST of MULAN 15 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_15_slices_pelvis["boxes"][0], gts_all=proposals_mulan_15_slices_pelvis["gts"][0], avgFP=avgFP, iou_th=iou_th)
    print(f'Abount of PV lesions: {len(proposals_mulan_15_slices_pelvis["gts"][0])}')
    print(f'Sensitvity lesion type PV of MULAN 15 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    print()
    print('--------- Sensitivity lesion size MULAN 15 slices ---------')
    proposals_mulan_15_slices_size_lt_10 = loadmat('../proposal_files/mulan_proposal_files/proposals_lesion_type_and_size/MULAN_5_image_3_slice_lr_0.002_first_test_size_lt_10.mat')
    proposals_mulan_15_slices_size_gte_10_lte_30 = loadmat('../proposal_files/mulan_proposal_files/proposals_lesion_type_and_size/MULAN_5_image_3_slice_lr_0.002_first_test_size_gte_10_lte_30.mat')
    proposals_mulan_15_slices_size_gt_30_lte_50 = loadmat('../proposal_files/mulan_proposal_files/proposals_lesion_type_and_size/MULAN_5_image_3_slice_lr_0.002_first_test_size_gt_30_lte_50.mat')
    proposals_mulan_15_slices_size_gt_50_lte_70 = loadmat('../proposal_files/mulan_proposal_files/proposals_lesion_type_and_size/MULAN_5_image_3_slice_lr_0.002_first_test_size_gt_50_lte_70.mat')
    proposals_mulan_15_slices_size_gt_70_lte_100 = loadmat('../proposal_files/mulan_proposal_files/proposals_lesion_type_and_size/MULAN_5_image_3_slice_lr_0.002_first_test_size_gt_70_lte_100.mat')
    proposals_mulan_15_slices_size_gt_100 = loadmat('../proposal_files/mulan_proposal_files/proposals_lesion_type_and_size/MULAN_5_image_3_slice_lr_0.002_first_test_size_gt_100.mat')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_15_slices_size_lt_10["boxes"][0], gts_all=proposals_mulan_15_slices_size_lt_10["gts"][0], avgFP=avgFP, iou_th=iou_th)
    print(f'Abount of lesions size < 10: {len(proposals_mulan_15_slices_size_lt_10["gts"][0])}')
    print(f'Sensitvity lesions size < 10 of MULAN 15 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_15_slices_size_gte_10_lte_30["boxes"][0], gts_all=proposals_mulan_15_slices_size_gte_10_lte_30["gts"][0], avgFP=avgFP, iou_th=iou_th)
    print(f'Abount of lesions 10 <= size <= 30: {len(proposals_mulan_15_slices_size_gte_10_lte_30["gts"][0])}')
    print(f'Sensitvity lesions 10 <= size <= 30 of MULAN 15 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_15_slices_size_gt_30_lte_50["boxes"][0], gts_all=proposals_mulan_15_slices_size_gt_30_lte_50["gts"][0], avgFP=avgFP, iou_th=iou_th)
    print(f'Abount of lesions 30 < size <= 50: {len(proposals_mulan_15_slices_size_gt_30_lte_50["gts"][0])}')
    print(f'Sensitvity lesions 30 < size <= 50 of MULAN 15 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_15_slices_size_gt_50_lte_70["boxes"][0], gts_all=proposals_mulan_15_slices_size_gt_50_lte_70["gts"], avgFP=avgFP, iou_th=iou_th)
    print(f'Abount of lesions 50 < size <= 70: {len(proposals_mulan_15_slices_size_gt_50_lte_70["gts"])}')
    print(f'Sensitvity lesions 50 < size <= 70 of MULAN 15 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_15_slices_size_gt_70_lte_100["boxes"][0], gts_all=proposals_mulan_15_slices_size_gt_70_lte_100["gts"], avgFP=avgFP, iou_th=iou_th)
    print(f'Abount of lesions 70 < size <= 100: {len(proposals_mulan_15_slices_size_gt_70_lte_100["gts"])}')
    print(f'Sensitvity 70 < size <= 100 of MULAN 15 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_15_slices_size_gt_100["boxes"][0], gts_all=proposals_mulan_15_slices_size_gt_100["gts"], avgFP=avgFP, iou_th=iou_th)
    print(f'Abount of lesions size > 100: {len(proposals_mulan_15_slices_size_gt_100["gts"])}')
    print(f'Sensitvity lesions size > 100 of MULAN 15 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')
    
    print()
    print("Calculate mAP...")
    mAP = em.voc_pascal_mAP(proposals_mulan_9_slices_lr_000283["boxes"][0], proposals_mulan_9_slices_lr_000283["gts"][0], proposals_mulan_9_slices_lr_000283["imname"])
    print(f"VOC PASCAL mAP MULAN 9 slices: {mAP}")

    mAP = em.voc_pascal_mAP(proposals_mulan_15_slices_lr_0002["boxes"][0], proposals_mulan_15_slices_lr_0002["gts"][0], proposals_mulan_15_slices_lr_0002["imname"])
    print(f"VOC PASCAL mAP MULAN 15 slices: {mAP}")

    mAP = em.coco_mAP(proposals_mulan_9_slices_lr_000283["boxes"][0], proposals_mulan_9_slices_lr_000283["gts"][0], proposals_mulan_9_slices_lr_000283["imname"])
    print(f"COCO mAP MULAN 9 slices: {mAP}")

    mAP = em.coco_mAP(proposals_mulan_15_slices_lr_0002["boxes"][0], proposals_mulan_15_slices_lr_0002["gts"][0], proposals_mulan_15_slices_lr_0002["imname"])
    print(f"COCO mAP MULAN 15 slices: {mAP}")


    # Sensitivity all
    print('--------Sensitivity IoBB all--------')
    sens = em.sens_at_FP(boxes_all=proposals_mulan_9_slices["boxes"][0], gts_all=proposals_mulan_9_slices["gts"][0], avgFP=avgFP, iou_th=iou_th, iobb=True)
    print(f'Sensitvity all of MULAN 9 slices BS=4:_____________{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')
    sens = em.sens_at_FP(boxes_all=proposals_mulan_9_slices_lr_0002["boxes"][0], gts_all=proposals_mulan_9_slices_lr_0002["gts"][0], avgFP=avgFP, iou_th=iou_th, iobb=True)
    print(f'Sensitvity all of MULAN 9 slices BS=4 lr=0.002:____{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')
    sens = em.sens_at_FP(boxes_all=proposals_mulan_9_slices_lr_000283["boxes"][0], gts_all=proposals_mulan_9_slices_lr_000283["gts"][0], avgFP=avgFP, iou_th=iou_th, iobb=True)
    print(f'Sensitvity all of MULAN 9 slices BS=4 lr=0.00283:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')
    sens = em.sens_at_FP(boxes_all=proposals_mulan_15_slices_lr_0002["boxes"][0], gts_all=proposals_mulan_15_slices_lr_0002["gts"][0], avgFP=avgFP, iou_th=iou_th, iobb=True)
    print(f'Sensitivity all of MULAN 15 slices BS=2 lr=0.002:{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    print()
    print('--------- Sensitivity IoBB lesion type MULAN 9 slices ---------')
  
    sens = em.sens_at_FP(boxes_all=proposals_mulan_9_slices_bone["boxes"][0], gts_all=proposals_mulan_9_slices_bone["gts"], avgFP=avgFP, iou_th=iou_th, iobb=True)
    print(f'Abount of BN lesions: {len(proposals_mulan_9_slices_bone["gts"])}')
    print(f'Sensitvity lesion type BN of MULAN 9 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_9_slices_abdomen["boxes"][0], gts_all=proposals_mulan_9_slices_abdomen["gts"][0], avgFP=avgFP, iou_th=iou_th, iobb=True)
    print(f'Abount of AB lesions: {len(proposals_mulan_9_slices_abdomen["gts"][0])}')
    print(f'Sensitvity lesion type AB of MULAN 9 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_9_slices_mediastinum["boxes"][0], gts_all=proposals_mulan_9_slices_mediastinum["gts"][0], avgFP=avgFP, iou_th=iou_th, iobb=True)
    print(f'Abount of ME lesions: {len(proposals_mulan_9_slices_mediastinum["gts"][0])}')
    print(f'Sensitvity lesion type ME of MULAN 9 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_9_slices_liver["boxes"][0], gts_all=proposals_mulan_9_slices_liver["gts"][0], avgFP=avgFP, iou_th=iou_th, iobb=True)
    print(f'Abount of LV lesions: {len(proposals_mulan_9_slices_liver["gts"][0])}')
    print(f'Sensitvity lesion type LV of MULAN 9 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_9_slices_lung["boxes"][0], gts_all=proposals_mulan_9_slices_lung["gts"][0], avgFP=avgFP, iou_th=iou_th, iobb=True)
    print(f'Abount of LU lesions: {len(proposals_mulan_9_slices_lung["gts"][0])}')
    print(f'Sensitvity lesion type LU of MULAN 9 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_9_slices_kidney["boxes"][0], gts_all=proposals_mulan_9_slices_kidney["gts"][0], avgFP=avgFP, iou_th=iou_th, iobb=True)
    print(f'Abount of KD lesions: {len(proposals_mulan_9_slices_kidney["gts"][0])}')
    print(f'Sensitvity lesion type KD of MULAN 9 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_9_slices_soft_tissue["boxes"][0], gts_all=proposals_mulan_9_slices_soft_tissue["gts"][0], avgFP=avgFP, iou_th=iou_th, iobb=True)
    print(f'Abount of ST lesions: {len(proposals_mulan_9_slices_soft_tissue["gts"][0])}')
    print(f'Sensitvity lesion type ST of MULAN 9 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_9_slices_pelvis["boxes"][0], gts_all=proposals_mulan_9_slices_pelvis["gts"][0], avgFP=avgFP, iou_th=iou_th, iobb=True)
    print(f'Abount of PV lesions: {len(proposals_mulan_9_slices_pelvis["gts"][0])}')
    print(f'Sensitvity lesion type PV of MULAN 9 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    print()
    print('--------- Sensitivity IoBB lesion size MULAN 9 slices---------')
    
    sens = em.sens_at_FP(boxes_all=proposals_mulan_9_slices_size_lt_10["boxes"][0], gts_all=proposals_mulan_9_slices_size_lt_10["gts"][0], avgFP=avgFP, iou_th=iou_th, iobb=True)
    print(f'Abount of lesions size < 10: {len(proposals_mulan_9_slices_size_lt_10["gts"][0])}')
    print(f'Sensitvity lesions size < 10 of MULAN 9 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_9_slices_size_gte_10_lte_30["boxes"][0], gts_all=proposals_mulan_9_slices_size_gte_10_lte_30["gts"][0], avgFP=avgFP, iou_th=iou_th, iobb=True)
    print(f'Abount of lesions 10 <= size <= 30: {len(proposals_mulan_9_slices_size_gte_10_lte_30["gts"][0])}')
    print(f'Sensitvity lesions 10 <= size <= 30 of MULAN 9 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_9_slices_size_gt_30_lte_50["boxes"][0], gts_all=proposals_mulan_9_slices_size_gt_30_lte_50["gts"][0], avgFP=avgFP, iou_th=iou_th, iobb=True)
    print(f'Abount of lesions 30 < size <= 50: {len(proposals_mulan_9_slices_size_gt_30_lte_50["gts"][0])}')
    print(f'Sensitvity lesions 30 < size <= 50 of MULAN 9 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_9_slices_size_gt_50_lte_70["boxes"][0], gts_all=proposals_mulan_9_slices_size_gt_50_lte_70["gts"], avgFP=avgFP, iou_th=iou_th, iobb=True)
    print(f'Abount of lesions 50 < size <= 70: {len(proposals_mulan_9_slices_size_gt_50_lte_70["gts"])}')
    print(f'Sensitvity lesions 50 < size <= 70 of MULAN 9 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_9_slices_size_gt_70_lte_100["boxes"][0], gts_all=proposals_mulan_9_slices_size_gt_70_lte_100["gts"], avgFP=avgFP, iou_th=iou_th, iobb=True)
    print(f'Abount of lesions 70 < size <= 100: {len(proposals_mulan_9_slices_size_gt_70_lte_100["gts"])}')
    print(f'Sensitvity 70 < size <= 100 of MULAN 9 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_9_slices_size_gt_100["boxes"][0], gts_all=proposals_mulan_9_slices_size_gt_100["gts"], avgFP=avgFP, iou_th=iou_th, iobb=True)
    print(f'Abount of lesions size > 100: {len(proposals_mulan_9_slices_size_gt_100["gts"])}')
    print(f'Sensitvity lesions size > 100 of MULAN 9 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    print()
    print('--------- Sensitivity IoBB lesion type MULAN 15 slices---------')
   
    sens = em.sens_at_FP(boxes_all=proposals_mulan_15_slices_bone["boxes"][0], gts_all=proposals_mulan_15_slices_bone["gts"], avgFP=avgFP, iou_th=iou_th, iobb=True)
    print(f'Abount of BN lesions: {len(proposals_mulan_15_slices_bone["gts"])}')
    print(f'Sensitvity lesion type BN of MULAN 15 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_15_slices_abdomen["boxes"][0], gts_all=proposals_mulan_15_slices_abdomen["gts"][0], avgFP=avgFP, iou_th=iou_th, iobb=True)
    print(f'Abount of AB lesions: {len(proposals_mulan_15_slices_abdomen["gts"][0])}')
    print(f'Sensitvity lesion type AB of MULAN 15 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_15_slices_mediastinum["boxes"][0], gts_all=proposals_mulan_15_slices_mediastinum["gts"][0], avgFP=avgFP, iou_th=iou_th, iobb=True)
    print(f'Abount of ME lesions: {len(proposals_mulan_15_slices_mediastinum["gts"][0])}')
    print(f'Sensitvity lesion type ME of MULAN 15 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_15_slices_liver["boxes"][0], gts_all=proposals_mulan_15_slices_liver["gts"][0], avgFP=avgFP, iou_th=iou_th, iobb=True)
    print(f'Abount of LV lesions: {len(proposals_mulan_15_slices_liver["gts"][0])}')
    print(f'Sensitvity lesion type LV of MULAN 15 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_15_slices_lung["boxes"][0], gts_all=proposals_mulan_15_slices_lung["gts"][0], avgFP=avgFP, iou_th=iou_th, iobb=True)
    print(f'Abount of LU lesions: {len(proposals_mulan_15_slices_lung["gts"][0])}')
    print(f'Sensitvity lesion type LU of MULAN 15 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_15_slices_kidney["boxes"][0], gts_all=proposals_mulan_15_slices_kidney["gts"][0], avgFP=avgFP, iou_th=iou_th, iobb=True)
    print(f'Abount of KD lesions: {len(proposals_mulan_15_slices_kidney["gts"][0])}')
    print(f'Sensitvity lesion type KD of MULAN 15 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_15_slices_soft_tissue["boxes"][0], gts_all=proposals_mulan_15_slices_soft_tissue["gts"][0], avgFP=avgFP, iou_th=iou_th, iobb=True)
    print(f'Abount of ST lesions: {len(proposals_mulan_15_slices_soft_tissue["gts"][0])}')
    print(f'Sensitvity lesion type ST of MULAN 15 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_15_slices_pelvis["boxes"][0], gts_all=proposals_mulan_15_slices_pelvis["gts"][0], avgFP=avgFP, iou_th=iou_th, iobb=True)
    print(f'Abount of PV lesions: {len(proposals_mulan_15_slices_pelvis["gts"][0])}')
    print(f'Sensitvity lesion type PV of MULAN 15 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    print()
    print('--------- Sensitivity IoBB lesion size MULAN 15 slices ---------')
   
    sens = em.sens_at_FP(boxes_all=proposals_mulan_15_slices_size_lt_10["boxes"][0], gts_all=proposals_mulan_15_slices_size_lt_10["gts"][0], avgFP=avgFP, iou_th=iou_th, iobb=True)
    print(f'Abount of lesions size < 10: {len(proposals_mulan_15_slices_size_lt_10["gts"][0])}')
    print(f'Sensitvity lesions size < 10 of MULAN 15 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_15_slices_size_gte_10_lte_30["boxes"][0], gts_all=proposals_mulan_15_slices_size_gte_10_lte_30["gts"][0], avgFP=avgFP, iou_th=iou_th, iobb=True)
    print(f'Abount of lesions 10 <= size <= 30: {len(proposals_mulan_15_slices_size_gte_10_lte_30["gts"][0])}')
    print(f'Sensitvity lesions 10 <= size <= 30 of MULAN 15 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_15_slices_size_gt_30_lte_50["boxes"][0], gts_all=proposals_mulan_15_slices_size_gt_30_lte_50["gts"][0], avgFP=avgFP, iou_th=iou_th, iobb=True)
    print(f'Abount of lesions 30 < size <= 50: {len(proposals_mulan_15_slices_size_gt_30_lte_50["gts"][0])}')
    print(f'Sensitvity lesions 30 < size <= 50 of MULAN 15 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_15_slices_size_gt_50_lte_70["boxes"][0], gts_all=proposals_mulan_15_slices_size_gt_50_lte_70["gts"], avgFP=avgFP, iou_th=iou_th, iobb=True)
    print(f'Abount of lesions 50 < size <= 70: {len(proposals_mulan_15_slices_size_gt_50_lte_70["gts"])}')
    print(f'Sensitvity lesions 50 < size <= 70 of MULAN 15 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_15_slices_size_gt_70_lte_100["boxes"][0], gts_all=proposals_mulan_15_slices_size_gt_70_lte_100["gts"], avgFP=avgFP, iou_th=iou_th, iobb=True)
    print(f'Abount of lesions 70 < size <= 100: {len(proposals_mulan_15_slices_size_gt_70_lte_100["gts"])}')
    print(f'Sensitvity 70 < size <= 100 of MULAN 15 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    sens = em.sens_at_FP(boxes_all=proposals_mulan_15_slices_size_gt_100["boxes"][0], gts_all=proposals_mulan_15_slices_size_gt_100["gts"], avgFP=avgFP, iou_th=iou_th, iobb=True)
    print(f'Abount of lesions size > 100: {len(proposals_mulan_15_slices_size_gt_100["gts"])}')
    print(f'Sensitvity lesions size > 100 of MULAN 15 slices:__{sens}')
    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

