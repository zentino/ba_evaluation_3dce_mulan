import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from evaluation_metrics import get_FROC_fn


if __name__ == "__main__":

    # Load 3DCE proposals .mat files
    proposals_3dce_9_slices = loadmat('../proposal_files/3dce_proposal_files/proposals_all_lesions/proposals_test_3DCE_3_image_3_slice.mat')
    proposals_3dce_15_slices = loadmat('../proposal_files/3dce_proposal_files/proposals_all_lesions/proposals_test_3DCE_5_image_3_slice.mat')
    proposals_3dce_21_slices = loadmat('../proposal_files/3dce_proposal_files/proposals_all_lesions/proposals_test_3DCE_7_image_3_slice.mat')

    # Load MULAN proposals .mat files
    proposals_mulan_9_slices = loadmat('../proposal_files/mulan_proposal_files/proposals_all_lesions/MULAN_3_image_3_slice_lr_0.00283_test.mat')
    proposals_mulan_15_slices = loadmat('../proposal_files/mulan_proposal_files/proposals_all_lesions/MULAN_5_image_3_slice_lr_0.002_first_test.mat')
    
    froc_fn_3dce_9_slices = get_FROC_fn(boxes_all=proposals_3dce_9_slices["boxes"][0], gts_all=proposals_3dce_9_slices["gts"][0], iou_th=0.5)
    froc_fn_3dce_15_slices = get_FROC_fn(boxes_all=proposals_3dce_15_slices["boxes"][0], gts_all=proposals_3dce_15_slices["gts"][0], iou_th=0.5)
    froc_fn_3dce_21_slices = get_FROC_fn(boxes_all=proposals_3dce_21_slices["boxes"][0], gts_all=proposals_3dce_21_slices["gts"][0], iou_th=0.5)
    froc_fn_mulan_9_slices = get_FROC_fn(boxes_all=proposals_mulan_9_slices["boxes"][0], gts_all=proposals_mulan_9_slices["gts"][0], iou_th=0.5)
    froc_fn_mulan_15_slices = get_FROC_fn(boxes_all=proposals_mulan_9_slices["boxes"][0], gts_all=proposals_mulan_9_slices["gts"][0], iou_th=0.5)


    X = np.arange(0.5, 20, 0.1)
    plt.plot(X, froc_fn_3dce_9_slices(X), '--', linewidth=1.07, label="3DCE mit 9 Bildern")
    plt.plot(X, froc_fn_3dce_15_slices(X), ':', linewidth=1.07, label="3DCE mit 15 Bildern")
    plt.plot(X, froc_fn_3dce_21_slices(X), '-.', linewidth=1.07, label="3DCE mit 21 Bildern")
    plt.plot(X, froc_fn_mulan_9_slices(X), '--', linewidth=1.07, label="MULAN mit 9 Bildern")
    plt.plot(X, froc_fn_mulan_15_slices(X), ':', linewidth=1.07, label="MULAN mit 15 Bildern")

    # x axis
    plt.xlabel('Ø falsche Kandidaten pro CT-Bild')
    # y axis
    plt.ylabel('Sensitivity IoU (%)')
    plt.title('FROC-Kurven aller trainierten Modelle')
    plt.legend()
    plt.show()