import pandas as pd
from scipy.io import loadmat
import evaluation_metrics as em
import matplotlib.pyplot as plt
import numpy as np
import glob

if __name__ == "__main__":
    # Sensitivity at .5, 1, 2 ... avg. FPs per image
    avgFP = [.5, 1, 2, 4, 8, 16]
    # IoU threshold
    iou_th = .5
    # Load proposals .mat files
    exp_names = ["MULAN_3_image_3_slice", "MULAN_5_image_3_slice"]
    lesion_sizes = ["lt_10", "gte_10_lte_30", "gt_30_lte_50", "gt_50_lte_70", "gt_70_lte_100", "gt_100"]
    proposal_files_paths = glob.glob(pathname='../proposal_files/mulan_proposal_files/proposals_all_lesions/*.mat')
  
    print('-------- Sensitivity IoU all lesions --------')
    for exp_name in exp_names:
        for proposal_file_path in proposal_files_paths:
            if exp_name in proposal_file_path:
                proposal_file = loadmat(proposal_file_path)
                sens = em.sens_at_FP(boxes_all=proposal_file["boxes"][0], gts_all=proposal_file["gts"], avgFP=avgFP, iou_th=iou_th)
                if "tcga_350" in proposal_file_path:
                    print(f'Sensitvity all of {exp_name}: {sens}')
                    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')
                    mAP = em.coco_mAP(proposal_file["boxes"][0], proposal_file["gts"], proposal_file["imname"])
                    print(f"COCO mAP {exp_name}: {mAP}")
                    mAP = em.voc_pascal_mAP(proposal_file["boxes"][0], proposal_file["gts"], proposal_file["imname"])
                    print(f"VOC PASCAL mAP {exp_name}: {mAP}")
    
    proposal_files_paths_lesion_types = glob.glob(pathname='../proposal_files/mulan_proposal_files/proposals_lesion_types_and_size/*[OV|KD|LV|LU].mat')
    
    print('-------- Sensitivity IoU lesion types OV, KD, LV & LU --------')
    for exp_name in exp_names:
        for proposal_file_path in proposal_files_paths_lesion_types:
            if exp_name in proposal_file_path:
                proposal_file = loadmat(proposal_file_path)
                try:
                    sens = em.sens_at_FP(boxes_all=proposal_file["boxes"][0], gts_all=proposal_file["gts"], avgFP=avgFP, iou_th=iou_th)
                except:
                    sens = em.sens_at_FP(boxes_all=proposal_file["boxes"][0], gts_all=proposal_file["gts"][0], avgFP=avgFP, iou_th=iou_th)
                if "tcga_350" in proposal_file_path:
                    print(f'Sensitvity lesion type {proposal_file_path.split("_")[-1].replace(".mat", "")} of {exp_name}: {sens}')
                    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    proposal_files_paths_lesion_sizes = glob.glob(pathname='../proposal_files/mulan_proposal_files/proposals_lesion_types_and_size/*size*.mat')

    print('-------- Sensitivity IoU lesion sizes --------')
    for exp_name in exp_names:
        for proposal_file_path in proposal_files_paths_lesion_sizes:
            for lesion_size in lesion_sizes:
                if exp_name in proposal_file_path and lesion_size in proposal_file_path:
                    proposal_file = loadmat(proposal_file_path)
                    try:
                        sens = em.sens_at_FP(boxes_all=proposal_file["boxes"][0], gts_all=proposal_file["gts"], avgFP=avgFP, iou_th=iou_th)
                    except:
                        sens = em.sens_at_FP(boxes_all=proposal_file["boxes"][0], gts_all=proposal_file["gts"][0], avgFP=avgFP, iou_th=iou_th)
                    if "tcga_350" in proposal_file_path:
                        print(f'Sensitvity lesion size = {lesion_size} of {exp_name}: {sens}')
                        print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    
    print('-------- Sensitivity IoBB all lesions --------')
    for exp_name in exp_names:
        for proposal_file_path in proposal_files_paths:
            if exp_name in proposal_file_path:
                proposal_file = loadmat(proposal_file_path)
                try:
                    sens = em.sens_at_FP(boxes_all=proposal_file["boxes"][0], gts_all=proposal_file["gts"], avgFP=avgFP, iou_th=iou_th, iobb=True)
                except:
                    sens = em.sens_at_FP(boxes_all=proposal_file["boxes"][0], gts_all=proposal_file["gts"][0], avgFP=avgFP, iou_th=iou_th, iobb=True)
                if "tcga_350" in proposal_file_path:
                    print(f'Sensitvity all of {exp_name}: {sens}')
                    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')
    
    proposal_files_paths_lesion_types = glob.glob(pathname='../proposal_files/mulan_proposal_files/proposals_lesion_types_and_size/*[OV|KD|LV|LU].mat')
    
    print('-------- Sensitivity IoBB lesion types OV, KD, LV & LU --------')
    for exp_name in exp_names:
        for proposal_file_path in proposal_files_paths_lesion_types:
            if exp_name in proposal_file_path:
                proposal_file = loadmat(proposal_file_path)
                try:
                    sens = em.sens_at_FP(boxes_all=proposal_file["boxes"][0], gts_all=proposal_file["gts"], avgFP=avgFP, iou_th=iou_th, iobb=True)
                except:
                    sens = em.sens_at_FP(boxes_all=proposal_file["boxes"][0], gts_all=proposal_file["gts"][0], avgFP=avgFP, iou_th=iou_th, iobb=True)
                if "tcga_350" in proposal_file_path:
                    print(f'Sensitvity lesion type {proposal_file_path.split("_")[-1].replace(".mat", "")} of {exp_name}: {sens}')
                    print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

    proposal_files_paths_lesion_sizes = glob.glob(pathname='../proposal_files/mulan_proposal_files/proposals_lesion_types_and_size/*size*.mat')

    print('-------- Sensitivity IoBB lesion sizes --------')
    for exp_name in exp_names:
        for proposal_file_path in proposal_files_paths_lesion_sizes:
            for lesion_size in lesion_sizes:
                if exp_name in proposal_file_path and lesion_size in proposal_file_path:
                    proposal_file = loadmat(proposal_file_path)
                    try:
                        sens = em.sens_at_FP(boxes_all=proposal_file["boxes"][0], gts_all=proposal_file["gts"], avgFP=avgFP, iou_th=iou_th, iobb=True)
                    except:
                        sens = em.sens_at_FP(boxes_all=proposal_file["boxes"][0], gts_all=proposal_file["gts"][0], avgFP=avgFP, iou_th=iou_th, iobb=True)
                    if "tcga_350" in proposal_file_path:
                        print(f'Sensitvity lesion size = {lesion_size} of {exp_name}: {sens}')
                        print(f'Mean of [.5, 1, 2, 4] {sum(sens[:4]/len(sens[:4]))}')

