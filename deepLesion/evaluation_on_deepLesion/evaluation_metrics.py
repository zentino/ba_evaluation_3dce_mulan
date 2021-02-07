import numpy as np
from scipy.io import loadmat
from scipy import interpolate
import pandas as pd
from itertools import compress
import matplotlib.pyplot as plt
from mean_average_precision import MetricBuilder

def IOU(box1, gts):
    # compute overlaps
    # intersection
    ixmin = np.maximum(gts[:, 0], box1[0])
    iymin = np.maximum(gts[:, 1], box1[1])
    ixmax = np.minimum(gts[:, 2], box1[2])
    iymax = np.minimum(gts[:, 3], box1[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    uni = ((box1[2] - box1[0] + 1.) * (box1[3] - box1[1] + 1.) +
           (gts[:, 2] - gts[:, 0] + 1.) *
           (gts[:, 3] - gts[:, 1] + 1.) - inters)

    overlaps = inters / uni

    return overlaps

def IOBB(box1, gts):
    # compute overlaps
    # intersection
    ixmin = np.maximum(gts[:, 0], box1[0])
    iymin = np.maximum(gts[:, 1], box1[1])
    ixmax = np.minimum(gts[:, 2], box1[2])
    iymax = np.minimum(gts[:, 3], box1[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # BB area
    bb = (box1[2] - box1[0] + 1.) * (box1[3] - box1[1] + 1.)

    overlaps = inters / bb

    return overlaps

def num_true_positive(boxes, gts, num_box, iou_th):
    # only count once if one gt is hit multiple times
    hit = np.zeros((gts.shape[0],), dtype=np.bool)
    scores = boxes[:, -1]
    boxes = boxes[scores.argsort()[::-1], :4]

    for i, box1 in enumerate(boxes):
        if i == num_box:
            break
        overlaps = IOU(box1, gts)
        hit = np.logical_or(hit, overlaps >= iou_th)

    tp = np.count_nonzero(hit)

    return tp

def recall_all(boxes_all, gts_all, num_box, iou_th):
    # Compute the recall at num_box candidates per image
    nCls = len(boxes_all)
    nImg = len(boxes_all[0])
    recs = np.zeros((nCls, len(num_box)))
    nGt = np.zeros((nCls,), dtype=np.float)

    for cls in range(nCls):
        for i in range(nImg):
            nGt[cls] += gts_all[cls][i].shape[0]
            for n in range(len(num_box)):
                tp = num_true_positive(
                    boxes_all[cls][i], gts_all[cls][i], num_box[n], iou_th)
                recs[cls, n] += tp

    recs /= nGt
    return recs

def FROC(boxes_all, gts_all, iou_th):
    # Compute the FROC curve, for single class only
    nImg = len(boxes_all)
    img_idxs = np.hstack([[i]*len(boxes_all[i]) for i in range(nImg)])
    boxes_cat = np.vstack(boxes_all)
    scores = boxes_cat[:, -1]
    ord = np.argsort(scores)[::-1]
    boxes_cat = boxes_cat[ord, :4]
    img_idxs = img_idxs[ord]
    hits = [np.zeros((len(gts),), dtype=bool) for gts in gts_all]
    nHits = 0
    nMiss = 0
    tps = []
    fps = []
    for i in range(len(boxes_cat)):
        overlaps = IOU(boxes_cat[i, :], gts_all[img_idxs[i]])
        if overlaps.max() < iou_th:
            nMiss += 1
        else:
            for j in range(len(overlaps)):
                if overlaps[j] >= iou_th and not hits[img_idxs[i]][j]:
                    hits[img_idxs[i]][j] = True
                    nHits += 1

        tps.append(nHits)
        fps.append(nMiss)

    nGt = len(np.vstack(gts_all))
    sens = np.array(tps, dtype=float) / nGt
    fp_per_img = np.array(fps, dtype=float) / nImg
    return sens, fp_per_img

def FROC_IOBB(boxes_all, gts_all, iobb_th):
    # Compute the FROC curve, for single class only
    nImg = len(boxes_all)
    img_idxs = np.hstack([[i]*len(boxes_all[i]) for i in range(nImg)])
    boxes_cat = np.vstack(boxes_all)
    scores = boxes_cat[:, -1]
    ord = np.argsort(scores)[::-1]
    boxes_cat = boxes_cat[ord, :4]
    img_idxs = img_idxs[ord]
    hits = [np.zeros((len(gts),), dtype=bool) for gts in gts_all]
    nHits = 0
    nMiss = 0
    tps = []
    fps = []
    for i in range(len(boxes_cat)):
        overlaps = IOBB(boxes_cat[i, :], gts_all[img_idxs[i]])
        if overlaps.max() < iobb_th:
            nMiss += 1
        else:
            for j in range(len(overlaps)):
                if overlaps[j] >= iobb_th and not hits[img_idxs[i]][j]:
                    hits[img_idxs[i]][j] = True
                    nHits += 1

        tps.append(nHits)
        fps.append(nMiss)

    nGt = len(np.vstack(gts_all))
    sens = np.array(tps, dtype=float) / nGt
    fp_per_img = np.array(fps, dtype=float) / nImg
    return sens, fp_per_img

def voc_pascal_mAP(boxes_all, gts_all, imgs):
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)
    for i in range(0, len(imgs)):
        # [xmin, ymin, xmax, ymax, class_id, confidence]
        preds = np.insert(boxes_all[i], 4, values=0, axis=1)
        # [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
        gts = np.insert(gts_all[i], 4, values=0, axis=1)
        gts = np.insert(gts, 5, values=0, axis=1)
        gts = np.insert(gts, 6, values=0, axis=1)
        metric_fn.add(preds, gts)
    mAP = metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']
    return mAP


def coco_mAP(boxes_all, gts_all, imgs):
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)
    for i in range(0, len(imgs)):
        # [xmin, ymin, xmax, ymax, class_id, confidence]
        preds = np.insert(boxes_all[i], 4, values=0, axis=1)
        # [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
        gts = np.insert(gts_all[i], 4, values=0, axis=1)
        gts = np.insert(gts, 5, values=0, axis=1)
        gts = np.insert(gts, 6, values=0, axis=1)
        metric_fn.add(preds, gts)
        
    mAP = metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
    return mAP

def sens_at_FP(boxes_all, gts_all, avgFP, iou_th, iobb=False):
    sens = None
    fp_per_img = None
    # compute the sensitivity at avgFP (average FP per image)
    if not iobb:
        sens, fp_per_img = FROC(boxes_all, gts_all, iou_th)
    else:
        sens, fp_per_img = FROC_IOBB(boxes_all, gts_all, iou_th)
    
    f = interpolate.interp1d(fp_per_img, sens, fill_value='extrapolate')
    res = f(np.array(avgFP))

    return res

def get_FROC_fn(boxes_all, gts_all, iou_th, iobb=False):
    sens = None
    fp_per_img = None
    if not iobb:
        sens, fp_per_img = FROC(boxes_all, gts_all, iou_th)
    else:
        sens, fp_per_img = FROC_IOBB(boxes_all, gts_all, iou_th)
    
    f = interpolate.interp1d(fp_per_img, sens, fill_value='extrapolate')
    return f



def filter_proposals_by_lesion_type(proposals, dl_info, lesion_type):
    filter = np.full((len(proposals['imname'])), False)
    # Filter by lesion type
    filtered_dl_info = dl_info.loc[(dl_info.Coarse_lesion_type == lesion_type)]
    # Filter by dataset (Train_Val_Test)
    filtered_dl_info = filtered_dl_info.loc[(
        filtered_dl_info.Train_Val_Test == 3)]
    filtered_dl_info = filtered_dl_info.loc[(
        filtered_dl_info.Possibly_noisy == 0)]
    print(len(filtered_dl_info))
    proposals_copy = proposals.copy()
    for _, row in filtered_dl_info.iterrows():
        last_underscore_index = row['File_name'].rfind('_')
        processed_filename = row['File_name'][:last_underscore_index] + \
            "/" + row['File_name'][last_underscore_index+1:]
        result = np.where(proposals['imname'] == processed_filename)
        if len(result[0]) > 0:
            filter[result[0]] = True

    proposals_copy['imname'] = list(compress(proposals['imname'], filter))
    proposals_copy['boxes'] = [list(compress(proposals['boxes'][0], filter))]
    proposals_copy['gts'] = [list(compress(proposals['gts'][0], filter))]
    return proposals_copy

def filter_proposals_by_lesion_size(proposals, dl_info, lesion_diameter_lower=None, lesion_diameter_upper=None):
    filter = np.full((len(proposals['imname'])), False)
    proposals_copy = proposals.copy()
    # Filter by dataset split (Train_Val_Test)
    filtered_dl_info = dl_info.loc[(dl_info.Train_Val_Test == 3)]
    filtered_dl_info = filtered_dl_info.loc[(
        filtered_dl_info.Possibly_noisy == 0)]
    print(len(filtered_dl_info))
    for _, row in filtered_dl_info.iterrows():
        lesion_diameter_px = eval(row['Lesion_diameters_Pixel_'].split(',')[0])
        pixel_spacing = eval(row['Spacing_mm_px_'].split(',')[0])
        lesion_diameter_mm = lesion_diameter_px * pixel_spacing
        if(eval(row['Spacing_mm_px_'].split(',')[0]) != eval(row['Spacing_mm_px_'].split(',')[1])):
            print("Different pixel spacing found")

        keep_entry = True

        if lesion_diameter_upper and lesion_diameter_lower:
            keep_entry = (
                lesion_diameter_mm >= lesion_diameter_lower and lesion_diameter_mm <= lesion_diameter_upper)
        elif lesion_diameter_upper:
            keep_entry = (lesion_diameter_mm < lesion_diameter_upper)
        elif lesion_diameter_lower:
            keep_entry = (lesion_diameter_mm > lesion_diameter_lower)

        last_underscore_index = row['File_name'].rfind('_')
        processed_filename = row['File_name'][:last_underscore_index] + \
            "/" + row['File_name'][last_underscore_index+1:]
        result = np.where(proposals['imname'] == processed_filename)
        if len(result[0]) > 0:
            filter[result[0]] = keep_entry

    proposals_copy['imname'] = list(compress(proposals['imname'], filter))
    proposals_copy['boxes'] = [list(compress(proposals['boxes'][0], filter))]
    proposals_copy['gts'] = [list(compress(proposals['gts'][0], filter))]
    return proposals_copy