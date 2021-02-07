import cv2
import os
import pydicom
import numpy as np
import sys
import pandas as pd
import time
import shutil
import glob
from scipy.io import loadmat
import math


def draw_bbs(proposals, dataset_path, out_folder, annotations, bb_scale_factor=1):
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 

    fontScale = 1

    # Blue color in BGR 
    text_color = (0, 255, 0)
    text_org = (25, 25) 

    thickness_font = 1
    # Ground-truth color
    gt_color = (255, 255, 0)

    fp_color = (0, 0, 255) 

    line_color = (0, 255, 0)

    syn_orthogonal = (0,255,255)

    for i in range(0, len(proposals['imname'])):
        for idx, row in annotations.iterrows():
            if row['patientID'] == proposals_tcga['imname'][i].split('/')[0]:
                print(dataset_path + proposals['imname'][i])
                image = cv2.imread(dataset_path + proposals['imname'][i], -1) 
                # 3DCE & MULAN use a full range window of [-1024, 3071]
                img_min = -1024
                img_max = 3071
               
                image = convert_image_like_DL(im1=image, img_min=img_min, img_max=img_max, intercept=int(row['intercept']), slope=int(row['slope']))
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                # Window name in which image is displayed 
                window_name = 'Image'

                # x1,y1 of gt
                gt_start_point = (int(proposals_tcga['gts'][i][0][0] * bb_scale_factor), int(proposals_tcga['gts'][i][0][1] * bb_scale_factor))

                # x2, y2 of gt
                gt_end_point = (int(proposals_tcga['gts'][i][0][2] * bb_scale_factor), int(proposals_tcga['gts'][i][0][3] * bb_scale_factor))

                # line bookmark
                lb_start_point = (int(row['start_x']), int(row['start_y']))
                lb_end_point = (int(row['end_x']), int(row['end_y']))

                # Line thickness of 2 px 
                thickness_line = 2

                # Calculate perp
                x1,y1,x2,y2 = get_orthogonal(lb_start_point[0], lb_start_point[1], lb_end_point[0], lb_end_point[1], row['length'] , 0.5) 
                
                # Draw ortogonal
                image = cv2.line(image, (x1,y1), (x2,y2), syn_orthogonal, thickness_line)

                # Draw line bookmark
                image = cv2.line(image, lb_start_point, lb_end_point, line_color, thickness_line)
                # # Calculate new bounding box like in DeepLesion
                # xmin = min(x1,x2, start_point[0], end_point[0])
                # xmax = max(x1,x2, start_point[0], end_point[0])
                # ymin = min(y1, y2, start_point[1], end_point[1])
                # ymax = max(y1, y2, start_point[1], end_point[1])
                # start = (int(xmin) - 5, int(ymin) - 5)
                # end = (int(xmax) + 5, int(ymax) + 5)
                # # replace diagonal with pseudo recist diagonal
                # annotations = change_diagonal_in_csv(start[0],start[1],end[0],end[1],idx,annotations)

                # Draw groundtruth bounding box
                image = cv2.rectangle(image, gt_start_point, gt_end_point, gt_color, thickness_line)

                for j in range(0, len(proposals_tcga['boxes'][0][i])):
                    # Score >= 0.7
                    if proposals_tcga['boxes'][0][i][j][4] >= 0.7:
                        start_x = proposals_tcga['boxes'][0][i][j][0] * bb_scale_factor
                        start_y = proposals_tcga['boxes'][0][i][j][1] * bb_scale_factor
                        end_x = proposals_tcga['boxes'][0][i][j][2] * bb_scale_factor
                        end_y = proposals_tcga['boxes'][0][i][j][3] * bb_scale_factor
                        start_point = (int(start_x), int(start_y))

                        end_point = (int(end_x), int(end_y))
                        image = cv2.rectangle(image, start_point, end_point, fp_color, thickness_line)
                        image = cv2.putText(image, str(round(proposals_tcga['boxes'][0][i][j][4], 3)), (int(proposals_tcga['boxes'][0][i][j][0] * bb_scale_factor), int(proposals_tcga['boxes'][0][i][j][1] * bb_scale_factor)-5), font, 0.5, (100, 0, 200), thickness_font, cv2.LINE_AA)
                        
                # Using cv2.putText() method 
                image = cv2.putText(image, proposals['imname'][i], text_org, font, fontScale, text_color, thickness_font, cv2.LINE_AA)
                image = cv2.putText(image, row['anatomy'], (25, 50), font, fontScale, text_color, thickness_font, cv2.LINE_AA)

                # Save image
                cv2.imwrite(out_folder + proposals['imname'][i].replace('/','_'), image)
        
        # Save .csv with pseudo recist
        # annotations.to_csv('annotation_file.csv', index=False)

# https://stackoverflow.com/questions/8664866/draw-perpendicular-line-to-a-line-in-opencv                
def get_orthogonal(aX, aY, bX, bY, length, ratio):
    length = math.sqrt(math.pow(bX-aX, 2) + math.pow(bY-aY, 2)) * ratio
    # Get the direction vector going from A to B
    vX = bX-aX
    vY = bY-aY
 
    mag = math.sqrt(vX*vX + vY*vY)
    vX = vX / mag
    vY = vY / mag
    temp = vX
    vX = 0-vY
    vY = temp

    # Create a new line at mid point
    mid_x = (aX + bX)/2
    mid_y = (aY + bY)/2

    cX = mid_x + vX * length
    cY = mid_y + vY * length

    dX = mid_x - vX * length
    dY = mid_y - vY * length
    return int(cX), int(cY), int(dX), int(dY)

def change_diagonal_in_csv(x1,y1,x2,y2,index,df):
    df.at[float(index),'start_x'] = x1
    df.at[float(index),'start_y'] = y1
    df.at[float(index),'end_x'] = x2
    df.at[float(index),'end_y'] = y2
    return df

def convert_image_like_DL(im1, img_min, img_max, intercept, slope):
    im1 = im1.astype(np.float32, copy=False) - 32768
    im1 = im1.astype(float)
    im1 -= img_min
    im1 /= img_max - img_min
    im1[im1 > 1] = 1
    im1[im1 < 0] = 0
    im1 *= 255
   
    return im1.astype(np.uint8)

if __name__ == "__main__":
    out_folder = "./tcga/images_with_bbs/"
    # Path to TCGA images
    dataset_path = 'TCGA_restructured/Images_png/'
    annotations = pd.read_csv('./tcga/filtered_tcga_annotations/tcga_info_line_bookmark_350.csv')
    proposals_tcga = loadmat('./tcga/proposal_files/3dce_proposal_files/proposals_lesion_types_and_size/proposals_test_3DCE_3_image_3_slice_tcga_350_size_lt_10.mat')
    draw_bbs(proposals=proposals_tcga, dataset_path=dataset_path, out_folder=out_folder, annotations=annotations, bb_scale_factor=1)




            