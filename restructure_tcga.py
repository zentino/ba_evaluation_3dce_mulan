import cv2
import os
import pydicom
import numpy as np
import sys
import pandas as pd
import time
import shutil
import glob


def check_slice_idx(inputdir, images, annotations_df):
    dir_list = [f for f in  os.listdir(inputdir)]
    for index, row in annotations_df.iterrows():
        # sub_dir is TCGA-LIHC ...
        for sub_dir in dir_list:
            # All patient ids
            patient_dirs = [f for f in  os.listdir(inputdir + sub_dir)]
            
            for patient_dir in patient_dirs:
                if row['patientID'] == patient_dir:
                    # Lesion type?
                    lesion_type_dirs = [f for f in  os.listdir(os.path.join(inputdir, sub_dir, patient_dir))]
                    image_dirs = [f for f in  os.listdir(os.path.join(inputdir, sub_dir, patient_dir, lesion_type_dirs[0]))]
                    # Choose first image dir
                    images = [f.replace('.dcm', '') for f in  os.listdir(os.path.join(inputdir, sub_dir, patient_dir, lesion_type_dirs[0], image_dirs[0]))]
                    slice_idx_found = False
                    for image in images:
                        # Study?
                        study, idx = image.split('-')
                        if row['sliceIndex'] == int(idx) and int(study) == 1:
                            slice_idx_found = True
                            break
                    
                    if not slice_idx_found:
                        print(row['patientID'])
                        print(row['sliceIndex'])
                        print(images)
                        time.sleep(1)

def dcm_to_png(input_dir, images, outdir, intercept, slope ):
    offset = 32768
    for i in range(0, len(images)):
        try:
            ds = pydicom.read_file(input_dir + '/' + images[i]) # read dicom image
            # Calculate HUs and add offset
            img = ds.pixel_array * slope + intercept + offset
            img = img.astype(np.uint16)
            cv2.imwrite(outdir + '/' + str(i) + '.png',img) # write png image
        except Exception as e:
            print(e)

def remove_spaces(inputdir, annotations_df):
    dir_list = [f for f in  os.listdir(inputdir)]
    # keep track of checked patients
    patients = []  
    for index, row in annotations_df.iterrows():
        # sub_dir is TCGA-LIHC ...
        for sub_dir in dir_list:
            # All patient ids
            patient_dirs = [f for f in  os.listdir(inputdir + sub_dir)]
            patient_found = False
            patient_dir = ''
            for p in patient_dirs:
                if row['patientID'] == p:
                    patient_found = True
                    patient_dir = p
                    
            if patient_found and row['patientID'] not in patients:
                lesion_type_dirs = [f for f in  os.listdir(os.path.join(inputdir, sub_dir, patient_dir))]
                image_dirs = [f for f in  os.listdir(os.path.join(inputdir, sub_dir, patient_dir, lesion_type_dirs[0]))]

                for image_dir in image_dirs:
                    os.rename(os.path.join(inputdir, sub_dir, patient_dir, lesion_type_dirs[0], image_dir), os.path.join(inputdir, sub_dir, patient_dir, lesion_type_dirs[0], image_dir.replace(' ','-')))

                patients.append(row['patientID'])

   
def restructure_data(inputdir, annotations_df):
    remove_spaces(inputdir, annotations_df)

    dir_list = [f for f in  os.listdir(inputdir)]
    c = 0
    # keep track of checked patients
    for index, row in annotations_df.iterrows():
        # sub_dir is TCGA-LIHC ...
        for sub_dir in dir_list:
            # All patient ids
            patient_dirs = [f for f in  os.listdir(inputdir + sub_dir)]
            patient_found = False
            patient_dir = ''
            image_found = False
            for p in patient_dirs:
                if row['patientID'] == p:
                    patient_found = True
                    patient_dir = p
                    
            if patient_found:
                # Lesion type?
                lesion_type_dirs = [f for f in  os.listdir(os.path.join(inputdir, sub_dir, patient_dir))]
                image_dirs = [f for f in  os.listdir(os.path.join(inputdir, sub_dir, patient_dir, lesion_type_dirs[0]))]    
                for image_dir in image_dirs:
                    # Check if file ends with .dcm
                    images = [f for f in  os.listdir(os.path.join(inputdir, sub_dir, patient_dir, lesion_type_dirs[0], image_dir)) if '.dcm' in f]
                    input_dir = os.path.join(inputdir, sub_dir, patient_dir, lesion_type_dirs[0], image_dir)
                    out_dir = os.path.join(inputdir, sub_dir, patient_dir, lesion_type_dirs[0], image_dir) + '/images_png'

                    # for image_dir in image_dirs:
                    #     out_dir = os.path.join(inputdir, sub_dir, patient_dir, lesion_type_dirs[0], image_dir) + '/images_png'
                    #     shutil.rmtree(out_dir, ignore_errors=True)
                    #     print(out_dir)
                    #     print("removed")
                                                    
                    # Convert dicom to png
                    image_found = extract_dicom_info(images=images, input_dir=input_dir, out_dir=out_dir, row=row, index=index,  annotations_df=annotations_df, image_dir=image_dir)
                    if image_found:
                        c+=1
                        break
            if image_found:
                break
    print(c)
    # Was only used because some dicom files don't have a dicom window. Since DeepLesion methods only use one window
    # annotations_df = remove_row_when_nan(annotations_df)
    annotations_df.to_csv('tcga_16_01_1413.csv', index=False)
    return annotations_df

# def extract_dicom_info(images, input_dir, out_dir, row, index, annotations_df, patients):
#     image_found = False
#     for i in range(0, len(images)):
#         try:
#             data = pydicom.read_file(input_dir + '/' + images[i])
#             if str(data[('0008','0018')].value) == row['instanceUID']:
#                 try:
#                     os.mkdir(out_dir)
#                     patients[row['patientID']] += 1
#                 except:
#                     print(f"Images png (folder {input_dir}) for patient {row['patientID']} already exists")

#                 image_found = True
#                 ps_x, ps_y = get_pixel_spacing(data)
#                 slice_thickness = get_slice_thickness(data)
#                 #window_center , window_width, intercept, slope = get_windowing(data)
#                 intercept, slope = get_windowing(data)

#                 # Not needed for now. DeepLesion methods use only one window of [-1024, 3071]
#                 # img_min = window_center - window_width//2
#                 # img_max = window_center + window_width//2
#                 # ', ' -> sonst wird als Zahl erkannt
#                 # DICOM_window = str(img_min) + ', ' + str(img_max)
#                 # annotations_df.at[float(index),'DICOM_window'] =  DICOM_window

#                 annotations_df.at[float(index),'filename'] =  str(row['patientID']) + '_' + str(patients[row['patientID']]) + '-' + str(i)
#                 annotations_df.at[float(index),'sliceIndex'] =  str(patients[row['patientID']]) + '-' + str(i)
#                 annotations_df.at[float(index),'Spacing_mm_px_']  = str(ps_x) + ',' + str(ps_y) + ',' + str(slice_thickness)
#                 annotations_df.at[float(index),'intercept'] = str(intercept)
#                 annotations_df.at[float(index),'slope'] = str(slope)
#                 # 3 is Test in DeepLesion
#                 annotations_df.at[float(index),'Train_Val_Test']  = 3
                
#                 print(f"convert dcms from folder {input_dir} tp pngs for patient {row['patientID']}")
#                 dcm_to_png(input_dir=input_dir, images=images, outdir=out_dir, intercept=intercept, slope=slope, patients=patients, row=row)
#                 break
#         except Exception as e:
#             print("Error in extract function")
#             print(e)

#     return image_found

def extract_dicom_info(images, input_dir, out_dir, row, index, annotations_df, image_dir):
    image_found = False
    for i in range(0, len(images)):
        try:
            data = pydicom.read_file(input_dir + '/' + images[i])
            if str(data[('0008','0018')].value) == row['instanceUID']:
               
                image_found = True
                ps_x, ps_y = get_pixel_spacing(data)
                slice_thickness = get_slice_thickness(data)
                intercept, slope = get_windowing(data)
                annotations_df.at[float(index),'Spacing_mm_px_']  = str(ps_x) + ',' + str(ps_y) + ',' + str(slice_thickness)
                annotations_df.at[float(index),'intercept'] = str(intercept)
                annotations_df.at[float(index),'slope'] = str(slope)
                # 3 is testset in DeepLesion
                annotations_df.at[float(index),'Train_Val_Test']  = 3
                annotations_df.at[float(index),'filename'] =  str(row['patientID']) + '/' + image_dir + '_'  + str(i)
                annotations_df.at[float(index),'sliceIndex'] = str(i)

                # Check if images_png already exists 
                if not os.path.isdir(out_dir):
                    os.mkdir(out_dir)
                    print(f"convert dcms from folder {image_dir} to pngs for patient {row['patientID']}")
                    dcm_to_png(input_dir=input_dir, images=images, outdir=out_dir, intercept=intercept, slope=slope)
                else:
                    print(f"Images png (folder {image_dir}) for patient {row['patientID']} already exists")

                break
        except Exception as e:
            print("Error in extract function")
            print(e)

    return image_found

def get_slice_thickness(data):
    return data[('0018', '0050')].value

def get_pixel_spacing(data):
    pixel_spacing = data[('0028','0030')].value

    return float(pixel_spacing[0]), float(pixel_spacing[1])
                            
# https://www.kaggle.com/omission/eda-view-dicom-images-with-correct-windowing
def get_windowing(data):
    # Dicom window not needed for DeepLesion methods. Some .dcm files don't have a Dicom window which leads to an error
    dicom_fields = [#data[('0028','1050')].value, #window center
                    #data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    windowing = [get_first_of_dicom_field_as_int(x) for x in dicom_fields]
    return windowing

# https://www.kaggle.com/omission/eda-view-dicom-images-with-correct-windowing
def get_first_of_dicom_field_as_int(x):
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)

def remove_row_when_nan(df):
    
    df = df[df.DICOM_window.notnull()]
    return df

def filter_annotations(df):
    '''
        Filter by radiologst_status and remove duplicate
        entries by slice index and instanceUID
    '''
    # 1414
    filtered_df = df.loc[(df.radiologist_status == "radiologist")]
    # 1008
    filtered_df = filtered_df.drop_duplicates(['instanceUID', 'sliceIndex'], keep='last')
    # 350
    filtered_df = filtered_df.drop_duplicates(['patientID'], keep='last')
    #filtered_df.sort_values(by=['patientID'])
    
    return filtered_df

def move_images_to_patient_folder(inputdir, annotations_df):
    dir_list = [f for f in  os.listdir(inputdir)]
    patients = []
    for index, row in annotations_df.iterrows():
        # sub_dir is TCGA-LIHC ...
        for sub_dir in dir_list:
            # All patient ids
            patient_dirs = [f for f in  os.listdir(inputdir + sub_dir)]
            
            for patient_dir in patient_dirs:
                if row['patientID'] == patient_dir and row['patientID'] not in patients:
                    patients.append(row['patientID'])
                    # Lesion type?
                    lesion_type_dirs = [f for f in  os.listdir(os.path.join(inputdir, sub_dir, patient_dir))]
                    patient_dir_path = os.path.join(inputdir, sub_dir, patient_dir)

                    lesion_type_dirs = list(filter(lambda a: ('.png' not in a), lesion_type_dirs))
                    if lesion_type_dirs:
                        image_dirs = [f for f in  os.listdir(os.path.join(inputdir, sub_dir, patient_dir, lesion_type_dirs[0]))]
                        
                        # Move
                        for image_dir in image_dirs:
                            if os.path.isdir(os.path.join(inputdir, sub_dir, patient_dir, lesion_type_dirs[0], image_dir, 'images_png')):
                                image_dir_path = os.path.join(inputdir, sub_dir, patient_dir, lesion_type_dirs[0], image_dir)
                                images_png_path = os.path.join(image_dir_path, 'images_png')
                                for path in glob.glob(image_dir_path + '\*.dcm'):
                                    # remove dcm files
                                    os.remove(path)
                                   
                                #shutil.rmtree(images_png2_path, ignore_errors=True)
                
                                for path in glob.glob(images_png_path + '\*'):
                                    shutil.move(path, image_dir_path)
                                   
                                shutil.rmtree(images_png_path, ignore_errors=True)
                                shutil.move(image_dir_path, patient_dir_path)
                              
                        # Delete
                        for lesion_type_dir in lesion_type_dirs:
                            print(patient_dir_path + '/' + lesion_type_dir)
                            shutil.rmtree(patient_dir_path + '/' + lesion_type_dir, ignore_errors=True)

def remove_image_png_dir(input_dir=''):
    dirs = glob.glob(input_dir + "/images_png*/")
    for dir in dirs:
        shutil.rmtree(dir, ignore_errors=True)

if __name__ == "__main__":
    restructured_tcga_dir = './TCGA_data/'
    #df = pd.read_csv('CrowdsCureCancer2017Annotations.csv')
    #filtered_df = filter_annotations(df)
    #filtered_df = filtered_df.to_csv('tcga_15_01_1413.csv', index=False)
    #df = restructure_data(inputdir=restructured_tcga_dir, annotations_df=filtered_df)
    # # Move images png to patient folder
    df = pd.read_csv('tcga_16_01_1413.csv')
    move_images_to_patient_folder(inputdir= restructured_tcga_dir + '/', annotations_df=df)


    



