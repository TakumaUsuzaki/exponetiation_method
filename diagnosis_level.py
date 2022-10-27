import pandas as pd
import glob
import shutil

df = pd.read_csv('diagnosis_level.csv')
m, n = df.shape
file_path = glob.glob('C:/Users/takumau/Dropbox/Unvisualizable_augmentation_CT/Cropped_images_radiomics/*/*.png')

for i in range(m):
    PatientID = df.iloc[i, 0]
    Diagnosis_Level = df.iloc[i, 1]
    print(PatientID, Diagnosis_Level)
    for f_name in file_path:
        if PatientID in f_name:
            if "mask" not in f_name:
                shutil.copy2(f_name, 'C:/Users/takumau/Dropbox/Unvisualizable_augmentation_CT/Cropped_image_diagnosis/diagnosis{}'.format(Diagnosis_Level))
            if "mask" in f_name:
                shutil.copy2(f_name, 'C:/Users/takumau/Dropbox/Unvisualizable_augmentation_CT/Cropped_image_diagnosis/diagnosis{}_mask'.format(Diagnosis_Level))
