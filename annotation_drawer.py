import pylidc as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manim
from skimage.measure import find_contours
import pandas as pd
from pylidc.utils import consensus
import types
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import cv2
import time
from tqdm import tqdm
import matplotlib.path as mplpath

#for i in range(1,2):
#    id_number = '{}'.format(i)
#    pid = 'LIDC-IDRI-' + id_number.zfill(4)
#    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).person()
#    if scan is not None:
#        nods = scan.cluster_annotations()
#        print(nods)
#        if len(nods) > 0:
#            print("{0} has {1} nodules.".format(pid, len(nods)))

for i in range(1, 6):
    anns = pl.query(pl.Annotation).filter(pl.Annotation.malignancy == i)
    for person in anns:
      images = person.scan.load_all_dicom_images()
      p_id = images[0].PatientID
      contours = sorted(person.contours, key=lambda c: c.image_z_position)
      fnames = person.scan.sorted_dicom_file_names.split(',')
      index_of_contour = [fnames.index(c.dicom_file_name) for c in contours]
      min_slice = min(index_of_contour)
      max_slice = max(index_of_contour)
      current_slice = min_slice
      #print(len(contours))

      for j, c in enumerate(contours):
          #fig = plt.figure(figsize=(1,1))
          #ax_image = fig.add_axes([0, 0, 1, 1])
          #img = ax_image.imshow(images[current_slice + j].pixel_array, cmap=plt.cm.gray)
         if current_slice + j < len(images):
              im = images[current_slice + j].pixel_array
              arr = c.to_matrix(include_k=False)

              dcm_wc = images[current_slice + j].WindowCenter
              dcm_ww = images[current_slice + j].WindowWidth

              if hasattr(dcm_wc, "__iter__"):
                  if dcm_ww[0] < dcm_ww[1]:
                    dcm_wc = dcm_wc[0]
                    dcm_ww = dcm_ww[0]
                  else:
                    dcm_wc = dcm_wc[1]
                    dcm_ww = dcm_ww[1]

              window_max = dcm_wc + dcm_ww/2
              window_min = dcm_wc - dcm_ww/2
              im = 255*(im - window_min)/(window_max - window_min)
              im[im > 255] = 255
              im[im < 0] = 0

              x_list = arr[:,1]
              y_list = arr[:,0]
              x_max = max(x_list)
              x_mini = min(x_list)
              y_max = max(y_list)
              y_mini = min(y_list)
              wid = x_max - x_mini
              hei = y_max - y_mini
              #ax_image.axis('off')

              ni, nj = im.shape
              ii, jj = np.indices((ni, nj))
              test_points = np.c_[ii.flatten(), jj.flatten()]
              path = mplpath.Path(arr, closed=True)
              contains_pts = path.contains_points(test_points)
              contains_pts = contains_pts.reshape(ni, nj)

              mask = np.zeros((ni, nj), dtype=np.bool)
              mask = np.logical_or(mask, contains_pts)

              im_cut = im[y_mini-10:y_max+10, x_mini-10:x_max+10]
              mask_cut = mask[y_mini-10:y_max+10, x_mini-10:x_max+10]

              #plt.imshow(im, cmap='gray')
              #plt.show()

              #plt.imshow(mask, cmap='gray')
              #plt.show()

              #img = ax_image.imshow(im_cut)

              if im_cut.size >= 128:
                if np.mean(im_cut) < 250:
                    im_cut = Image.fromarray(im_cut)
                    im_cut = im_cut.convert("RGB")
                    mask_cut = Image.fromarray(mask_cut)
                    mask_cut = mask_cut.convert("RGB")
                    #im_cut = im_cut.resize(size=(256, 256), resample=Image.LANCZOS)
                    im_cut.save('D:/non-visual-data/Cropped_images_radiomics/Malignancy{0}/mal{0}_{1}_{2}.png'.format(i, p_id, j))
                    mask_cut.save('D:/non-visual-data/Cropped_images_radiomics/Malignancy{0}_mask/mal{0}_{1}_{2}_mask.png'.format(i, p_id, j))
                #print("Malignancy {}".format(i))
              """
              if im_cut.size > 100:
                np.save('D:/non-visual-data/Cropped_images/Malignancy{0}np100/img_test/mal{0}_{1}_{2}'.format(i, p_id, j), im_cut)
              """
              """
              if im_cut.size > 400:
                np.save('D:/non-visual-data/Cropped_images/Malignancy{0}np400/mal{0}_{1}_{2}'.format(i, p_id, j), im_cut)
              """
          #df.to_csv('E:/non-visual-data/Malignancy{0}/mal{0}_{1}_{2}.csv'.format(i, p_id, j), index=False)
          #plt.clf()
          #plt.close()
          #plt.show()

""""
r = patches.Rectangle(xy=(x_mini-round(0.5*wid), y_mini-round(0.5*hei)), width=wid*2, height=hei*2, ec='#000000', fill=False)
df = pd.DataFrame(arr, columns=['x', 'y', 'z'])
#ax_image.plot(arr[:,1], arr[:,0], '-r')
#ax_image.set_xlim(-0.5,511.5); ax_image.set_ylim(511.5,-0.5)
ax_image.add_patch(r)
"""
