import pylidc as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manim
from skimage.measure import find_contours
import pandas as pd
from pylidc.utils import consensus
import types

#for i in range(1,2):
#    id_number = '{}'.format(i)
#    pid = 'LIDC-IDRI-' + id_number.zfill(4)
#    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
#    if scan is not None:
#        nods = scan.cluster_annotations()
#        print(nods)
#        if len(nods) > 0:
#            print("{0} has {1} nodules.".format(pid, len(nods)))


for i in range(1, 6):
    anns = pl.query(pl.Annotation).filter(pl.Annotation.malignancy == i)
    first = anns.first()

    images = first.scan.load_all_dicom_images()
    p_id = images[0].PatientID
    contours = sorted(first.contours, key=lambda c: c.image_z_position)
    fnames = first.scan.sorted_dicom_file_names.split(',')
    index_of_contour = [fnames.index(c.dicom_file_name) for c in contours]

    min_slice = min(index_of_contour)
    max_slice = max(index_of_contour)
    current_slice = min_slice
    #print(len(contours))

    for j, c in enumerate(contours):
        fig = plt.figure(figsize=(8,8))
        ax_image = fig.add_axes([0, 0, 1, 1])
        img = ax_image.imshow(images[current_slice + j].pixel_array, cmap=plt.cm.gray)
        arr = c.to_matrix()
        df = pd.DataFrame(arr, columns=['x', 'y', 'z'])
        df.to_csv('csv{0}_{1}.csv'.format(i, j), index=False)
        ax_image.plot(arr[:,1], arr[:,0], '-r')
        ax_image.set_xlim(-0.5,511.5); ax_image.set_ylim(511.5,-0.5)
        ax_image.axis('off')
        #fig.savefig("img{0}_{1}.png".format(i, j))
        #plt.show()

















