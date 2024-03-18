import numpy as np
import cv2
from PIL import Image
import pandas as pd
from matplotlib import pyplot as plt
from skimage.measure import label as l
from scipy.ndimage.filters import gaussian_filter
import os
import random


# open example image + mask + background image
directory = r"C:\Users\VEYSEL\Desktop\Datasets_AI\mask_rcnn_8bit_standard"
background_directory = r"C:\Users\VEYSEL\Desktop\Datasets_AI\background imgs"
object_data = []

c = 1
l = len(os.listdir(os.path.join(directory, "raw")))
for name in os.listdir(os.path.join(directory, "raw")):

    img = np.array(Image.open(os.path.join(directory, "raw", name)))
    mask = np.array(Image.open(os.path.join(directory, "masks", name)))

    # copy droplet region using mask
    objects_dict = {}

    for n_obj in np.unique(mask)[1:]:
        nor_obj = []
        # find region + values of one object
        obj_coordinates = np.where(mask == n_obj)
        obj_values = img[obj_coordinates]
        center = (int((max(obj_coordinates[0]) + min(obj_coordinates[0])) / 2.), int((max(obj_coordinates[1]) + min(obj_coordinates[1])) / 2.))

        # list with [0] normalized y coords, [1] normalized x coords and [2] object values
        nor_obj.append(obj_coordinates[0] - center[0])
        nor_obj.append(obj_coordinates[1] - center[1])
        nor_obj.append(obj_values)

        objects_dict[str(n_obj)] = nor_obj

        # x, y, values, these will be exported
        object_data.append(nor_obj)




    background_img = np.copy(img) # np.zeros(img.shape)
    new_mask = np.copy(mask)

    for n in range(8):
        n_d = random.choice(np.unique(mask)[1:])
        x = np.random.randint(10, 130)
        y = np.random.randint(10, 1014)
        new_y = objects_dict[str(n_d)][0] + y
        new_x = objects_dict[str(n_d)][1] + x
        values = objects_dict[str(n_d)][2]

        background_img[new_y, new_x] = values



        # create new mask
        m_max = max(np.unique(new_mask))
        new_mask[new_y, new_x] = m_max +1

    # blur over droplet regions
    l_mask = np.array(new_mask, dtype=np.bool)
    not_l_mask = np.logical_not(l_mask)
    d_img = l_mask * background_img
    b_img = not_l_mask * background_img
    gd_img = gaussian_filter(d_img, sigma=0.75) # this leads to image with blured droplets on 0 bg
    background_img = gd_img

    # TODO: add local gaussian blur (only over new droplets)
    #background = gaussian_filter(background, sigma=0.85)

    """ax1 = plt.subplot(141)
    plt.imshow(img, cmap = "gray")

    plt.subplot(142, sharex = ax1, sharey = ax1)
    plt.imshow(mask, cmap = "gray")

    plt.subplot(143, sharex = ax1, sharey = ax1)
    plt.imshow(background_img, cmap ="gray")

    plt.subplot(144, sharex=ax1, sharey=ax1)
    plt.imshow(new_mask, cmap="gray")


    plt.show()"""
    print(f"[{c}/{l}]")
    c +=1

full_data = pd.DataFrame(object_data, columns = ("x_coordinates", "y_coordinates", "values"))
full_data.to_json(os.path.join(r"C:\Users\VEYSEL\Desktop\Datasets_AI\half_synthetic_imgs", "object_coordinates.json"), orient="records")