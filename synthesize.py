import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
import os
import pandas as pd
import random
from scipy.ndimage import gaussian_filter, sobel, prewitt, shift, affine_transform

example_im = Image.open(r"F:\Trainingsdaten_v06\cropd\raw\S14_C2_01_r1_10_B00010_row1_col3.tif")
example_im_np = np.array(example_im)
background = np.ones((432, 512,3), dtype=np.uint8)*100
background_im = Image.fromarray(background)

draw = ImageDraw.Draw(background_im)

num_pits = np.random.randint(1,20)
for i in range(num_pits):
    x1 = np.random.randint(10, 500)
    y1 = np.random.randint(10, 390)
    sx = np.random.randint(3, 50)
    sy = np.random.randint(3, 50)
    ox = np.random.randint(3, 20)
    oy = np.random.randint(3, 20)
    int = np.random.randint(80, 180)

    draw.ellipse((y1, x1, y1+sy, x1+sx), fill='black', outline='black')





background = np.array(background_im, dtype=np.int32)
backgroundcam = sobel(background, axis=0)
#back_cam_shifted = shift(backgroundcam, (-30, 0, 0))


"""p = 2
affine = affine_transform(background, [[np.cos(p),np.sin(p), 0], [np.sin(p), np.cos(p),0], [0, 0, 1]])"""
backgroundcam[backgroundcam < 0] = 0
backgroundcam = backgroundcam / np.max(backgroundcam) * 180

synth = np.array(background + backgroundcam, dtype=np.uint8)


p1 = plt.figure(1)
plt.imshow(background)

p1 = plt.figure(2)
plt.imshow(synth)
"""
p2 = plt.figure(3)
plt.imshow(example_im_np)


p2 = plt.figure(3)
plt.imshow(backgroundcam)

p2 = plt.figure(4)
plt.imshow(synth)"""

plt.show()