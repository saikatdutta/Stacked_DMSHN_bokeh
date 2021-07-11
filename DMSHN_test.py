from __future__ import absolute_import, division, print_function
import cv2

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision.utils import save_image

from skimage.measure import compare_psnr,compare_ssim
from tqdm import tqdm

import math
import numbers
import sys

import matplotlib.pyplot as plt



from DMSHN import DMSHN 

device = torch.device("cuda:0")


feed_width = 1536
feed_height = 1024


bokehnet = DMSHN().to(device)
# bokehnet = nn.DataParallel(bokehnet)
bokehnet.load_state_dict(torch.load('checkpoints/DMSHN/dmshn.pth',map_location=device))



os.makedirs('outputs/DMSHN',exist_ok=True)


with torch.no_grad():
    for i in tqdm(range(4400,4694)):

        image_path = '/media/data2/saikat/bokeh_data/Training/original/' + str(i) + '.jpg'   # change input path

        # Load image and preprocess
        input_image = pil.open(image_path).convert('RGB')
        original_width, original_height = input_image.size

        input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        # PREDICTION
        input_image = input_image.to(device)

        bok_pred = bokehnet(input_image)

        bok_pred = F.interpolate(bok_pred,(original_height,original_width),mode = 'bilinear')
        
        save_image(bok_pred,'./outputs/DMSHN/'+ str(i)+'.png')
        

        del bok_pred
        del input_image
    

