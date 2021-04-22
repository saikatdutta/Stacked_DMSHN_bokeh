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


from stacked_DMSHN import stacked_DMSHN 

device = torch.device("cuda:0")


feed_width = 1536
feed_height = 1024

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

bokehnet = stacked_DMSHN().to(device)
bokehnet = nn.DataParallel(bokehnet)
# bokehnet.load_state_dict(torch.load('stacked/dmshn-4-0.pth',map_location=device))
bokehnet.load_state_dict(torch.load('../vanilla_models/stacked2/dmshn-12-0.pth',map_location=device))

# bokehnet.load_state_dict(torch.load('vanilla3/dmshn-19-0.pth',map_location=device))
# bokehnet.load_state_dict(torch.load('vanilla2_asl/dmshn-19-0.pth',map_location=device))



import time

total_time=0 


# os.makedirs('outputs/stacked2_val294/',exist_ok= True)
# os.makedirs('outputs/stacked2_val/',exist_ok= True)
os.makedirs('outputs/stacked_DMSHN/',exist_ok= True)


# org_dir = '../PyNET-Bokeh/test_ut/'
# listfiles = os.listdir(org_dir)

# os.makedirs('analysis',exist_ok=True)

with torch.no_grad():
    for i in tqdm(range(4400,4694)):
    #for i in tqdm(range(0,200)):
    # for file in listfiles : 

        # image_path = '/media/data2/saikat/bokeh_data/ValidationBokehFree/' + str(i) + '.png'
        image_path = '/media/data2/saikat/bokeh_data/Training/original/' + str(i) + '.jpg'
        #image_path = '/media/data2/saikat/bokeh_data/TestBokehFree/' + str(i) + '.png'

        # image_path = org_dir + file

        # Load image and preprocess
        input_image = pil.open(image_path).convert('RGB')
        original_width, original_height = input_image.size

        org_image = input_image
        org_image = transforms.ToTensor()(org_image).unsqueeze(0)

        input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        # PREDICTION
        org_image = org_image.to(device)
        input_image = input_image.to(device)

        start_time = time.time()
        bok_pred = bokehnet(input_image)
        # print (bok_pred.shape)
        total_time += time.time() - start_time

        

        bok_pred = F.interpolate(bok_pred,(original_height,original_width),mode = 'bilinear')
        
        
        save_image(bok_pred,'./outputs/stacked_DMSHN/'+ str(i)+'.png')


        del bok_pred
        del input_image
    



