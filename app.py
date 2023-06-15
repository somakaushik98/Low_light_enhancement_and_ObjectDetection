import streamlit as st
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
from matplotlib import pyplot as plt
import cv2 as cv
import os
import sys
import argparse
import time
import numpy as np
from PIL import Image
from torchvision import transforms
import time
from objectDetection import detection_recognition
from lowlightenhancement import enhance_net_nopool

def lowlight(image_path):
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    data_lowlight = Image.open(image_path)
    data_lowlight = data_lowlight.convert('RGB')
    data_lowlight = (np.asarray(data_lowlight)/255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2,0,1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)

    DCE_net = enhance_net_nopool().cuda()
    DCE_net.load_state_dict(torch.load('./Epoch99.pth'))
    start = time.time()
    _,enhanced_image,_ = DCE_net(data_lowlight)

    end_time = (time.time() - start)
    print(end_time)
    image_path = image_path.replace('transit','result')
    result_path = image_path
    if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
        os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))

    torchvision.utils.save_image(enhanced_image, './test.png')
st.title('Low Light Image Enhancement and Object Detection')

# set header
st.header('Please upload a image')
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

if file is not None:
    with torch.no_grad():
        img1 = Image.open(file)
        img1.save('./transit.png')
        lowlight('./transit.png')
    c = st.radio(label = 'Choose an option', options = ['Denoising Image','Without Denoising Image'])
    img = cv.imread('test.png')
    if c == 'Denoising Image':
        dst = cv.fastNlMeansDenoisingColored(img,None,10,10,7,21)
        image = Image.fromarray(dst)
        image.save("test_uni.png")
        image1 = detection_recognition('test_uni.png')
    else:
        image1 = detection_recognition('test.png')
    st.image(file, width=200, caption='Low Light Images')
    st.write('<style>div.row-widget.stVertical {flex-wrap: wrap;}</style>', unsafe_allow_html=True)
    st.image(img, width=200, caption='Enhanced Image')
    st.image(image1, width=200, caption='Object Detected Image')

    





