# ECE 285 UCSD - Spring'23 - Final Project
# Low Light Enhancement And Object Detection

A Project By -

1) Sai Kaushik Soma - (PID A59020013)
2) Venkata Harsha Vardhan Gangala - (PID A59019872)
<br>

# Project Description

This project presents a novel approach to enhance visibility and enable object detection in low-light and twilight scenarios. We employ the Zero-DCE method, a deep learning-based approach, to effectively enhance low-light images by improving their brightness, contrast, and color balance. By extending this technique to twilight images, we bridge the gap between low-light and twilight conditions. Additionally, we utilize the YOLOv3 object detection algorithm, pretrained on the COCO dataset, to detect objects in the enhanced twilight images. The experimental results demonstrate the effectiveness of our approach in significantly improving visibility, preserving details, and accurately detecting objects in challenging low-light and twilight scenarios. This comprehensive solution holds promise for applications in surveillance, autonomous driving, and nighttime image analysis, where accurate object detection under challenging lighting conditions is crucial.
<br>

# Installation
To install Python dependencies and modules, use
```bash
pip install -r requirements.txt
```

# Dataset
a) Low-Light Dataset: The Low-Light (LoL) dataset is a challenging benchmark for evaluating
low-light image enhancement methods. It consists of 500 pairs of low-light and correspond-
ing normal-light images, covering a diverse range of scenes and lighting conditions. The
dataset has been used to evaluate the performance of various low-light image enhancement
algorithms, including deep learning-based methods. Due to its realistic and challenging na-
ture, the LoL dataset has become a popular benchmark for researchers working on low-light
image enhancement.
<br>
<br>
b) Dark-Face Dataset: Dark Face is a dataset containing images captured in both dark and
normal lighting conditions. The dataset contains a total of 789 image pairs, where each pair
consists of a dark image and a corresponding normal image. The dataset was designed to
evaluate image enhancement algorithms that aim to improve the visibility and quality of
dark images. The dataset also includes ground truth images for evaluation purposes.
 <br>
 <br>
 c) The COCO (Common Objects in Context) dataset is a popular and widely used dataset for object detection and segmentation tasks in computer vision. It consists of over 200,000 labeled images with more than 80 object categories, covering various everyday objects and scenes. The dataset includes bounding box annotations for object localization, segmentation masks for pixel-level labeling, and captions for image captioning. Its comprehensive and diverse nature makes it valuable for training and evaluating models in the field of computer vision.

 # Zero DCE (Low Light Enhancement)

ZERO-DCE framework is devised with Deep Curve Estimation Network (DCE-Net) that estimates a group of Light-Enhancement curves (LE-curves) that best fit the input image.It is a neural network that is designed to learn the relationship between an input low-light image and its corresponding curve parameter maps. These maps represent the best-fitting curves for the input image. The network is composed of seven convolutional layers with 32 kernels each,
which are activated using the ReLU function. The last convolutional layer is followed by the Tanh activation function, which generates 24 parameter maps for eight iterations. Each iteration requires three parameter maps for the three color channels. The Zero DCE-Net is a lightweight network with
symmetrical concatenation.

![electronics-11-02750-g002](https://github.com/somakaushik98/Low_light_enhancement_and_ObjectDetection/assets/63076797/35d7be80-8163-4edb-99c1-fa7d51513e90)


# YOLOv3 (Object Detection)

YOLOv3, short for You Only Look Once version 3, is a popular object detection algorithm that employs a single deep neural network to achieve real-time and accurate object recognition in images and videos. It divides the input image into a grid and predicts bounding boxes and class probabilities for multiple objects within each grid cell, enabling efficient detection across the entire image. YOLOv3 employs a feature pyramid network and three detection scales to handle objects of different sizes, enhancing its ability to detect both small and large objects. Additionally, it incorporates anchor boxes to improve localization precision. With its efficient architecture and strong performance, YOLOv3 has become widely adopted for various computer vision tasks, including autonomous driving, surveillance systems, and interactive applications.

 # Directory Structure

 1) Final_project_1.ipynb : The Jupyter Notebook consists of complete project pipeline starting from data preprocessing to final object detection after light enhancement. <br>
 2) app.py : A webapp built using streamlit where we developed an UI to test our model and present our results. <br>
 3) coco.names : It is a text file in YOLOv3 containing the names of objects the model can detect from the COCO dataset, providing labels for detected objects during inference. <br>
 4) lowlightenhancement.py : The python script is utilised to import the necessary functions related to low light enhancement while building our webapp.
 5) objectDetection.py : The python script is utilised to import the necessary functions related to object detection using yolo while building our webapp.
 6) yolov3.cfg : It is a configuration file in YOLO models, specifying the network architecture and parameters.
It defines how the model is structured and operates during training and inference.


# Project Workflow

![WhatsApp Image 2023-06-12 at 00 26 44](https://github.com/somakaushik98/Low_light_enhancement_and_ObjectDetection/assets/63076797/22698657-7888-4bff-af86-7baf1f1da884)

1) After installation of necessary libraries we run the following command.
   ```bash
   streamlit run app.py
   ```
2) Once the webapp is hosted locally we provide a dark input image as the input image file.
3) Then we are prompted to choose between Denoising and Without Denoising.
4) Finally the outputs are displayed on the UI of the webapp.

# Results 
![Untitled Diagram drawio (1)](https://github.com/somakaushik98/Low_light_enhancement_and_ObjectDetection/assets/63076797/df587ee0-9d0a-4d11-b383-0924c208dc05)

# Demo

https://github.com/somakaushik98/Low_light_enhancement_and_ObjectDetection/assets/63076797/1bf45b58-2c48-430e-9765-666ce2d7714d





