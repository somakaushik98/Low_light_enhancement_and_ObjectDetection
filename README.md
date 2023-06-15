# ECE 285 UCSD - Spring'23 - Final Project
# Low Light Enhancement And Object Detection
<br>
This project presents a novel approach to enhance visibility and enable object detection in low-light and twilight scenarios. We employ the Zero-DCE method, a deep learning-based approach, to effectively enhance low-light images by improving their brightness, contrast, and color balance. By extending this technique to twilight images, we bridge the gap between low-light and twilight conditions. Additionally, we utilize the YOLOv3 object detection algorithm, pretrained on the COCO dataset, to detect objects in the enhanced twilight images. The experimental results demonstrate the effectiveness of our approach in significantly improving visibility, preserving details, and accurately detecting objects in challenging low-light and twilight scenarios. This comprehensive solution holds promise for applications in surveillance, autonomous driving, and nighttime image analysis, where accurate object detection under challenging lighting conditions is crucial.
<br>

https://github.com/somakaushik98/Low_light_enhancement_and_ObjectDetection/assets/63076797/f5287557-e947-4448-8ab6-2de00b232093

# Installation
To install Python dependencies and modules, use
```bash
pip install -r requirements.txt
```

# Dataset

a) The LOL dataset is composed of 500 low-light and normal-light image pairs and is divided into 485 training pairs and 15 testing pairs. The low-light images contain noise produced during the photo capture process. Most of the images are indoor scenes. All the images have a resolution of 400×600. The dataset was introduced in the paper Deep Retinex Decomposition for Low-Light Enhancement.
<br>
<br>
 b) The Exclusively Dark (ExDARK) dataset is a collection of 7,363 low-light images from very low-light environments to twilight (i.e 10 different conditions) with 12 object classes (similar to PASCAL VOC) annotated on both image class level and local object bounding boxes.
 <br>
 <br>
 c) The COCO (Common Objects in Context) dataset is a popular and widely used dataset for object detection and segmentation tasks in computer vision. It consists of over 200,000 labeled images with more than 80 object categories, covering various everyday objects and scenes. The dataset includes bounding box annotations for object localization, segmentation masks for pixel-level labeling, and captions for image captioning. Its comprehensive and diverse nature makes it valuable for training and evaluating models in the field of computer vision.

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
