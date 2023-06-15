# ECE 285 UCSD - Spring'23 - Final Project
# Low Light Enhancement And Object Detection
<br>
This project presents a novel approach to enhance visibility and enable object detection in low-light and twilight scenarios. We employ the Zero-DCE method, a deep learning-based approach, to effectively enhance low-light images by improving their brightness, contrast, and color balance. By extending this technique to twilight images, we bridge the gap between low-light and twilight conditions. Additionally, we utilize the YOLOv3 object detection algorithm, pretrained on the COCO dataset, to detect objects in the enhanced twilight images. The experimental results demonstrate the effectiveness of our approach in significantly improving visibility, preserving details, and accurately detecting objects in challenging low-light and twilight scenarios. This comprehensive solution holds promise for applications in surveillance, autonomous driving, and nighttime image analysis, where accurate object detection under challenging lighting conditions is crucial.
<br>

https://github.com/somakaushik98/Low_light_enhancement_and_ObjectDetection/assets/63076797/f5287557-e947-4448-8ab6-2de00b232093

#Dataset

a) The LOL dataset is composed of 500 low-light and normal-light image pairs and is divided into 485 training pairs and 15 testing pairs. The low-light images contain noise produced during the photo capture process. Most of the images are indoor scenes. All the images have a resolution of 400×600. The dataset was introduced in the paper Deep Retinex Decomposition for Low-Light Enhancement.
<br>
 b) The Exclusively Dark (ExDARK) dataset is a collection of 7,363 low-light images from very low-light environments to twilight (i.e 10 different conditions) with 12 object classes (similar to PASCAL VOC) annotated on both image class level and local object bounding boxes.
