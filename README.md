# Satellite-Imagery-Object-Detection
Satellite Imagery Object Detection with YOLO Models

**Objective:**

This project aims to develop robust object detection models using different versions of the YOLO algorithm for classifying objects in satellite imagery. The primary goal is to build accurate models and compare their performance and inference speed to identify the most suitable architecture for this specific task.

**Dataset:**

The project utilizes the "Satellite Imagery Multi-vehicles Dataset (SIMD)" for training and evaluation. The dataset is available through the following links:
- GitHub Repository: https://github.com/ihians/simd
- Pre-trained Model Example: https://github.com/asimniazi63/Object-Detection-on-Satellite-Images

For testing purposes, as the "Test" folder is missing, we will leverage the "Validation" folder. Furthermore, approximately 15% of the "Training" folder will be used for validation.

**Project Outline:**

1. **YOLOv5 and YOLOv8 Training:**
   - Train YOLOv5 and YOLOv8 algorithms using the ultralytics library with the SIMD dataset.
   - Evaluate the models' detection accuracy and compare their inference speeds to determine the most efficient option.

2. **YOLOv6 with SqueezeNet:**
   - Develop a customized YOLOv6 model with a SqueezeNet backbone for object detection.
   - Train this new model from scratch on the SIMD dataset.
   - Report the model's performance and compare it with the previous YOLO versions.

3. **YOLOv6 with MobileNet V2:**
   - Modify the YOLOv6 architecture to adopt MobileNet V2 as the backbone.
   - Utilize pre-trained weights for MobileNet V2 and freeze them during model training.
   - Evaluate the model's performance and compare it with the SqueezeNet-based model.
   - Ensure the use of small-sized models for efficiency.

**Results Presentation:**

For each part of the project, we will present processed satellite imagery samples, along with their respective classes and bounding boxes. The report will highlight the accuracies achieved, model evaluation metrics, and any notable insights gained during the experimentation.
