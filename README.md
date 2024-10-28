# Face-Detection-using-Viola-Jones-VJ-Algorithm

# Face Detection using Viola-Jones (VJ) Algorithm

This project implements a real-time face detection system using the Viola-Jones (VJ) algorithm with OpenCV. Our goal is to develop an efficient and accurate face detection model that performs well under different conditions. This repository contains the code, dataset, and scripts used for training, testing, and evaluating the model.

## Project Overview

Face detection is a crucial technology in computer vision and has applications in security, human-computer interaction, and biometric recognition. This project leverages the Viola-Jones algorithm, a popular method for real-time face detection, and applies it with Haar features and a cascade classifier to efficiently identify faces in images.

### Objectives
1. Develop a real-time face detection model using the Viola-Jones algorithm.
2. Evaluate the model's performance under varying conditions, including minimum and maximum detectable face sizes.
3. Measure average detection time to assess computational efficiency.
4. Address common challenges in object detection such as viewpoint variation, occlusion, and illumination conditions.

## Features and Implementation

### 1. Dataset Preparation
The dataset was divided into:
- **Positive dataset**: 900 face images (cropped and processed for optimal detection).
- **Negative dataset**: 3700 non-face images, including backgrounds, objects, and animals.

Scripts for preprocessing the dataset, including cropping, resizing, and adding padding to images, are included in the `image_database_management` folder.

### 2. Training Process
We used OpenCV's Cascade Trainer GUI to train the model with carefully selected parameters:
- **Positive image usage** was set to 85% to avoid a common issue in OpenCV, where positive samples are sometimes removed during training.
- **Detection window size**: Increased to 40x40 pixels to improve accuracy and reduce false detections, albeit with longer training times.

### 3. Evaluation and Results
- **Minimum/Maximum Face Size Detection**: The model was tested to find the smallest and largest detectable face sizes.
- **Average Detection Time**: A Python script measures the time taken to detect faces in each image, providing insights into processing time per image and per detected face.
- **False Detection Rate**: By adjusting the negative dataset and detection parameters, we reduced the number of false positives.

### 4. Analysis of Challenges
The model was evaluated under various conditions to analyze its robustness against common detection challenges:
- **Viewpoint Variation**
- **Occlusion**
- **Illumination Changes**
- **Cluttered Backgrounds**
- **Intra-Class Variation**

## Future Improvements

- **Expand the Dataset**: Add more diverse images to improve generalization.
- **Optimize Training Parameters**: Further fine-tuning of the detection window size and image usage percentage could enhance performance.
- **Explore Alternative Algorithms**: Consider experimenting with deep learning models for more complex face detection tasks.

## Contributors

- **Chaw Thiri San** 
- **Cyrius** 
- **Guillaume**
