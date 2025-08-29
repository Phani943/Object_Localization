# Object Localization and Classification

![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=flat&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)
![MIT License](https://img.shields.io/badge/License-MIT-green.svg)

A deep learning project that simultaneously performs object classification and localization using a multitask CNN architecture built with TensorFlow/Keras.

## Overview

This project implements a multitask learning approach to identify and locate household appliances in images. The model can classify objects into 5 categories and predict their bounding box coordinates simultaneously using a single neural network based on VGG16 architecture.

## Dataset

The project uses a custom dataset containing images of household appliances with corresponding XML annotations in PASCAL VOC format.

**Classes (5 categories):**
- Air Conditioner (ac)
- Fan 
- Hair Dryer (dryer)
- Refrigerator (fridge)
- Stove

**Dataset Structure:**
```
dataset/
├── data/
│   ├── Ac/
│   ├── Fan/
│   ├── Stove/
│   ├── Refrigerator/
│   └── Hair_Dryer/
└── Annotations/
    ├── ac_annotations/
    ├── fan_annotations/
    ├── stove_annotations/
    ├── refrigerator_annotations/
    └── hair_dryer_annotations/
```

## Model Architecture

**Base Model:** VGG16 (pre-trained on ImageNet)
- **Input:** 224×224×3 RGB images
- **Feature Extraction:** Frozen VGG16 convolutional layers
- **Shared Dense Layer:** 256 units with ReLU activation
- **Dual Output Branches:**
  - **Classification Branch:** 256 → 5 units (softmax activation)
  - **Localization Branch:** 256 → 4 units (linear activation for bbox coordinates)

**Model Statistics:**
- Total params: 21,271,369 (81.14 MB)
- Trainable params: 6,556,681 (25.01 MB)
- Non-trainable params: 14,714,688 (56.13 MB)

[📊 View Model Architecture Diagram](model/model.png)

## Key Features

- **Multitask Learning:** Single network for both classification and bounding box regression
- **Transfer Learning:** Utilizes pre-trained VGG16 for robust feature extraction
- **Custom Data Generator:** Efficient batch processing with coordinate normalization
- **Bounding Box Rescaling:** Automatic rescaling from original 600×600 to 224×224
- **Early Stopping:** Prevents overfitting with validation accuracy monitoring
- **Reproducible Results:** Fixed random seeds for consistent training

## Installation

```
# Clone the repository
git clone https://github.com/Phani943/Object_Localization.git
cd Object_Localization

# Install required dependencies
pip install tensorflow
pip install opencv-python
pip install matplotlib
pip install numpy
pip install xml
```

## Usage

### 1. Training the Model

```
# Run the complete training pipeline
jupyter notebook notebook/localization-classification-training.ipynb
```

### 2. Using Pre-trained Model

```
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('model/localizer.keras')

# Class names
class_names = ['ac', 'fan', 'dryer', 'fridge', 'stove']

# Preprocess image
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Make prediction
def predict_image(img_path):
    img = preprocess_image(img_path)
    class_pred, bbox_pred = model.predict(img)
    
    class_name = class_names[np.argmax(class_pred)]
    bbox_coords = bbox_pred  # [x_min, y_min, x_max, y_max] normalized
    
    return class_name, bbox_coords

# Example usage
class_name, bbox = predict_image('path/to/image.jpg')
print(f"Detected: {class_name}")
print(f"Bounding box: {bbox}")
```

## Model Performance

**Training Results:**
- **Final Training Accuracy:** 99.99%
- **Final Validation Accuracy:** 94.20%
- **Test Accuracy:** 96.80%
- **Localization MSE:** 0.0177 (test set)
- **Training Epochs:** 12 (early stopping triggered)

**Training Configuration:**
- **Optimizer:** Adam
- **Loss Functions:** 
  - Classification: Categorical crossentropy (weight: 1.0)
  - Localization: Mean squared error (weight: 2.0)
- **Batch Size:** 16
- **Data Split:** 70% train, 20% validation, 10% test
- **Early Stopping:** Patience of 5 epochs monitoring validation accuracy

## Data Preprocessing

- **Image Resizing:** All images resized to 224×224 pixels
- **Normalization:** Pixel values normalized to [0,1] range
- **Bounding Box Normalization:** Coordinates normalized relative to image dimensions
- **Data Augmentation:** Shuffling enabled during training

## File Structure

```
Object_Localization/
├── model/
│   ├── localizer.keras          # Trained model file
│   └── model.png               # Model architecture diagram
├── notebook/
│   └── localization-classification-training.ipynb  # Complete training pipeline
├── .gitattributes
├── LICENSE                     # MIT License
└── README.md
```

## Results Visualization

The model successfully demonstrates:
- Accurate object classification across all 5 appliance categories
- Precise bounding box prediction with minimal localization error
- Robust performance on unseen test data
- Real-time inference capability

## Technical Implementation

**Custom Data Generator Features:**
- Batch-wise data loading and preprocessing
- Automatic coordinate rescaling and normalization
- One-hot encoding for classification labels
- Memory-efficient data pipeline

**Loss Function Strategy:**
- Weighted multi-task loss combining classification and regression objectives
- Higher weight (2.0) assigned to localization task for precise bbox prediction
- Categorical crossentropy for classification accuracy
- MSE for smooth bounding box regression

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TensorFlow/Keras team for the deep learning framework
- VGG16 architecture and ImageNet pre-trained weights
- PASCAL VOC annotation format for object detection datasets
