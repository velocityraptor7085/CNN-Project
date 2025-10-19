# CNN Image Classification Project

## Overview
This project implements a Convolutional Neural Network (CNN) for image classification using PyTorch. The project is designed as an educational resource with detailed mathematical explanations and comments for deep learning students.

## Project Structure
```
Deep-Learning/
│
├── data/                      # Data directory (auto-created)
│   ├── raw/                  # Raw downloaded datasets
│   └── processed/            # Processed and augmented data
│
├── models/                    # Saved model checkpoints
│   └── checkpoints/          # Training checkpoints
│
├── results/                   # Training results and visualizations
│   ├── plots/                # Training plots and charts
│   └── metrics/              # Performance metrics
│
├── src/                       # Source code
│   ├── data_preprocessing.py # Data loading and augmentation
│   ├── model.py              # CNN architecture
│   ├── train.py              # Training loop
│   ├── evaluate.py           # Model evaluation
│   └── visualization.py      # Visualization utilities
│
├── main.py                    # Main execution script
├── requirements.txt           # Project dependencies
└── PROJECT_REPORT.md          # Comprehensive project documentation

```

## Setup Instructions

### 1. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Project
```bash
python main.py
```

## Features

- **Data Preprocessing**: Image normalization, augmentation, and efficient data loading
- **CNN Architecture**: Multi-layer convolutional neural network with batch normalization
- **Training Pipeline**: Complete training loop with validation and checkpointing
- **Evaluation**: Comprehensive metrics including accuracy, precision, recall, and F1-score
- **Visualization**: Training curves, confusion matrices, and sample predictions

## Dataset

This project uses the CIFAR-10 dataset by default:
- 60,000 32x32 color images
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 50,000 training images and 10,000 test images

## Learning Objectives

This project demonstrates:
1. **Convolutional Layers**: How convolution operations extract features from images
2. **Pooling Layers**: Dimensionality reduction and translation invariance
3. **Batch Normalization**: Stabilizing training and faster convergence
4. **Dropout**: Regularization to prevent overfitting
5. **Optimization**: Gradient descent and adaptive learning rates
6. **Backpropagation**: How gradients flow through the network

## Author
Educational CNN Project for Deep Learning Course
