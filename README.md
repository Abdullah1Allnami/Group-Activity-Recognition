# Volleyball Deep Learning Project

This project focuses on building a deep learning pipeline for image analysis in volleyball gameplay scenarios. The system processes annotated images, applies data augmentation, and trains a baseline model to classify and predict activities.

## Table of Contents
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Acknowledgments](#acknowledgments)

---

## Features

- **Data Parsing and Visualization**:
  - Parses annotation files to extract bounding boxes and activity labels.
  - Visualizes images with annotated bounding boxes.

- **Data Augmentation**:
  - Includes random transformations such as rotations, color jittering, and resizing.

- **Model Training**:
  - Implements a baseline ResNet-50 model for activity classification.
  - Supports transfer learning by freezing ResNet-50 layers except for the final layer.

- **Exploratory Data Analysis (EDA)**:
  - Visualizes label distributions and image properties (e.g., dimensions, formats).

---

## Setup

### Required Libraries
Install the required Python libraries:

```bash
pip install torch torchvision numpy scikit-learn matplotlib seaborn pillow opencv-python
