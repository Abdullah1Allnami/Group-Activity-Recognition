# Group Activity Recognition in Volleyball Videos

This repository implements the deep learning pipeline for group activity recognition in volleyball, following the methodology proposed in Dr. Mostafa Saad Ibrahim's paper:

> **Paper Reference:**  
> Mostafa Saad Ibrahim, et al. "Hierarchical Deep Temporal Models for Group Activity Recognition."  
> [Link to Paper](https://arxiv.org/abs/1707.02786)

## Overview

The project aims to classify group activities in volleyball matches using annotated video frames. It leverages hierarchical deep learning models to process both global scene context and individual player actions, closely following the architecture and training strategies described in the referenced paper.

## Features

- **Data Parsing & Preprocessing:**  
  - Loads and parses volleyball dataset annotations.
  - Extracts bounding boxes and player actions.
  - Preprocesses images and player crops.

- **Data Augmentation:**  
  - Applies random transformations (rotation, color jitter, resizing) for robust training.

- **Hierarchical Model Architecture:**  
  - Implements a baseline model inspired by the paper, using ResNet backbone and temporal modeling.
  - Supports transfer learning and fine-tuning.

- **Training & Evaluation:**  
  - Early stopping, validation, and test accuracy reporting.
  - Model checkpointing.

- **Exploratory Data Analysis (EDA):**  
  - Visualizes activity and player action distributions.

## Setup

### Requirements

Install dependencies:

```bash
pip install torch torchvision numpy scikit-learn matplotlib seaborn pillow opencv-python
```

### Dataset

- Download and organize the volleyball dataset as described in the paper.
- Place data in the `data/` directory, following the expected structure.

## Usage

1. **Train the Model:**

   ```bash
   python main.py
   ```

2. **Evaluate/Test:**

   - Results and metrics will be printed after training.

## Project Structure

- `main.py` — Entry point for training and evaluation.
- `dataset.py` — Data loading and preprocessing.
- `models.py` — Model architecture.
- `train.py` — Training loop and early stopping.
- `evaluation.py` — Testing and metrics.
- `EDA.py` — Exploratory data analysis utilities.
- `utils.py` — Helper functions.
- `requirements.txt` — Python dependencies.

## Acknowledgments

- This implementation is based on Dr. Mostafa Saad Ibrahim's research.
- Thanks to the original authors for their dataset and methodology.

---

For questions or contributions, please open an issue or pull request.
