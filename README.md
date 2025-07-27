# Group Activity Recognition in Volleyball Videos

This repository implements a deep learning pipeline for **group activity recognition in volleyball**, based on the methodology proposed in the following paper:

> **Paper Reference**
> Mostafa S. Ibrahim, Srikanth Muralidharan, Zhiwei Deng, Arash Vahdat, and Greg Mori.
> **"Hierarchical Deep Temporal Models for Group Activity Recognition"**
> *CVPR 2016*
> [https://www.cs.sfu.ca/\~mori/research/papers/ibrahim-cvpr16.pdf](https://www.cs.sfu.ca/~mori/research/papers/ibrahim-cvpr16.pdf)

---

## 📌 Overview

The project classifies **group activities** in volleyball matches using annotated video frames. It leverages **hierarchical deep learning models** to process both global scene context and individual player actions, as described in the referenced paper.
<img src="https://github.com/Abdullah1Allnami/Group-Activity-Recognition/blob/main/img/fig1.png" alt="Figure 1" height="400" >
---

## ✨ Features

### 🔄 Data Parsing & Preprocessing

* Loads volleyball dataset annotations.
* Extracts bounding boxes and individual player actions.
* Preprocesses image frames and player crops.

### 🧪 Data Augmentation

* Random transformations: rotation, color jitter, resizing, etc.

### 🧠 Hierarchical Model Architecture

* Implements a model inspired by the paper with:

  * **ResNet** backbone.
  * **Temporal modeling** across player and scene representations.
* Supports transfer learning and fine-tuning.

### 🏋️ Training & Evaluation

* Training with early stopping and validation monitoring.
* Test-time evaluation and metrics reporting.
* Model checkpoint saving.

### 📊 Exploratory Data Analysis (EDA)

* Visualization of group activity and individual action distributions.

---

## ⚙️ Setup

### Requirements

Install dependencies:

```bash
pip install torch torchvision numpy scikit-learn matplotlib seaborn pillow opencv-python
```

### Dataset

* Download and structure the **volleyball dataset** as described in the original paper.
* Place it in the `data/` directory with the expected structure:

  ```
  data/
  ├── frames/
  └── annotations/
  ```

---

## 🚀 Usage

### Train the Model

```bash
python main.py
```

### Evaluate / Test

* After training, results and evaluation metrics will be printed to the console.

---

## 📁 Project Structure

* `main.py` — Entry point for training and evaluation.
* `dataset.py` — Data loading, annotation parsing, preprocessing.
* `models.py` — Model architecture.
* `train.py` — Training loop and logic.
* `evaluation.py` — Testing and accuracy metrics.
* `EDA.py` — Data visualization and analysis tools.
* `utils.py` — Helper functions.
* `requirements.txt` — Python dependencies.

---

## 🙏 Acknowledgments

* This implementation is inspired by **Dr. Mostafa Saad Ibrahim** and collaborators’ work on group activity recognition.
* Thanks to the original authors for their **dataset** and **methodology**.

---

For questions, suggestions, or contributions, please **open an issue or a pull request**.
