# Human Activity Recognition with Continual Learning (PyTorch)

This project implements Human Activity Recognition (HAR) using wearable sensor data from the MHEALTH dataset.  
The goal is to compare standard joint training with continual learning strategies and study catastrophic forgetting when new activities are learned sequentially.

The project includes:
- Data loading and preprocessing
- Sliding window segmentation
- 1D CNN models for time-series classification
- Baseline joint training
- Continual learning (naive vs rehearsal)
- Memory buffer replay
- Forgetting measurement
- Hyperparameter experiments
- Visualization of results

The code was developed in Google Colab using PyTorch.


---

## Dataset

Dataset used: **MHEALTH dataset (UCI Machine Learning Repository)**  
Contains body sensor signals from 10 subjects performing physical activities.

Activities used in this project:

- Standing
- Sitting
- Lying
- Walking
- Stairs
- Cycling
- Jogging

The dataset is downloaded automatically from the UCI repository.

Source:
https://archive.ics.uci.edu/ml/datasets/mhealth+dataset


---

## Project Pipeline

### 1. Data loading
- Dataset downloaded from UCI
- Files extracted automatically
- Subject files detected using regex
- Stored in Google Drive

### 2. Cleaning
- Remove NaN / inf values
- Remove label 0
- Keep only selected activities

### 3. Window segmentation
Sliding windows:
- window = 3 seconds
- overlap = 50%
- sample rate = 50 Hz

Each window is assigned the majority label.

### 4. Normalization
Z-score normalization computed on training set only.

### 5. Dataset split

Train subjects:
1–7

Validation:
8

Test:
9–10


---

## Models

Two CNN models are implemented.

### SmallCNN1

- 1 Conv1D layer
- BatchNorm
- Global average pooling
- Fully connected layer

### SmallCNN2

- 2 Conv1D layers
- BatchNorm
- Global pooling
- Fully connected layer

Input shape:

(batch, time, features)


---

## Baseline Training

Joint training on all classes at once.

Metrics:

- Accuracy
- Macro F1
- Confusion matrix
- Classification report


---

## Continual Learning

Classes are introduced in phases.

Phases:

1 → standing, sitting, walking  
2 → lying  
3 → cycling  
4 → jogging  
5 → stairs  


Two strategies:

### Naive

Train only on new classes  
→ causes catastrophic forgetting

### Rehearsal

Memory buffer stores old samples  
→ replayed during training

Memory buffer size:

m samples per class


---

## Memory Buffer

Stores limited examples per class.

Used to reduce forgetting.
MemoryBuffer(m_per_class)




Supports:

- add_examples
- sample_all


---

## Evaluation

Metrics:

- Accuracy on seen classes
- Macro F1
- Forgetting

Forgetting = best previous accuracy − current accuracy


---

## Experiments

The project includes:

- CNN1 vs CNN2
- Naive vs Rehearsal
- Different memory sizes
- Different learning rates
- Phase-by-phase evaluation

Plots:

- Accuracy vs phase
- Forgetting vs phase
- Final accuracy comparison
- Learning rate sweep


---

## Results

Findings:

- Joint training gives highest accuracy
- Naive continual learning causes strong forgetting
- Rehearsal reduces forgetting
- Larger memory improves performance
- CNN2 performs better than CNN1


---

## Requirements

Python 3.10+

Libraries:
torch
numpy
pandas
scikit-learn
matplotlib


---

## How to run

1. Open notebook in Google Colab
2. Mount Google Drive
3. Run all cells
4. Dataset will download automatically
5. Training will start


---

## Author

Shima Abdollahi Barfjan — Machine Learning / Deep Learning / Continual Learning

Implemented in PyTorch for Human Activity Recognition.

