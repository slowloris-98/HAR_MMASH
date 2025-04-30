# MMASH Activity Recognition Project

## 📌 Project Overview

This project focuses on building machine learning and deep learning models for human activity recognition using the **MMASH (MultiModal Activities for Stress and Health)** dataset. The goal is to classify user activities based on sensor data collected from wearable devices and to evaluate the generalization of models using cross-validation and transfer learning.

---

## 🗂️ Dataset

- **MMASH Dataset**: Includes actigraphy (accelerometer) data with timestamps, steps, heart rate, and posture labels.
- **Used Columns**: `Axis1`, `Axis2`, `Axis3`, `HR`, `Steps`, `Inclinometer`.

---

## 🧹 Data Preprocessing

- **Missing Values**: Handled using linear interpolation.
- **Noise Reduction**: Applied Butterworth low-pass filtering on accelerometer signals.
- **Normalization**: Z-score normalization of continuous features.
- **Segmentation**: Data segmented into 1-second overlapping windows at 50 Hz for model input.

---

## 🧠 Feature Engineering

- **Time-domain features**: Mean, std, min, max, peak-to-peak, skewness, kurtosis, signal magnitude area (SMA).
- **Frequency-domain features**: Dominant frequency and signal energy via FFT.

---

## 🔍 Model Development

- **Baseline Models**: Decision Tree, K-NN, Naive Bayes.
- **Advanced Models**: Random Forest, SVM, CNN, and LSTM (using Keras).
- **Input Format**: Raw segments of shape `(50, 3)` for CNN/LSTM models.
- **Evaluation**:
  - **k-Fold Cross-Validation** (k=5)
  - **Leave-One-Subject-Out (LOSO)** for generalization across users.
  - **Metrics**: Accuracy, Precision, Recall, F1-Score

---

## 🔄 Transfer Learning

- **Approach**: Adopted architecture of a public HAR-CNN model trained on UCI HAR.
- **Strategy**:
  - Recreated HAR-CNN architecture: `Conv1D → MaxPooling → Conv1D → GlobalMaxPooling → Dense`.
  - Trained this architecture from scratch on MMASH using LOSO evaluation.
  - Input segments were zero-padded to shape `(200, 3)` for compatibility.

---

## ✅ Key Results

- **Random Forest** and **CNNs** achieved best accuracy (~56%).
- **LSTM** showed strong user-level generalization under LOSO.
- **Architecture-based transfer learning** using HAR CNN led to effective adaptation to MMASH.

---

## 💡 Learning Outcomes

This project gave me hands-on experience in working with real-world **time-series sensor data**, from preprocessing to deploying deep learning models. I learned how to handle missing values, apply signal filtering, extract features, and evaluate models rigorously using LOSO. I particularly enjoyed building and comparing deep models like CNNs and LSTMs, and implementing transfer learning using pretrained architectures.

---

## 🔭 Future Work

- **Multimodal Fusion**: Combine HR, stress, and questionnaire data with actigraphy.
- **Advanced Architectures**: Explore Bi-LSTMs, Temporal CNNs, and Transformers.

---

## 📁 Files

- `CS256-MMASH_Demo1_Udayan.ipynb`: Data exploration and preprocessing
- `CS256-MMASH_Demo2_Udayan.ipynb`: Model training, evaluation, and transfer learning
