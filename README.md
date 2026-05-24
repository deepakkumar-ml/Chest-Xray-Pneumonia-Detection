# 🫁 Chest X-Ray Pneumonia Detection Using Deep Learning

This repository contains a comprehensive Deep Learning pipeline to detect Pneumonia from Chest X-Ray images. The project evaluates and compares standalone Convolutional Neural Networks (CNNs) against state-of-the-art Transfer Learning architectures.

## 📊 Dataset Overview
* **Source:** Chest X-Ray Images (Pneumonia) available via Kaggle.
* **Classes:** `NORMAL` and `PNEUMONIA`.
* **Training Data Distribution:** 1,341 Normal images vs. 3,875 Pneumonia images (Imbalanced Dataset handled via class weights).
* **Test Dataset:** 624 images.

### Programmatic Download:
```python
import kagglehub
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
```

---

## ⚙️ Core Configurations
* **Image Target Size:** 224 x 224 pixels (RGB)
* **Batch Size:** 32
* **Base Learning Rate:** 1e-4 with Adam Optimizer
* **Training Optimization:** Mixed Precision (`mixed_float16`) enabled for faster GPU computation.

---

## 🧠 Model Architectures Evaluated
We systematically designed and tested four distinct configurations to observe optimization and overfitting behaviors:

1. **Base CNN:** A foundational 3-layer sequential convolution pipeline optimized with Batch Normalization.
2. **Advanced CNN:** Built with Data Augmentation (Flip, Rotation, Zoom, Contrast) and Separable Convolutions to reduce overfitting.
3. **EfficientNetB0 (Transfer Learning):** Feature extractor initialized with ImageNet weights (Pretrained base frozen).
4. **ResNet50V2 (Transfer Learning):** Fully unfrozen deep residual architecture trained with calculated class weights to counter dataset imbalance.

---

## 📈 Final Performance Comparison
Below is the empirical performance comparison extracted from the training logs:


| Model Architecture | Best Val Accuracy | Test Overall Accuracy | Normal Recall | Pneumonia Recall | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Base CNN** | 62.50% | 82.00% | 66.00% | 91.00% | Overfitting |
| **Advanced CNN** | 62.50% | 56.00% | 99.00% | 31.00% | Underperforming |
| **EfficientNetB0** | 93.75% | 82.00% | 59.00% | 96.00% | Highly Sensitive |
| **ResNet50V2** | **93.75%** | **93.00%** | **88.00%** | **96.00%** | **Best Balanced Model** |

### 🎯 Key Medical Insight
In medical diagnostics, **Recall (Sensitivity)** is a critical metric to prevent False Negatives (missing a sick patient). **ResNet50V2** achieved an exceptional **96.00% Recall** for Pneumonia detection while maintaining an overall test accuracy of **93.00%**.

---

## 🚀 How to Run the Project
1. Open the `.ipynb` notebook from the `notebooks/` folder inside Google Colab.
2. Make sure your runtime environment is set to **GPU**.
3. Execute all cells sequentially to trigger automated data pipeline optimization via TensorFlow `cache()` and `prefetch()`.
