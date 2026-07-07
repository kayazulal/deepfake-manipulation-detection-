# Deepfake Manipulation Detection and Analysis

An end-to-end deep learning framework based on **PyTorch** designed to detect and analyze AI-generated multimedia face manipulations on the FaceForensics++ dataset. 

This project explores the progression of mitigating overfitting in high-resolution video classification through iterative model optimizations, including MTCNN face cropping, data augmentation, regularization techniques, and learning rate scheduling.

---

## 📊 Ablation Study & Project Evolution

The development process consists of 6 distinct experimental phases, highlighting how data leakage was prevented and how overfitting was systematically resolved:

| Experiment | Approach | Split Type | Frames / Video | Test Accuracy | AUC | Identified Issues & Key Takeaways |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Deneme 1** | Frame-based | Random (Invalid) | 10 frames | %99.00 | 0.9999 | **Data Leakage!** Frames from the same video co-existed in train and test sets. Unrealistic metric. |
| **Deneme 2** | Video-based | Leak-free (70/15/15) | 10 frames | %68.00 | 0.7377 | Severe performance drop, signaling early onset of overfitting. |
| **Deneme 3** | Augmentation | Video-based (70/15/15) | 20 frames | %83.00 | 0.8858 | Augmentation helped, but significant Train/Val loss gap remained. |
| **Deneme 4** | Full Frames | Video-based (70/15/15) | ~450 frames (366K total) | %88.00 | 0.9321 | High visual redundancy among consecutive frames failed to fully solve overfitting. |
| **Deneme 5** | MTCNN Crop | Video-based (70/15/15) | 50 equally-spaced frames | %97.00 | 0.9969 | Face cropping boosted accuracy immensely, but Val loss remained unstable. |
| **Deneme 6** | **Dropout + LR Scheduler** | **Video-based (55/30/15)** | **50 equally-spaced frames** | **%97.96** | **0.9980** | **Overfitting Resolved!** Regularization stabilized curves; Train/Val losses converged seamlessly. |

---

## 📈 Final Experimental Results (Deneme 6)

Below are the training curves and the test confusion matrix achieved in the final pipeline using **EfficientNet-B0 + 40% Dropout** and a **ReduceLROnPlateau** scheduler:

![Experimental Results](Ekran%20Görüntüsü%20(1866).png)

### Key Interpretations:
* **Overfitting Mitigation:** The Training and Validation Loss curves converge robustly toward the final epochs, proving the success of the 40% Dropout layer in enforcing generalized feature learning rather than memorization.
* **Balanced Performance:** The confusion matrix displays high equilibrium, successfully classifying **2499 Real** and **2497 Fake** frames with minimal false positives or false negatives.

---

## 🛠️ Tech Stack & Dependencies
* **Core Framework:** Python, PyTorch, Torchvision
* **Model Backbone:** `timm` (EfficientNet-B0)
* **Face Detection:** `facenet-pytorch` (MTCNN)
* **Metrics & Analytics:** Scikit-Learn, Matplotlib, OpenCV
