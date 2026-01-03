
# 🔐 MATLAB Robust Facial Recognition & Authentication System

A secure biometric verification system built in MATLAB that combines geometric feature matching with advanced texture analysis and multi-method segmentation. This project was designed to achieve high-accuracy identity verification while demonstrating advanced computer vision techniques.

## 🌟 Key Features

### 1. Biometric Verification
* **Euclidean Distance Matching:** Identifies users by comparing feature vectors (Mean, Standard Deviation, Histograms) against a known database.
* **Tuned Sensitivity:** Uses a statistically derived cut-off threshold (**0.675**) to eliminate false positives.

### 2. Advanced Security Layers
* **Texture Analysis (LBP):** Implements **Local Binary Patterns** to analyze skin micro-textures, ensuring the system verifies actual skin topography and not just face shape.
* **Interactive Authentication:** Features a secure "Sign-off" stage where verified users must digitally sign the image using a mouse-driven interface.

### 3. Smart Segmentation Analysis
To ensure robust Region of Interest (ROI) extraction, the system visually compares four different segmentation algorithms side-by-side:
* **HSV Skin Detection:** Isolates skin pixels based on Hue/Saturation.
* **K-Means Clustering:** Unsupervised segmentation of image regions.
* **Viola-Jones:** Automatic face detection and cropping.
* **Adaptive Thresholding:** SVM-style binary masking.

---

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `main.m` | **Entry point.** Runs the full pipeline: Recognition -> Signature -> Analysis. |
| `FaceDB/` | Database folder containing reference images (Positive & Negative samples). |
| `query/` | Input folder containing the image to be verified. |
| `compare_segmentations.m`| Helper function that generates the 5-way segmentation grid. |
| `analyze_lbp.m` | Helper function for LBP texture comparison. |
| `add_signature.m` | Logic for the interactive signing interface. |
| `extract_features.m` | Core algorithm for statistical feature extraction. |
| `preprocess_image.m` | Standardizes images to 256x256 grayscale. |

---

## 🚀 How to Run

### Prerequisites
* MATLAB (R2021a or later recommended)
* **Image Processing Toolbox**
* **Computer Vision Toolbox**

### Setup
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/MATLAB-Facial-Recognition.git](https://github.com/YourUsername/MATLAB-Facial-Recognition.git)
    ```
2.  **Prepare the Database:**
    * Place known reference photos in the `FaceDB/` folder.
    * *Note: For privacy, the uploaded repo contains dummy/empty placeholders.*
3.  **Set the Query:**
    * Place the image you want to test inside the `query/` folder.

### Execution
1.  Open `main.m` in MATLAB.
2.  Run the script.
3.  **If a match is found:** A window will pop up asking for a signature. Draw with your mouse and **double-click** the line to confirm.
4.  The system will then generate performance graphs and segmentation analysis windows.

---

## 📊 Performance Metrics

The system was evaluated against a mixed dataset of positive and negative samples, achieving:

* **Accuracy:** 100% (on test set)
* **False Acceptance Rate (FAR):** 0%
* **False Rejection Rate (FRR):** 0%

### Visual Output Examples
* **Confusion Matrix:** visualizes the perfect classification accuracy.
* **Distance Bar Chart:** Displays the safety margin between the Target User and Imposters relative to the 0.675 threshold.

---

## 📝 License
This project is open-source and available for educational and research purposes.
