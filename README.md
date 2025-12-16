# 🌸 Saffron Image Processing

A computer vision project to automatically detect **saffron flowers** in agricultural images using image processing techniques.

![Saffron Example](https://github.com/MDVR9980/saffronImageProcess/blob/main/Out/Grouped_Saffron_Flowers.jpg)

---

## 📌 Overview

This project implements an image analysis pipeline using **Python** and **OpenCV** to locate saffron flowers in an image. It utilizes color space conversion (RGB → HSV/HSI), thresholding, and morphological filtering to identify flower regions based on their distinct purple/violet color characteristics.

> 🔬 **Application:** A useful tool for smart farming, agricultural monitoring, and AI-based plant detection systems.

---

## 🚀 Features

*   🌈 **Color Space Conversion:** Efficient RGB to HSV/HSI transformation.
*   🎯 **Precise Detection:** Identifies flowers using color masking thresholds.
*   🧹 **Noise Reduction:** implementation of morphological operations (Erosion/Dilation).
*   📦 **Modular Design:** Easy to extend with Machine Learning models.
*   🖼 **Visual Output:** Generates annotated images with bounding boxes/contours.

---

## 🧠 How It Works

1.  **Input:** Load the saffron field image using OpenCV.
2.  **Preprocessing:** Convert the image into HSV color space for better color separation.
3.  **Masking:** Apply lower and upper thresholds to isolate the saffron purple color.
4.  **Cleaning:** Perform morphological operations to remove noise and fill gaps.
5.  **Output:** Mark detected regions (contours/centroids) and save the result.

---

## 📂 Project Structure

```text
saffronImageProcess/
│
├── saffron_detector.py   # Main image processing script
├── utils.py              # Helper functions (optional)
├── sample_input.jpg      # Input image of saffron flowers
├── sample_output.jpg     # Output after processing
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## ▶️ Usage

### 1. Install dependencies

If you have a `requirements.txt` file:
```bash
pip install -r requirements.txt
```

Or install the libraries manually:
```bash
pip install opencv-python numpy matplotlib
```

### 2. Run the detector script
Make sure to place your input image in the project directory (e.g., `sample_input.jpg`) or update the path in the code.

```bash
python saffron_detector.py
```

## 🖼 Example Results

| Input Image | Output Image |
| :---: | :---: |
| ![Input](sample_input.jpg) | ![Output](sample_output.jpg) |

## 🛠 Technologies Used
*   **Python 3.x**
*   **OpenCV** (Image Processing)
*   **NumPy** (Array Manipulations)
*   **Matplotlib** (Visualization)

## 📈 Future Improvements
*   🔬 Add shape/contour analysis for better accuracy.
*   🤖 Integrate Deep Learning (CNNs) for flower classification.
*   🌐 Add a web-based interface (Streamlit or FastAPI).
*   📤 Export counting results as structured data (CSV/JSON).

## 🙋 About the Developer
**Mohammad Davood Vahhab Rajaee**

📫 **Email:** [mdvahhabrajaee@gmail.com](mailto:mdvahhabrajaee@gmail.com) | [mdvr9980@gmail.com](mailto:mdvr9980@gmail.com)
