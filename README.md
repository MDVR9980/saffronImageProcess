# 🌸 Saffron Image Processing

A computer vision project to automatically detect **saffron flowers** in agricultural images using image processing techniques.

![Saffron Example](https://raw.githubusercontent.com/MDVR9980/saffronImageProcess/main/sample_output.jpg)

---

## 📌 Overview

This project implements an image analysis pipeline using Python and OpenCV to locate saffron flowers in an image. It utilizes color space conversion (RGB → HSV/HSI), thresholding, and filtering to identify flower regions based on their color characteristics.

> 🔬 A useful tool for smart farming, agricultural monitoring, and AI-based plant detection systems.

---

## 🚀 Features

- 🌈 RGB to HSV and HSI color conversion
- 🎯 Flower color detection using masking
- 🧹 Noise removal using morphological operations
- 📦 Modular code (easy to extend with ML models)
- 🖼 Annotated visual output

---

## 🧠 How It Works

1. Load the saffron field image using OpenCV
2. Convert the image into HSV or HSI color space
3. Apply thresholds to isolate saffron-colored regions
4. Perform morphological operations to clean noise
5. Mark detected regions and display/save the result

---

## 📂 Project Structure

saffronImageProcess/
│
├── saffron_detector.py # Main image processing script
├── utils.py # Optional helpers for color conversion
├── sample_input.jpg # Input image of saffron flowers
├── sample_output.jpg # Output after processing
├── requirements.txt # Python dependencies
└── README.md # Project documentation

yaml
Copy
Edit

---

## ▶️ Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
If requirements.txt not available, install manually:

bash
Copy
Edit
pip install opencv-python numpy matplotlib
2. Run the detector script
bash
Copy
Edit
python saffron_detector.py
Make sure to place your input image as sample_input.jpg or modify the image path in the script.

🖼 Example
Input Image:

Output Image:

🛠 Technologies Used
Python 3.x

OpenCV

NumPy

Matplotlib

SciPy (optional for advanced color space support)

📈 Future Improvements
🔬 Add shape/contour analysis for better accuracy

🤖 Integrate deep learning (CNNs for classification)

🌐 Add web-based interface (Streamlit or FastAPI)

📤 Export results as structured data (CSV/JSON)

🙋 About the Developer
👨‍💻 Mohammad Davood Vahab Rajaee
📫 Email: mdvahhabrajaee@gmail.com | mdvr9980@gmail.com 
