# AI_Skin_Analysis
## Overview
This application utilizes machine learning models to analyze facial images for skin type classification, facial skin problems detection, and age prediction. It is built with Streamlit for the user interface and integrates DeepFace for age prediction and YOLO for facial problem detection.
## Features
Upload an image or use the webcam for real-time analysis.
Predict skin type and confidence score.
Detect facial skin problems with visual segmentation.
Recommend skincare products based on detected issues.
Age prediction using DeepFace.
## Requirements
You can install the required libraries using pip:
!pip install streamlit deepface ultralytics opencv-python numpy Pillow tensorflow pandas
## Model Files
keras_model.h5: Pre-trained skin type classification model.
labels.txt: File containing skin type labels.
best.pt: YOLO model for facial skin problem detection.
skincare_products.xlsx: Excel file containing skincare product recommendations.

"Ensure these files are located in the same directory as your application or update the paths accordingly."
## How to Run
! wget -q -O - ipv4.icanhazip.com
!streamlit run /content/AI_Skin_Analysis.py & npx localtunnel --port 8501
## Usage
Choose to upload an image or use your webcam.
If uploading, select an image file (supported formats: jpg, jpeg, png, webp).
The application will display the uploaded image, predicted age, skin type, detected skin problems, and recommended products.
If using the webcam, click "Start Webcam" to begin real-time analysis.
## Notes
The application caches data for improved performance using @st.cache_data.
Ensure your environment has access to a webcam if using the webcam feature.
## Acknowledgments
This application was developed by [Kholoud Khaled.] (https://github.com/kholoudanran/AI_Skin_Analysis.git).
