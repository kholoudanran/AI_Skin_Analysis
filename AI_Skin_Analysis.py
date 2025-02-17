import streamlit as st
from deepface import DeepFace
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import pandas as pd  # For reading the Excel file

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the skin type classification model
skin_model = load_model("/content/AI_Skin_Analysis/keras_model.h5", compile=False)
skin_class_names = open("/content/AI_Skin_Analysis/labels.txt", "r").readlines()

# Load the YOLO model for facial skin problem detection
yolo_model = YOLO("/content/AI_Skin_Analysis/best.pt")  # Replace with the path to your best.pt file

# Define the segmentation classes and their corresponding colors (in RGB format)
classes = ['darkcircle', 'melasma', 'redness', 'vascular', 'wrinkle']
colors = {
    "darkcircle": (213, 108, 108),  
    "melasma": (90, 174, 115),    
    "redness": (243, 80, 85), 
    "vascular": (49, 174, 169),    
    "wrinkle": (170, 105, 193)      
}

# Load skincare product recommendations from Excel
@st.cache_data  # Cache the data for better performance
def load_skincare_data():
    return pd.read_excel("/content/AI_Skin_Analysis/skincare_products.xlsx")  # Replace with the path to your Excel file

skincare_data = load_skincare_data()

# Title of the application
st.title("🎭 Facial Analysis Application - NTI")

# Sidebar for user input
st.sidebar.title("Options")
option = st.sidebar.radio("Select Input Source", ["Upload Image", "Use Webcam"])

# Function to preprocess the image for skin type classification
def preprocess_image(image):
    # Resize the image to (224, 224)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    # Convert the image to a numpy array and reshape it to the model's input shape
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    # Normalize the image array
    image = (image / 127.5) - 1
    return image

# Function to predict the skin type
def predict_skin_type(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    # Predict the skin type
    prediction = skin_model.predict(processed_image)
    index = np.argmax(prediction)
    class_name = skin_class_names[index]
    confidence_score = prediction[0][index]
    return class_name[2:], confidence_score

# Function to process the image for facial skin problem detection
def process_image(image_np):
    """
    Process the input image, perform segmentation, and return the image with segmentation masks and unique detected classes.
    """
    # Perform inference using the YOLO model
    results = yolo_model.predict(image_np, conf=0.25, imgsz=640, task="segment")

    # Set to store unique detected classes
    detected_classes = set()

    # Create a copy of the image to draw on
    output_image = image_np.copy()

    # Process results
    for result in results:
        if result.masks is not None:
            for mask, class_id in zip(result.masks.xy, result.boxes.cls):
                class_name = classes[int(class_id)]
                detected_classes.add(class_name)

                # Get the color for the current class (convert RGB to BGR for OpenCV)
                color = colors[class_name][::-1]  # Convert RGB to BGR

                # Plot masks with unique colors
                mask = mask.astype(np.int32)
                cv2.fillPoly(output_image, [mask], color)

    # Convert the output image to RGB for display in Streamlit
    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    return output_image_rgb, detected_classes

# Function to display detected classes with corresponding colors
def display_detected_classes(detected_classes):
    """Display detected classes with corresponding colors."""
    for class_name in detected_classes:
        color = colors[class_name]  # Use RGB color for HTML
        st.write(f"<span style='color:rgb{color};'>{class_name.capitalize()}</span>", unsafe_allow_html=True)

# Function to recommend skincare products
def recommend_products(detected_classes):
    """Recommend skincare products based on detected skin problems."""
    st.write("### Recommended Skincare Products:")
    for problem in detected_classes:
        # Filter the DataFrame for the given skin problem
        products = skincare_data[skincare_data["Skin Problem"] == problem]
        if not products.empty:
            st.write(f"**For {problem.capitalize()}:**")
            for _, row in products.iterrows():
                st.write(f"- **{row['Product Name']}** by {row['Brand']}")
                st.write(f"  {row['Description']}")
        else:
            st.write(f"No recommendations found for {problem}.")

# Main application logic
if option == "Upload Image":
    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        # Read the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)  # Display uploaded image

        # Convert the image to OpenCV format
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Perform age prediction using DeepFace
        try:
            analysis = DeepFace.analyze(image_np, actions=['age'])
            age = analysis[0]["age"]
            st.write(f"**Predicted Age:** {age}")
        except Exception as e:
            st.error(f"An error occurred during age prediction: {e}")

        # Perform skin type classification
        skin_type, confidence = predict_skin_type(image_np)
        st.write(f"**Predicted Skin Type:** {skin_type}")

        # Perform facial skin problem detection
        output_image, detected_classes = process_image(image_np)

        # Display the results
        st.write("### Detected Facial Skin Problems:")
        display_detected_classes(detected_classes)

        # Display the resulting image with segmentation masks
        st.image(output_image, caption="The resulting image", width=400)

        # Recommend skincare products
        recommend_products(detected_classes)

elif option == "Use Webcam":
    # Webcam input
    st.write("Using Webcam...")
    run_webcam = st.checkbox("Start Webcam")

    if run_webcam:
        # Open the webcam
        cap = cv2.VideoCapture(0)

        # Placeholder for the webcam feed
        frame_placeholder = st.empty()

        while run_webcam:
            # Read a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from webcam.")
                break

            # Display the webcam feed
            frame_placeholder.image(frame, channels="BGR", caption="Webcam Feed", width=300)

            # Perform age prediction using DeepFace
            try:
                analysis = DeepFace.analyze(frame, actions=['age'])
                age = analysis[0]["age"]
                st.write(f"**Predicted Age:** {age}")
            except Exception as e:
                st.error(f"An error occurred during age prediction: {e}")

            # Perform skin type classification
            skin_type, confidence = predict_skin_type(frame)
            st.write(f"**Predicted Skin Type:** {skin_type}")

            # Perform facial skin problem detection
            output_image, detected_classes = process_image(frame)

            # Display the results
            st.write("### Detected Facial Skin Problems:")
            display_detected_classes(detected_classes)

            # Display the resulting image with segmentation masks
            st.image(output_image, caption="The resulting image", width=400)

            # Recommend skincare products
            recommend_products(detected_classes)

            # Stop the webcam if the user clicks the stop button
            if st.button("Stop Webcam"):
                run_webcam = False
                cap.release()
                st.write("Webcam stopped.")

        # Release the webcam
        cap.release()

st.markdown("---")
st.markdown("This application combines skin type classification, face age, and facial skin problem detection.")
st.markdown("🌐 Developed by [Kholoud Khaled.] (https://github.com/kholoudanran/AI_Skin_Analysis.git)")
