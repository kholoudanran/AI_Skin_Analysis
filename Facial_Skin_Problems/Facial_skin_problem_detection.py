import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Title of the application
st.title("Facial Skin Problem Detection - NTI")

# Load the trained YOLOv8 model (best.pt)
model = YOLO("best.pt")  # Replace with the path to your best.pt file

# Define the segmentation classes and their corresponding colors (in RGB format)
classes = ['darkcircle', 'melasma', 'redness', 'vascular', 'wrinkle']
colors = {
    "darkcircle": (213, 108, 108),  
    "melasma": (90, 174, 115),    
    "redness": (243, 80, 85),
    "vascular": (49, 174, 169),    
    "wrinkle": (170, 105, 193)     
}

# Sidebar for user input
st.sidebar.title("Options")
option = st.sidebar.radio("Select Input Source", ["Upload Image", "Use Webcam"])

def process_image(image_np):
    """
    Process the input image, perform segmentation, and return the image with segmentation masks and unique detected classes.
    """
    # Perform inference using the model
    results = model.predict(image_np, conf=0.25, imgsz=640, task="segment")

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

def display_detected_classes(detected_classes):
    """Display detected classes with corresponding colors."""
    for class_name in detected_classes:
        color = colors[class_name]  # Use RGB color for HTML
        st.write(f"<span style='color:rgb{color};'>{class_name.capitalize()}</span>", unsafe_allow_html=True)

if option == "Upload Image":
    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Upload an image of your face", type=["jpg", "jpeg", "png", 'webp'])

    if uploaded_file is not None:
        # Read the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)  # Display uploaded image

        # Convert the image to OpenCV format
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Process the image and get the output image with segmentation masks and detected classes
        output_image, detected_classes = process_image(image_np)

        # Display the results
        st.write("### Detected Facial Skin Problems:")
        display_detected_classes(detected_classes)

        # Display the resulting image with segmentation masks
        st.image(output_image, caption="The resulting image", width=400)

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

            # Process the frame and get the output image with segmentation masks and detected classes
            output_image, detected_classes = process_image(frame)

            # Display the results
            st.write("### Detected Facial Skin Problems:")
            display_detected_classes(detected_classes)

            # Display the resulting image with segmentation masks
            st.image(output_image, caption="The resulting image", width=400)

            # Stop the webcam if the user clicks the stop button
            if st.button("Stop Webcam"):
                run_webcam = False
                cap.release()
                st.write("Webcam stopped.")

        # Release the webcam
        cap.release()
st.markdown("---")
st.markdown("This application detect facial skin problems.")
st.markdown("🌐 Developed by [Kholoud Khaled.]")