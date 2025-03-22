import streamlit as st
import requests
from PIL import Image
import io
import os
from model import FurnitureDetector

st.set_page_config(
    page_title="LEVisions - Furniture Detector",
    page_icon="🪑",
    layout="wide"
)

# Initialize the detector
@st.cache_resource
def get_detector():
    return FurnitureDetector()

# Get the backend URL from environment variable or use a default production URL
BACKEND_URL = os.getenv("BACKEND_URL", "https://furniture-detector-backend.vercel.app")

st.title("LEVisions - Furniture Detection & Recommendation")

# Sidebar
st.sidebar.title("Options")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Demo mode toggle
demo_mode = st.sidebar.checkbox("Demo Mode", value=True)

# Backend status
st.sidebar.markdown("### Backend Status")
try:
    response = requests.get(f"{BACKEND_URL}/health")
    if response.status_code == 200:
        st.sidebar.success("Backend Connected")
    else:
        st.sidebar.error("Backend Error")
except:
    st.sidebar.error("Backend Offline")

# Main content
st.write("""
## Upload your interior image
Upload a 3D-rendered interior image and we'll detect the furniture items for you.
""")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Create two columns for the analyze button and progress
    col1, col2 = st.columns([1, 3])
    
    with col1:
        analyze_button = st.button("Analyze Image")
    
    if analyze_button:
        with st.spinner("Detecting furniture..."):
            try:
                if demo_mode:
                    # Demo mode - show sample results
                    st.success("Demo Mode: Showing sample results")
                    demo_results = {
                        "detections": [
                            {
                                "class": "Chair",
                                "confidence": 0.95,
                                "bbox": [100, 200, 300, 400]
                            },
                            {
                                "class": "Table",
                                "confidence": 0.88,
                                "bbox": [150, 250, 350, 450]
                            }
                        ]
                    }
                    
                    # Display demo results
                    st.subheader("Detected Furniture (Demo Mode)")
                    
                    for item in demo_results["detections"]:
                        with st.expander(f"{item['class']} - Confidence: {item['confidence']:.2f}"):
                            st.write(f"Location: {item['bbox']}")
                            
                            if st.button(f"Find similar {item['class']} items", key=f"find_{item['class']}"):
                                st.write("Demo: Searching for similar items...")
                                st.info("This is a demo feature. In production, this would show real furniture recommendations.")
                else:
                    # Get the detector
                    detector = get_detector()
                    
                    # Run detection
                    results = detector.detect_furniture(image, confidence_threshold)
                    
                    # Display results
                    st.subheader("Detected Furniture")
                    
                    # Create columns for detected items
                    for item in results["detections"]:
                        with st.expander(f"{item['class']} - Confidence: {item['confidence']:.2f}"):
                            st.write(f"Location: {item['bbox']}")
                            
                            if st.button(f"Find similar {item['class']} items", key=f"find_{item['class']}"):
                                st.write("Searching for similar items...")
                                # Add recommendation functionality here
                    
                    # Display processed image
                    if "image" in results:
                        st.image(
                            results["image"],
                            caption="Processed Image with Detections",
                            use_container_width=True
                        )
                        
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by LEVisions") 