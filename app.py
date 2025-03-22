import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(
    page_title="LEVisions - Furniture Detector",
    page_icon="🪑",
    layout="wide"
)

st.title("LEVisions - Furniture Detection & Recommendation")

# Sidebar
st.sidebar.title("Options")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

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
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Create two columns for the analyze button and progress
    col1, col2 = st.columns([1, 3])
    
    with col1:
        analyze_button = st.button("Analyze Image")
    
    if analyze_button:
        with st.spinner("Detecting furniture..."):
            try:
                # Convert image to bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format)
                img_byte_arr = img_byte_arr.getvalue()

                # Send image to FastAPI backend
                files = {"file": ("image.jpg", img_byte_arr, "image/jpeg")}
                response = requests.post(
                    "http://localhost:8000/detect",
                    files=files,
                    params={"confidence": confidence_threshold}
                )
                
                if response.status_code == 200:
                    results = response.json()
                    
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
                            use_column_width=True
                        )
                else:
                    st.error("Error processing the image. Please try again.")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by LEVisions") 