import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from datetime import datetime
import json
from github import Github

# Labels from Teachable Machine
LABELS = ["happy", "sad", "frustrated"]

# Page configuration
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="😊",
    layout="centered"
)

# Load the model
@st.cache_resource
def load_model():
    custom_objects = {
        "DepthwiseConv2D": lambda **kwargs: tf.keras.layers.DepthwiseConv2D(
            **{k: v for k, v in kwargs.items() if k != "groups"}
        )
    }
    try:
        model = tf.keras.models.load_model(
            "keras_model.h5",
            compile=False,
            custom_objects=custom_objects,
            safe_mode=False,
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# GitHub integration for data storage
class GitHubDataStore:
    def __init__(self, token, repo_name, file_path='predictions.json'):
        self.token = token
        self.repo_name = repo_name
        self.file_path = file_path
        self.github = Github(token) if token else None
        
    def save_prediction(self, prediction_data):
        if not self.github:
            st.warning("GitHub token not configured. Data will not be saved.")
            return False
            
        try:
            repo = self.github.get_repo(self.repo_name)
            
            try:
                file = repo.get_contents(self.file_path)
                existing_data = json.loads(file.decoded_content.decode())
                existing_data.append(prediction_data)
                repo.update_file(
                    self.file_path,
                    f"Add prediction at {prediction_data['timestamp']}",
                    json.dumps(existing_data, indent=2),
                    file.sha
                )
            except Exception:
                repo.create_file(
                    self.file_path,
                    "Initialize predictions file",
                    json.dumps([prediction_data], indent=2)
                )
            
            return True
        except Exception as e:
            st.error(f"Error saving to GitHub: {str(e)}")
            return False

# Preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image)
    normalized_image = (image_array.astype(np.float32) / 127.5) - 1
    return normalized_image.reshape((1, 224, 224, 3))

# Main app
def main():
    st.title("😊 Emotion Detection App")
    st.markdown("Detect emotions (**Happy**, **Sad**, **Frustrated**) from an image or webcam photo.")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("### GitHub Data Storage")
        github_token = st.text_input(
            "GitHub Personal Access Token",
            type="password"
        )
        repo_name = st.text_input(
            "Repository (username/repo-name)"
        )
        save_to_github = st.checkbox("Save predictions to GitHub", value=False)
        st.markdown("---")
        st.info("This app uses a Teachable Machine image model.")

    github_store = None
    if save_to_github and github_token and repo_name:
        github_store = GitHubDataStore(github_token, repo_name)

    model = load_model()
    if model is None:
        st.info("Ensure 'keras_model.h5' is in the same directory.")
        return

    # Image source selection
    st.markdown("### 📸 Image Source")
    image_source = st.radio(
        "Choose how you want to provide an image:",
        ["Upload Image", "Use Webcam"]
    )

    uploaded_file = None

    if image_source == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"]
        )
    else:
        uploaded_file = st.camera_input("Take a photo")

    # Prediction
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Input Image", use_container_width=True)

        with col2:
            with st.spinner("Analyzing emotion..."):
                processed_image = preprocess_image(image)
                predictions = model.predict(processed_image, verbose=0)
                predicted_class = int(np.argmax(predictions[0]))
                confidence = float(predictions[0][predicted_class])

                st.markdown("### 🎯 Prediction")
                st.success(f"**Emotion:** {LABELS[predicted_class].upper()}")
                st.metric("Confidence", f"{confidence * 100:.2f}%")

                st.markdown("### 📊 Probabilities")
                for i, label in enumerate(LABELS):
                    prob = float(predictions[0][i])
                    st.progress(prob, text=f"{label.capitalize()}: {prob * 100:.1f}%")

        # Save to GitHub
        if save_to_github and github_store:
            if st.button("💾 Save Prediction to GitHub"):
                prediction_data = {
                    "timestamp": datetime.now().isoformat(),
                    "source": image_source.lower().replace(" ", "_"),
                    "predicted_emotion": LABELS[predicted_class],
                    "confidence": confidence,
                    "all_probabilities": {
                        LABELS[i]: float(predictions[0][i])
                        for i in range(len(LABELS))
                    }
                }

                with st.spinner("Saving to GitHub..."):
                    if github_store.save_prediction(prediction_data):
                        st.success("✅ Saved successfully!")
                    else:
                        st.error("❌ Failed to save prediction.")

    # Help section
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        **Steps**
        1. Choose **Upload Image** or **Use Webcam**
        2. Provide an image
        3. View detected emotion and confidence
        4. (Optional) Save prediction to GitHub

        Predictions are stored in `predictions.json` in your repository.
        """)

if __name__ == "__main__":
    main()
