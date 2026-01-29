import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from datetime import datetime
import json
from github import Github

# Labels from Teachable Machine (happy, sad, frustrated)
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

# Preprocess image for the model
def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image)
    normalized_image = (image_array.astype(np.float32) / 127.5) - 1
    return normalized_image.reshape((1, 224, 224, 3))

# Main app
def main():
    st.title("😊 Emotion Detection App")
    st.markdown("Upload an image to detect the emotion: **Happy**, **Sad**, or **Frustrated**")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("### GitHub Data Storage")
        github_token = st.text_input(
            "GitHub Personal Access Token",
            type="password",
            help="Generate a token at: https://github.com/settings/tokens"
        )
        repo_name = st.text_input(
            "Repository (username/repo-name)",
            help="e.g., yourusername/emotion-data"
        )
        save_to_github = st.checkbox("Save predictions to GitHub", value=False)
        st.markdown("---")
        st.markdown("### About")
        st.info("This app uses a Teachable Machine model to classify emotions from images.")
    
    github_store = None
    if save_to_github and github_token and repo_name:
        github_store = GitHubDataStore(github_token, repo_name)
    
    model = load_model()
    if model is None:
        st.info("Make sure 'keras_model.h5' is in the same directory as this app.")
        return
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image to detect the emotion"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption='Uploaded Image', use_container_width=True)
        
        with col2:
            with st.spinner('Analyzing emotion...'):
                processed_image = preprocess_image(image)
                predictions = model.predict(processed_image, verbose=0)
                predicted_class = int(np.argmax(predictions[0]))
                confidence = float(predictions[0][predicted_class])
                
                st.markdown("### 🎯 Prediction Results")
                st.success(f"**Detected Emotion:** {LABELS[predicted_class].upper()}")
                st.metric("Confidence", f"{confidence * 100:.2f}%")
                
                st.markdown("### 📊 All Probabilities")
                for i, label in enumerate(LABELS):
                    prob = float(predictions[0][i])
                    st.progress(prob, text=f"{label.capitalize()}: {prob * 100:.1f}%")
        
        if save_to_github and github_store:
            if st.button("💾 Save Prediction to GitHub"):
                prediction_data = {
                    'timestamp': datetime.now().isoformat(),
                    'predicted_emotion': LABELS[predicted_class],
                    'confidence': confidence,
                    'all_probabilities': {LABELS[i]: float(predictions[0][i]) for i in range(len(LABELS))}
                }
                with st.spinner('Saving to GitHub...'):
                    if github_store.save_prediction(prediction_data):
                        st.success("✅ Prediction saved to GitHub successfully!")
                    else:
                        st.error("❌ Failed to save prediction to GitHub")
    
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        1. **Upload an Image**: Click on the upload button and select an image
        2. **View Results**: The app will display the detected emotion and confidence score
        3. **Configure GitHub** (Optional):
           - Create a GitHub Personal Access Token with 'repo' scope
           - Enter your token and repository name in the sidebar
           - Enable "Save predictions to GitHub"
           - Click "Save Prediction to GitHub" button after each prediction
        
        **Note**: Your predictions will be stored in a `predictions.json` file in your GitHub repository.
        """)

if __name__ == "__main__":
    main()
