import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import resnet_v2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models


@st.cache_resource
def load_models():
    """Load both TensorFlow and PyTorch models (cached)."""
    with st.spinner("Loading models... ⏳"):
        tf_model = tf.keras.models.load_model('./models/tensorflow_model.keras')

        torch_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        torch_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        torch_model.load_state_dict(torch.load('./models/pytorch_model.pth', map_location='cpu'))
        torch_model.eval()

    return tf_model, torch_model


def preprocess_image(image, model_type, image_size=256):
    """Preprocess the image based on the selected model."""
    if model_type == "PyTorch":
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)  # Add batch dimension
    else:
        image_tensor = tf.convert_to_tensor(image)
        image_tensor = tf.image.resize(image_tensor, (image_size, image_size))
        image_tensor = tf.cast(image_tensor, tf.float32)
        image_tensor = resnet_v2.preprocess_input(image_tensor)
        return tf.expand_dims(image_tensor, axis=0)  # Shape: (1, 256, 256, 3)


def make_prediction(model, image_tensor, model_type):
    """Make a prediction using the selected model."""
    if model_type == "PyTorch":
        with torch.no_grad():
            output = model(image_tensor)
            return torch.sigmoid(output).item()
    else:
        return model.predict(image_tensor)[0][0]


# ---- STREAMLIT UI ----
st.set_page_config(page_title="Pneumonia Detection", page_icon="🩺", layout="centered")

# ---- HEADER ----
st.title("🩺 Pneumonia Detection using CNNs")
st.markdown("### Upload an X-ray image to detect pneumonia")
st.divider()

# ---- Load Models ----
tf_model, torch_model = load_models()

# ---- Sidebar - Model Selection ----
with st.sidebar:
    st.markdown("## 🔍 **Model Selection**")
    selected_model = st.radio("Choose a model:", ["TensorFlow", "PyTorch"])

    # ---- File Upload ----
    uploaded_file = st.file_uploader("📂 Upload an X-ray image", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(uploaded_file, caption="🖼️ Uploaded X-ray", use_container_width=True)

    with col2:
        # Convert image
        image = Image.open(uploaded_file).convert("RGB")
        image_tensor = preprocess_image(image, selected_model)

        # Display loading message
        with st.spinner("🧠 Analyzing X-ray..."):
            prediction = make_prediction(torch_model if selected_model == "PyTorch" else tf_model, image_tensor,
                                         selected_model)

        # ---- Display result ----
        st.markdown("## **Prediction Result**")
        confidence = prediction * 100

        if confidence > 90:
            st.error(f"🔴 **High Risk**: {confidence:.2f}% chance of pneumonia!", icon="🚨")
        else:
            st.success(f"🟢 **Low Risk**: {confidence:.2f}% chance of pneumonia!", icon="✅")

        st.markdown("### 📊 **Model Used:**")
        st.info(f"✔ You used **{selected_model}** model for this prediction.")

# ---- Footer ----
st.markdown("---")
st.markdown(
    "🚀 **Built with Streamlit, TensorFlow & PyTorch** | Made by **[Davron Abdukhakimov](https://www.linkedin.com/in/davron-abdukhakimov/)**\n\n"
    "📌 Check out the repo on [Hugging Face 🤗](https://huggingface.co/spaces/davron04/CNN_Pneumonia_detection/tree/main)\n\n"
    "📑 The dataset on [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)"
)