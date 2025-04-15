import requests
import joblib
import base64
import os
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from PIL import Image

# Define a utility function to fetch and save models
def fetch_model(api_url, local_path):
    if not os.path.exists(local_path):
        response = requests.post(api_url)
        if response.status_code == 200:
            model_data = base64.b64decode(response.json()["model"])
            with open(local_path, "wb") as f:
                f.write(model_data)
        else:
            raise Exception(f"Failed to fetch model from {api_url}")

# API URLs for the models
base_url = "https://introduction-to-genai-captsone.onrender.com"
best_model_url = f"{base_url}/best_model"
kmeans_model_url = f"{base_url}/kmeans_model"
preprocessor_url = f"{base_url}/preprocessor"
scaler_url = f"{base_url}/scaler"

# Local paths for the models
best_model_path = "best_model.pkl"
kmeans_model_path = "kmeans_model.pkl"
preprocessor_path = "preprocessor.pkl"
scaler_path = "scaler.pkl"

# Fetch and save the models
fetch_model(best_model_url, best_model_path)
fetch_model(kmeans_model_url, kmeans_model_path)
fetch_model(preprocessor_url, preprocessor_path)
fetch_model(scaler_url, scaler_path)

# Load the models
best_model = joblib.load(best_model_path)
kmeans_model = joblib.load(kmeans_model_path)
preprocessor = joblib.load(preprocessor_path)
scaler = joblib.load(scaler_path)

# Streamlit App layout
st.title("Student Data Prediction and Clustering, and Image Generator")

# Create tabs for tasks
task = st.radio("Select Task", ["Clustering", "Prediction", "Generate"])

if task == "Clustering":
    st.subheader("Clustering Task")
    studytime = st.number_input("Study Time", min_value=1, max_value=4, value=1)
    G1 = st.number_input("Grade G1", min_value=0, max_value=20, value=10)
    G2 = st.number_input("Grade G2", min_value=0, max_value=20, value=10)

    input_data_cluster = pd.DataFrame({
        'studytime': [studytime],
        'G1': [G1],
        'G2': [G2]
    })
    input_data_cluster['G_combined'] = (input_data_cluster['G1'] + input_data_cluster['G2']) / 2

    scaled_cluster_data = scaler.transform(input_data_cluster[['studytime', 'G_combined']])

    if st.button('Run Clustering'):
        prediction = kmeans_model.predict(scaled_cluster_data)
        cluster_mapping = {0: "Beginner", 1: "Advanced", 2: "Intermediate"}
        cluster_label = cluster_mapping.get(prediction[0], "Unknown")
        st.write(f"Assigned to {cluster_label} Class")

        fig, ax = plt.subplots()
        sns.scatterplot(x=input_data_cluster['studytime'], y=input_data_cluster['G_combined'], hue=[cluster_label], palette='viridis', ax=ax)
        ax.set_title("Cluster Distribution (studytime vs G_combined)")
        st.pyplot(fig)

elif task == "Prediction":
    st.subheader("Prediction Task")
    studytime = st.number_input("Study Time", min_value=1, max_value=4, value=1)
    G1 = st.number_input("Grade G1", min_value=0, max_value=20, value=10)
    G2 = st.number_input("Grade G2", min_value=0, max_value=20, value=10)
    freetime = st.number_input("Free Time", min_value=0, max_value=20, value=10)
    sex = st.selectbox("Sex", ["M", "F"])
    absences = st.number_input("Absences", min_value=0, max_value=100, value=0)
    failures = st.number_input("Failures", min_value=0, max_value=10, value=0)
    goout = st.number_input("Go Out", min_value=0, max_value=20, value=10)
    address = st.selectbox("Address", ["U", "R"])

    input_data_full = pd.DataFrame({
        'studytime': [studytime],
        'G1': [G1],
        'G2': [G2],
        'freetime': [freetime],
        'sex': [sex],
        'absences': [absences],
        'failures': [failures],
        'goout': [goout],
        'address': [address]
    })

    transformed_data_full = preprocessor.transform(input_data_full)

    if st.button('Run Prediction'):
        prediction = best_model.predict(transformed_data_full)
        if prediction[0] == 0:
            st.write("Prediction: Fail")
        else:
            st.write("Prediction: Pass")

elif task == "Generate":
    st.subheader("Image Generation")
    seed_value = st.slider("Seed for Latent Vector", 0, 100, 42)

    if st.button("Generate Image"):
        with st.spinner("Generating image... Please wait."):
            response = requests.post(f"{base_url}/generate", json={"seed": seed_value})
            if response.status_code == 200:
                img_data = base64.b64decode(response.json()["image"])
                image = Image.open(BytesIO(img_data))
                st.image(image, caption="Generated Image", use_container_width=True)
            else:
                st.error("Error generating image!")
