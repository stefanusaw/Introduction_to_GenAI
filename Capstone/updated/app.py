# import os

# file_path = "/Users/albertwelong/Github_Site/Cognizant/GenAI/Capstone/updated//full_pipeline.pkl"

# if not os.path.exists(file_path):
#     print("File not found!")
# else:
#     print("File exists!")


import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import base64
from io import BytesIO
from PIL import Image


# Load the models and preprocessor
# full_pipeline = joblib.load('/Users/albertwelong/cognizant_genai/.venv/capstone_project/updated/full_pipeline.pkl')
# best_model = joblib.load('/Users/albertwelong/cognizant_genai/.venv/capstone_project/updated/best_model.pkl') 
# kmeans_model = joblib.load('/Users/albertwelong/cognizant_genai/.venv/capstone_project/updated/kmeans_model.pkl') 
# preprocessor = joblib.load('/Users/albertwelong/cognizant_genai/.venv/capstone_project/updated/preprocessor.pkl') 
# scaler = joblib.load('/Users/albertwelong/cognizant_genai/.venv/capstone_project/updated/scaler.pkl') 

full_pipeline = joblib.load('full_pipeline.pkl')
best_model = joblib.load('best_model.pkl') 
kmeans_model = joblib.load('kmeans_model.pkl') 
preprocessor = joblib.load('preprocessor.pkl') 
scaler = joblib.load('scaler.pkl') 

# Define Generator Model
class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 3 * 64 * 64)  # Output: 64x64x3
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

    def forward(self, z):
        x = self.relu(self.fc1(z))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(-1, 3, 64, 64)  # Reshape to 64x64 image with 3 channels
        return self.tanh(x)

# Parameters
z_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained Generator model
generator = Generator(z_dim).to(device)
generator.load_state_dict(torch.load("generator_epoch_100.pth", map_location=device))
generator.eval()


# Streamlit App layout
st.title("Student Data Prediction and Clustering")

# Create tabs for tasks
task = st.radio("Select Task", ["Clustering", "Prediction", "Generator"])

# Shared input fields
studytime = st.number_input("Study Time", min_value=1, max_value=4, value=1)
G1 = st.number_input("Grade G1", min_value=0, max_value=20, value=10)
G2 = st.number_input("Grade G2", min_value=0, max_value=20, value=10)

if task == "Clustering":
    # Clustering specific input (only studytime and combined grade matter)
    st.subheader("Clustering Task")

    # Prepare the input for clustering task (studytime and combined G1/G2)
    input_data_cluster = pd.DataFrame({
        'studytime': [studytime],
        'G1': [G1],
        'G2': [G2]
    })
    input_data_cluster['G_combined'] = (input_data_cluster['G1'] + input_data_cluster['G2']) / 2  # Add G_combined feature

    # Apply scaling to the clustering data
    scaled_cluster_data = scaler.transform(input_data_cluster[['studytime', 'G_combined']])

    # Button to trigger clustering
    if st.button('Run Clustering'):
        # Check scaled data
        st.write("Scaled Data:", scaled_cluster_data)
        
        # For clustering task (kmeans)
        prediction = kmeans_model.predict(scaled_cluster_data)
        cluster_mapping = {0: "Beginner", 1: "Advanced", 2: "Intermediate"}
        cluster_label = cluster_mapping.get(prediction[0], "Unknown")
        st.write(f"Assigned to {cluster_label} Class")

        # Visualize the clusters
        st.subheader("Cluster Visualization")
        sns.scatterplot(x=input_data_cluster['studytime'], y=input_data_cluster['G_combined'], hue=[cluster_label], palette='viridis')
        plt.title("Cluster Distribution (studytime vs G_combined)")
        st.pyplot()

elif task == "Prediction":
    # Prediction specific input (all 11 columns are needed for prediction)
    st.subheader("Prediction Task")

    # Input fields specific to prediction task
    freetime = st.number_input("Free Time", min_value=0, max_value=20, value=10)
    sex = st.selectbox("Sex", ["M", "F"])
    absences = st.number_input("Absences", min_value=0, max_value=100, value=0)
    failures = st.number_input("Failures", min_value=0, max_value=10, value=0)
    goout = st.number_input("Go Out", min_value=0, max_value=20, value=10)
    address = st.selectbox("Address", ["U", "R"])

    # Prepare the input for prediction task (all 11 columns)
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

    # Apply preprocessing to the input data for prediction (includes all features)
    transformed_data_full = preprocessor.transform(input_data_full)

    # Button to trigger prediction
    if st.button('Run Prediction'):
        # For prediction task (e.g., regression or classification)
        prediction = best_model.predict(transformed_data_full)
        if prediction[0] == 0:
            st.write("Prediction: Fail")
        else:
            st.write("Prediction: Pass")

elif task == "Generate":
    st.subheader("Image Generation")
    seed_value = st.slider("Seed for Latent Vector", 0, 100, 42)

    if st.button("Generate Image"):
        response = requests.post("https://introduction-to-genai-captsone.onrender.com/generate", json={"seed": seed_value})
        if response.status_code == 200:
            img_data = base64.b64decode(response.json()["image"])
            image = Image.open(BytesIO(img_data))
            st.image(image, caption="Generated Image", use_container_width=True)
        else:
            st.error("Error generating image!")
