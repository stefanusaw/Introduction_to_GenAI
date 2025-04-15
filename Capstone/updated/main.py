from fastapi import FastAPI
import joblib
import os
import base64
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import numpy as np
from io import BytesIO
from PIL import Image

# FastAPI app
app = FastAPI()

# Define a utility function to encode files as Base64
def encode_file_as_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Paths to the .pkl files
"""
best_model_path = "/Users/albertwelong/cognizant_genai/.venv/capstone_project/updated/best_model.pkl"
kmeans_model_path = "/Users/albertwelong/cognizant_genai/.venv/capstone_project/updated/kmeans_model.pkl"
preprocessor_path = "/Users/albertwelong/cognizant_genai/.venv/capstone_project/updated/preprocessor.pkl"
scaler_path = "/Users/albertwelong/cognizant_genai/.venv/capstone_project/updated/scaler.pkl"
"""

best_model_path = "best_model.pkl"
kmeans_model_path = "kmeans_model.pkl"
preprocessor_path = "preprocessor.pkl"
scaler_path = "scaler.pkl"

# Ensure the files exist
for file_path in [best_model_path, kmeans_model_path, preprocessor_path, scaler_path]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found!")

# Serve the models as Base64-encoded files
@app.post("/best_model")
def get_best_model():
    return {"model": encode_file_as_base64(best_model_path)}

@app.post("/kmeans_model")
def get_kmeans_model():
    return {"model": encode_file_as_base64(kmeans_model_path)}

@app.post("/preprocessor")
def get_preprocessor():
    return {"model": encode_file_as_base64(preprocessor_path)}

@app.post("/scaler")
def get_scaler():
    return {"model": encode_file_as_base64(scaler_path)}

# Generator model and /generate endpoint
class Generator(torch.nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.fc1 = torch.nn.Linear(z_dim, 256)
        self.fc2 = torch.nn.Linear(256, 512)
        self.fc3 = torch.nn.Linear(512, 1024)
        self.fc4 = torch.nn.Linear(1024, 3 * 64 * 64)
        self.relu = torch.nn.ReLU(True)
        self.tanh = torch.nn.Tanh()

    def forward(self, z):
        x = self.relu(self.fc1(z))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(-1, 3, 64, 64)
        return self.tanh(x)

z_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(z_dim).to(device)
generator.load_state_dict(torch.load("generator_epoch_100.pth", map_location=device))
generator.eval()

class GenerateInput(BaseModel):
    seed: int

@app.post("/generate")
def generate_image(data: GenerateInput):
    torch.manual_seed(data.seed)
    z = torch.randn(1, z_dim).to(device)
    
    with torch.no_grad():
        generated_image = generator(z).squeeze(0)
    
    generated_image = F.interpolate(generated_image.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False).squeeze(0)
    generated_image = (generated_image + 1) / 2  # Normalize to [0,1]
    generated_image = generated_image.permute(1, 2, 0).cpu().numpy()

    img = Image.fromarray((generated_image * 255).astype(np.uint8))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return {"image": img_str}
