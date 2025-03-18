from fastapi import FastAPI
import torch
import torch.nn.functional as F
import numpy as np
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import base64

# FastAPI app
app = FastAPI()

# Load the model
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

# Set up model
z_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(z_dim).to(device)
generator.load_state_dict(torch.load("generator_epoch_100.pth", map_location=device))
generator.eval()

# Define input schema
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

    # Convert to PIL image
    img = Image.fromarray((generated_image * 255).astype(np.uint8))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return {"image": img_str}
