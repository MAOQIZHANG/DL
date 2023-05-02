import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from math import e
import numpy as np
from unet_model import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet()
model = model.to(device)
model.load_state_dict(torch.load('DL2/unet_best_model.pth', map_location=device))
model.eval()

print('load model success')
# Load and preprocess the image
dir_path = 'predicted_22nd_frames_test2'
output_folder = "mask_full"
os.makedirs(output_folder, exist_ok=True)


image_path = 'image_21.png'
image = Image.open(image_path).convert('RGB')
transform = transforms.Compose([
    transforms.ToTensor()  # Convert the image to a PyTorch tensor
])
image = transform(image)
image = image.to(device)  # Send the image to the device
# Run the image through the model
with torch.no_grad():
    output = model(image)

# The output is a batch of tensors with shape (num_classes, height, width)
# We take the argmax to get the class with the highest probability for each pixel
mask = torch.argmax(output, dim=1)

# Convert the mask to a numpy array and squeeze to remove the batch dimension
mask = mask.cpu().numpy()
mask = np.squeeze(mask)
