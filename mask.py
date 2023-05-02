import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from math import e

import src 
from unet_model import Unet


model = UNet()
model.load_state_dict(torch.load('./best_model.pth'))
model = model.to(device)
model.eval()  # Set the model to evaluation mode

# Load and preprocess the image
image_path = '../Dataset_Student/val/video_1000/image_10.png'
image = Image.open(image_path).convert('RGB')
transform = transforms.Compose([
    transforms.ToTensor()  # Convert the image to a PyTorch tensor
])
image = transform(image)
image = image.unsqueeze(0)  # Add a batch dimension
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