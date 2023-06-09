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
model.load_state_dict(torch.load('DL/model/unet_best_model.pth', map_location=device))
model.eval()

print('load model success')
# Load and preprocess the image
dir_path = 'predicted_22nd_frames_64'
output_folder = "mask_64"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(dir_path):
    print('begin')
    if os.path.isfile(os.path.join(dir_path, filename)):
        file_, ext = os.path.splitext(filename)
        tensor = torch.load(os.path.join(dir_path, filename))
        image = tensor.to(device)  # Send the image to the device
        with torch.no_grad():
            output = model(image)
            print('generate success')
        mask = torch.argmax(output, dim=1)
        mask = mask.cpu().numpy()
        mask = np.squeeze(mask)
        output_path = os.path.join(output_folder, file_)+".npy"
        np.save(output_path, mask)


# Specify the directory
dir_path = 'mask_64'

# Initialize an empty list to hold the numpy arrays
arrays = []

# Loop over the file numbers
for i in range(15000, 17000):  # Modify this range as needed
    # Construct the file name
    file_name = f'22nd_frame_video_{i}.npy'
    file_path = os.path.join(dir_path, file_name)

    # Check if the file exists
    if os.path.isfile(file_path):
        # Load the numpy array from the file
        array = np.load(file_path)
        # Append the array to the list
        arrays.append(array)

# Stack the arrays along the first dimension (axis=0)
stacked_array = np.stack(arrays, axis=0)

# Save the stacked array to a file
np.save('final/final_64_3.npy', stacked_array)

print('finish')
'''
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
mask = np.squeeze(mask)'''