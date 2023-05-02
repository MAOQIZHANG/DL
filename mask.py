import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from math import e
 
from unet_model import UNet


model = UNet()
model.load_state_dict(torch.load('unet_best_model.pth'))
model = model.to(device)
model.eval()  # Set the model to evaluation mode
print('load model success')
# Load and preprocess the image
dir_path = 'predicted_22nd_frames_32'
output_folder = "mask_32"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(dir_path):
    print('begin')
    if os.path.isfile(os.path.join(dir_path, filename)):
        file_, ext = os.path.splitext(filename)
    image = torch.load(os.path.join(dir_path, filename))
    image = image.unsqueeze(0)  # Add a batch dimension
    image = image.to(device)  # Send the image to the device
    with torch.no_grad():
        output = model(image)
    mask = torch.argmax(output, dim=1)
    mask = mask.cpu().numpy()
    mask = np.squeeze(mask)
    output_path = os.path.join(output_folder, file_)
    np.save(output_path, mask)

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