from unet_model import UNet
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import os
import torch.utils.data as data
from tqdm import tqdm
import numpy as np

class UNetSegmentationDataset(data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.video_dirs = os.listdir(root_dir)

    def __len__(self):
        return len(self.video_dirs) * 22

    def __getitem__(self, idx):
        video_idx = idx // 22
        frame_idx = idx % 22
        video_dir = os.path.join(self.root_dir, self.video_dirs[video_idx])

        # Load images
        image = Image.open(video_dir + '/image_' + str(frame_idx) + '.png').convert('RGB')

        # Load segmentation mask
        segm_name = 'mask.npy'
        segm_mask = np.load(os.path.join(video_dir, segm_name))[frame_idx]
        max_class = 48
        invalid_mask = segm_mask > max_class
        segm_mask[invalid_mask] = 0

        # Define the transform
        transform = transforms.Compose([transforms.ToTensor()])

        # transform to torch
        image = transform(image)
        segm_mask = torch.from_numpy(segm_mask).long()

        return image, segm_mask

def load_data(root):
    # Load the data
    dataset_train = UNetSegmentationDataset(root_dir=root+'/train')
    dataloader_train = data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    dataset_val = UNetSegmentationDataset(root_dir=root+'/val')
    dataloader_val = data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

    return dataloader_train, dataloader_val

if __name__ == "__main__":
    # Set hyperparameters
    batch_size = 16
    learning_rate = 1e-4
    epochs = 30

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Load data
    dataloader_train, dataloader_val = load_data('../data')

    # Initialize the model
    model = UNet()
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for i, data in tqdm(enumerate(dataloader_train, 0)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Validate the model
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in tqdm(dataloader_val):
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        # Calculate loss
        epoch_loss /= len(dataloader_train)
        val_loss /= len(dataloader_val)

        print(f"Epoch: {epoch}, Train Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "../model/unet_best_model.pth")
