# -*- coding: utf-8 -*-
"""nextframe_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1U7PPXLG7uZk2djQGZyNnYo4XWPRiNV__
"""

import os
import torch
import imageio.v3 as iio
import numpy as np
from math import e
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

class VideoDataset(Dataset):
    def __init__(self, root_dirs,num=None):
        self.root_dirs = root_dirs #can only accept root_dirs with at most 2 root_dir

        if len(self.root_dirs)==1:
          self.lendir=-1# this variable saves the length of the first root_dir if there are 2 root_dir, else -1
        else:
          self.lendir=len(sorted([d for d in os.listdir(root_dirs[0]) if os.path.isdir(os.path.join(root_dirs[0], d))]))
        self.video_folders=[]
        for root_dir in root_dirs:
          self.video_folders.extend(sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]))

        if num:
          self.video_folders = self.video_folders[:num]


    def __len__(self):
        return len(self.video_folders) * 11

    def __getitem__(self, idx):
        video_folder_idx = idx // 11
        start_frame_idx = idx % 11

        if (self.lendir==-1) or (video_folder_idx<self.lendir):
          video_folder = os.path.join(self.root_dirs[0], self.video_folders[video_folder_idx])
        else:
          video_folder = os.path.join(self.root_dirs[1], self.video_folders[video_folder_idx])

        input_frames = []
        for i in range(start_frame_idx, start_frame_idx + 11):
            image_path = os.path.join(video_folder, f'image_{i}.png')
            image = Image.open(image_path).convert('RGB')
            image = transforms.ToTensor()(image)
            input_frames.append(image)
        input_frames = torch.stack(input_frames)

        target_frame_idx = start_frame_idx + 11
        target_image_path = os.path.join(video_folder, f'image_{target_frame_idx}.png')
        target_image = Image.open(target_image_path).convert('RGB')
        target_frame = transforms.ToTensor()(target_image)

        return input_frames, target_frame

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

# ConvLSTM Model
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(self.input_channels + self.hidden_channels, 4 * self.hidden_channels, self.kernel_size, padding=self.padding, bias=self.bias)
        self.bn = nn.BatchNorm2d(4 * self.hidden_channels)

    def forward(self, input, hidden_state):
        h_cur, c_cur = hidden_state
        combined = torch.cat([input, h_cur], dim=1)
        conv_output = self.conv(combined)
        conv_output = self.bn(conv_output)

        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers, seq_length):
        super(ConvLSTM, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.seq_length = seq_length

        self.preprocess = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(hidden_channels, hidden_channels),
            ResidualBlock(hidden_channels, hidden_channels),
        )
        self.preprocess_dropout = nn.Dropout2d(p=0.5)

        layers = []
        for i in range(self.num_layers):
            in_channels = self.hidden_channels
            cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size)
            layers.append(cell)
        self.layers = nn.ModuleList(layers)

        self.dropout = nn.Dropout2d(p=0.5)

        self.postprocess = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.ReLU(inplace=True),
            ResidualBlock(hidden_channels, hidden_channels),
            ResidualBlock(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, input_channels, 1),
        )

    def forward(self, input):
        batch_size, seq_length, _, height, width = input.shape
        layer_hidden_states = [layer.init_hidden(batch_size, (height // 2, width // 2)) for layer in self.layers]

        for t in range(self.seq_length):
            cur_input = input[:, t]
            cur_input = self.preprocess(cur_input)
            cur_input = self.preprocess_dropout(cur_input)
            new_hidden_states = []
            for i, (layer, hidden_state) in enumerate(zip(self.layers, layer_hidden_states)):
                h_cur, c_cur = hidden_state
                h_next, c_next = layer(cur_input, (h_cur, c_cur))
                new_hidden_states.append((h_next, c_next))
                cur_input = h_next
                if i < len(self.layers) - 1:  # Apply dropout to all layers except the last one
                    cur_input = self.dropout(cur_input)

            layer_hidden_states = new_hidden_states

        last_output = cur_input
        return self.postprocess(last_output)


if __name__ == "__main__":
  #!gdown https://drive.google.com/uc?id=1fs3AmGYzHWOaMnG4rURC3M1QMStcz11-
  #!unzip Dataset_Student_V2.zip
  # Configuration
  train_root_dirs= ['Dataset_Student/train/','Dataset_Student/unlabeled/']
  valid_root_dirs=['Dataset_Student/val']


  batch_size = 32
  epochs = 100
  learning_rate = 0.001
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  scaler = GradScaler()

  # Create dataset and dataloader
  #number could be added here 
  train_dataset = VideoDataset(train_root_dirs)
  val_dataset = VideoDataset(valid_root_dirs)
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
  val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

  # Initialize the ConvLSTM model
  input_channels = 3  # For RGB images
  hidden_channels = 32
  kernel_size = 3
  num_layers = 3
  seq_length = 11
  model = ConvLSTM(input_channels, hidden_channels, kernel_size, num_layers, seq_length).to(device)

  # Optimizer and loss function
  optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

  criterion = nn.MSELoss().to(device)

  # Training function
  def train(model, dataloader, criterion, optimizer, device):
      model.train()
      epoch_loss = 0.0
      for input_batch, target_batch in dataloader:
          input_batch = input_batch.to(device)
          target_batch = target_batch.to(device)
          # Use autocast to enable mixed precision training
          with autocast():
              output = model(input_batch)
              loss = criterion(output, target_batch)

          # Scale the loss and perform backpropagation using the GradScaler
          optimizer.zero_grad()
          scaler.scale(loss).backward()
          scaler.step(optimizer)
          scaler.update()

          epoch_loss += loss.item()
      return epoch_loss / len(dataloader)


  # Validation function
  def validate(model, dataloader, criterion, device):
      model.eval()
      epoch_loss = 0.0
      with torch.no_grad():
          for input_batch, target_batch in dataloader:
              input_batch = input_batch.to(device)
              target_batch = target_batch.to(device)
              # Use autocast to enable mixed precision during validation
              with autocast():
                  output = model(input_batch)
                  loss = criterion(output, target_batch)
              epoch_loss += loss.item()
      return epoch_loss / len(dataloader)
      
  # Training and validation loop
  patience = 10
  epochs_without_improvement = 0
  early_stop = False

  best_val_loss = float('inf')
  print('Training started!')
  for epoch in range(1, epochs + 1):
      train_loss = train(model, train_dataloader, criterion, optimizer, device)
      val_loss = validate(model, val_dataloader, criterion, device)

      scheduler.step(val_loss)

      print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

      if val_loss < best_val_loss:
          best_val_loss = val_loss
          torch.save(model.state_dict(), "model_output/best_conv_lstm_model_full.pth")
          epochs_without_improvement = 0
      else:
          epochs_without_improvement += 1
          if epochs_without_improvement >= patience:
              early_stop = True
              print("Early stopping triggered.")
              break