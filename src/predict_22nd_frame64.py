# -*- coding: utf-8 -*-
"""predict_22nd_frame2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10b0-8bKt59rBFU0fs80QoQflF33Mzxi_
"""

import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from math import e

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
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  input_channels = 3  # For RGB images
  hidden_channels = 32
  kernel_size = 3
  num_layers = 3
  seq_length = 11
  model = ConvLSTM(input_channels, hidden_channels, kernel_size, num_layers, seq_length).to(device)
  # Load the trained model
  model_path = "model_output/best_conv_lstm_model_b64_l3_e50.pth"
  model = model.to(device)
  state_dict = torch.load(model_path, map_location=device)
  new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()} 
  
  model = model.to(device)
  model.load_state_dict(new_state_dict)
  model.eval()

  print("load model success")
  # Function to generate the 22nd frame
  def generate_22nd_frame(model, input_frames):
      input_frames = input_frames.unsqueeze(0).to(device)
      for _ in range(11):
          output = model(input_frames)
          input_frames = torch.cat((input_frames[:, 1:], output.unsqueeze(1)), dim=1)
      return output

  # Process the hidden set
  hidden_dir = "hidden/"
  
  # Create a folder to save the predicted 22nd frames
  output_folder = "predicted_22nd_frames_64"
  os.makedirs(output_folder, exist_ok=True)
  print("mkdir success")

  for folder in os.listdir(hidden_dir):
      video_folder_path = os.path.join(hidden_dir, folder)
      
      input_frames = []
      print("finish 1")
      for i in range(11):
          image_path = os.path.join(video_folder_path, f"image_{i}.png")
          image = Image.open(image_path).convert("RGB")
          print("image open finish")
          image = transforms.ToTensor()(image)
          input_frames.append(image)
      
      print("image load success")
      input_frames = torch.stack(input_frames)
      print("image stack success")
      # Generate the 22nd frame using the model
      with torch.no_grad():
          predicted_22nd_frame = generate_22nd_frame(model, input_frames)
      print("predict success")
      # Save the predicted 22nd frame
      output_filename = os.path.join(output_folder, f"22nd_frame_{folder}.pt")
      torch.save(predicted_22nd_frame.cpu(), output_filename)
      print("save predict success")

  print("finish")