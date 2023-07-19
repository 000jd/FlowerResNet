import os
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from utils.dataset import load_dataset
from models.flower_resnet import FlowerResNet
from utils.training import train, evaluate

data_path = 'data/flower299'
batch_size = 128
num_classes = 299
learning_rate = 0.001
checkpoint_path = 'checkpoint/res_checkpoint_quantized.pth'

train_loader, val_loader = load_dataset(data_path, batch_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FlowerResNet(num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)

start_epoch = 0
best_val_loss = float('inf')


if os.path.exists(checkpoint_path):
    try:
      checkpoint = torch.load(checkpoint_path)
    except Exception as e:
      checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['val_loss']
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    train_accuracies = checkpoint['train_accuracies']
    val_accuracies = checkpoint['val_accuracies']