import torch
import torch.nn as nn
from utils.dataset import load_dataset
from models.flower_resnet import FlowerResNet
from utils.training import evaluate

data_path = 'data/flower299'
batch_size = 128
num_classes = 299
checkpoint_path = 'checkpoint/res_checkpoint_quantized.pth'

_, test_loader = load_dataset(data_path, batch_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FlowerResNet(num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()

checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])

evaluate(model, test_loader, criterion, device)