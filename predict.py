import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from models import flower_resnet
from utils import classes

def predict_flower(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    image = transform(image).unsqueeze(0)

    model = flower_resnet.FlowerResNet(num_classes=299)
    model.load_state_dict(torch.load('res_checkpoint.pth', map_location=torch.device('cpu'))['model_state_dict'])
    model.eval()

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    flower_name = classes.class_names[predicted.item()]
    return flower_name