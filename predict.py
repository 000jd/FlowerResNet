import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import tqdm
from models import flower_resnet
from utils import classes
from google_drive_downloader import GoogleDriveDownloader as gdd

def download_model():
    model_id = '1SxqhmJZEgw5h38V6SdrCFeFDsRM-xSnu'  # Replace with your Google Drive model ID
    # Get the base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Set the model path to the base directory
    model_path = os.path.join(base_dir, 'res_checkpoint.pth')
    with tqdm.tqdm(total=100, unit='B', unit_scale=True) as progress_bar:
        gdd.download_file_from_google_drive(file_id=model_id, dest_path=model_path, showsize=True, overwrite=True,
                                            progress_bar=progress_bar)
    print('Download complete.')
    
def predict_flower(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    image = transform(image).unsqueeze(0)

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res_checkpoint.pth')
    if not os.path.exists(model_path):
        print('Model not found. Downloading the model...')
        download_model()
        
    model = flower_resnet.FlowerResNet(num_classes=299)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])
    model.eval()

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    flower_name = classes.class_names[predicted.item()]
    return flower_name
