from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import os
from torchvision import transforms

app = Flask(__name__)
UPLOAD_FOLDER = 'static/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
             nn.MaxPool2d(kernel_size=2, stride=2),
           nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * 4, 64),  # Adjusted the input size to match the output of the final convolutional layer
            nn.ReLU(),
            nn.Dropout(0.5),  # Added dropout for regularization
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Create an instance of the CNN model
model = CNN()
model.load_state_dict(torch.load('model/zyro_model.pth'))
model.eval()

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = preprocess(img)
    img = img.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(img)
    _, predicted_class = torch.max(output, 1)
    class_names = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 
                   5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
    class_name = class_names[predicted_class.item()]
    return class_name

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            prediction = predict_image(file_path)
            return render_template('index.html', prediction=prediction, image_path=file_path)

    return render_template('index.html', prediction=None, image_path=None)

if __name__== '__main__':
    app.run(debug=True)