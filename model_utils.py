import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class WeldClassifier(nn.Module):

    def __init__(self, num_classes=3):
        super(WeldClassifier, self).__init__()
        
        backbone = models.efficientnet_b1(weights=None)
        in_features = 1280
                   
        for param in backbone.parameters():
            param.requires_grad = False
        
        backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        self.model = backbone
    
    def forward(self, x):
        return self.model(x)

def load_model(model_path, device='cpu'):

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    class_names = checkpoint.get('class_names', ['bad_weld', 'good_weld', 'no_weld'])
    num_classes = len(class_names)
    
    model = WeldClassifier(num_classes=num_classes)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    config = {
        'INPUT_SIZE': 224,
        'NUM_CLASSES': num_classes
    }
    
    return model, config, class_names

def get_transforms(input_size=224):
    return transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def predict_image(model, image, transform, device='cpu', class_names=None):
   
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    confidence = confidence.item()
    predicted_idx = predicted_idx.item()
    all_probs = probabilities.cpu().numpy()[0]
    
    if class_names:
        pred_class = class_names[predicted_idx]
    else:
        pred_class = f"Class {predicted_idx}"
    
    return pred_class, confidence, all_probs