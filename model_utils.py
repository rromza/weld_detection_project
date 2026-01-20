import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

class WeldClassifier(nn.Module):
    """Класс модели для классификации сварных швов"""
    def __init__(self, config):
        super(WeldClassifier, self).__init__()
        self.config = config
        
        # Загружаем предобученную модель
        if config['MODEL_NAME'] == 'efficientnet_b0':
            backbone = models.efficientnet_b0(weights=None)
            in_features = 1280
        elif config['MODEL_NAME'] == 'efficientnet_b1':
            backbone = models.efficientnet_b1(weights=None)
            in_features = 1280
        elif config['MODEL_NAME'] == 'efficientnet_b2':
            backbone = models.efficientnet_b2(weights=None)
            in_features = 1408
        else:
            raise ValueError(f"Model {config['MODEL_NAME']} not supported")
        
        # Замораживаем backbone (опционально)
        for param in backbone.parameters():
            param.requires_grad = False
        
        # Меняем классификатор
        backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, config['NUM_CLASSES'])
        )
        
        self.model = backbone
    
    def forward(self, x):
        return self.model(x)

def load_model(model_path, device='cpu'):
    """
    Загрузка обученной модели из файла
    
    Args:
        model_path: путь к файлу модели
        device: устройство для загрузки (cpu или cuda)
    
    Returns:
        model: загруженная модель
        config: конфигурация модели
        class_names: названия классов
    """
    # Загружаем сохраненные данные
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    class_names = checkpoint['classes']
    
    # Создаем модель
    model = WeldClassifier(config)
    
    # Загружаем веса
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # Переводим в режим инференса
    
    print(f"✅ Model loaded from {model_path}")
    print(f"📊 Classes: {class_names}")
    print(f"⚙️  Device: {device}")
    
    return model, config, class_names

def get_transforms(input_size=224):
    """
    Создание трансформаций для изображений
    
    Args:
        input_size: размер входного изображения
    
    Returns:
        transform: композиция трансформаций
    """
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def predict_image(model, image, transform, device='cpu', class_names=None):
    """
    Предсказание класса для одного изображения
    
    Args:
        model: обученная модель
        image: PIL Image
        transform: трансформации для изображения
        device: устройство для вычислений
        class_names: названия классов
    
    Returns:
        pred_class: предсказанный класс (строка)
        confidence: уверенность модели
        all_probs: вероятности для всех классов
    """
    # Применяем трансформации
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Предсказание
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    # Преобразуем в numpy
    confidence = confidence.item()
    predicted_idx = predicted_idx.item()
    all_probs = probabilities.cpu().numpy()[0]
    
    # Получаем название класса
    if class_names:
        pred_class = class_names[predicted_idx]
    else:
        pred_class = f"Class {predicted_idx}"
    
    return pred_class, confidence, all_probs

def load_and_prepare_image(image_path, max_size=800):
    """
    Загрузка и подготовка изображения для отображения
    
    Args:
        image_path: путь к изображению
        max_size: максимальный размер для отображения
    
    Returns:
        image: PIL Image
    """
    image = Image.open(image_path).convert('RGB')
    
    # Масштабируем для отображения (сохраняя пропорции)
    width, height = image.size
    if max(width, height) > max_size:
        ratio = max_size / max(width, height)
        new_size = (int(width * ratio), int(height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    return image