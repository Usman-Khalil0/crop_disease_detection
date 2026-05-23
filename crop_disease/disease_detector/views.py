import os
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from django.contrib import messages

# CORRECT PATH - add the parent directory of ml_model
ML_MODEL_PATH = os.path.join(settings.BASE_DIR, 'disease_detector')
import sys
if ML_MODEL_PATH not in sys.path:
    sys.path.insert(0, ML_MODEL_PATH)

# Initialize predictor
PREDICTOR = None

def get_predictor():
    global PREDICTOR
    if PREDICTOR is None:
        try:
            # Import from ml_model (now in path)
            from ml_model.predict import DiseasePredictor
            
            # Path to model
            model_path = os.path.join(ML_MODEL_PATH, 'ml_model', 'best_model.pth')
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            PREDICTOR = DiseasePredictor(model_path=model_path, device='cpu')
            print(f" Model loaded from {model_path}")
            
        except Exception as e:
            print(f"Warning: ML model not loaded: {e}")
            PREDICTOR = None
    
    return PREDICTOR

def index(request):
    context = {
        'title': 'Crop Disease Detection',
        'languages': ['en', 'hi', 'ur']
    }
    return render(request, 'disease_detector/index.html', context)

def predict_disease(request):
    if request.method != 'POST':
        return render(request, 'disease_detector/index.html')
    
    predictor = get_predictor()
    if predictor is None:
        messages.error(request, 'ML model not loaded. Please contact administrator.')
        return render(request, 'disease_detector/index.html')
    
    uploaded_file = request.FILES.get('leaf_image')
    if not uploaded_file:
        messages.error(request, 'Please upload an image file.')
        return render(request, 'disease_detector/index.html')
    
    content_type = uploaded_file.content_type
    allowed_types = ['image/jpeg', 'image/png', 'image/jpg']
    if content_type not in allowed_types:
        messages.error(request, 'Please upload JPEG or PNG image (max 5MB).')
        return render(request, 'disease_detector/index.html')
    
    if uploaded_file.size > 5 * 1024 * 1024:
        messages.error(request, 'File too large. Maximum size is 5MB.')
        return render(request, 'disease_detector/index.html')
    
    language = request.POST.get('language', 'en')
    
    try:
        file_name = default_storage.save(
            os.path.join('uploads', uploaded_file.name),
            ContentFile(uploaded_file.read())
        )
        file_path = default_storage.path(file_name)
        
        result = predictor.predict(file_path, language=language)
        
        default_storage.delete(file_name)
        
        context = {
            'result': result,
            'language': language,
            'image_name': uploaded_file.name
        }
        return render(request, 'disease_detector/result.html', context)
        
    except Exception as e:
        messages.error(request, f'Prediction error: {str(e)}')
        return render(request, 'disease_detector/index.html')

def about(request):
    context = {
        'title': 'About',
        'crops_supported': ['Tomato'],
        'diseases_supported': 7,
        'languages_supported': ['English', 'Hindi', 'Urdu'],
        'model_accuracy': '93.1%'
    }
    return render(request, 'disease_detector/about.html', context)

def contact(request):
    return render(request, 'disease_detector/contact.html')