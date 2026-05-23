import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
import json
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Constants (no external config dependency)
IMAGE_SIZE = 299
CONFIDENCE_THRESHOLD = 0.60

CLASS_NAMES = [
    'Healthy',
    'Early_blight',
    'Late_blight',
    'Leaf_mold',
    'Bacterial_spot',
    'Yellow_leaf_curl_virus',
    'Mosaic_virus'
]


class DiseasePredictor:
    def __init__(self, model_path=None, device='cpu'):
    
        self.device = device
        
        # Set model path
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'best_model.pth')
        self.model_path = model_path
        
        self.class_names = CLASS_NAMES
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self.image_size = IMAGE_SIZE
        self.model = None
        
        self.load_model()
    
    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at {self.model_path}\n"
                f"Please ensure best_model.pth exists in this folder"
            )
        
        try:
            # Import model architecture from same folder
            try:
                from model import get_model
            except ImportError:
                from .model import get_model
            
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Get number of classes from checkpoint if available, otherwise use CLASS_NAMES
            if 'class_names' in checkpoint:
                checkpoint_classes = checkpoint['class_names']
                if len(checkpoint_classes) != len(self.class_names):
                    print(f"Warning: Checkpoint has {len(checkpoint_classes)} classes but expected {len(self.class_names)}")
                    print(f"Using checkpoint classes: {checkpoint_classes}")
                    self.class_names = checkpoint_classes
            
            self.model = get_model(num_classes=len(self.class_names), device=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print(f" Model loaded successfully")
            print(f"  Path: {self.model_path}")
            print(f"  Device: {self.device}")
            print(f"  Classes: {len(self.class_names)}")
            print(f"  Class names: {self.class_names}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def preprocess_image(self, image_path):
        """Preprocess image for model input"""
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0)  # Add batch dimension
    
    def predict(self, image_path, language='en'):
        """Predict disease from image"""
        
        # Check if file exists
        if not os.path.exists(image_path):
            return {
                'status': 'error',
                'message': f"Image file not found: {image_path}",
                'success': False
            }
        
        # Check if model is loaded
        if self.model is None:
            return {
                'status': 'error',
                'message': "Model not loaded. Please check model file.",
                'success': False
            }
        
        try:
            # Preprocess and predict
            image_tensor = self.preprocess_image(image_path).to(self.device)
            
            with torch.no_grad():
                probabilities = self.model(image_tensor)
                confidence, predicted_idx = torch.max(probabilities, 1)
                confidence = confidence.item()
                print(f"DEBUG - Raw confidence: {confidence}")
                predicted_class = self.class_names[predicted_idx.item()]
                
                # Get top-3 predictions (convert to percentages)
                top3_probs, top3_indices = torch.topk(probabilities, min(3, len(self.class_names)))
                top3_predictions = [
                    {'class': self.class_names[idx], 'confidence': prob.item() * 100}  # Multiply by 100 for percentage
                    for idx, prob in zip(top3_indices[0], top3_probs[0])
                ]
            
            # Check confidence threshold
            if confidence < self.confidence_threshold:
                messages = {
                    'en': f"⚠️ Low confidence ({confidence:.1%}). Please upload a clearer, well-lit image.",
                    'hi': f"⚠️ कम विश्वसनीयता ({confidence:.1%})। कृपया स्पष्ट छवि अपलोड करें।",
                    'ur': f"⚠️ کم اعتماد ({confidence:.1%})۔ براہ کرم واضح تصویر اپ لوڈ کریں۔"
                }
                return {
                    'status': 'low_confidence',
                    'confidence': confidence,
                    'predicted_class': predicted_class,
                    'top3_predictions': top3_predictions,
                    'message': messages.get(language, messages['en']),
                    'success': False
                }
            
            # Get treatment suggestion
            treatment = self._get_treatment(predicted_class, language)
            
            return {
                'status': 'success',
                'disease': predicted_class,
                'confidence': confidence * 100,
                'treatment': treatment,
                'top3_predictions': top3_predictions,
                'success': True
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Prediction error: {str(e)}",
                'success': False
            }
    
    def _get_treatment(self, disease, language='en'):
        """Get treatment recommendations for the disease"""
        
        treatments = {
            'Healthy': {
                'en': "✓ Your plant appears healthy.\n\nRecommendations:\n• Continue regular watering and fertilization\n• Monitor weekly for any changes\n• Maintain proper spacing for air circulation",
                'hi': "✓ आपका पौधा स्वस्थ दिखता है।\n\nसुझाव:\n• नियमित पानी और खाद देना जारी रखें\n• किसी भी बदलाव के लिए साप्ताहिक निगरानी करें\n• वायु संचार के लिए उचित दूरी बनाए रखें",
                'ur': "✓ آپ کا پودا صحت مند نظر آتا ہے۔\n\nتجاویز:\n• باقاعدہ پانی اور کھاد ڈالتے رہیں\n• کسی بھی تبدیلی کے لیے ہفتہ وار نگرانی کریں\n• ہوا کی گردش کے لیے مناسب فاصلہ برقرار رکھیں"
            },
            'Early_blight': {
                'en': "⚠️ EARLY BLIGHT DETECTED\n\nTREATMENT PLAN:\n1. Remove all infected leaves immediately\n2. Apply copper-based fungicide every 7-10 days\n3. Improve air circulation through pruning\n4. Water at base only - avoid wetting leaves\n5. Mulch around plants to prevent soil splash\n\nPrevention: Use resistant varieties next season.",
                'hi': "⚠️ अर्ली ब्लाइट का पता चला\n\nउपचार योजना:\n1. सभी संक्रमित पत्तियों को तुरंत हटाएं\n2. हर 7-10 दिनों में कॉपर-आधारित फफूंदनाशक लगाएं\n3. छंटाई द्वारा वायु संचार में सुधार करें\n4. केवल आधार पर पानी दें - पत्तियों को गीला करने से बचें\n5. मिट्टी के छींटे रोकने के लिए पौधों के आसपास गीली घास डालें\n\nरोकथाम: अगले सीजन में प्रतिरोधी किस्मों का उपयोग करें।",
                'ur': "⚠️ ارلی بلائٹ کی نشاندہی\n\nعلاج کا منصوبہ:\n1. تمام متاثرہ پتوں کو فوری طور پر ہٹا دیں\n2. ہر 7-10 دن بعد کاپر پر مبنی فنگسائڈ لگائیں\n3. کٹائی کے ذریعے ہوا کی گردش بہتر بنائیں\n4. صرف بنیاد پر پانی دیں - پتوں کو گیلا کرنے سے گریز کریں\n5. مٹی کے چھینٹے روکنے کے لیے پودوں کے ارد گرد ملچ ڈالیں\n\nبچاؤ: اگلے سیزن میں مزاحم اقسام استعمال کریں۔"
            },
            'Late_blight': {
                'en': "🚨 LATE BLIGHT DETECTED - SEVERE\n\nIMMEDIATE ACTION REQUIRED:\n1. Remove and destroy infected plants immediately (burn or bag and dispose)\n2. DO NOT compost infected material\n3. Apply chlorothalonil or mancozeb fungicide urgently\n4. Maintain 7-day spray schedule for 3 weeks\n5. Remove all plant debris after harvest\n\nNext season: Practice 3-4 year crop rotation and use resistant varieties.",
                'hi': "🚨 लेट ब्लाइट का पता चला - गंभीर\n\nतत्काल कार्रवाई आवश्यक:\n1. संक्रमित पौधों को तुरंत हटाएं और नष्ट करें (जलाएं या बैग करके फेंकें)\n2. संक्रमित सामग्री को खाद में न डालें\n3. तत्काल क्लोरोथैलोनिल या मैन्कोजेब फफूंदनाशक लगाएं\n4. 3 सप्ताह तक 7-दिवसीय स्प्रे शेड्यूल बनाए रखें\n5. कटाई के बाद सभी पौधों के मलबे को हटा दें\n\nअगला सीजन: 3-4 साल की फसल चक्र अपनाएं और प्रतिरोधी किस्मों का उपयोग करें।",
                'ur': "🚨 لیٹ بلائٹ کی نشاندہی - شدید\n\nفوری کارروائی ضروری:\n1. متاثرہ پودوں کو فوری طور پر ہٹا کر تلف کریں (جلائیں یا بیگ کر کے پھینک دیں)\n2. متاثرہ مواد کو کھاد میں استعمال نہ کریں\n3. فوری طور پر کلوروتھالونیل یا مینکوزیب فنگسائڈ لگائیں\n4. 3 ہفتوں تک 7 روزہ سپرے شیڈول برقرار رکھیں\n5. کٹائی کے بعد تمام پودوں کا ملبہ ہٹا دیں\n\nاگلا سیزن: 3-4 سال کی فصل کی گردش کریں اور مزاحم اقسام استعمال کریں۔"
            },
            'Leaf_mold': {
                'en': "🍃 LEAF MOLD DETECTED\n\nTREATMENT PLAN:\n1. Reduce humidity immediately (improve ventilation)\n2. Remove severely infected leaves\n3. Apply sulfur or copper-based fungicide\n4. Water at base only, early morning\n5. Increase plant spacing for better airflow\n\nPrevention: Avoid overhead irrigation and maintain lower humidity.",
                'hi': "🍃 लीफ मोल्ड का पता चला\n\nउपचार योजना:\n1. तुरंत नमी कम करें (वेंटिलेशन में सुधार करें)\n2. गंभीर रूप से संक्रमित पत्तियों को हटाएं\n3. सल्फर या कॉपर-आधारित फफूंदनाशक लगाएं\n4. केवल आधार पर, सुबह जल्दी पानी दें\n5. बेहतर वायुप्रवाह के लिए पौधों की दूरी बढ़ाएं\n\nरोकथाम: ऊपर से सिंचाई से बचें और कम नमी बनाए रखें।",
                'ur': "🍃 لیف مولڈ کی نشاندہی\n\nعلاج کا منصوبہ:\n1. فوری طور پر نمی کم کریں (وینٹیلیشن بہتر بنائیں)\n2. شدید متاثرہ پتوں کو ہٹا دیں\n3. سلفر یا کاپر پر مبنی فنگسائڈ لگائیں\n4. صرف بنیاد پر، صبح سویرے پانی دیں\n5. بہتر ہوا کے بہاؤ کے لیے پودوں کا فاصلہ بڑھائیں\n\nبچاؤ: اوپر سے آبپاشی سے گریز کریں اور کم نمی برقرار رکھیں۔"
            },
            'Bacterial_spot': {
                'en': "🦠 BACTERIAL SPOT DETECTED\n\nTREATMENT PLAN:\n1. Remove infected leaves immediately\n2. Apply copper-based bactericide\n3. Avoid working with plants when wet\n4. Use disease-free seeds next season\n5. Disinfect gardening tools between plants\n\nNote: Antibiotic treatments are not recommended for home gardens.",
                'hi': "🦠 बैक्टीरियल स्पॉट का पता चला\n\nउपचार योजना:\n1. संक्रमित पत्तियों को तुरंत हटाएं\n2. कॉपर-आधारित बैक्टीरिसाइड लगाएं\n3. गीले मौसम में पौधों के साथ काम करने से बचें\n4. अगले सीजन में रोग-मुक्त बीजों का उपयोग करें\n5. पौधों के बीच बागवानी उपकरणों को कीटाणुरहित करें\n\nनोट: एंटीबायोटिक उपचार घरेलू बगीचों के लिए अनुशंसित नहीं हैं।",
                'ur': "🦠 بیکٹیریل سپاٹ کی نشاندہی\n\nعلاج کا منصوبہ:\n1. متاثرہ پتوں کو فوری طور پر ہٹا دیں\n2. کاپر پر مبنی بیکٹیرسائڈ لگائیں\n3. گیلی حالت میں پودوں کے ساتھ کام کرنے سے گریز کریں\n4. اگلے سیزن میں بیماری سے پاک بیج استعمال کریں\n5. پودوں کے درمیان باغبانی کے اوزار جراثیم سے پاک کریں\n\nنوٹ: اینٹی بائیوٹک علاج گھریلو باغات کے لیے تجویز نہیں کیے جاتے۔"
            },
            'Yellow_leaf_curl_virus': {
                'en': "🟡 YELLOW LEAF CURL VIRUS DETECTED - NO CURE\n\nIMMEDIATE ACTIONS:\n1. Remove and destroy infected plants immediately (burn or deep bury)\n2. Control whitefly population (use yellow sticky traps)\n3. Apply neem oil to repel whiteflies\n4. Use virus-resistant varieties next season\n5. Install insect screens in greenhouse/garden\n\nNote: This virus is incurable. Prevention is the only solution.",
                'hi': "🟡 येलो लीफ कर्ल वायरस का पता चला - कोई इलाज नहीं\n\nतत्काल कार्रवाई:\n1. संक्रमित पौधों को तुरंत हटाएं और नष्ट करें (जलाएं या गहराई से दबाएं)\n2. सफेद मक्खी की आबादी को नियंत्रित करें (पीले चिपचिपे जाल का उपयोग करें)\n3. सफेद मक्खियों को भगाने के लिए नीम का तेल लगाएं\n4. अगले सीजन में वायरस-प्रतिरोधी किस्मों का उपयोग करें\n5. ग्रीनहाउस/बगीचे में कीट स्क्रीन लगाएं\n\nनोट: यह वायरस लाइलाज है। रोकथाम ही एकमात्र समाधान है।",
                'ur': "🟡 یلو لیف کرل وائرس کی نشاندہی - کوئی علاج نہیں\n\nفوری کارروائی:\n1. متاثرہ پودوں کو فوری طور پر ہٹا کر تلف کریں (جلائیں یا گہرائی میں دفن کریں)\n2. سفید مکھی کی آبادی کو کنٹرول کریں (پیلے چپچپے جال استعمال کریں)\n3. سفید مکھیوں کو بھگانے کے لیے نیم کا تیل لگائیں\n4. اگلے سیزن میں وائرس سے مزاحم اقسام استعمال کریں\n5. گرین ہاؤس/باغ میں کیڑوں کی اسکرینیں لگائیں\n\nنوٹ: یہ وائرس لاعلاج ہے۔ روک تھام ہی واحد حل ہے۔"
            },
            'Mosaic_virus': {
                'en': "🟢 MOSAIC VIRUS DETECTED - NO CURE\n\nIMMEDIATE ACTIONS:\n1. Remove and destroy infected plants immediately\n2. Control aphid population (main virus vectors)\n3. Disinfect gardening tools between plants (bleach solution)\n4. DO NOT save seeds from infected plants\n5. Wash hands thoroughly after handling infected plants\n\nLong-term: Use resistant varieties and control insect vectors.",
                'hi': "🟢 मोज़ेक वायरस का पता चला - कोई इलाज नहीं\n\nतत्काल कार्रवाई:\n1. संक्रमित पौधों को तुरंत हटाएं और नष्ट करें\n2. एफिड आबादी को नियंत्रित करें (मुख्य वायरस वाहक)\n3. पौधों के बीच बागवानी उपकरणों को कीटाणुरहित करें (ब्लीच घोल)\n4. संक्रमित पौधों से बीज न बचाएं\n5. संक्रमित पौधों को छूने के बाद हाथों को अच्छी तरह धोएं\n\nदीर्घकालिक: प्रतिरोधी किस्मों का उपयोग करें और कीट वाहकों को नियंत्रित करें।",
                'ur': "🟢 موزیک وائرس کی نشاندہی - کوئی علاج نہیں\n\nفوری کارروائی:\n1. متاثرہ پودوں کو فوری طور پر ہٹا کر تلف کریں\n2. ایفڈ آبادی کو کنٹرول کریں (اہم وائرس ویکٹر)\n3. پودوں کے درمیان باغبانی کے اوزار جراثیم سے پاک کریں (بلیچ حل)\n4. متاثرہ پودوں سے بیج محفوظ نہ کریں\n5. متاثرہ پودوں کو چھونے کے بعد ہاتھوں کو اچھی طرح دھوئیں\n\nطویل مدتی: مزاحم اقسام استعمال کریں اور کیڑے ویکٹر کو کنٹرول کریں۔"
            }
        }
        
        if disease in treatments:
            return treatments[disease].get(language, treatments[disease]['en'])
        return "Consult local agricultural expert for proper diagnosis and treatment."


def predict_batch(image_dir, predictor, language='en'):
    """Predict diseases for all images in a directory"""
    results = []
    
    if not os.path.exists(image_dir):
        return [{'status': 'error', 'message': f'Directory not found: {image_dir}'}]
    
    for image_file in os.listdir(image_dir):
        if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_dir, image_file)
            result = predictor.predict(image_path, language)
            result['image_file'] = image_file
            results.append(result)
    
    return results


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("Disease Predictor - Test Mode")
    print("=" * 60)
    
    # Check if model exists
    model_path = os.path.join(os.path.dirname(__file__), 'best_model.pth')
    
    if not os.path.exists(model_path):
        print(f"\n Model not found at: {model_path}")
        print("\nPlease ensure best_model.pth exists in this folder")
    else:
        try:
            # Initialize predictor
            predictor = DiseasePredictor(device='cpu')
            
            # Test with user input
            test_image = input("\nEnter path to test image: ").strip()
            if os.path.exists(test_image):
                result = predictor.predict(test_image, language='en')
                
                print("\n" + "=" * 60)
                print("PREDICTION RESULT")
                print("=" * 60)
                
                if result['success']:
                    print(f"\n Disease: {result['disease']}")
                    print(f" Confidence: {result['confidence']:.1%}")
                    print(f"\n Treatment:\n{result['treatment']}")
                    
                    if result.get('top3_predictions'):
                        print("\n Top predictions:")
                        for i, pred in enumerate(result['top3_predictions'], 1):
                            print(f"   {i}. {pred['class']}: {pred['confidence']:.1%}")
                else:
                    print(f"\n {result.get('message', 'Prediction failed')}")
            else:
                print(f"\n Image not found: {test_image}")
                
        except Exception as e:
            print(f"\n Error: {str(e)}")