
"""
Simple inference script for quick testing
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplePredictor:
    def __init__(self, model_path=None):
        self.model_path = model_path or Config.MODELS_DIR / "roberta_fake_news"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load model and tokenizer"""
        try:
            # Check if the local model path exists
            if hasattr(self.model_path, 'exists') and self.model_path.exists():
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
                logger.info(f"Loaded trained model from {self.model_path}")
            else:
                # Check if it's a string path that could be a huggingface model
                if isinstance(self.model_path, str) and not self.model_path.startswith('/') and not '\\' in self.model_path:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                    self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
                    logger.info(f"Loaded model from Hugging Face: {self.model_path}")
                else:
                    raise FileNotFoundError("Local model not found")
                    
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Use a pretrained model as fallback
            logger.info("Using pretrained RoBERTa model as fallback")
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            self.model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Pretrained RoBERTa model loaded successfully")
    
    def predict(self, text):
        """Make a prediction"""
        inputs = self.tokenizer(
            text, 
            truncation=True, 
            padding=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][prediction].item()
            fake_prob = probabilities[0][1].item()
        
        return {
            'prediction': prediction,
            'label': 'FAKE' if prediction == 1 else 'REAL',
            'confidence': confidence,
            'fake_probability': fake_prob
        }

def test_predictions():
    """Test the predictor with sample texts"""
    predictor = SimplePredictor()
    
    test_texts = [
        "Scientists at MIT develop new AI technology for medical diagnosis",
        "SHOCKING: This one weird trick will cure all diseases doctors hate it!",
        "The Federal Reserve announced interest rate changes today",
        "BREAKING: Aliens control government with secret mind control rays!"
    ]
    
    print("Testing Fake News Predictor:")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        result = predictor.predict(text)
        print(f"\nTest {i}: {text[:50]}...")
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Fake Probability: {result['fake_probability']:.3f}")

if __name__ == "__main__":
    test_predictions()
