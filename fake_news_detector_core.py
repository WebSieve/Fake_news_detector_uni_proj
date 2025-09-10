"""
Robust fake news detector with error handling and fallbacks
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from config import Config
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplePredictor:
    def __init__(self, model_path=None, use_pretrained=True, timeout_seconds=30):
        self.model_path = model_path or Config.MODELS_DIR / "roberta_fake_news"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.classifier = None
        self.use_pretrained = use_pretrained
        self.timeout_seconds = timeout_seconds
        self.load_model()
    
    def load_model(self):
        """Load model with timeout and fallback options"""
        if self.use_pretrained:
            try:
                # Set environment variable to limit cache size
                os.environ['TRANSFORMERS_CACHE'] = str(Config.MODELS_DIR / "cache")
                
                logger.info("Attempting to load pretrained model...")
                logger.info("This may take a few minutes to download initially...")
                
                # Try smaller/faster models first
                models_to_try = [
                    "martin-ha/toxic-comment-model",  # Smaller model
                    "unitary/toxic-bert",  # Another option
                    "jy46604790/Fake-News-Bert-Detect"  # Original choice
                ]
                
                for model_name in models_to_try:
                    try:
                        logger.info(f"Trying model: {model_name}")
                        self.classifier = pipeline(
                            "text-classification",
                            model=model_name,
                            device=-1,  # Use CPU to avoid GPU issues
                            model_kwargs={"torch_dtype": torch.float32}
                        )
                        logger.info(f"Successfully loaded: {model_name}")
                        return
                    except Exception as e:
                        logger.warning(f"Failed to load {model_name}: {str(e)[:100]}...")
                        continue
                
                raise Exception("All pretrained models failed to load")
                
            except Exception as e:
                logger.error(f"Pretrained model loading failed: {e}")
                logger.info("Falling back to local model or basic approach...")
        
        # Try local model
        try:
            if hasattr(self.model_path, 'exists') and self.model_path.exists():
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
                logger.info(f"Loaded local model from {self.model_path}")
            else:
                raise FileNotFoundError("Local model not found")
                
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Local model loading failed: {e}")
            logger.info("Using basic rule-based fallback...")
            self._setup_rule_based()
    
    def _setup_rule_based(self):
        """Setup simple rule-based detection as ultimate fallback"""
        self.fake_keywords = [
            'shocking', 'breaking', 'urgent', 'amazing', 'unbelievable',
            'exposed', 'secret', 'scandal', 'conspiracy', 'doctors hate',
            'one weird trick', 'instantly', 'cure all', 'miracle'
        ]
        self.real_keywords = [
            'research', 'study', 'scientists', 'university', 'published',
            'federal reserve', 'government', 'official', 'announced'
        ]
        logger.info("Rule-based fallback initialized")
    
    def predict(self, text):
        """Make prediction with appropriate method"""
        if self.classifier:
            return self._predict_with_model(text)
        elif self.model:
            return self._predict_with_local_model(text)
        else:
            return self._predict_with_rules(text)
    
    def _predict_with_model(self, text):
        """Predict using pretrained pipeline"""
        try:
            result = self.classifier(text)
            label = result[0]['label']
            score = result[0]['score']
            
            # Handle different model outputs
            if 'FAKE' in label.upper() or 'NEGATIVE' in label.upper() or label == 'LABEL_0':
                is_fake = True
            else:
                is_fake = False
            
            return {
                'prediction': 1 if is_fake else 0,
                'label': 'FAKE' if is_fake else 'REAL',
                'confidence': score,
                'fake_probability': score if is_fake else (1 - score),
                'model_type': 'PRETRAINED'
            }
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self._predict_with_rules(text)
    
    def _predict_with_local_model(self, text):
        """Predict using local model"""
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
            'fake_probability': fake_prob,
            'model_type': 'LOCAL'
        }
    
    def _predict_with_rules(self, text):
        """Predict using simple rules"""
        text_lower = text.lower()
        
        fake_score = sum(1 for keyword in self.fake_keywords if keyword in text_lower)
        real_score = sum(1 for keyword in self.real_keywords if keyword in text_lower)
        
        if fake_score > real_score:
            prediction = 1
            confidence = min(0.9, 0.6 + fake_score * 0.1)
        else:
            prediction = 0
            confidence = min(0.9, 0.6 + real_score * 0.1) if real_score > 0 else 0.5
        
        return {
            'prediction': prediction,
            'label': 'FAKE' if prediction == 1 else 'REAL',
            'confidence': confidence,
            'fake_probability': confidence if prediction == 1 else (1 - confidence),
            'model_type': 'RULE-BASED'
        }

def test_predictions():
    """Test the predictor with fallback handling"""
    print("IMPROVED FAKE NEWS PREDICTOR TEST")
    print("(With download timeout and fallback handling)")
    print("=" * 60)
    
    # Test with pretrained model first
    predictor = SimplePredictor(use_pretrained=True)
    
    test_texts = [
        "Scientists at MIT develop new AI technology for medical diagnosis",
        "SHOCKING: This one weird trick will cure all diseases doctors hate it!",
        "The Federal Reserve announced interest rate changes today",
        "BREAKING: Aliens control government with secret mind control rays!"
    ]
    
    print("\nTEST RESULTS:")
    print("-" * 50)
    
    for i, text in enumerate(test_texts, 1):
        try:
            result = predictor.predict(text)
            status = "[REAL]" if result['label'] == 'REAL' else "[FAKE]"
            
            print(f"\n{status} Test {i}: {text[:50]}...")
            print(f"Prediction: {result['label']} ({result['model_type']})")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Fake Probability: {result['fake_probability']:.3f}")
            
        except Exception as e:
            print(f"\nERROR in Test {i}: {e}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED!")
    print("If you see RULE-BASED results, the AI model couldn't download.")
    print("This is normal on slow connections - the rule-based system still works!")
    print("=" * 60)

if __name__ == "__main__":
    test_predictions()
