
"""
Configuration file for Fake News Detection Project
"""
import torch
import os
from pathlib import Path

class Config:
    """Main configuration class"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    MODEL_SAVE_PATH = MODELS_DIR  # Path where trained models are saved
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # Model configuration
    MODEL_NAME = "roberta-base"
    PRETRAINED_MODEL = "jy46604790/Fake-News-Bert-Detect"  # Advanced pretrained model
    USE_PRETRAINED = True  # Set to True to use pretrained model by default
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    
    # Data configuration
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    RANDOM_SEED = 42
    
    # Training configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4 if torch.cuda.is_available() else 2
    GRADIENT_ACCUMULATION_STEPS = 2
    MAX_GRAD_NORM = 1.0
    
    # Evaluation metrics
    METRICS = ["accuracy", "precision", "recall", "f1"]
    
    # App configuration
    APP_TITLE = "AI-Powered Fake News Detector"
    APP_DESCRIPTION = """
    This tool uses a PRETRAINED RoBERTa model trained on 40,000+ real news articles.
    Simply paste a news article and get an instant, accurate prediction!
    """
    
    # Pretrained model options
    AVAILABLE_PRETRAINED_MODELS = {
        "jy46604790": "jy46604790/Fake-News-Bert-Detect",  # 40K+ samples, most popular
        "hamzab": "hamzab/roberta-fake-news-classification"  # 100% accuracy on benchmark
    }
    
    # Dataset URLs
    FAKENEWSNET_URL = "https://github.com/KaiDMML/FakeNewsNet"
    LIAR_URL = "https://huggingface.co/datasets/liar"
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("Current Configuration:")
        print(f"   Model: {cls.MODEL_NAME}")
        print(f"   Pretrained Model: {cls.PRETRAINED_MODEL}")
        print(f"   Use Pretrained: {cls.USE_PRETRAINED}")
        print(f"   Device: {cls.DEVICE}")
        print(f"   Batch Size: {cls.BATCH_SIZE}")
        print(f"   Learning Rate: {cls.LEARNING_RATE}")
        print(f"   Max Length: {cls.MAX_LENGTH}")
        print(f"   Epochs: {cls.NUM_EPOCHS}")
    
    @classmethod
    def create_directories(cls):
        """Create necessary project directories"""
        for dir_path in [cls.DATA_DIR, cls.MODELS_DIR, cls.LOGS_DIR]:
            dir_path.mkdir(exist_ok=True)
