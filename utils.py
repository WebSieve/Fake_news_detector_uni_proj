
"""
Utility functions for fake news detection project
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

def setup_logging(name='fake_news_detector', log_level=logging.INFO):
    """Setup logging configuration and return logger"""
    
    # Create logs directory
    create_dir('logs')
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/fake_news_detector.log'),
            logging.StreamHandler()
        ]
    )
    
    # Return named logger
    return logging.getLogger(name)

def check_gpu():
    """Check GPU availability"""
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("GPU not available, using CPU")
        return False

def create_directories():
    """Create necessary project directories"""
    dirs = ['data', 'models', 'logs', 'results']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
    print("Project directories created")

def save_results(results, filename):
    """Save results to file"""
    if isinstance(results, dict):
        import json
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
    elif isinstance(results, pd.DataFrame):
        results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def clean_text(text):
    """Clean and preprocess text for model input"""
    import re
    
    if not isinstance(text, str):
        text = str(text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:\-\'\"()]', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Strip and normalize
    text = text.strip()
    
    return text

def save_json(data, filepath):
    """Save data as JSON file"""
    import json
    from pathlib import Path
    
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(filepath):
    """Load data from JSON file"""
    import json
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_dir(path):
    """Create directory if it doesn't exist"""
    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    setup_logging()
    check_gpu()
    create_directories()
