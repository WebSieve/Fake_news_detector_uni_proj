"""
Advanced Model Training Module for Fake News Detection
Production-ready training pipeline with RoBERTa fine-tuning
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
    EarlyStoppingCallback, TrainingArguments, Trainer
)
try:
    from transformers import AdamW
except ImportError:
    from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import Config
from utils import setup_logging, save_json, load_json, create_dir
from data_preparation import DatasetLoader


class FakeNewsDataset(Dataset):
    """Custom dataset class for fake news detection"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class AdvancedModelTrainer:
    """Advanced training pipeline for fake news detection"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging('model_training')
        
        # Initialize tokenizer and model
        self.tokenizer = RobertaTokenizer.from_pretrained(config.MODEL_NAME)
        self.model = None
        
        # Training metrics
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }
    
    def create_model(self) -> RobertaForSequenceClassification:
        """Create and configure the RoBERTa model"""
        self.logger.info(f"Creating model: {self.config.MODEL_NAME}")
        
        model = RobertaForSequenceClassification.from_pretrained(
            self.config.MODEL_NAME,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False
        )
        
        # Move to device
        model = model.to(self.config.DEVICE)
        
        return model
    
    def prepare_data(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
        """Prepare data loaders for training"""
        self.logger.info("Preparing data loaders...")
        
        # Create datasets
        train_dataset = FakeNewsDataset(
            train_data['text'].tolist(),
            train_data['label'].tolist(),
            self.tokenizer,
            self.config.MAX_LENGTH
        )
        
        val_dataset = FakeNewsDataset(
            val_data['text'].tolist(),
            val_data['label'].tolist(),
            self.tokenizer,
            self.config.MAX_LENGTH
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=0  # Set to 0 for Windows compatibility
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        
        self.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.info(f"Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, optimizer, scheduler) -> float:
        """Train model for one epoch"""
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.config.DEVICE)
            attention_mask = batch['attention_mask'].to(self.config.DEVICE)
            labels = batch['labels'].to(self.config.DEVICE)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(train_loader)
    
    def evaluate(self, model: nn.Module, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set"""
        model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating", leave=False):
                input_ids = batch['input_ids'].to(self.config.DEVICE)
                attention_mask = batch['attention_mask'].to(self.config.DEVICE)
                labels = batch['labels'].to(self.config.DEVICE)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                # Get predictions
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict[str, float]:
        """Complete training pipeline"""
        self.logger.info("Starting training pipeline...")
        
        # Create model
        self.model = self.create_model()
        
        # Prepare data
        train_loader, val_loader = self.prepare_data(train_data, val_data)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            eps=1e-8
        )
        
        total_steps = len(train_loader) * self.config.NUM_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training loop
        best_f1 = 0
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(self.config.NUM_EPOCHS):
            self.logger.info(f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS}")
            
            # Train
            train_loss = self.train_epoch(self.model, train_loader, optimizer, scheduler)
            
            # Evaluate
            val_metrics = self.evaluate(self.model, val_loader)
            
            # Log metrics
            self.logger.info(f"Train Loss: {train_loss:.4f}")
            self.logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            self.logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            self.logger.info(f"Val F1: {val_metrics['f1']:.4f}")
            
            # Save metrics
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['val_accuracy'].append(val_metrics['accuracy'])
            self.training_history['val_f1'].append(val_metrics['f1'])
            
            # Early stopping and model saving
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                
                # Save best model
                self.save_model(self.model, f"best_model_f1_{best_f1:.4f}")
                self.logger.info(f"New best model saved with F1: {best_f1:.4f}")
            else:
                patience_counter += 1
                
            if patience_counter >= 3:  # Early stopping patience
                self.logger.info(f"Early stopping triggered. Best F1: {best_f1:.4f}")
                break
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        return {
            'best_f1': best_f1,
            'final_accuracy': val_metrics['accuracy'],
            'final_precision': val_metrics['precision'],
            'final_recall': val_metrics['recall']
        }
    
    def save_model(self, model: nn.Module, name: str = "final_model"):
        """Save model and tokenizer"""
        model_path = self.config.MODEL_SAVE_PATH / name
        create_dir(model_path)
        
        # Save model
        model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        # Save training history
        save_json(self.training_history, model_path / "training_history.json")
        
        # Save config
        config_dict = {
            'model_name': self.config.MODEL_NAME,
            'max_length': self.config.MAX_LENGTH,
            'batch_size': self.config.BATCH_SIZE,
            'learning_rate': self.config.LEARNING_RATE,
            'epochs': self.config.NUM_EPOCHS
        }
        save_json(config_dict, model_path / "model_config.json")
        
        self.logger.info(f"Model saved to: {model_path}")
    
    def get_detailed_report(self, test_data: pd.DataFrame) -> Dict:
        """Generate detailed evaluation report"""
        if not self.model:
            raise ValueError("No trained model found. Train model first.")
        
        self.logger.info("Generating detailed evaluation report...")
        
        # Prepare test data
        test_dataset = FakeNewsDataset(
            test_data['text'].tolist(),
            test_data['label'].tolist(),
            self.tokenizer,
            self.config.MAX_LENGTH
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        
        # Get predictions
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                input_ids = batch['input_ids'].to(self.config.DEVICE)
                attention_mask = batch['attention_mask'].to(self.config.DEVICE)
                labels = batch['labels'].to(self.config.DEVICE)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # Classification report
        class_report = classification_report(
            all_labels, all_predictions,
            target_names=['REAL', 'FAKE'],
            output_dict=True
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'classification_report': class_report,
            'predictions': all_predictions,
            'true_labels': all_labels,
            'probabilities': all_probabilities
        }


def main():
    """Main training function"""
    config = Config()
    
    # Setup logging
    logger = setup_logging('main_training')
    logger.info("Starting fake news detection training...")
    
    try:
        # Load data
        logger.info("Loading training data...")
        data_loader = DatasetLoader()
        
        # For this example, we'll use the sample data
        # In production, use: train_data, val_data, test_data = data_loader.load_full_dataset()
        sample_data = data_loader.create_sample_dataset(n_samples=5000)
        
        # Split data
        train_size = int(0.7 * len(sample_data))
        val_size = int(0.2 * len(sample_data))
        
        train_data = sample_data[:train_size].reset_index(drop=True)
        val_data = sample_data[train_size:train_size + val_size].reset_index(drop=True)
        test_data = sample_data[train_size + val_size:].reset_index(drop=True)
        
        logger.info(f"Training data: {len(train_data)} samples")
        logger.info(f"Validation data: {len(val_data)} samples")
        logger.info(f"Test data: {len(test_data)} samples")
        
        # Initialize trainer
        trainer = AdvancedModelTrainer(config)
        
        # Train model
        results = trainer.train(train_data, val_data)
        
        # Generate detailed report on test set
        test_report = trainer.get_detailed_report(test_data)
        
        # Log final results
        logger.info("Training completed!")
        logger.info(f"Best F1 Score: {results['best_f1']:.4f}")
        logger.info(f"Test Accuracy: {test_report['accuracy']:.4f}")
        logger.info(f"Test F1 Score: {test_report['f1']:.4f}")
        
        # Save final results
        final_results = {
            'training_results': results,
            'test_results': test_report,
            'training_history': trainer.training_history
        }
        
        results_path = config.MODEL_SAVE_PATH / "training_results.json"
        save_json(final_results, results_path)
        
        if test_report['accuracy'] >= 0.95:
            logger.info("SUCCESS: Model achieved target accuracy of 95%+!")
        else:
            logger.info(f"Model accuracy: {test_report['accuracy']:.4f} - Consider hyperparameter tuning")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()