"""
Advanced Evaluation Module for Fake News Detection
Comprehensive model evaluation with visualizations and analysis
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, accuracy_score, f1_score
)
from sklearn.preprocessing import label_binarize
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import Config
from utils import setup_logging, load_json, create_dir
from simple_inference import SimplePredictor


class AdvancedEvaluator:
    """Comprehensive evaluation system for fake news detection"""
    
    def __init__(self, model_path: str, config: Config):
        self.config = config
        self.model_path = Path(model_path)
        self.logger = setup_logging('evaluation')
        
        # Load model and tokenizer
        self.predictor = SimplePredictor(str(self.model_path))
        
        # Results storage
        self.evaluation_results = {}
        self.plots_dir = config.PROJECT_ROOT / "evaluation_plots"
        create_dir(self.plots_dir)
    
    def evaluate_model(self, test_data: pd.DataFrame) -> Dict:
        """Comprehensive model evaluation"""
        self.logger.info("Starting comprehensive evaluation...")
        
        # Get predictions
        predictions = []
        probabilities = []
        true_labels = test_data['label'].tolist()
        
        self.logger.info("Generating predictions...")
        for text in test_data['text']:
            result = self.predictor.predict(text)
            predictions.append(1 if result['label'] == 'FAKE' else 0)
            probabilities.append(result['confidence'])
        
        # Calculate basic metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')
        
        # Generate classification report
        class_report = classification_report(
            true_labels, predictions,
            target_names=['REAL', 'FAKE'],
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # ROC curve data
        fpr, tpr, _ = roc_curve(true_labels, probabilities)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(true_labels, probabilities)
        pr_auc = auc(recall, precision)
        
        self.evaluation_results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'predictions': predictions,
            'probabilities': probabilities,
            'true_labels': true_labels,
            'roc_data': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
            'pr_data': {'precision': precision.tolist(), 'recall': recall.tolist()}
        }
        
        self.logger.info(f"Evaluation complete - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        return self.evaluation_results
    
    def create_confusion_matrix_plot(self):
        """Create confusion matrix visualization"""
        cm = np.array(self.evaluation_results['confusion_matrix'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['REAL', 'FAKE'],
            yticklabels=['REAL', 'FAKE'],
            cbar_kws={'label': 'Number of Samples'}
        )
        plt.title('Confusion Matrix - Fake News Detection', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        # Add percentage annotations
        total = cm.sum()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                percentage = cm[i, j] / total * 100
                plt.text(j + 0.5, i + 0.3, f'({percentage:.1f}%)', 
                        ha='center', va='center', fontsize=10, color='red')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Confusion matrix plot saved")
    
    def create_roc_curve_plot(self):
        """Create ROC curve visualization"""
        roc_data = self.evaluation_results['roc_data']
        fpr = roc_data['fpr']
        tpr = roc_data['tpr']
        roc_auc = self.evaluation_results['roc_auc']
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Fake News Detection', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("ROC curve plot saved")
    
    def create_precision_recall_plot(self):
        """Create Precision-Recall curve visualization"""
        pr_data = self.evaluation_results['pr_data']
        precision = pr_data['precision']
        recall = pr_data['recall']
        pr_auc = self.evaluation_results['pr_auc']
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve - Fake News Detection', fontsize=16, fontweight='bold')
        plt.legend(loc="lower left", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Precision-Recall curve plot saved")
    
    def create_metrics_summary_plot(self):
        """Create metrics summary visualization"""
        class_report = self.evaluation_results['classification_report']
        
        # Extract metrics for both classes
        metrics = ['precision', 'recall', 'f1-score']
        real_scores = [class_report['REAL'][metric] for metric in metrics]
        fake_scores = [class_report['FAKE'][metric] for metric in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars1 = ax.bar(x - width/2, real_scores, width, label='REAL News', color='lightgreen', alpha=0.8)
        bars2 = ax.bar(x + width/2, fake_scores, width, label='FAKE News', color='lightcoral', alpha=0.8)
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Classification Metrics by Class', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.title() for m in metrics])
        ax.legend()
        ax.set_ylim([0, 1.0])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        autolabel(bars1)
        autolabel(bars2)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'metrics_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Metrics summary plot saved")
    
    def create_confidence_distribution_plot(self):
        """Create confidence score distribution plot"""
        probabilities = self.evaluation_results['probabilities']
        predictions = self.evaluation_results['predictions']
        true_labels = self.evaluation_results['true_labels']
        
        # Separate by correctness
        correct_preds = []
        incorrect_preds = []
        
        for i, (pred, true_label, prob) in enumerate(zip(predictions, true_labels, probabilities)):
            if pred == true_label:
                correct_preds.append(prob)
            else:
                incorrect_preds.append(prob)
        
        plt.figure(figsize=(12, 8))
        plt.hist(correct_preds, bins=50, alpha=0.7, label='Correct Predictions', 
                color='green', density=True)
        plt.hist(incorrect_preds, bins=50, alpha=0.7, label='Incorrect Predictions', 
                color='red', density=True)
        
        plt.xlabel('Confidence Score', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Distribution of Confidence Scores', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Confidence distribution plot saved")
    
    def create_interactive_dashboard(self, test_data: pd.DataFrame):
        """Create interactive Plotly dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Confusion Matrix', 'ROC Curve', 'Class Distribution', 'Confidence Scores'),
            specs=[[{"type": "heatmap"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )
        
        # Confusion matrix heatmap
        cm = np.array(self.evaluation_results['confusion_matrix'])
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=['REAL', 'FAKE'],
                y=['REAL', 'FAKE'],
                colorscale='Blues',
                showscale=False,
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16}
            ),
            row=1, col=1
        )
        
        # ROC Curve
        roc_data = self.evaluation_results['roc_data']
        fig.add_trace(
            go.Scatter(
                x=roc_data['fpr'],
                y=roc_data['tpr'],
                mode='lines',
                name=f"ROC (AUC={self.evaluation_results['roc_auc']:.3f})",
                line=dict(color='orange', width=3)
            ),
            row=1, col=2
        )
        
        # Random line for ROC
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(color='navy', width=2, dash='dash')
            ),
            row=1, col=2
        )
        
        # Class distribution
        class_counts = test_data['label'].value_counts()
        fig.add_trace(
            go.Bar(
                x=['REAL', 'FAKE'],
                y=[class_counts[0], class_counts[1]],
                marker_color=['lightgreen', 'lightcoral'],
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Confidence distribution
        probabilities = self.evaluation_results['probabilities']
        fig.add_trace(
            go.Histogram(
                x=probabilities,
                nbinsx=30,
                marker_color='lightblue',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Fake News Detection - Model Evaluation Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        # Save interactive plot
        fig.write_html(self.plots_dir / 'interactive_dashboard.html')
        self.logger.info("Interactive dashboard saved")
    
    def analyze_misclassified_samples(self, test_data: pd.DataFrame, n_samples: int = 10):
        """Analyze misclassified samples"""
        predictions = self.evaluation_results['predictions']
        true_labels = self.evaluation_results['true_labels']
        probabilities = self.evaluation_results['probabilities']
        
        # Find misclassified samples
        misclassified_indices = []
        for i, (pred, true_label) in enumerate(zip(predictions, true_labels)):
            if pred != true_label:
                misclassified_indices.append(i)
        
        if not misclassified_indices:
            self.logger.info("No misclassified samples found!")
            return
        
        # Sort by confidence (most confident wrong predictions first)
        misclassified_data = []
        for idx in misclassified_indices:
            misclassified_data.append({
                'index': idx,
                'text': test_data.iloc[idx]['text'],
                'true_label': 'FAKE' if true_labels[idx] == 1 else 'REAL',
                'predicted_label': 'FAKE' if predictions[idx] == 1 else 'REAL',
                'confidence': probabilities[idx]
            })
        
        # Sort by confidence descending
        misclassified_data.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Save analysis
        analysis_results = {
            'total_misclassified': len(misclassified_indices),
            'misclassification_rate': len(misclassified_indices) / len(predictions),
            'samples': misclassified_data[:n_samples]
        }
        
        # Save to file
        with open(self.plots_dir / 'misclassified_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Misclassified analysis saved. Total errors: {len(misclassified_indices)}")
    
    def generate_word_clouds(self, test_data: pd.DataFrame):
        """Generate word clouds for real vs fake news"""
        try:
            # Separate real and fake news
            real_news = test_data[test_data['label'] == 0]['text'].str.cat(sep=' ')
            fake_news = test_data[test_data['label'] == 1]['text'].str.cat(sep=' ')
            
            # Create word clouds
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            
            # Real news word cloud
            wordcloud_real = WordCloud(
                width=800, height=400,
                background_color='white',
                colormap='Greens',
                max_words=100
            ).generate(real_news)
            
            ax1.imshow(wordcloud_real, interpolation='bilinear')
            ax1.set_title('Real News - Most Common Words', fontsize=16, fontweight='bold')
            ax1.axis('off')
            
            # Fake news word cloud
            wordcloud_fake = WordCloud(
                width=800, height=400,
                background_color='white',
                colormap='Reds',
                max_words=100
            ).generate(fake_news)
            
            ax2.imshow(wordcloud_fake, interpolation='bilinear')
            ax2.set_title('Fake News - Most Common Words', fontsize=16, fontweight='bold')
            ax2.axis('off')
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'word_clouds.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("Word clouds generated")
            
        except ImportError:
            self.logger.warning("WordCloud not available. Skipping word cloud generation.")
        except Exception as e:
            self.logger.error(f"Error generating word clouds: {str(e)}")
    
    def generate_full_report(self, test_data: pd.DataFrame) -> str:
        """Generate comprehensive evaluation report"""
        self.logger.info("Generating comprehensive evaluation report...")
        
        # Run evaluation
        results = self.evaluate_model(test_data)
        
        # Create all visualizations
        self.create_confusion_matrix_plot()
        self.create_roc_curve_plot()
        self.create_precision_recall_plot()
        self.create_metrics_summary_plot()
        self.create_confidence_distribution_plot()
        self.create_interactive_dashboard(test_data)
        self.analyze_misclassified_samples(test_data)
        self.generate_word_clouds(test_data)
        
        # Generate text report
        report = f"""
FAKE NEWS DETECTION MODEL - EVALUATION REPORT
=============================================

OVERALL PERFORMANCE:
- Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)
- F1 Score: {results['f1_score']:.4f}
- ROC AUC: {results['roc_auc']:.4f}
- PR AUC: {results['pr_auc']:.4f}

CLASSIFICATION REPORT:
{self._format_classification_report(results['classification_report'])}

CONFUSION MATRIX:
{np.array(results['confusion_matrix'])}

MODEL PERFORMANCE ANALYSIS:
- Total test samples: {len(test_data)}
- Correct predictions: {sum(np.array(results['predictions']) == np.array(results['true_labels']))}
- Misclassified samples: {len(results['predictions']) - sum(np.array(results['predictions']) == np.array(results['true_labels']))}

ACHIEVEMENT STATUS:
{'✅ SUCCESS: Model achieved target accuracy of 96%+!' if results['accuracy'] >= 0.96 else '⚠️  Model accuracy below target. Consider hyperparameter tuning.'}

FILES GENERATED:
- Confusion Matrix: evaluation_plots/confusion_matrix.png
- ROC Curve: evaluation_plots/roc_curve.png
- Precision-Recall Curve: evaluation_plots/precision_recall_curve.png
- Metrics Summary: evaluation_plots/metrics_summary.png
- Confidence Distribution: evaluation_plots/confidence_distribution.png
- Interactive Dashboard: evaluation_plots/interactive_dashboard.html
- Word Clouds: evaluation_plots/word_clouds.png
- Misclassified Analysis: evaluation_plots/misclassified_analysis.json
"""
        
        # Save report
        with open(self.plots_dir / 'evaluation_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Save results as JSON
        with open(self.plots_dir / 'evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info("Full evaluation report generated!")
        return report
    
    def _format_classification_report(self, report_dict: Dict) -> str:
        """Format classification report for text output"""
        formatted = "\n"
        for class_name in ['REAL', 'FAKE']:
            if class_name in report_dict:
                metrics = report_dict[class_name]
                formatted += f"{class_name:>10}: "
                formatted += f"Precision={metrics['precision']:.3f} "
                formatted += f"Recall={metrics['recall']:.3f} "
                formatted += f"F1={metrics['f1-score']:.3f}\n"
        
        # Add overall metrics
        if 'weighted avg' in report_dict:
            metrics = report_dict['weighted avg']
            formatted += f"{'Weighted Avg':>10}: "
            formatted += f"Precision={metrics['precision']:.3f} "
            formatted += f"Recall={metrics['recall']:.3f} "
            formatted += f"F1={metrics['f1-score']:.3f}\n"
        
        return formatted


def main():
    """Main evaluation function"""
    config = Config()
    logger = setup_logging('main_evaluation')
    
    try:
        # Check for trained model
        model_path = config.MODEL_SAVE_PATH / "best_model_f1_0.9000"  # Adjust based on actual model
        if not model_path.exists():
            logger.error(f"No trained model found at {model_path}")
            logger.info("Please train the model first using: python model_training.py")
            return
        
        # Load test data (in production, load from actual test set)
        from data_preparation import DatasetLoader
        data_loader = DatasetLoader()
        sample_data = data_loader.create_sample_dataset(n_samples=1000)
        
        # Use last 200 samples as test set
        test_data = sample_data[-200:].reset_index(drop=True)
        
        logger.info(f"Evaluating model on {len(test_data)} test samples...")
        
        # Initialize evaluator
        evaluator = AdvancedEvaluator(str(model_path), config)
        
        # Generate full report
        report = evaluator.generate_full_report(test_data)
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETED!")
        print("="*60)
        print(report)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()