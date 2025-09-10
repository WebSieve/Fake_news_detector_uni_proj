"""
Fake News Detection Web Application
Production-ready Gradio web interface for fake news detection
"""

import os
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from config import Config
from fake_news_inference import SimplePredictor
from utils import setup_logging, clean_text


class FakeNewsWebApp:
    """Production-ready web application for fake news detection"""
    
    def __init__(self):
        self.config = Config()
        self.logger = setup_logging('web_app')
        
        # Initialize predictor
        self.predictor = None
        self.model_loaded = False
        self.load_model()
        
        # Statistics tracking
        self.prediction_history = []
        self.daily_stats = {'total': 0, 'fake': 0, 'real': 0}
    
    def load_model(self):
        """Load the trained model"""
        try:
            # Try pretrained model for best accuracy
            try:
                self.predictor = SimplePredictor(use_pretrained=True)
                self.model_loaded = True
                self.logger.info("Using pretrained model (40K+ training samples)")
                return
            except Exception as e:
                self.logger.warning(f"Pretrained model failed: {e}")
            
            # Try to find the best local model
            model_dir = self.config.MODEL_SAVE_PATH
            if model_dir.exists():
                # Look for best model
                best_models = list(model_dir.glob("best_model_*"))
                if best_models:
                    model_path = str(best_models[0])
                    self.predictor = SimplePredictor(model_path, use_pretrained=False)
                    self.model_loaded = True
                    self.logger.info(f"Model loaded from: {model_path}")
                else:
                    # Fallback to pretrained model
                    self.predictor = SimplePredictor(use_pretrained=True)
                    self.model_loaded = True
                    self.logger.info("Using pretrained model as fallback")
            else:
                # Fallback to pretrained model
                self.predictor = SimplePredictor(use_pretrained=True)
                self.model_loaded = True
                self.logger.info("Using pretrained model (no local model found)")
                
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            self.model_loaded = False
    
    def predict_news(self, text: str) -> Tuple[str, str, str, str]:
        """Main prediction function for Gradio interface"""
        if not self.model_loaded:
            return (
                "❌ Model not loaded",
                "Please check model files",
                "0.00",
                "Unable to process"
            )
        
        if not text or len(text.strip()) < 10:
            return (
                "⚠️ Invalid Input",
                "Please enter at least 10 characters of news text",
                "0.00",
                "Text too short"
            )
        
        try:
            # Clean and validate text
            cleaned_text = clean_text(text)
            
            # Get prediction
            start_time = time.time()
            result = self.predictor.predict(cleaned_text)
            processing_time = time.time() - start_time
            
            # Extract results
            label = result['label']
            confidence = result['confidence']
            
            # Format output
            if label == 'FAKE':
                status = f"🚨 FAKE NEWS DETECTED"
                color_class = "fake-news"
                interpretation = "This article appears to contain misinformation or fake news."
            else:
                status = f"✅ LEGITIMATE NEWS"
                color_class = "real-news"
                interpretation = "This article appears to be legitimate news."
            
            confidence_text = f"{confidence:.2%}"
            processing_info = f"Processed in {processing_time:.2f}s"
            
            # Update statistics
            self.update_stats(label)
            
            # Log prediction
            self.logger.info(f"Prediction: {label} (confidence: {confidence:.3f})")
            
            return status, interpretation, confidence_text, processing_info
            
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            return (
                "❌ Error",
                f"Prediction failed: {str(e)}",
                "0.00",
                "Error occurred"
            )
    
    def update_stats(self, label: str):
        """Update prediction statistics"""
        self.daily_stats['total'] += 1
        if label == 'FAKE':
            self.daily_stats['fake'] += 1
        else:
            self.daily_stats['real'] += 1
        
        # Store in history
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'label': label
        })
        
        # Keep only last 100 predictions
        if len(self.prediction_history) > 100:
            self.prediction_history.pop(0)
    
    def get_stats_plot(self):
        """Generate statistics plot"""
        try:
            if not self.prediction_history:
                # Return empty plot
                fig = go.Figure()
                fig.add_annotation(
                    text="No predictions yet",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font_size=20
                )
                return fig
            
            # Create pie chart of predictions
            labels = ['Real News', 'Fake News']
            values = [self.daily_stats['real'], self.daily_stats['fake']]
            colors = ['#2E8B57', '#DC143C']  # Sea green, Crimson
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                marker_colors=colors,
                textinfo='label+percent',
                textfont_size=14
            )])
            
            fig.update_layout(
                title={
                    'text': f"Today's Predictions ({self.daily_stats['total']} total)",
                    'x': 0.5,
                    'font': {'size': 18, 'family': 'Arial Black'}
                },
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=400,
                margin=dict(t=80, b=40, l=40, r=40)
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating stats plot: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(
                text="Error loading statistics",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def get_recent_predictions(self) -> str:
        """Get recent predictions summary"""
        if not self.prediction_history:
            return "No recent predictions"
        
        # Get last 5 predictions
        recent = self.prediction_history[-5:]
        summary = "🕒 **Recent Predictions:**\n"
        
        for i, pred in enumerate(recent, 1):
            timestamp = pred['timestamp'].strftime("%H:%M:%S")
            label = pred['label']
            emoji = "🚨" if label == 'FAKE' else "✅"
            summary += f"{i}. {timestamp} - {emoji} {label}\n"
        
        return summary
    
    def analyze_batch(self, file) -> Tuple[str, str]:
        """Batch analysis of news articles from file"""
        if file is None:
            return "No file uploaded", ""
        
        try:
            # Read file
            if file.name.endswith('.csv'):
                df = pd.read_csv(file.name)
            elif file.name.endswith('.txt'):
                with open(file.name, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                df = pd.DataFrame({'text': [line.strip() for line in lines if line.strip()]})
            else:
                return "Unsupported file format. Please use CSV or TXT.", ""
            
            if 'text' not in df.columns:
                return "CSV file must have a 'text' column", ""
            
            # Limit to 50 articles for demo
            df = df.head(50)
            
            # Process articles
            results = []
            for text in df['text']:
                if len(str(text).strip()) > 10:
                    result = self.predictor.predict(str(text))
                    results.append(result)
                else:
                    results.append({'label': 'UNKNOWN', 'confidence': 0.0})
            
            # Create summary
            fake_count = sum(1 for r in results if r['label'] == 'FAKE')
            real_count = len(results) - fake_count
            
            summary = f"""
## Batch Analysis Results
            
📊 **Summary:**
- Total articles analyzed: {len(results)}
- 🚨 Fake news detected: {fake_count} ({fake_count/len(results)*100:.1f}%)
- ✅ Legitimate news: {real_count} ({real_count/len(results)*100:.1f}%)
            
⚠️ **High-risk articles:** {sum(1 for r in results if r['label'] == 'FAKE' and r['confidence'] > 0.8)}
"""
            
            # Create detailed results
            detailed = "## Detailed Results\n\n"
            for i, (text, result) in enumerate(zip(df['text'], results), 1):
                emoji = "🚨" if result['label'] == 'FAKE' else "✅"
                confidence = result['confidence']
                preview = str(text)[:100] + "..." if len(str(text)) > 100 else str(text)
                detailed += f"**{i}. {emoji} {result['label']}** (Confidence: {confidence:.2%})\n"
                detailed += f"*{preview}*\n\n"
            
            return summary, detailed
            
        except Exception as e:
            self.logger.error(f"Batch analysis error: {str(e)}")
            return f"Error processing file: {str(e)}", ""
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        # Custom CSS for better styling
        css = """
        .fake-news { background-color: #ffebee !important; border-left: 5px solid #f44336 !important; }
        .real-news { background-color: #e8f5e8 !important; border-left: 5px solid #4caf50 !important; }
        .main-header { text-align: center; color: #1976d2; font-size: 2.5em; font-weight: bold; margin: 20px 0; }
        .subtitle { text-align: center; color: #666; font-size: 1.2em; margin-bottom: 30px; }
        .stats-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; }
        """
        
        with gr.Blocks(css=css, theme=gr.themes.Soft(), title="Fake News Detector") as interface:
            
            # Header
            gr.HTML("""
                <div class="main-header">🔍 Advanced Fake News Detector</div>
                <div class="subtitle">AI-Powered News Authenticity Analysis | Powered by RoBERTa</div>
            """)
            
            with gr.Tab("🔍 Single Article Analysis"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 📝 Enter News Article")
                        news_input = gr.Textbox(
                            placeholder="Paste your news article here... (minimum 10 characters)",
                            lines=8,
                            label="News Text"
                        )
                        
                        with gr.Row():
                            analyze_btn = gr.Button(
                                "🔍 Analyze Article",
                                variant="primary",
                                size="lg"
                            )
                            clear_btn = gr.Button("🗑️ Clear", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Quick Stats")
                        stats_plot = gr.Plot(
                            label="Prediction Statistics",
                            value=self.get_stats_plot()
                        )
                
                # Results section
                gr.Markdown("### 🎯 Analysis Results")
                with gr.Row():
                    with gr.Column():
                        result_status = gr.Textbox(
                            label="🎯 Detection Result",
                            interactive=False,
                            container=True
                        )
                        
                        result_interpretation = gr.Textbox(
                            label="📋 Interpretation",
                            interactive=False,
                            lines=2
                        )
                    
                    with gr.Column():
                        confidence_score = gr.Textbox(
                            label="📈 Confidence Score",
                            interactive=False
                        )
                        
                        processing_time = gr.Textbox(
                            label="⚡ Processing Info",
                            interactive=False
                        )
                
                # Recent predictions
                with gr.Row():
                    recent_predictions = gr.Markdown(
                        value=self.get_recent_predictions(),
                        label="Recent Activity"
                    )
            
            with gr.Tab("📁 Batch Analysis"):
                gr.Markdown("""
                    ### 📁 Upload Multiple Articles
                    Upload a CSV file with a 'text' column or a TXT file with one article per line.
                    *Maximum 50 articles per batch for demo purposes.*
                """)
                
                with gr.Row():
                    with gr.Column():
                        file_input = gr.File(
                            label="Upload File (CSV or TXT)",
                            file_types=[".csv", ".txt"]
                        )
                        batch_btn = gr.Button(
                            "📊 Analyze Batch",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column():
                        batch_summary = gr.Markdown(label="Summary")
                
                batch_details = gr.Markdown(label="Detailed Results")
            
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                    ## 🔍 About This Fake News Detector
                    
                    ### 🤖 Model Information
                    - **Model**: PRETRAINED RoBERTa Fake News Detector
                    - **Training Data**: 40,000+ real news articles  
                    - **Source**: jy46604790/Fake-News-Bert-Detect
                    - **Accuracy**: Proven on real-world data
                    - **Languages**: Primarily English
                    
                    ### 🎯 How It Works
                    1. **Text Processing**: The article is cleaned and tokenized
                    2. **Analysis**: Machine learning model analyzes linguistic patterns
                    3. **Confidence Scoring**: Provides probability-based confidence
                    4. **Real-time Results**: Instant feedback on authenticity
                    
                    ### ⚠️ Important Notes
                    - This tool uses an advanced machine learning model
                    - Trained on 40,000+ news articles
                    - Provides reliable accuracy for detection
                    - Human verification is always recommended
                    - Consider multiple sources for important news
                    
                    ### 🔧 Technical Details
                    - **Framework**: Transformers, PyTorch
                    - **Base Model**: RoBERTa fine-tuned for fake news
                    - **Interface**: Gradio
                    - **Deployment**: Production-ready with Docker support
                    
                    ### 📞 Support
                    For technical issues or questions, please refer to the documentation.
                """)
            
            # Event handlers
            analyze_btn.click(
                fn=self.predict_news,
                inputs=[news_input],
                outputs=[result_status, result_interpretation, confidence_score, processing_time]
            ).then(
                fn=lambda: (self.get_stats_plot(), self.get_recent_predictions()),
                outputs=[stats_plot, recent_predictions]
            )
            
            clear_btn.click(
                fn=lambda: ("", "", "", "", ""),
                outputs=[news_input, result_status, result_interpretation, confidence_score, processing_time]
            )
            
            batch_btn.click(
                fn=self.analyze_batch,
                inputs=[file_input],
                outputs=[batch_summary, batch_details]
            )
            
            # Note: Auto-refresh disabled for compatibility
            # interface.load(
            #     fn=lambda: self.get_stats_plot(),
            #     outputs=[stats_plot],
            #     every=30
            # )
        
        return interface
    
    def launch(self, share: bool = False, debug: bool = False):
        """Launch the web application"""
        if not self.model_loaded:
            self.logger.error("Cannot launch app: Model not loaded")
            return
        
        self.logger.info("Launching Fake News Detection Web App...")
        
        interface = self.create_interface()
        
        # Launch with custom settings
        interface.launch(
            share=True,
            debug=debug,
            server_name="0.0.0.0",  # Allow external connections
            server_port=7860,
            show_error=True,
            quiet=False,
            favicon_path=None,
            ssl_verify=False
        )


def main():
    """Main function to run the web application"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fake News Detection Web App')
    parser.add_argument('--share', action='store_true', help='Create public shareable link')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    # Create and launch app
    app = FakeNewsWebApp()
    
    print("\n" + "="*60)
    print("🔍 FAKE NEWS DETECTOR - WEB APPLICATION")
    print("="*60)
    print("🚀 Starting application...")
    print("📡 Server will be available at: http://localhost:7860")
    if args.share:
        print("🌐 Public link will be generated...")
    print("="*60 + "\n")
    
    try:
        app.launch(share=args.share, debug=args.debug)
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()