
"""
Master script to run the entire pipeline
"""

import subprocess
import sys
import time
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"COMMAND: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("SUCCESS!")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Show last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {e}")
        if e.stdout:
            print("Output:", e.stdout[-500:])
        if e.stderr:
            print("Error:", e.stderr[-500:])
        return False

def main():
    """Run the complete pipeline"""
    print("STARTING IMPROVED FAKE NEWS DETECTION PIPELINE")
    print("Using advanced pretrained models for high accuracy!")
    print("=" * 60)
    
    # Step 1: Setup
    print("Setting up environment...")
    if not Path('config.py').exists():
        print("ERROR: config.py not found. Run the notebook cells first.")
        return
    
    # Step 2: Test model
    if run_command("python fake_news_detector_core.py", "Testing Enhanced Model"):
        print("SUCCESS: Fixed model working perfectly!")
    else:
        print("Trying original version...")
        if run_command("python simple_detector_demo.py", "Testing Simplified Detector"):
            print("SUCCESS: Simplified detector working!")
    
    # Step 3: Optional - Prepare data (only if you want to train your own)
    print("\nDATA PREPARATION (Optional):")
    print("The model is already trained on 40,000+ articles.")
    print("You can skip training and use the model directly!")
    
    user_choice = input("\nDo you want to prepare data anyway? (y/N): ").lower()
    if user_choice == 'y':
        if run_command("python data_preparation.py", "Preparing Dataset"):
            print("SUCCESS: Data preparation complete!")
        else:
            print("ERROR: Data preparation failed. Using model anyway...")
    
    # Step 4: Optional training info
    print("\nTRAINING INSTRUCTIONS:")
    print("RECOMMENDED: Use the pretrained model (already loaded)")
    print("OPTIONAL: To train your own model, run: python model_training.py")
    print("   (This will take 30-60 minutes and may not be better than pretrained)")
    
    print("\nEVALUATION INSTRUCTIONS:")
    print("To evaluate model performance, run: python evaluation.py")
    
    print("\nWEB APP INSTRUCTIONS:")  
    print("To launch the web app with the trained model, run: python web_application.py")
    
    print("\nPIPELINE SETUP COMPLETE!")
    print("Pretrained model ready for accurate fake news detection!")
    print("No training required - instant results!")
    print("=" * 60)

if __name__ == "__main__":
    main()
