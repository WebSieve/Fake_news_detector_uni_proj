
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
    print("🚀 STARTING FAKE NEWS DETECTION PIPELINE")
    print("=" * 60)
    
    # Step 1: Setup
    print("Setting up environment...")
    if not Path('config.py').exists():
        print("❌ config.py not found. Run the notebook cells first.")
        return
    
    # Step 2: Test simple inference
    if run_command("python simple_inference.py", "Testing Simple Inference"):
        print("✅ Simple inference working!")
    
    # Step 3: Prepare data
    if run_command("python data_preparation.py", "Preparing Dataset"):
        print("✅ Data preparation complete!")
    else:
        print("❌ Data preparation failed. Continuing anyway...")
    
    # Step 4: Check if we can run training (this would take time)
    print("\n📝 TRAINING INSTRUCTIONS:")
    print("To train the model, run: python model_training.py")
    print("This will take 30-60 minutes depending on your hardware.")
    print("\n📝 EVALUATION INSTRUCTIONS:")
    print("After training, run: python evaluation.py")
    print("\n📝 WEB APP INSTRUCTIONS:")  
    print("To launch the web app, run: python app.py")
    
    print("\n🎉 PIPELINE SETUP COMPLETE!")
    print("All files are ready for execution.")

if __name__ == "__main__":
    main()
