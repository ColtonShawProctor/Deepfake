#!/usr/bin/env python3
"""
Check training progress for all models
"""

import os
import sys
import time
from pathlib import Path

def check_training_progress():
    """Check the progress of all training processes"""
    
    print("🔍 Checking Training Progress")
    print("=" * 50)
    
    # Check for model files
    models_dir = Path("models")
    if models_dir.exists():
        print("\n📁 Model Files:")
        for model_file in models_dir.glob("*.pth"):
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"  {model_file.name:25} | {size_mb:6.1f} MB")
    else:
        print("\n📁 No models directory found")
    
    # Check for training logs or output files
    print("\n📊 Training Status:")
    
    # Check if training scripts are still running
    import subprocess
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if 'train_missing_models.py' in result.stdout:
            print("  ✅ MesoNet/Xception training: RUNNING")
        else:
            print("  ⏸️  MesoNet/Xception training: NOT RUNNING")
            
        if 'fix_resnet_overfitting.py' in result.stdout:
            print("  ✅ ResNet retraining: RUNNING")
        else:
            print("  ⏸️  ResNet retraining: NOT RUNNING")
    except:
        print("  ❓ Could not check process status")
    
    # Check for any new model files
    print("\n🆕 Recent Model Files:")
    if models_dir.exists():
        recent_files = []
        for model_file in models_dir.glob("*.pth"):
            if model_file.stat().st_mtime > time.time() - 3600:  # Last hour
                recent_files.append(model_file)
        
        if recent_files:
            for file in recent_files:
                age_minutes = (time.time() - file.stat().st_mtime) / 60
                print(f"  {file.name:25} | {age_minutes:5.1f} min ago")
        else:
            print("  No new model files in the last hour")
    
    # Check expected models
    print("\n🎯 Expected Models:")
    expected_models = [
        "mesonet_weights.pth",
        "xception_weights.pth", 
        "resnet_weights_fixed.pth"
    ]
    
    for model in expected_models:
        model_path = models_dir / model
        if model_path.exists():
            print(f"  ✅ {model}")
        else:
            print(f"  ❌ {model}")

if __name__ == "__main__":
    check_training_progress()
