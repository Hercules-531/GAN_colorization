"""
EASY TRAINING SCRIPT
Run this file to start training immediately
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import main components
from colozier_gan import train_pix2pix
import torch

def main():
    """Easy training launcher"""
    
    print("=" * 70)
    print("IMAGE COLORIZATION - TRAINING LAUNCHER")
    print("=" * 70)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU Available: {gpu_name}")
        print(f"✅ CUDA Version: {torch.version.cuda}")
        print(f"✅ Mixed Precision: ENABLED")
    else:
        device = torch.device('cpu')
        print("⚠️  GPU Not Available - Training on CPU")
        print("⚠️  WARNING: CPU training is VERY SLOW")
        print("⚠️  Consider using Google Colab or GPU instance")
    
    print("=" * 70)
    
    # Training configuration
    TRAIN_DIR = r'GAN_color_images'  # Relative path
    NUM_EPOCHS = 100
    BATCH_SIZE = 32 if torch.cuda.is_available() else 4
    LEARNING_RATE = 0.0002
    LAMBDA_L1 = 100
    CHECKPOINT_DIR = 'checkpoints'
    
    # Display configuration
    print("\nTraining Configuration:")
    print(f"  Dataset: {TRAIN_DIR}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Lambda L1: {LAMBDA_L1}")
    print(f"  Checkpoint Dir: {CHECKPOINT_DIR}")
    
    # Check dataset
    if not os.path.exists(TRAIN_DIR):
        print(f"\n❌ ERROR: Dataset directory not found: {TRAIN_DIR}")
        print("Please ensure GAN_color_images folder exists")
        return
    
    # Count images
    image_count = len([f for f in os.listdir(TRAIN_DIR) if f.endswith('.jpg')])
    print(f"\n✅ Found {image_count} images in dataset")
    
    # Confirm start
    print("\n" + "=" * 70)
    print("Ready to start training!")
    print("=" * 70)
    
    if torch.cuda.is_available():
        print(f"\nEstimated time: ~8-10 hours on RTX 3080")
        print("Checkpoints will be saved every 10 epochs")
    else:
        print(f"\n⚠️  CPU training will take VERY LONG")
        print("Consider reducing NUM_EPOCHS to 10 for testing")
    
    print("\nPress Ctrl+C to cancel training at any time")
    print("Training will start in 3 seconds...\n")
    
    import time
    time.sleep(3)
    
    # Start training
    try:
        train_pix2pix(
            train_dir=TRAIN_DIR,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            lambda_L1=LAMBDA_L1,
            save_interval=10,
            checkpoint_dir=CHECKPOINT_DIR
        )
        
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\nModel saved to: {CHECKPOINT_DIR}/generator_final.pth")
        print("\nNext steps:")
        print("1. Run: python colozier_gan.py")
        print("2. This will launch the Gradio interface")
        print("3. Upload grayscale images to colorize")
        print("4. View attention maps for explainability")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Checkpoints saved so far can be found in:", CHECKPOINT_DIR)
        
    except Exception as e:
        print(f"\n\n❌ Training failed with error:")
        print(f"{str(e)}")
        print("\nPlease check:")
        print("1. Dataset path is correct")
        print("2. Sufficient disk space available")
        print("3. Dependencies are installed")
        print("4. CUDA drivers are up to date (if using GPU)")


if __name__ == '__main__':
    main()
