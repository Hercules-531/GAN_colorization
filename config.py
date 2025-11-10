"""
Configuration file for Image Colorization with Pix2Pix GAN
Modify these settings before training
"""

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Dataset
TRAIN_DIR = r'C:\Users\shiva\OneDrive\Desktop\DLI_LAB_PROJECT\GAN_color_images'

# Training Hyperparameters
NUM_EPOCHS = 100          # Number of training epochs (100 recommended)
BATCH_SIZE = 32           # Batch size (32 optimized for mixed precision)
LEARNING_RATE = 0.0002    # Learning rate for both G and D
BETA1 = 0.5               # Adam beta1
BETA2 = 0.999             # Adam beta2
LAMBDA_L1 = 100           # Weight for L1 loss (pixel-wise reconstruction)

# Image Settings
IMG_SIZE = 256            # Image resolution (256x256)

# Checkpointing
CHECKPOINT_DIR = 'checkpoints'
SAVE_INTERVAL = 10        # Save checkpoint every N epochs

# Data Loading
NUM_WORKERS = 4           # Number of data loading workers
PIN_MEMORY = True         # Pin memory for faster GPU transfer

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

# Generator (U-Net)
GEN_IN_CHANNELS = 1       # Input: L channel (grayscale)
GEN_OUT_CHANNELS = 2      # Output: ab channels (color)

# Discriminator (PatchGAN)
DISC_IN_CHANNELS = 3      # Input: L + ab channels

# =============================================================================
# INFERENCE CONFIGURATION
# =============================================================================

# Model Path
GENERATOR_PATH = 'checkpoints/generator_final.pth'

# Gradio Settings
SHARE_GRADIO = True       # Create public sharing link
SERVER_NAME = '0.0.0.0'   # Listen on all network interfaces
SERVER_PORT = 7860        # Default Gradio port

# =============================================================================
# ADVANCED SETTINGS (MODIFY WITH CAUTION)
# =============================================================================

# Mixed Precision Training
USE_AMP = True            # Automatic Mixed Precision (recommended for GPU)

# Weight Initialization
INIT_TYPE = 'normal'      # Weight initialization type
INIT_GAIN = 0.02          # Initialization standard deviation

# GradCAM
GRADCAM_TARGET_LAYER_NAME = 'up7'  # Target layer for attention visualization

# Dataset Subset (for testing)
USE_SUBSET = None         # None = use all images, or specify number (e.g., 1000)

# =============================================================================
# PERFORMANCE TUNING
# =============================================================================

# For RTX 3080 / RTX 3090 (24GB VRAM)
# BATCH_SIZE = 32
# NUM_WORKERS = 4
# IMG_SIZE = 256

# For RTX 2060 / RTX 3060 (8-12GB VRAM)
# BATCH_SIZE = 16
# NUM_WORKERS = 2
# IMG_SIZE = 256

# For GTX 1080 Ti (11GB VRAM)
# BATCH_SIZE = 12
# NUM_WORKERS = 2
# IMG_SIZE = 256

# For CPU Training (Not Recommended)
# BATCH_SIZE = 4
# NUM_WORKERS = 0
# NUM_EPOCHS = 10  # Just for testing

# =============================================================================
# EXPERIMENTAL SETTINGS
# =============================================================================

# Different Loss Weights (experiment with these)
# LAMBDA_L1 = 50    # Less pixel-wise accuracy, more creativity
# LAMBDA_L1 = 200   # More pixel-wise accuracy, less creativity

# Different Learning Rates
# LEARNING_RATE = 0.0001  # Slower, more stable
# LEARNING_RATE = 0.0005  # Faster, might be unstable

# Different Image Sizes
# IMG_SIZE = 128   # Faster training, lower quality
# IMG_SIZE = 512   # Slower training, higher quality (needs more VRAM)

# =============================================================================
# NOTES
# =============================================================================

"""
1. BATCH_SIZE:
   - Higher = faster training but more memory
   - Lower = slower training but less memory
   - 32 is optimal for mixed precision on modern GPUs

2. LAMBDA_L1:
   - Higher = more accurate colors, less creative
   - Lower = more creative, less accurate
   - 100 is standard for Pix2Pix

3. NUM_EPOCHS:
   - 10-20: Quick test
   - 50: Decent results
   - 100: Good results (recommended)
   - 200+: Best results (diminishing returns)

4. IMG_SIZE:
   - 128: Fast, low quality
   - 256: Standard, good quality
   - 512: Slow, high quality (8x more compute)

5. Mixed Precision (USE_AMP):
   - Always enable on CUDA GPUs
   - 2x faster training
   - 40% less memory
   - No quality loss
"""
