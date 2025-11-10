"""
SYSTEM CHECK SCRIPT
Run this before training to verify your setup
"""

import sys
import os

def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"  ✅ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"  ❌ Python {version.major}.{version.minor}.{version.micro} (Need 3.8+)")
        return False

def check_packages():
    """Check required packages"""
    print("\nChecking required packages...")
    
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'skimage': 'scikit-image',
        'matplotlib': 'Matplotlib',
        'gradio': 'Gradio',
        'tqdm': 'tqdm'
    }
    
    all_installed = True
    
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"  ✅ {name} installed")
        except ImportError:
            print(f"  ❌ {name} NOT installed")
            all_installed = False
    
    return all_installed

def check_cuda():
    """Check CUDA availability"""
    print("\nChecking CUDA/GPU...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            
            print(f"  ✅ CUDA available")
            print(f"  ✅ CUDA version: {cuda_version}")
            print(f"  ✅ GPU count: {device_count}")
            print(f"  ✅ GPU 0: {gpu_name}")
            
            # Check memory
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  ✅ GPU Memory: {total_memory:.1f} GB")
            
            if total_memory >= 8:
                print(f"  ✅ Sufficient memory for batch size 32")
                return True
            elif total_memory >= 6:
                print(f"  ⚠️  Limited memory - consider batch size 16")
                return True
            else:
                print(f"  ⚠️  Low memory - batch size 8 recommended")
                return True
        else:
            print(f"  ⚠️  CUDA not available - will use CPU")
            print(f"  ⚠️  Training will be VERY slow on CPU")
            return False
            
    except ImportError:
        print(f"  ❌ PyTorch not installed - cannot check CUDA")
        return False

def check_dataset():
    """Check dataset directory"""
    print("\nChecking dataset...")
    
    dataset_path = 'GAN_color_images'
    
    if not os.path.exists(dataset_path):
        print(f"  ❌ Dataset directory not found: {dataset_path}")
        return False
    
    # Count images
    image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    image_count = len(image_files)
    
    if image_count == 0:
        print(f"  ❌ No images found in {dataset_path}")
        return False
    
    print(f"  ✅ Dataset directory found")
    print(f"  ✅ Found {image_count} images")
    
    if image_count < 100:
        print(f"  ⚠️  Small dataset - results may not be optimal")
    elif image_count < 1000:
        print(f"  ✅ Decent dataset size")
    else:
        print(f"  ✅ Large dataset - excellent!")
    
    return True

def check_disk_space():
    """Check available disk space"""
    print("\nChecking disk space...")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        
        free_gb = free / (1024**3)
        print(f"  ℹ️  Free space: {free_gb:.1f} GB")
        
        if free_gb >= 20:
            print(f"  ✅ Plenty of space for checkpoints")
            return True
        elif free_gb >= 10:
            print(f"  ✅ Sufficient space")
            return True
        else:
            print(f"  ⚠️  Low disk space - may run out during training")
            return False
            
    except Exception as e:
        print(f"  ⚠️  Could not check disk space: {e}")
        return True

def check_files():
    """Check required files"""
    print("\nChecking project files...")
    
    required_files = [
        'colozier_gan.py',
        'requirements.txt',
        'README.md',
        'QUICKSTART.md'
    ]
    
    all_present = True
    
    for filename in required_files:
        if os.path.exists(filename):
            print(f"  ✅ {filename}")
        else:
            print(f"  ❌ {filename} NOT FOUND")
            all_present = False
    
    return all_present

def print_summary(results):
    """Print summary and recommendations"""
    print("\n" + "=" * 70)
    print("SYSTEM CHECK SUMMARY")
    print("=" * 70)
    
    all_pass = all(results.values())
    
    if all_pass:
        print("\n✅ ALL CHECKS PASSED!")
        print("\nYour system is ready for training.")
        print("\nNext steps:")
        print("1. Run: python train.py")
        print("2. Or uncomment training section in colozier_gan.py")
        print("3. Training will take 8-10 hours on GPU (RTX 3080)")
        print("4. Monitor progress via progress bars")
        print("5. Checkpoints saved every 10 epochs")
        
    else:
        print("\n⚠️  SOME CHECKS FAILED")
        print("\nRequired actions:")
        
        if not results['python']:
            print("- Upgrade Python to 3.8 or higher")
        
        if not results['packages']:
            print("- Install missing packages: pip install -r requirements.txt")
        
        if not results['dataset']:
            print("- Ensure GAN_color_images folder exists with images")
        
        if not results['files']:
            print("- Restore missing project files")
        
        if not results['cuda']:
            print("- Optional: Install CUDA for GPU acceleration")
            print("  (Training possible on CPU but very slow)")
    
    print("\n" + "=" * 70)

def main():
    """Run all checks"""
    print("=" * 70)
    print("IMAGE COLORIZATION - SYSTEM CHECK")
    print("=" * 70)
    print()
    
    results = {
        'python': check_python_version(),
        'packages': check_packages(),
        'cuda': check_cuda(),
        'dataset': check_dataset(),
        'disk': check_disk_space(),
        'files': check_files()
    }
    
    print_summary(results)
    
    # Return exit code
    return 0 if all(results.values()) else 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
