# Quick Start Guide - Image Colorization

## 🚀 Installation (5 minutes)

1. **Install Python 3.8+** (if not already installed)

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Or install individually:
   ```bash
   pip install torch torchvision gradio numpy scikit-image Pillow matplotlib tqdm
   ```

## 🎯 Training (8-10 hours on GPU)

### Step 1: Open colozier_gan.py

Find the training section (around line 657) and **UNCOMMENT** these lines:

```python
# Training configuration
TRAIN_DIR = r'C:\Users\shiva\OneDrive\Desktop\DLI_LAB_PROJECT\GAN_color_images'
NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.0002
LAMBDA_L1 = 100
CHECKPOINT_DIR = 'checkpoints'

# Start training
train_pix2pix(
    train_dir=TRAIN_DIR,
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    lambda_L1=LAMBDA_L1,
    save_interval=10,
    checkpoint_dir=CHECKPOINT_DIR
)
```

### Step 2: Run Training

```bash
python colozier_gan.py
```

**What to expect:**
- Progress bars showing G_loss, D_loss, L1 loss
- Checkpoints saved every 10 epochs
- ~8-10 hours on RTX 3080
- ~4-5 hours on RTX 4090

**Output:**
```
Training on: cuda
Found 7129 images in C:\Users\shiva\OneDrive\Desktop\DLI_LAB_PROJECT\GAN_color_images

Training Configuration:
- Epochs: 100
- Batch Size: 32
- Training Images: 7129
- Learning Rate: 0.0002
- Lambda L1: 100
- Mixed Precision: True

Epoch 1/100: 100%|███████| 223/223 [01:30<00:00, G_loss: 12.3456, D_loss: 0.4567, L1: 0.1234]
...
```

### Step 3: Quick Test (Optional)

Want to test before full training? Change:
```python
NUM_EPOCHS = 10  # Instead of 100
```

## 🎨 Inference (Immediate after training)

### Option A: Automatic Launch

After training completes, the Gradio interface launches automatically.

### Option B: Manual Launch

1. **Ensure training is complete** (checkpoints/generator_final.pth exists)

2. **Comment out training section** in colozier_gan.py

3. **Run:**
   ```bash
   python colozier_gan.py
   ```

4. **Open browser** at shown URL (e.g., http://127.0.0.1:7860)

5. **Share publicly** using the provided share link

## 🖼️ Using the Interface

### Input
- Upload any grayscale image
- Or upload color image (will auto-convert to grayscale)

### Output (3 images)
1. **Colorized Image**: Your grayscale image with predicted colors
2. **Red-Green Attention**: Shows where model focused for red/green colors
3. **Blue-Yellow Attention**: Shows where model focused for blue/yellow colors

### Attention Map Legend
- 🔴 **Red areas**: High attention (model focused here)
- 🟡 **Yellow areas**: Medium attention
- 🔵 **Blue areas**: Low attention (less important)

## 📊 Monitor Training

### Loss Values
- **G_loss**: Generator loss (should decrease)
- **D_loss**: Discriminator loss (should stabilize around 0.5)
- **L1**: Pixel reconstruction loss (should decrease)

### Good Training Signs
✅ G_loss decreases over epochs
✅ D_loss stabilizes (not going to 0 or infinity)
✅ L1 loss consistently decreasing
✅ No NaN or Inf values

### Bad Training Signs
❌ D_loss = 0 (discriminator too strong)
❌ G_loss exploding (learning rate too high)
❌ NaN values (reduce learning rate)
❌ Out of memory (reduce batch size)

## 🔧 Common Issues & Solutions

### Issue 1: CUDA Out of Memory
```python
# In colozier_gan.py, change:
BATCH_SIZE = 16  # or 8 instead of 32
```

### Issue 2: Model checkpoint not found
```
ERROR: Model checkpoint not found at checkpoints/generator_final.pth
```
**Solution:** Train the model first (uncomment training section)

### Issue 3: Slow training
```python
# Quick test with fewer epochs
NUM_EPOCHS = 10
```

### Issue 4: No GPU available
```
Training on: cpu
```
**Solution:** Install CUDA-enabled PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue 5: Import errors
```bash
# Reinstall all dependencies
pip install -r requirements.txt --upgrade
```

## 📂 File Structure After Training

```
DLI_LAB_PROJECT/
│
├── colozier_gan.py          ✅ Main code
├── requirements.txt          ✅ Dependencies
├── README.md                 ✅ Full documentation
├── QUICKSTART.md             ✅ This file
│
├── GAN_color_images/         ✅ Your dataset (7,129 images)
│
└── checkpoints/              ✅ Created during training
    ├── generator_epoch_10.pth
    ├── generator_epoch_20.pth
    ├── ...
    ├── generator_final.pth    ⭐ Use this for inference
    └── discriminator_final.pth
```

## ⚡ Performance Tips

### For Faster Training
```python
# Enable mixed precision (already enabled by default)
# Uses FP16 on GPU = 2× faster
```

### For Better Quality
```python
# Train longer
NUM_EPOCHS = 200

# Or increase L1 weight
LAMBDA_L1 = 200
```

### For Less Memory
```python
# Reduce batch size
BATCH_SIZE = 16

# Or reduce image size
# In LandscapeColorizationDataset:
img_size=128  # instead of 256
```

## 🎓 Understanding the Output

### Colorized Image
- Full color version of your grayscale input
- Colors predicted based on learned patterns
- Best results on landscape/nature images

### Attention Maps
- **Purpose**: Show where model "looks" for color prediction
- **Red areas**: Model pays attention here
- **Blue areas**: Model ignores these areas
- **Use case**: Understand model decision-making

### Example Interpretation
If colorizing a landscape:
- **Sky attention**: High in blue-yellow map
- **Grass attention**: High in red-green map
- **Shadows**: Low attention in both maps

## 📈 Training Progress Timeline

| Epoch | G_loss | D_loss | L1 Loss | Quality |
|-------|--------|--------|---------|---------|
| 1-10  | ~20-15 | ~0.8   | ~0.3    | Poor    |
| 11-30 | ~15-10 | ~0.6   | ~0.2    | Fair    |
| 31-60 | ~10-8  | ~0.5   | ~0.15   | Good    |
| 61-100| ~8-6   | ~0.5   | ~0.10   | Great   |

## 🎯 Next Steps

After successful training:

1. ✅ Test with your own grayscale images
2. ✅ Share Gradio link with others
3. ✅ Experiment with different images
4. ✅ Compare attention maps across images
5. ✅ Fine-tune on specific image types

## 💡 Pro Tips

1. **Save interesting results**: Right-click → Save image in Gradio
2. **Batch processing**: Modify code to loop through folder
3. **Compare checkpoints**: Load different epoch checkpoints
4. **Attention analysis**: Study what features model focuses on
5. **Dataset specific**: Best results on images similar to training data

## 🆘 Need Help?

1. Check full README.md for detailed documentation
2. Review error messages carefully
3. Verify all dependencies installed
4. Ensure CUDA/GPU properly configured
5. Check disk space for checkpoints (~2GB)

## ✅ Checklist Before Training

- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] CUDA/GPU available (check with `torch.cuda.is_available()`)
- [ ] Dataset folder exists (GAN_color_images/)
- [ ] At least 10GB free disk space
- [ ] Training section uncommented
- [ ] TRAIN_DIR path is correct

## ✅ Checklist Before Inference

- [ ] Training completed
- [ ] generator_final.pth exists in checkpoints/
- [ ] Training section commented out
- [ ] GENERATOR_PATH points to correct file
- [ ] Browser ready for Gradio interface

---

**Ready? Start training now!** 🚀

```bash
python colozier_gan.py
```

**Questions?** Check README.md for full details.
